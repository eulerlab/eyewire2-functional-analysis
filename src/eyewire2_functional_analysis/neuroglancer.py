import copy

import matplotlib.colors as mcolors
from matplotlib import pyplot as plt


def add_layer(ids, layer_name, colormap, data_template, visible=True, hidden_ids=None, flat_color=None):
    layer = copy.deepcopy(data_template['layers'][1])

    # The viewer uses `segments` for visible and `hiddenSegments` for hidden
    layer['segments'] = ids
    if hidden_ids:
        layer['hiddenSegments'] = hidden_ids

    # Colors for all ids (visible + hidden). Either one flat color for every
    # id (e.g. to color-code by field), or a gradient sampled from `colormap`.
    all_ids = ids + (hidden_ids or [])
    if flat_color is not None:
        colors = [mcolors.to_hex(flat_color)] * len(all_ids)
    else:
        cmap = plt.get_cmap(colormap)
        colors = [
            mcolors.to_hex(cmap((i + len(all_ids) // 2) / max(1, len(all_ids) * 2)))
            for i in range(len(all_ids))
        ]
    layer['segmentColors'] = dict(zip(all_ids, colors))

    layer['name'] = layer_name
    layer['visible'] = visible
    data_template['layers'].append(layer)


def _parse_coord(coord_str):
    """Parse a "x, y, z" string (as used in the EM-2p mapping CSVs) into an [x, y, z] int list."""
    return [int(v) for v in str(coord_str).split(',')]


def line_annotation(point_a, point_b, description=''):
    """Build a Neuroglancer line-annotation dict between two [x, y, z] voxel coordinates."""
    return {'type': 'line', 'id': '', 'pointA': list(point_a), 'pointB': list(point_b), 'description': description}


def point_annotation(point, description=''):
    """Build a Neuroglancer point-annotation dict at an [x, y, z] voxel coordinate."""
    return {'type': 'point', 'id': '', 'point': list(point), 'description': description}


def add_annotation_layer(annotations, layer_name, color, data_template, visible=True, annotation_layer_idx=2):
    """
    Add an annotation layer by cloning the empty annotation-layer template at
    `data_template['layers'][annotation_layer_idx]` (this carries the coordinate
    space/transform already set up for the viewer, so annotation points/lines
    can be given directly in the same raw voxel units as the segments) and
    filling it with `annotations` (a list of dicts, e.g. built by
    `line_annotation`/`point_annotation`).

    Parameters
    ----------
    annotations : list of dict
    layer_name : str
    color : matplotlib color
        Single color used for every annotation in the layer.
    data_template : dict
        Neuroglancer state dict to append the layer to (mutated in place).
    visible : bool
    annotation_layer_idx : int
        Index of the empty annotation layer to clone from `data_template['layers']`.
    """
    layer = copy.deepcopy(data_template['layers'][annotation_layer_idx])
    layer['annotations'] = annotations
    layer['annotationColor'] = mcolors.to_hex(color)
    layer['name'] = layer_name
    layer['visible'] = visible
    data_template['layers'].append(layer)


def spawn_field_mapping_link(client, df_map, df_estimates, colormap='tab10',
                              state_id=4697418519019520, annotation_layer_idx=2):
    """
    Build a Neuroglancer link with, per 2p field, up to 3 layers:
      - "<field> segments": segmentation layer of the matched cells' "Latest NucID"
      - "<field> lines": line annotations from the estimated EM coordinate to the
        real "Nuc Coords", for ROIs that are matched to an EM cell
      - "<field> unmatched": point annotations at the estimated EM coordinate,
        for ROIs with no EM match yet (omitted if a field has none). ROIs
        flagged 'not_a_cell' (confirmed not a real cell, e.g. an overlay of
        two cells) are excluded from this layer, since there's nothing to
        find for them.

    All layers for a given field share one color.

    Parameters
    ----------
    client : CAVEclient
        Client used to fetch/upload the Neuroglancer state.
    df_map : pd.DataFrame
        Must have '2p-Field', '2p-ROI', 'Latest NucID', 'Nuc Coords' columns
        (as in the EM-2p mapping CSV).
    df_estimates : pd.DataFrame
        Must have '2p-Field', '2p-ROI', 'matched', 'Estimated EM Coords' columns
        (as produced by em-2p-mapping.py). An optional 'not_a_cell' boolean
        column excludes those ROIs from the "unmatched" layer.
    colormap : str
        Matplotlib colormap name used to assign one color per field.
    state_id : int
        Neuroglancer state to use as a template (cloned, not modified in place).
        Must have a segmentation layer at index 1 and an empty annotation
        layer at `annotation_layer_idx` (like `add_layer` and
        `add_annotation_layer` expect, respectively).
    annotation_layer_idx : int
        Index of the empty annotation layer to clone for the "lines" and
        "unmatched" layers.

    Returns
    -------
    str
        The new Neuroglancer link.
    """
    # Only pull in columns df_estimates doesn't already have (it may already
    # carry 'Nuc Coords' itself), to avoid the merge suffixing both away.
    map_cols = ['2p-Field', '2p-ROI'] + [
        c for c in ('Latest NucID', 'Nuc Coords') if c not in df_estimates.columns
    ]
    df = df_estimates.merge(df_map[map_cols], on=['2p-Field', '2p-ROI'], how='left')
    if 'not_a_cell' not in df.columns:
        df['not_a_cell'] = False

    data_template = client.state.get_state_json(state_id)

    fields = sorted(df['2p-Field'].unique())
    cmap = plt.get_cmap(colormap)

    for i, field in enumerate(fields):
        color = cmap(i / max(1, len(fields) - 1))
        df_field = df[df['2p-Field'] == field]
        df_matched = df_field[df_field['matched']]
        df_unmatched = df_field[(~df_field['matched']) & (~df_field['not_a_cell'])]

        nuc_ids = df_matched['Latest NucID'].dropna().astype('int64').astype(str).tolist()
        add_layer(nuc_ids, f'{field} segments', colormap, data_template, visible=True, flat_color=color)

        line_annotations = [
            line_annotation(_parse_coord(est), _parse_coord(real), description=f'2p-ROI {roi_id}')
            for est, real, roi_id in zip(
                df_matched['Estimated EM Coords'], df_matched['Nuc Coords'], df_matched['2p-ROI'],
            )
        ]
        add_annotation_layer(
            line_annotations, f'{field} lines', color, data_template, visible=True,
            annotation_layer_idx=annotation_layer_idx,
        )

        point_annotations = [
            point_annotation(_parse_coord(est), description=f'2p-ROI {roi_id}')
            for est, roi_id in zip(df_unmatched['Estimated EM Coords'], df_unmatched['2p-ROI'])
        ]
        if point_annotations:
            add_annotation_layer(
                point_annotations, f'{field} unmatched', color, data_template, visible=True,
                annotation_layer_idx=annotation_layer_idx,
            )

    link_template = "https://spelunker.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/"
    new_id = client.state.upload_state_json(data_template)
    return f'{link_template}{new_id}'


def spawn_example_cells_link(client, example_cells, df_map, colormap='tab10',
                              state_id=4697418519019520, annotation_layer_idx=2):
    """
    Build a Neuroglancer link with one segmentation layer per example cell
    (so its full EM reconstruction, e.g. an axon, can be traced), plus, per
    2p field, a point-annotation layer marking every matched EM cell's real
    "Nuc Coords" (colored by field, as in `spawn_field_mapping_link`) -- so
    an example cell's true EM position can be checked against its field's
    full matched-cell point cloud.

    Parameters
    ----------
    client : CAVEclient
        Client used to fetch/upload the Neuroglancer state.
    example_cells : dict
        Maps a label (used as the layer name, e.g. "GCL0: RGC") to that
        cell's `("Latest NucID", "Latest SegID")` -- both are shown in the
        cell's segmentation layer, since "Latest NucID" (the nucleus
        detection) can lag behind "Latest SegID" (the up-to-date root ID
        covering the cell's full current proofread extent, e.g. after its
        axon was traced further).
    df_map : pd.DataFrame
        Must have '2p-Field', '2p-ROI', 'Nuc Coords' columns (as in the
        EM-2p mapping CSV).
    colormap : str
        Matplotlib colormap name used to assign one color per field.
    state_id : int
        Neuroglancer state to use as a template (cloned, not modified in
        place). Must have a segmentation layer at index 1 and an empty
        annotation layer at `annotation_layer_idx`.
    annotation_layer_idx : int
        Index of the empty annotation layer to clone for the per-field
        point-annotation layers.

    Returns
    -------
    str
        The new Neuroglancer link.
    """
    data_template = client.state.get_state_json(state_id)

    example_cmap = plt.get_cmap('Set1')
    for i, (label, (nuc_id, seg_id)) in enumerate(example_cells.items()):
        color = example_cmap(i / max(1, len(example_cells) - 1))
        ids = sorted({str(int(nuc_id)), str(int(seg_id))})
        add_layer(ids, f'example: {label}', colormap, data_template, visible=True, flat_color=color)

    df_valid = df_map[df_map['Nuc Coords'].notna()]
    fields = sorted(df_valid['2p-Field'].unique())
    field_cmap = plt.get_cmap(colormap)

    for i, field in enumerate(fields):
        color = field_cmap(i / max(1, len(fields) - 1))
        df_field = df_valid[df_valid['2p-Field'] == field]
        point_annotations = [
            point_annotation(_parse_coord(coord), description=f'2p-ROI {roi_id}')
            for coord, roi_id in zip(df_field['Nuc Coords'], df_field['2p-ROI'])
        ]
        add_annotation_layer(
            point_annotations, f'{field} matched cells', color, data_template, visible=True,
            annotation_layer_idx=annotation_layer_idx,
        )

    link_template = "https://spelunker.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/"
    new_id = client.state.upload_state_json(data_template)
    return f'{link_template}{new_id}'


def spawn_link(client, ids, names, colors, visibles, hidden_ids=None,
               state_id=6631492699553792):
    # https://spelunker.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/6631492699553792

    data_template = client.state.get_state_json(state_id)

    for i, (color, id_list, name, visible) in enumerate(zip(colors, ids, names, visibles)):
        h_ids = hidden_ids[i] if hidden_ids is not None else None
        add_layer(id_list, name, color, data_template, visible=visible, hidden_ids=h_ids)

    link_template = "https://spelunker.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/"
    new_id = client.state.upload_state_json(data_template)
    new_link = f'{link_template}{new_id}'

    return new_link


    
def show_layers(data):

    """
    Print the available layers in a state dictionary.

    Parameters
    ----------
    data : dict
        Dictionary containing a 'layers' key with layer information.

    Returns
    -------
    nb_layers : int
        Number of layers found in the state.
    """

    
    nb_layers = len(data['layers'])
    for layer in range(nb_layers):
        name =  data['layers'][layer]['name']
        print(f'layer {layer} :{name}')

    return nb_layers