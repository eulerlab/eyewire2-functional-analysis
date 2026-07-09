import os

import numpy as np
import pandas as pd
from pathlib import Path

from eyewire2_functional_analysis.io import restore_numpy_arrays


HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(HERE, "..", "..", "data")
DATA_2P = os.path.join(DATA_ROOT, "data-2p")
DATA_SS = os.path.join(DATA_ROOT, "spreadsheets")
DATA_SWC = os.path.join(DATA_ROOT, "swc")
DATA_REGISTRATION = os.path.join(DATA_ROOT, "registration")

MAIN_ALL_CELLS_SHEET = "Eyewire II Proofread Cells Main List - All Cells 2026-07-03.csv"
MAP_SHEET = "Eyewire II Proofread Cells Main List - EM-2p-mapping 2026-07-08e v2-final.csv"
EM_2P_REGISTRATION_FILE = "em_2p_registration.yaml"


def load_parquet_df(filepath):
    """Load a parquet file and restore any serialised numpy arrays.

    Args:
        filepath: Path to the ``.parquet`` file.

    Returns:
        pandas.DataFrame: DataFrame with numpy arrays restored in object columns.
    """
    df_flat = pd.read_parquet(filepath, engine='fastparquet')
    df = restore_numpy_arrays(df_flat)
    df = df.map(lambda x: np.array(x) if isinstance(x, list) else x)
    return df


def load_df_rois(data_folder: str | Path = DATA_2P) -> pd.DataFrame:
    """Load and concatenate all GCL ROI-level parquet files from ``data_folder``.

    Reads five parquet files named ``df_eyewire2_roi_level_GCL{0..4}.parquet``,
    concatenates them, and adds a boolean ``qfilt`` column based on quality indices.

    Args:
        data_folder: Path to the directory containing the parquet files.

    Returns:
        pandas.DataFrame: Combined ROI-level DataFrame with ``qfilt`` column.
    """
    df_rois = pd.concat([
        load_parquet_df(os.path.join(data_folder, f'df_eyewire2_roi_level_GCL{i}.parquet'))
        for i in range(5)])
    df_rois['qfilt'] = (df_rois['bar_qidx'] > 0.6) | (df_rois['chirp_qidx'] > 0.45)
    return df_rois


def load_df_fields(data_folder: str | Path = DATA_2P) -> pd.DataFrame:
    """Load the field-level parquet file from ``data_folder``.

    Args:
        data_folder: Path to the directory containing ``df_eyewire2_field_level.parquet``.

    Returns:
        pandas.DataFrame: Field-level DataFrame.
    """
    df_fields = load_parquet_df(os.path.join(data_folder, 'df_eyewire2_field_level.parquet'))
    return df_fields


def load_df_outline(data_folder: str | Path = DATA_2P) -> pd.DataFrame:
    """Load the outline parquet file from ``data_folder``.

    Args:
        data_folder: Path to the directory containing ``df_eyewire2_outline.parquet``.

    Returns:
        pandas.DataFrame: Outline DataFrame.
    """
    df_outline = load_parquet_df(os.path.join(data_folder, 'df_eyewire2_outline.parquet'))
    return df_outline


def load_all_dfs(data_folder: str | Path = DATA_2P) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load ROI-level, field-level, and outline DataFrames in one call.

    Args:
        data_folder: Path to the directory containing all required parquet files.

    Returns:
        tuple: ``(df_rois, df_fields, df_outline)`` – the three DataFrames.
    """
    df_rois = load_df_rois(data_folder)
    df_fields = load_df_fields(data_folder)
    df_outline = load_df_outline(data_folder)
    return df_rois, df_fields, df_outline


def load_df_rois_morph(
        data_folder: str | Path = DATA_2P,
        spreadsheet_folder: str | Path = DATA_SS,
        main_all_cells_sheet: str = MAIN_ALL_CELLS_SHEET,
        map_sheet: str = MAP_SHEET,
        nuc_col_master: str = 'Latest NucID',
        seg_col_master=('Latest SegID', 'Proofread SegID'),
        df_rois: pd.DataFrame | None = None,
        verbose: bool = False,
        ):
    """Load and merge the ROI-level DataFrame with the morphology master spreadsheet.

    ROI-level data has no direct nucleus ID of its own, so the EM-2p mapping sheet is
    used to link each 2p ROI (``field``/``roi_id``) to the EM cell it corresponds to
    (``2p-Field``/``2p-ROI`` -> nucleus ID), the same correspondence used to fit the
    2p<->EM coordinate registration in ``notebooks/preprocessing/em-2p-mapping.py``.
    That nucleus ID is then used to inner-join against the master list of proofread
    cells. ROIs with no nucleus ID in the mapping sheet (e.g. an overlay of two cells
    rather than a real single cell) are dropped.

    Args:
        data_folder: Path to the directory containing parquet files. Required when
            ``df_rois`` is ``None``.
        spreadsheet_folder: Path to the directory containing the morphology spreadsheet.
        main_all_cells_sheet: Filename of the CSV master list within
            ``spreadsheet_folder``
        map_sheet: Filename of the CSV EM-2p mapping sheet within ``spreadsheet_folder``.
        nuc_col_master: Column name (shared by the master CSV and the mapping CSV)
            used as the nucleus ID key.
        seg_col_master : Column or tuple of columns used to find Final SegID in the master spreadsheet.
            The function will check these columns in order and use the first non-null value as the 'Latest SegID'.
        df_rois: Pre-loaded ROI-level DataFrame. If ``None``, it is loaded from
            ``data_folder``.
        verbose: If ``True``, print diagnostics about the EM-2p mapping and the
            master-spreadsheet join (row counts, drops, unmatched ROIs).

    Returns:
        pandas.DataFrame: Merged DataFrame containing columns from the master
        spreadsheet, the EM-2p mapping sheet, and the ROI-level data, indexed by
        nucleus ID.

    Raises:
        AssertionError: If required columns are missing from the master spreadsheet,
            or if ``data_folder`` is not provided when ``df_rois`` is ``None``.
    """
    if df_rois is None:
        assert data_folder is not None, "data_folder must be provided if df_rois is None"
        df_rois = load_df_rois(data_folder)

    df_main = pd.read_csv(os.path.join(os.path.join(spreadsheet_folder, main_all_cells_sheet)), dtype=str).dropna(
        axis=1, how='all')

    df_map = pd.read_csv(os.path.join(os.path.join(spreadsheet_folder, map_sheet)), dtype=str).dropna(axis=1, how='all')

    assert nuc_col_master in df_main.columns, f"Column '{nuc_col_master}' not found in df_main {list(df_main.columns)}"
    assert nuc_col_master in df_map.columns, f"Column '{nuc_col_master}' not found in df_map {list(df_map.columns)}"

    # Normalise seg_col_master to a tuple so it works whether a single string or tuple was passed
    if isinstance(seg_col_master, str):
        seg_col_master = (seg_col_master,)

    missing = [c for c in seg_col_master if c not in df_main.columns]
    assert not missing, f"seg_col_master column(s) not found in df_main: {missing}"

    # Drop duplicates in df_main based on the nucleus ID column, keeping the first occurrence
    n_main_before = len(df_main)
    df_main = df_main.drop_duplicates(subset=nuc_col_master, keep='first')
    if verbose and n_main_before != len(df_main):
        print(f"df_main: dropped {n_main_before - len(df_main)} duplicate '{nuc_col_master}' row(s), {len(df_main)} remain")

    # Link each 2p ROI to its EM nucleus ID via the mapping sheet (dtype=str above
    # turns '2p-ROI' into an object column, so cast it to match df_rois' int roi_id).
    n_no_nuc_id = df_map[nuc_col_master].isna().sum()
    if verbose and n_no_nuc_id:
        print(f"df_map: excluding {n_no_nuc_id} / {len(df_map)} ROI(s) with no '{nuc_col_master}' (not a real cell)")
    df_map_valid = df_map.loc[df_map[nuc_col_master].notna(), ['2p-Field', '2p-ROI', nuc_col_master]].copy()
    df_map_valid['2p-ROI'] = df_map_valid['2p-ROI'].astype(df_rois['roi_id'].dtype)

    df_rois_mapped = df_rois.merge(
        df_map_valid, left_on=['field', 'roi_id'], right_on=['2p-Field', '2p-ROI'], how='inner',
    )
    if verbose:
        n_unmatched = len(df_map_valid) - len(df_rois_mapped)
        print(f"Matched {len(df_rois_mapped)} / {len(df_map_valid)} mapping-sheet ROI(s) to a 2p ROI in df_rois"
              + (f" ({n_unmatched} mapping row(s) had no corresponding 2p ROI)" if n_unmatched else ""))

    df_merged = pd.merge(
        df_main.set_index(nuc_col_master),
        df_rois_mapped.set_index(nuc_col_master),
        left_index=True, right_index=True, how='inner'
    ).reset_index()
    if verbose:
        n_dropped = len(df_rois_mapped) - len(df_merged)
        print(f"Joined {len(df_merged)} / {len(df_rois_mapped)} mapped ROI(s) to df_main on '{nuc_col_master}'"
              + (f" ({n_dropped} had no matching row in df_main)" if n_dropped else ""))

    # Fold seg_col_master columns in order, taking the first non-null value as Latest SegID
    latest = df_merged[seg_col_master[0]]
    for col in seg_col_master[1:]:
        latest = latest.combine_first(df_merged[col])
    df_merged['Latest SegID'] = latest

    return df_merged



def add_skels(df: pd.DataFrame, swc_dir: str | Path | None = DATA_SWC, inplace: bool = False) -> pd.DataFrame:
    import skeliner as sk

    if not inplace:
        df = df.copy()

    df['swc_path'] = ''
    df['skel'] = None

    df['swc_path'] = df['Latest SegID'].apply(lambda x: os.path.join(swc_dir, f"{x}.swc"))
    df['skel'] = df.apply(lambda row: sk.io.load_swc(row['swc_path']) if os.path.isfile(row['swc_path']) else None, axis=1)
    return df