import numpy as np
from skeliner import Skeleton


def transform_skel_xy(skel, matrix_2x2):
    """
    Apply a general 2x2 linear map to a skeleton's XY coordinates, about its
    soma center (e.g. a rotation, a mirror flip, or a rotation+flip composed
    via `eyewire2_functional_analysis.registration.similarity_matrix`/
    `get_field_rotation_matrix`). Z is left untouched.

    Parameters
    - skel: skeliner.core.Skeleton
        Input skeleton. Node 0 is assumed to be the soma centroid.
    - matrix_2x2: array-like, shape (2, 2)
        Column-vector convention: the transformed point is ``matrix_2x2 @ point``.

    Returns
    - skel_aug: skeliner.core.Skeleton
        The transformed skeleton (a copy; `skel` is untouched).
    """
    nodes = np.asarray(skel.nodes, dtype=np.float64).copy()
    if len(nodes) == 0:
        return skel

    # Soma center
    soma_center = np.asarray(
        skel.soma.center if skel.soma is not None else nodes[0],
        dtype=np.float64
    ).copy()

    # Embed the 2x2 XY map into a 3x3 matrix that leaves Z untouched
    M = np.eye(3, dtype=np.float64)
    M[:2, :2] = matrix_2x2

    # Apply about soma: x' = M @ (x - c) + c
    nodes = (nodes - soma_center) @ M.T + soma_center

    # Update soma geometry
    if skel.soma is not None:
        new_R = M @ skel.soma.R
        soma2 = skel.soma.__class__(
            soma_center.copy(),
            skel.soma.axes.copy(),
            new_R,
            verts=skel.soma.verts
        )
    else:
        soma2 = skel.soma

    # Build new Skeleton
    skel_aug = Skeleton(
        soma=soma2,
        nodes=nodes,
        radii={k: np.asarray(v).copy() for k, v in skel.radii.items()},
        edges=skel.edges.copy() if skel.edges is not None else None,
        ntype=np.asarray(skel.ntype).copy() if skel.ntype is not None else None,
        node2verts=skel.node2verts,
        vert2node=skel.vert2node,
        meta=dict(skel.meta) if hasattr(skel, "meta") and skel.meta is not None else {},
        extra={},
    )

    return skel_aug


def rotate_skel(skel, rotation_deg):
    """
    Apply rotation in the XY plane around the soma center.

    Parameters
    - skel: skeliner.core.Skeleton
        Input skeleton. Node 0 is assumed to be the soma centroid.
    - rotation_deg: float | None
        Counterclockwise rotation angle in degrees applied in the XY plane
        about the soma center. Use 0 or None for no rotation.

    Returns
    - skel_aug: skeliner.core.Skeleton
        The rotated skeleton.

    Example
    >>> skel_aug = rotate_skel(skel, rotation_deg=30.0)
    """
    theta = np.deg2rad(rotation_deg if rotation_deg is not None else 0.0)
    c, s = np.cos(theta), np.sin(theta)
    Rz = np.array([[c, -s], [s, c]], dtype=np.float64)
    return transform_skel_xy(skel, Rz)