# Copied from djimaging

from warnings import warn

import numpy as np

# Cluster ID, Cluster Name, Group ID, Supergroup
# Don't change this array, it is used in the classifier.
BADEN_CLUSTER_INFO = np.array([
    [1, '1', 1, 'OFF', ],
    [2, '2', 2, 'OFF', ],
    [3, '3', 3, 'OFF', ],
    [4, '4a', 4, 'OFF', ],
    [5, '4b', 4, 'OFF', ],
    [6, '5a', 5, 'OFF', ],
    [7, '5b', 5, 'OFF', ],
    [8, '5c', 5, 'OFF', ],
    [9, '6', 6, 'OFF', ],
    [10, '7', 7, 'OFF', ],
    [11, '8a', 8, 'OFF', ],
    [12, '8b', 8, 'OFF', ],
    [13, '9', 9, 'OFF', ],
    [14, '10', 10, 'ON-OFF', ],
    [15, '11a', 11, 'ON-OFF', ],
    [16, '11b', 11, 'ON-OFF', ],
    [17, '12a', 12, 'ON-OFF', ],
    [18, '12b', 12, 'ON-OFF', ],
    [19, '13', 13, 'ON-OFF', ],
    [20, '14', 14, 'ON-OFF', ],
    [21, '15', 15, 'Fast ON', ],
    [22, '16', 16, 'Fast ON', ],
    [23, '17a', 17, 'Fast ON', ],
    [24, '17b', 17, 'Fast ON', ],
    [25, '17c', 17, 'Fast ON', ],
    [26, '18a', 18, 'Fast ON', ],
    [27, '18b', 18, 'Fast ON', ],
    [28, '19', 19, 'Fast ON', ],
    [29, '20', 20, 'Fast ON', ],
    [30, '21', 21, 'Slow ON', ],
    [31, '22a', 22, 'Slow ON', ],
    [32, '22b', 22, 'Slow ON', ],
    [33, '23', 23, 'Slow ON', ],
    [34, '24', 24, 'Slow ON', ],
    [35, '25', 25, 'Slow ON', ],
    [36, '26', 26, 'Slow ON', ],
    [37, '27', 27, 'Slow ON', ],
    [38, '28a', 28, 'Slow ON', ],
    [39, '28b', 28, 'Slow ON', ],
    [40, '29', 29, 'Unc. ON', ],
    [41, '30', 30, 'Unc. ON', ],
    [42, '31a', 31, 'Unc. SbC', ],
    [43, '31b', 31, 'Unc. SbC', ],
    [44, '31c', 31, 'Unc. SbC', ],
    [45, '31d', 31, 'Unc. SbC', ],
    [46, '31e', 31, 'Unc. SbC', ],
    [47, '32a', 32, 'Unc. SbC', ],
    [48, '32b', 32, 'Unc. SbC', ],
    [49, '32c', 32, 'Unc. SbC', ],
    [50, '33', 33, 'dAC', ],
    [51, '34a', 34, 'dAC', ],
    [52, '34b', 34, 'dAC', ],
    [53, '35a', 35, 'dAC', ],
    [54, '35b', 35, 'dAC', ],
    [55, '36', 36, 'dAC', ],
    [56, '37a', 37, 'dAC', ],
    [57, '37b', 37, 'dAC', ],
    [58, '38a', 38, 'dAC', ],
    [59, '38b', 38, 'dAC', ],
    [60, '38c', 38, 'dAC', ],
    [61, '39', 39, 'dAC', ],
    [62, '40a', 40, 'dAC', ],
    [63, '40b', 40, 'dAC', ],
    [64, '41', 41, 'dAC', ],
    [65, '42a', 42, 'dAC', ],
    [66, '42b', 42, 'dAC', ],
    [67, '42c', 42, 'dAC', ],
    [68, '42d', 42, 'dAC', ],
    [69, '42e', 42, 'dAC', ],
    [70, '42f', 42, 'dAC', ],
    [71, '43', 43, 'dAC', ],
    [72, '44', 44, 'dAC', ],
    [73, '45', 45, 'dAC', ],
    [74, '46a', 46, 'dAC', ],
    [75, '46b', 46, 'dAC', ],
])

BADEN_GROUP_ID_NAMES = {
    1: 'Off local, OS',
    2: 'Off DS',
    3: 'Off step',
    4: 'Off slow',
    5: 'Off alpha sustained',
    6: '(On-)Off "JAM-B" mix',
    7: 'Off sustained',
    8: 'Off alpha transient',
    9: 'Off "mini" alpha transient',
    10: 'On-Off local-edge "W3"',
    11: 'On-Off local',
    12: 'On-Off DS 1',
    13: 'On-Off DS 2',
    14: '(On-)Off local, OS',
    15: 'On step',
    16: 'On DS transient',
    17: 'On local transient, OS',
    18: 'On transient',
    19: 'On alpha transient',
    20: 'On high frequency',
    21: 'On low frequency',
    22: 'On sustained',
    23: 'On "mini" alpha',
    24: 'On alpha sustained',
    25: 'On DS sustained 1',
    26: 'On DS sustained 2',
    27: 'On slow',
    28: 'On contrast suppressed',
    29: 'On DS sustained 3',
    30: 'On local sustained, OS',
    31: 'Off suppressed 1',
    32: 'Off suppressed 2',
    33: 'Off',
    34: 'On high frequency sustained 1',
    35: 'On high frequency transient',
    36: 'On-Off high frequency',
    37: 'On high frequency sustained 2',
    38: 'On sustained 1',
    39: 'On sustained 2',
    40: 'On sustained 3',
    41: 'On sustained 4',
    42: 'On "starburst"',
    43: 'On-Off local',
    44: 'On step',
    45: 'On local 1',
    46: 'On local 2',
}


def baden_cluster_id_to_cluster_name(cluster_id):
    """Return the Baden cluster name string for a given cluster ID (1-75).

    Args:
        cluster_id: Integer cluster ID in the range 1–75.

    Returns:
        str: The cluster name (e.g. ``'17a'``), or ``'Unknown'`` if out of range.
    """
    cluster_id = int(cluster_id)
    if cluster_id < 1 or cluster_id > 75:
        return 'Unknown'
    cluster_ids = BADEN_CLUSTER_INFO[:, 0].astype(int)
    cluster_names = BADEN_CLUSTER_INFO[:, 1].astype(str)
    i = np.where(cluster_ids == cluster_id)[0][0]
    return cluster_names[i]


def baden_cluster_name_to_cluster_id(cluster_name):
    """Return the Baden cluster ID integer for a given cluster name string.

    Args:
        cluster_name: Cluster name string (e.g. ``'17a'``). Pass ``'Unknown'``
            to receive ``-1``.

    Returns:
        int: The cluster ID (1–75), or ``-1`` for ``'Unknown'``.
    """
    cluster_name = str(cluster_name)
    if cluster_name == 'Unknown':
        return -1
    cluster_ids = BADEN_CLUSTER_INFO[:, 0].astype(int)
    cluster_names = BADEN_CLUSTER_INFO[:, 1].astype(str)
    i = np.where(cluster_names == cluster_name)[0][0]
    return cluster_ids[i]


def baden_cluster_id_to_group_id(cluster_id):
    """Return the Baden group ID for a given cluster ID.

    Args:
        cluster_id: Integer cluster ID in the range 1–75.

    Returns:
        int: The group ID (1–46), or ``-1`` if ``cluster_id`` is out of range.
    """
    cluster_id = int(cluster_id)
    if cluster_id < 1 or cluster_id > 75:
        return -1
    cluster_ids = BADEN_CLUSTER_INFO[:, 0].astype(int)
    group_ids = BADEN_CLUSTER_INFO[:, 2].astype(int)
    i = np.where(cluster_ids == cluster_id)[0][0]
    return group_ids[i]


def baden_cluster_id_to_supergroup(cluster_id):
    """Return the Baden supergroup label for a given cluster ID.

    Args:
        cluster_id: Integer cluster ID in the range 1–75.

    Returns:
        str: The supergroup label (e.g. ``'OFF'``, ``'Fast ON'``, ``'dAC'``),
            or ``'Unknown'`` if out of range.
    """
    cluster_id = int(cluster_id)
    if cluster_id < 1 or cluster_id > 75:
        return 'Unknown'
    cluster_ids = BADEN_CLUSTER_INFO[:, 0].astype(int)
    supergroups = BADEN_CLUSTER_INFO[:, 3].astype(str)
    i = np.where(cluster_ids == cluster_id)[0][0]
    return supergroups[i]


def baden_group_id_to_supergroup(group_id):
    """Return the Baden supergroup label for a given group ID.

    Args:
        group_id: Integer group ID in the range 1–46.

    Returns:
        str: The supergroup label (e.g. ``'OFF'``, ``'Slow ON'``, ``'dAC'``),
            or ``'Unknown'`` if out of range.
    """
    group_id = int(group_id)
    if group_id < 1 or group_id > 46:
        return 'Unknown'
    group_ids = BADEN_CLUSTER_INFO[:, 2].astype(int)
    supergroups = BADEN_CLUSTER_INFO[:, 3].astype(str)
    i = np.where(group_ids == group_id)[0][0]
    return supergroups[i]


def baden_group_id_to_group_name(group_id, shorten=False):
    """Return the descriptive name for a Baden group ID.

    Args:
        group_id: Integer group ID in the range 1–46.
        shorten: If ``True``, abbreviate common words (e.g. ``'sustained'`` →
            ``'sus.'``) via :func:`shorten_baden_name`.

    Returns:
        str: The group name (e.g. ``'Off DS'``), or ``'Unknown'`` if out of range.
    """
    group_id = int(group_id)
    if group_id < 1 or group_id > 46:
        return 'Unknown'
    group_name = BADEN_GROUP_ID_NAMES[group_id]
    if shorten:
        group_name = shorten_baden_name(group_name)
    return group_name


def shorten_baden_name(name):
    """Abbreviate common words in a Baden group name string.

    Replaces ``'frequency'`` → ``'freq.'``, ``'sustained'`` → ``'sus.'``,
    ``'transient'`` → ``'trans.'``, and ``'suppressed'`` → ``'suppr.'``.

    Args:
        name: Full group name string.

    Returns:
        str: The abbreviated name string.
    """
    name = name.replace('frequency', 'freq.')
    name = name.replace('sustained', 'sus.')
    name = name.replace('transient', 'trans.')
    name = name.replace('suppressed', 'suppr.')
    return name


def baden16_cluster_probs_to_info(probs):
    """Convert a vector of 75 per-cluster probabilities to aggregated classification info.

    Args:
        probs: Array-like of length 75 containing probabilities for each of the
            75 Baden clusters (sum need not equal 1).

    Returns:
        tuple: ``(cluster_id, group_id, supergroup, prob_cluster, prob_group,
        prob_supergroup, prob_class)`` where the ``prob_*`` values are summed
        probabilities at the respective level and ``prob_class`` is the probability
        of belonging to the predicted broad class (RGC vs. dAC).

    Raises:
        ValueError: If ``probs`` does not have exactly 75 elements.
    """
    if len(probs) != 75:
        raise ValueError(f"Expected 75 probabilities corresponding to 75 Baden clusters, got {len(probs)}.")

    cluster_id = np.argmax(probs) + 1  # Cluster IDs are 1-indexed
    group_id = baden_cluster_id_to_group_id(cluster_id)
    supergroup = baden_group_id_to_supergroup(group_id)
    prob_cluster = probs[cluster_id - 1]

    group_ids = BADEN_CLUSTER_INFO[:, 2].astype(int)
    supergroups = BADEN_CLUSTER_INFO[:, 3].astype(str)

    prob_group = np.sum(probs[group_ids == group_id])
    prob_supergroup = np.sum(probs[supergroups == supergroup])
    prob_rgc = np.sum(probs[supergroups != 'dAC'])
    prob_class = (1. - prob_rgc) if supergroup == 'dAC' else prob_rgc

    return cluster_id, group_id, supergroup, prob_cluster, prob_group, prob_supergroup, prob_class


def probs_per_cluster_to_level_probs(probs):
    """Aggregate a 75-length per-cluster probability vector into per-level probabilities.

    Args:
        probs: Array-like of length 75 containing probabilities for each of the
            75 Baden clusters (sum need not equal 1).

    Returns:
        tuple: ``(supergroup_probs, group_probs, cluster_probs)``, each a list of
        ``(label, supergroup, prob)`` tuples summed at that level. Entries are in
        `BADEN_CLUSTER_INFO`'s array order, which is already grouped contiguously
        by supergroup and then by group, so bars built from these lists stack
        comparably across levels. Group labels use the shortened group name
        (see :func:`shorten_baden_name`); cluster labels use the cluster name
        (e.g. ``'17a'``).

    Raises:
        ValueError: If ``probs`` does not have exactly 75 elements.
    """
    probs = np.asarray(probs, dtype=float)
    if len(probs) != 75:
        raise ValueError(f"Expected 75 probabilities corresponding to 75 Baden clusters, got {len(probs)}.")

    cluster_names = BADEN_CLUSTER_INFO[:, 1].astype(str)
    group_ids = BADEN_CLUSTER_INFO[:, 2].astype(int)
    supergroups = BADEN_CLUSTER_INFO[:, 3].astype(str)

    cluster_probs = list(zip(cluster_names, supergroups, probs))

    group_order, group_sum, group_supergroup = [], {}, {}
    for gid, sg, p in zip(group_ids, supergroups, probs):
        if gid not in group_sum:
            group_order.append(gid)
            group_sum[gid] = 0.0
            group_supergroup[gid] = sg
        group_sum[gid] += p
    group_probs = [(baden_group_id_to_group_name(gid, shorten=True), group_supergroup[gid], group_sum[gid])
                   for gid in group_order]

    sg_order, sg_sum = [], {}
    for sg, p in zip(supergroups, probs):
        if sg not in sg_sum:
            sg_order.append(sg)
            sg_sum[sg] = 0.0
        sg_sum[sg] += p
    supergroup_probs = [(sg, sg, sg_sum[sg]) for sg in sg_order]

    return supergroup_probs, group_probs, cluster_probs


def correct_probs_by_cellclass(cellclass, probs_per_cluster):
    """Zero out per-cluster probability mass inconsistent with a known ground-truth cell class,
    then renormalize -- corrects classifier confusion between RGCs and dACs (displaced amacrine
    cells) using independent information (e.g. EM proofreading) about which broad class the cell
    actually belongs to.

    Args:
        cellclass: ``'RGC'`` or ``'AC'`` (i.e. dAC). Anything else is treated as ``'AC'``, with a warning.
        probs_per_cluster: Array-like of length 75 containing probabilities for each of the
            75 Baden clusters (sum need not equal 1).

    Returns:
        numpy.ndarray: Renormalized length-75 probability vector, with all clusters outside
        `cellclass`'s broad class zeroed out.
    """
    cluster_is_ac = BADEN_CLUSTER_INFO[:, -1] == 'dAC'

    probs_per_cluster = np.asarray(probs_per_cluster, dtype=float).copy()
    if cellclass == 'RGC':
        probs_per_cluster[cluster_is_ac] = 0
    else:
        if not cellclass == 'AC':
            warn('cellclass is not AC or RGC, but will treat as AC')
        probs_per_cluster[~cluster_is_ac] = 0

    probs_per_cluster /= probs_per_cluster.sum()
    return probs_per_cluster


def infer_cellinfo_using_cellclass(cellclass, probs_per_cluster):
    probs_per_cluster = correct_probs_by_cellclass(cellclass, probs_per_cluster)
    cluster_id, group_id, supergroup, prob_cluster, prob_group, prob_supergroup, prob_class = (
        baden16_cluster_probs_to_info(probs_per_cluster))

    return cluster_id, group_id, supergroup, prob_cluster, prob_group, prob_supergroup, prob_class