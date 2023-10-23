import numpy as np

def graph_diff2(g1: np.ndarray, g2: np.ndarray) -> tuple[int, int, int]:
    tp = g1 & g2
    fp = g1 & (np.logical_not(g2))
    fn = (np.logical_not(g1)) & g2
    return tp.sum(), fp.sum(), fn.sum()

def graph_diff(
    g1: np.ndarray, g2: np.ndarray
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    edges1 = set(zip(np.where(g1)[0], np.where(g1)[1]))
    edges2 = set(zip(np.where(g2)[0], np.where(g2)[1]))

    g1_reversed = {(j, i) for (i, j) in edges1 if i != j}
    g2_reversed = {(j, i) for (i, j) in edges2 if i != j}

    additions = edges2 - edges1 - g1_reversed
    deletions = edges1 - edges2 - g2_reversed
    reversals = edges1 & g2_reversed

    return additions, deletions, reversals


def structural_hamming_distance(g1: np.ndarray, g2: np.ndarray) -> int:
    additions, deletions, reversals = graph_diff(g1, g2)
    return len(additions) + len(deletions) + len(reversals)


def perm_mat_from_perm(p: np.ndarray) -> np.ndarray:
    return p[..., None] == np.arange(len(p))[None, ...]

