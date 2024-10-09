import numpy as np
import scipy.signal
import pandas as pd
import os
import tempfile
import subprocess
import analysis.motif_util as motif_util

MEME_BASE_PATH = "/home/tsenga5/lib/meme/bin"

def compute_motif_cross_correlation(
    motif_1, motif_2, l1_norm=True, allow_overhangs=True, length_norm=True,
    pseudocount=1e-6
):
    """
    Given two motifs (e.g. as PFMs), computes the similarity between them as the
    maximal sliding cross correlation.
    Arguments:
        `motif_1`: an L1 x 4 NumPy array
        `motif_2`: an L2 x 4 NumPy array
        `l1_norm`: if True, L1 normalize each position independently prior to
            computing cross correlation
        `allow_overhangs`: if True, consider sliding windows where both motifs
            may have overhangs; otherwise, only consider sliding windows where
            the smaller motif fully fits in the larger one
        `length_norm`: if True, normalize the final similarity score by the
            length of overlap
        `pseudocount`: a small number to use in L1-normalization
    Returns a scalar similarity.
    """
    assert motif_1.shape[1] == 4 and motif_2.shape[1] == 4

    if l1_norm:
        # L1-normalize both
        motif_1 = (motif_1 + pseudocount) \
            / np.sum(np.abs(motif_1 + pseudocount), axis=1, keepdims=True)
        motif_2 = (motif_2 + pseudocount) \
            / np.sum(np.abs(motif_2 + pseudocount), axis=1, keepdims=True)

    min_len = min(len(motif_1), len(motif_2))
    if allow_overhangs:
        sims = scipy.signal.correlate(motif_1, motif_2, mode="full")[:, 3]
        # Pick best window by total similarity, then normalize is needed
        best_ind = np.argmax(sims)
        sim = sims[best_ind]
        if length_norm:
            overlap_sizes = np.full(len(motif_1) + len(motif_2) - 1, min_len)
            if min_len > 1:
                overlap_sizes[:min_len - 1] = np.arange(1, min_len)
                overlap_sizes[-(min_len - 1):] = np.flip(np.arange(1, min_len))
            sim = sim / overlap_sizes[best_ind]
    else:
        sim = np.max(scipy.signal.correlate(motif_1, motif_2, mode="valid"))
        if length_norm:
            sim = sim / min_len

    return sim


def compute_closest_motifs(
    query_motifs, target_motifs, sim_func=compute_motif_cross_correlation,
    top_k=10
):
    """
    For each motif in `query_motifs`, computes the top `top_k` matches in
    `target_motifs`.
    Arguments:
        `query_motifs`: a list or dictionary of motifs, where each motif is an
            L x 4 NumPy array (may be different Ls)
        `target_motifs`: a dictionary mapping keys to motifs, where each motif
            is an L x 4 NumPy array (may be different Ls)
        `sim_func`: a symmetric two-argument function between motifs which
            returns a similarity metric; defaults to
            `compute_motif_cross_correlation`
        `top_k`: the maximum number of matches to return; set to negative number
            to return all matches
    Returns a list or dictionary (parallel indices or keys to `query_motifs`),
    where the value is a list of matches in descending order of similarity. Each
    triplet is: 1) the key of the target motif matched; 2) whether or not the
    match was reverse complemented; and 3) the similarity score.
    """
    if type(query_motifs) is dict:
        result = {}
        query_iter = query_motifs.items()
    else:
        result = [None] * len(query_motifs)
        query_iter = enumerate(query_motifs)

    for query_key, query_motif in query_iter:
        query_motif_rc = np.flip(query_motif)

        sims = {
            (target_key, False) : sim_func(query_motif, target_motif)
            for target_key, target_motif in target_motifs.items()
        }
        sims.update({
            (target_key, True) : sim_func(query_motif_rc, target_motif)
            for target_key, target_motif in target_motifs.items()
        })

        top_matches = sorted(sims, key=(lambda k: -sims[k]))
        if top_k > 0:
            top_matches = top_matches[:top_k]
        result[query_key] = [pair + (sims[pair],) for pair in top_matches]
    return result


def run_tomtom(
    query_motifs, target_motifs, out_dir=None, return_subtables=True
):
    """
    Runs TOMTOM given the target and query motif files. The default threshold
    of q < 0.5 is used to filter for matches.
    Arguments:
        `query_motifs`: the set of query motifs to find matches for; may be the
            path to a file of motifs in MEME format, or a dictionary mapping
            keys to L x 4 NumPy arrays (may be different Ls)
        `target_motifs`: the set of target motifs to match against; may be the
            path to a file of motifs in MEME format, or a dictionary mapping
            keys to L x 4 NumPy arrays (may be different Ls)
        `out_dir`: path to directory to store results (a "tomtom" subdirectory
            will be created; if None, create a temporary directory which will be
            deleted right after
        `return_subtables`: if True, return a dictionary mapping query keys to
            subtable of matches; otherwise, returns just one big table
    Returns a Pandas DataFrame of match results, or a dictionary mapping each
    query motif key to a subtable to match results (keys without a match will
    not be in the dictionary).
    """
    if out_dir is None:
        temp_dir_obj = tempfile.TemporaryDirectory()
        out_dir = temp_dir_obj.name
    else:
        temp_dir_obj = None
    
    tomtom_dir = os.path.join(out_dir, "tomtom")
    os.makedirs(tomtom_dir, exist_ok=True)

    # If needed, write motif dictionaries to files
    if type(query_motifs) is dict:
        query_motifs_path = os.path.join(tomtom_dir, "query_motifs.txt")
        motif_util.export_meme_motifs(query_motifs, query_motifs_path)
    else:
        query_motifs_path = query_motifs
    if type(target_motifs) is dict:
        target_motifs_path = os.path.join(tomtom_dir, "target_motifs.txt")
        motif_util.export_meme_motifs(target_motifs, target_motifs_path)
    else:
        target_motifs_path = target_motifs

    # Run TOMTOM
    comm = [os.path.join(MEME_BASE_PATH, "tomtom")]
    comm += [query_motifs_path, target_motifs_path]
    comm += ["-oc", tomtom_dir]
    proc = subprocess.run(comm, capture_output=True)

    # Import results
    out_table = pd.read_csv(
        os.path.join(tomtom_dir, "tomtom.tsv"), sep="\t", header=0,
        index_col=False, comment="#"
    )

    # Cleanup
    if temp_dir_obj is not None:
        temp_dir_obj.cleanup()

    # If needed, split into subtables
    if return_subtables:
        return dict(tuple(out_table.groupby("Query_ID")))
    else:
        return out_table
