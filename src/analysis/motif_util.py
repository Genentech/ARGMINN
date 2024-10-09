import numpy as np
import h5py
import scipy.signal
import os

BACKGROUND_FREQS = np.array([0.25, 0.25, 0.25, 0.25])

def import_meme_motifs(motifs_paths):
    """
    Imports a set of motif weight matrices from a motif file in MEME format.
    Arguments:
        `motifs_path`: path to MEME-format motif file, or list of paths to motif
            files; later paths and later motifs in a single file overwrite
            earlier ones
    Returns dictionary mapping motif identifiers to NumPy arrays. Each NumPy
    array is of shape L x D for the length and alphabet depth of the motif. L
    and D may be different for each motif.
    """
    if type(motifs_paths) is str:
        motifs_paths = [motifs_paths]

    motifs = {}
    
    for motifs_path in motifs_paths:
        with open(motifs_path, "r") as f:
            motif = None  # Change to [] only when we see line before matrix
            for line in f:
                if line.startswith("MOTIF "):
                    if motif:
                        # Save previous motif
                        motifs[motif_id] = np.stack(motif)
                        motif = None  # Reset to None
                    
                    # Extract motif ID
                    tokens = line.split()
                    motif_id = "-".join([t.strip() for t in tokens[1:]])

                elif line.startswith("letter-probability matrix:"):
                    motif = []  # Next lines are matrix, so make it list
                
                elif type(motif) is list:
                    # We are ready to fill in the matrix
                    line = line.strip()
                    if not line:
                        # Empty line implies the matrix is done
                        motifs[motif_id] = np.stack(motif)
                        motif = None
                        continue
                    try:
                        vals = np.array([float(x) for x in line.split()])
                    except ValueError:
                        # Skip if there is no non-numerical value; this also
                        # implies the matrix is done
                        motifs[motif_id] = np.stack(motif)
                        motif = None
                        continue

                    motif.append(vals)
                 
        if motif:
            # Last motif in the file (if someone forgot the last blank line)
            motifs[motif_id] = np.stack(motif)
    return motifs


def export_meme_motifs(pfms, out_path):
    """
    Writes a set of PFMs in MEME format to the given output path.
    Arguments:
        `pfms`: dictionary mapping motif keys to normalized PFMs, each is an
            L x 4 NumPy array (Ls may be different)
        `out_path`: output path to write motifs to in MEME format; intermediate
            directories will be made if necessary
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as f:
        f.write("MEME version 4\n\n")
        f.write("ALPHABET = ACGT\n\n")

        for key, pfm in pfms.items():
            f.write("MOTIF %s\n" % key)
            f.write("letter-probability matrix:\n")
            for row in pfm:
                f.write(" ".join(["%.8f" % x for x in row]) + "\n")
            f.write("\n")


def import_modisco_motifs(modisco_results_path, order_by_seqlets=True):
    """
    Given the path to a MoDISco results object, imports the motifs.
    Arguments:
        `modisco_results_path`: path to MoDISco results object
        `order_by_seqlets`: if True, order the motifs by descending order of
            number of seqlets
    Returns a list of L x 4 NumPy arrays of PFMs, a parallel list of L x 4 NumPy
    arrays of CWMs, a parallel list of L x 4 NumPy arrays of hCWMs, and a
    parallel NumPy array of number of seqlets.
    """
    pfms, cwms, hcwms, num_seqlets = [], [], [], []
    with h5py.File(modisco_results_path, "r") as f:
        for metacluster_key in f.keys():
            metacluster = f[metacluster_key]
            for pattern_key in metacluster.keys():
                pattern = metacluster[pattern_key]
                num_seqlets.append(len(pattern["seqlets"]["sequence"]))
                pfms.append(pattern["sequence"][:])
                cwms.append(pattern["contrib_scores"][:])
                hcwms.append(pattern["hypothetical_contribs"][:])
    
    num_seqlets = np.array(num_seqlets)
    if order_by_seqlets:
        # Sort by number of seqlets
        inds = np.flip(np.argsort(num_seqlets))
        num_seqlets = np.array(num_seqlets)[inds]
        pfms = [pfms[i] for i in inds]
        cwms = [cwms[i] for i in inds]
        hcwms = [hcwms[i] for i in inds]
    
    return pfms, cwms, hcwms, num_seqlets


def pfm_info_content(pfm, pseudocount=0.001):
    """
    Given an L x 4 PFM, computes information content for each base and
    returns it as an L-array.
    """
    num_bases = pfm.shape[1]
    # Normalize track to probabilities along base axis
    pfm_norm = (pfm + pseudocount) / \
        (np.sum(pfm, axis=1, keepdims=True) + (num_bases * pseudocount))
    ic = pfm_norm * np.log2(pfm_norm / np.expand_dims(BACKGROUND_FREQS, axis=0))
    return np.sum(ic, axis=1)


def trim_motif_by_low_ic(motif, ic, min_ic=0.2, pad=0):
    """
    Given a motif and the information content at each position, trims the motif
    by cutting off flanks of low information content. If no positions pass the
    threshold, then no trimming is done.
    Arguments:
        `motif`: an L x 4 NumPy array of the motif (e.g. PFM or PWM)
        `ic`: an L-array of the information content at each position
        `min_ic`: minimum information content required to not be trimmed
        `pad`: if given, the trimmed flanks will be padded back by this amount
            on either side
    """
    pass_inds = np.where(ic >= min_ic)[0]
    
    if not pass_inds.size:
        return motif

    # Expand trimming to +/- pad bp on either side
    start = max(0, np.min(pass_inds) - pad)
    end = min(len(motif), np.max(pass_inds) + pad + 1)
    return motif[start:end]


def trim_motif_by_max_ic_window(motif, ic, window_size):
    """
    Given a motif and the information content at each position, trims the motif
    by identifying the window with the highest total information content.
    Arguments:
        `motif`: an L x 4 NumPy array of the motif (e.g. PFM or PWM)
        `ic`: an L-array of the information content at each position
        `window_size`: the size of window to consider (i.e. the size of the
            trimmed motif); must be at most L
    """
    ic_sums = scipy.signal.correlate(ic, np.ones(window_size), mode="valid")
    start = np.argmax(ic_sums)
    end = start + window_size
    return motif[start:end]
