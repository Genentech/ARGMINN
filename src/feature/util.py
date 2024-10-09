import subprocess
import numpy as np
import pandas as pd
import os

def file_line_count(filepath):
    """
    Returns the number of lines in the given file. If the file is gzipped (i.e.
    ends in ".gz"), unzips it first.
    Arguments:
        `filepath`: path to file to check, which may be gzipped
    Returns the number of lines in the file
    """
    if filepath.endswith(".gz"):
        cat_comm = ["zcat", filepath]
    else:
        cat_comm = ["cat", filepath]
    wc_comm = ["wc", "-l"]

    cat_proc = subprocess.Popen(cat_comm, stdout=subprocess.PIPE)
    wc_proc = subprocess.Popen(
        wc_comm, stdin=cat_proc.stdout, stdout=subprocess.PIPE
    )
    output, err = wc_proc.communicate()
    return int(output.strip())


def seqs_to_one_hot(seqs, alphabet="ACGT", to_upper=True, out_dtype=np.float64):
    """
    Converts a list of strings to one-hot encodings, where the position of 1s is
    ordered by the given alphabet.
    Arguments:
        `seqs`: a list of N strings, where every string is the same length L
        `alphabet`: string of length D containing the alphabet used to do
            the encoding; defaults to "ACGT", so that the position of 1s is
            alphabetical according to "ACGT"
        `to_upper`: if True, convert all bases to upper-case prior to performing
            the encoding
        `out_dtype`: NumPy datatype of the output one-hot sequences; defaults
            to `np.float64` but can be changed (e.g. `np.int8` drastically
            reduces memory usage)
    Returns an N x L x D NumPy array of one-hot encodings, in the same order as
    the input sequences. Any bases that are not in the alphabet will be given an
    encoding of all 0s.
    """
    seq_len = len(seqs[0])
    assert np.all(np.array([len(s) for s in seqs]) == seq_len)

    # Get ASCII codes of alphabet in order
    alphabet_codes = np.frombuffer(bytearray(alphabet, "utf8"), dtype=np.int8)

    # Join all sequences together into one long string, all uppercase
    seq_concat = "".join(seqs).upper() + alphabet
    # Add one example of each base, so np.unique doesn't miss indices later

    one_hot_map = np.identity(len(alphabet) + 1)[:, :-1].astype(out_dtype)

    # Convert string into array of ASCII character codes;
    base_vals = np.frombuffer(bytearray(seq_concat, "utf8"), dtype=np.int8)

    # Anything that's not in the alphabet gets assigned a higher code
    base_vals[~np.isin(base_vals, alphabet_codes)] = np.max(alphabet_codes) + 1

    # Convert the codes into indices, in ascending order by code
    _, base_inds = np.unique(base_vals, return_inverse=True)

    # Get the one-hot encoding for those indices, and reshape back to separate
    return one_hot_map[base_inds[:-len(alphabet)]].reshape(
        (len(seqs), seq_len, len(alphabet))
    )


def one_hot_to_seqs(one_hot, alphabet="ACGT", unk_token="N"):
    """
    Converts a one-hot encoding into a list of strings, where the position of 1s
    is ordered by the given alphabet.
    Arguments:
        `one_hot`: an N x L x D array of one-hot encodings
        `alphabet`: string of length D containing the alphabet used to do
            the decoding; defaults to "ACGT", so that the position of 1s is
            alphabetical according to "ACGT"
        `unk_token`: token to use for a one-hot encoding of all 0s
    Returns a list of N strings, each of length L, in the same order as the
    input array. The returned sequences will only consist of characters in the
    alphabet or `unk_token`. Any encodings that are all 0s will be translated to
    `unk_token`.
    """
    assert len(alphabet) == one_hot.shape[2]
    assert len(unk_token) == 1
    bases = np.array(list(alphabet) + [unk_token])

    # Create N x L array of all Ds
    one_hot_inds = np.tile(one_hot.shape[2], one_hot.shape[:2])

    # Get indices of where the 1s are
    batch_inds, seq_inds, base_inds = np.where(one_hot)

    # In each of the locations in the N x L array, fill in the location of the 1
    one_hot_inds[batch_inds, seq_inds] = base_inds

    # Fetch the corresponding base for each position using indexing
    seq_array = bases[one_hot_inds]
    return ["".join(seq) for seq in seq_array]


def import_peaks_bed(peaks_bed):
    """
    Imports a peaks BED file in NarrowPeak format as a Pandas DataFrame.
    Arguments:
        `peaks_bed`: a BED file (gzipped or not) containing peaks in ENCODE
            NarrowPeak format
    Returns a Pandas DataFrame.
    """
    return pd.read_csv(
        peaks_bed, sep="\t", header=None,  # Infer compression
        names=[
            "chrom", "peak_start", "peak_end", "name", "score", "strand",
            "signal", "pval", "qval", "summit_offset"
        ]
    )
