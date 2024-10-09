import torch
import captum.attr
import numpy as np
import extract.dinuc_shuffle as dinuc_shuffle
import tqdm

# Define device
if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


def shap_hyp_imp_projector(multipliers, inputs, baselines):
    """
    In order to obtain meaningful hypothetical contribution scores from
    DeepLIFTShap, the multipliers need to be modified for one-hot encoded
    inputs. For the multipliers coming from a given baseline, we need to
    subtract out the contribution for the base actuall present, and add in the
    contribution of the base it is hypothetically becoming.
    Arguments:
        `multipliers`: a singleton tuple of a B x L x 4 tensor of multipliers
            from DeepLIFTShap
        `inputs`: a singleton tuple of a B x L x 4 tensor of the one-hot encoded
            input sequence being queried for (duplicated across the B dimension)
        `baselines`: a singleton tuple of a B x L x 4 tensor of one-hot encoded
            baselines (B is the number of baselines)
    Returns the multipliers projected across all bases as a singleton tuple of
    a B x L x 4 tensor.
    """
    assert type(multipliers) is tuple and len(multipliers) == 1
    assert type(inputs) is tuple and len(inputs) == 1
    assert type(baselines) is tuple and len(baselines) == 1

    projections = torch.zeros_like(baselines[0], dtype=inputs[0].dtype)
    # Shape: B x L x 4

    for i in range(inputs[0].shape[2]):  # Iterate over bases
        hyp_input = torch.zeros_like(inputs[0], dtype=inputs[0].dtype)
        hyp_input[:, :, i] = 1  # If this base were present
        hyp_diffs = hyp_input - baselines[0]  # Difference of base/baselines
        hyp_imps = hyp_diffs * multipliers[0]  # Scale by multipliers
        projections[:, :, i] = torch.sum(hyp_imps, dim=2)

    return (projections,)


def compute_shap_scores(
    model, input_seqs, num_baselines=10, save_stem=None, verbose=True
):
    """
    Computes DeepLIFTShap importance scores (hypothetical contributions) for a
    given model and set of input sequences.
    Arguments:
        `model`: a trained model which takes in a B x L x 4 tensor of one-hot
            encoded sequences
        `input_seqs`: an N x L x 4 NumPy array or Torch tensor of one-hot
            sequences to perform interpretations for
        `num_baselines`: number of dinucleotide shuffles to create as baselines
            for each input sequence
        `save_stem`: if provided, save the input sequences to
            `{save_stem}onehots.npz` and the importance scores to
            `{save_stem}impscores.npz`; objects will be stored as N x 4 x L
            arrays (note the change in dimension)
        `verbose`: if True, show progress bar
    Returns a parallel N x L x 4 array of importance scores.
    """
    explainer = captum.attr.DeepLiftShap(model.to(DEVICE))
    
    if type(input_seqs) is np.ndarray:
        input_seqs_arr = input_seqs
    else:
        input_seqs_arr = input_seqs.detach().cpu().numpy()

    if save_stem is not None:
        np.savez(
            "%sonehots.npz" % save_stem,
            np.transpose(input_seqs_arr, (0, 2, 1))
        )

    imp_scores_arr = []
    t_iter = tqdm.trange(len(input_seqs_arr)) if verbose \
        else range(len(input_seqs_arr))
    for i in t_iter:
        one_hot_arr = input_seqs_arr[i]
        one_hot_ten = torch.tensor(one_hot_arr).float().to(DEVICE)
        baselines_arr = dinuc_shuffle.dinuc_shuffle(one_hot_arr, 10)
        baselines_ten = torch.tensor(baselines_arr).to(DEVICE).float()
        scores_ten = explainer.attribute(
            one_hot_ten[None], baselines_ten,
            custom_attribution_func=shap_hyp_imp_projector
        )
        imp_scores_arr.append(scores_ten.detach().cpu().numpy())
    imp_scores_arr = np.concatenate(imp_scores_arr)

    if save_stem is not None:
        np.savez(
            "%simpscores.npz" % save_stem,
            np.transpose(imp_scores_arr, (0, 2, 1))
        )

    return imp_scores_arr


def compute_ism_scores_single(input_seq, predict_func, slice_only=None):
    """
    Computes in-silico mutagenesis importance scores for a single input
    sequence.
    Arguments:
        `input_seq`: an L x 4 NumPy array of input sequences to explain
        `predict_func`: a function that takes in a B x L x 4 array of input
            sequences, and returns an N-array of output values (usually this is
            a logit, or an aggregate of logits); any batching must be done by
            this function
        `slice_only`: if provided, this is a slice along the input length
            dimension for which the ISM computation is limited to
    Returns an L' x 4 NumPy array of ISM scores, which consists of the difference
    between the output values for each possible mutation made, and the output
    value of the original sequence. L' = L if `slice_only` is None, and is
    shorter otherwise.
    """
    input_length, num_bases = input_seq.shape
    if slice_only:
        start, end = slice_only.start, slice_only.stop
        if not start:
            start = 0
        elif start < 0:
            start = start % input_length
        if not end:
            end = input_length
        elif end < 0:
            end = (end % input_length) + 1
    else:
        start, end = 0, input_length

    # A dictionary mapping a base index to all the other base indices; this will
    # be useful when making mutations (e.g. 2 -> [0, 1, 3])
    non_base_indices = {}
    for base_index in range(num_bases):
        non_base_inds = np.arange(num_bases - 1)
        non_base_inds[base_index:] += 1
        non_base_indices[base_index] = non_base_inds
    non_base_indices_misc = np.zeros(num_bases - 1)  # For miscellaneous bases

    # Allocate array to hold ISM scores, which are differences from original
    ism_scores = np.zeros_like(input_seq)  # Default 0, actual bases stay 0

    # Allocate array to hold the input sequences to feed in: original, plus
    # all mutations
    num_muts = (num_bases - 1) * (end - start)
    seqs_to_run = np.empty((num_muts + 1, input_length, num_bases))
    seqs_to_run[0] = input_seq  # Fill in original sequence
    i = 1  # Next location to fill in a sequence to `seqs_to_run`
    for pos_index in range(start, end):
        one_loc = np.where(input_seq[pos_index])[0]
        if len(one_loc) == 1:
            # There should always be exactly 1 position that's a 1
            base_index = one_loc[0]
            for mut_index in non_base_indices[base_index]:
                # For each base index that's not the actual base, make the
                # mutation and put it into `seqs_to_run`
                seqs_to_run[i] = input_seq
                seqs_to_run[i][pos_index][base_index] = 0  # Actual base to 0
                seqs_to_run[i][pos_index][mut_index] = 1  # Mutated base to 1
                i += 1
        else:
            # Something went wrong (e.g. undefined base), so just set to all 0s
            for _ in range(num_bases - 1):
                seqs_to_run[i] = input_seq
                i += 1
            
    # Make the predictions and get the outputs
    output_vals = predict_func(seqs_to_run)
   
    # Map back the output values to the proper location, and store
    # difference from original
    orig_val = output_vals[0]
    output_diffs = output_vals - orig_val
    i = 1  # Next location to read difference from `output_diffs`
    for pos_index in range(start, end):
        one_loc = np.where(input_seq[pos_index])[0]
        if len(one_loc) == 1:
            base_index = one_loc[0]
            for mut_index in non_base_indices[base_index]:
                # For each base index that's not the actual base, put the score
                # into the proper location; actual bases stay 0
                ism_scores[pos_index][mut_index] = output_diffs[i]
                i += 1
        else:
            # Base was zero or otherwise not well formed, so set all scores to 0
            ism_scores[pos_index] = 0
    
    return ism_scores


def compute_ism_scores(
    model, input_seqs, slice_only=None, mean_normalize=True, batch_size=128,
    save_stem=None, verbose=True
):
    """
    Computes importance scores via in-silico mutagenesis for a given model and
    set of input sequences.
    Arguments:
        `model`: a trained model which takes in a B x L x 4 tensor of one-hot
            encoded sequences
        `input_seqs`: an N x L x 4 NumPy array or Torch tensor of one-hot
            sequences to perform interpretations for
        `slice_only`: if provided, this is a slice along the input length
            dimension for which the ISM computation is limited to
        `mean_normalize`: if True, mean-normalize ISM scores at each position
            (across bases) for each track
        `batch_size`: batch size for model prediction
        `save_stem`: if provided, save the input sequences to
            `{save_stem}onehots.npz` and the importance scores to
            `{save_stem}impscores.npz`; objects will be stored as N x 4 x L'
            arrays (note the change in dimension)
        `verbose`: if True, show progress bar
    Returns a parallel N x L' x 4 array of importance scores. L' = L if
    `slice_only` is None, otherwise is shorter.
    """
    if type(input_seqs) is torch.Tensor:
        input_seqs = input_seqs.detach().cpu().numpy()

    if save_stem is not None:
        np.savez(
            "%sonehots.npz" % save_stem,
            np.transpose(input_seqs, (0, 2, 1))
        )

    all_ism_scores = np.empty_like(input_seqs)
    num_seqs = len(input_seqs)
    t_iter = tqdm.trange(num_seqs) if verbose else range(num_seqs)

    def predict_func(input_seq_batch):
        # Return the set of outputs for the batch of input sequences
        num_in_batch = len(input_seq_batch)
        output_vals = np.empty(num_in_batch)
        num_batches = int(np.ceil(num_in_batch / batch_size))
        for i in range(num_batches):
            batch_slice = slice(i * batch_size, (i + 1) * batch_size)
            preds = model(
                torch.tensor(
                    input_seq_batch[batch_slice], device=DEVICE
                ).float()
            ).detach().cpu().numpy()
            if len(preds.shape) > 1:
                # Average over non-batch dimensions
                preds = np.mean(preds, axis=tuple(range(1, len(preds.shape))))
            output_vals[batch_slice] = preds

        return output_vals

    for seq_index in t_iter:
        # Run ISM for this sequence
        ism_scores = compute_ism_scores_single(
            input_seqs[seq_index], predict_func, slice_only=slice_only
        )
        all_ism_scores[seq_index] = ism_scores

    if mean_normalize:
        all_ism_scores = \
            all_ism_scores - np.mean(all_ism_scores, axis=2, keepdims=True)

    if save_stem is not None:
        np.savez(
            "%simpscores.npz" % save_stem,
            np.transpose(all_ism_scores, (0, 2, 1))
        )

    return all_ism_scores


def compute_input_gradient_scores(
    model, input_seqs, slice_only=None, mean_normalize=True, batch_size=128,
    save_stem=None, verbose=True
):
    """
    Computes importance scores via input gradients for a given model and set of
    input sequences.
    Arguments:
        `model`: a trained model which takes in a B x L x 4 tensor of one-hot
            encoded sequences
        `input_seqs`: an N x L x 4 NumPy array or Torch tensor of one-hot
            sequences to perform interpretations for
        `slice_only`: if provided, this is a slice along the input length
            dimension, where only this slice of importance scores are saved
        `mean_normalize`: if True, mean-normalize importance scores at each
            position (across bases) for each track
        `batch_size`: batch size for model prediction
        `save_stem`: if provided, save the input sequences to
            `{save_stem}onehots.npz` and the importance scores to
            `{save_stem}impscores.npz`; objects will be stored as N x 4 x L'
            arrays (note the change in dimension)
        `verbose`: if True, show progress bar
    Returns a parallel N x L' x 4 array of importance scores. L' = L if
    `slice_only` is None, otherwise is shorter.
    """
    explainer = captum.attr.InputXGradient(model.to(DEVICE))
    
    if type(input_seqs) is np.ndarray:
        input_seqs_arr = input_seqs
    else:
        input_seqs_arr = input_seqs.detach().cpu().numpy()

    if save_stem is not None:
        if slice_only:
            input_seqs_arr_lim = input_seqs_arr[:, slice_only]
        else:
            input_seqs_arr_lim = input_seqs_arr
        np.savez(
            "%sonehots.npz" % save_stem,
            np.transpose(input_seqs_arr_lim, (0, 2, 1))
        )

    imp_scores_arr = []
    num_batches = int(np.ceil(len(input_seqs_arr) / batch_size))
    t_iter = tqdm.trange(num_batches) if verbose else range(num_batches)
    for batch_i in t_iter:
        start = batch_i * batch_size
        end = start + batch_size
        one_hots_ten = \
            torch.tensor(input_seqs_arr[start : end]).float().to(DEVICE)
        scores_ten = explainer.attribute(one_hots_ten)

        scores_arr = scores_ten.detach().cpu().numpy()
        
        if slice_only:
            scores_arr = scores_arr[:, slice_only]

        if mean_normalize:
            scores_arr = scores_arr - np.mean(scores_arr, axis=2, keepdims=True)

        imp_scores_arr.append(scores_arr)

    imp_scores_arr = np.concatenate(imp_scores_arr)

    if save_stem is not None:
        np.savez(
            "%simpscores.npz" % save_stem,
            np.transpose(imp_scores_arr, (0, 2, 1))
        )

    return imp_scores_arr
