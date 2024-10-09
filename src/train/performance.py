import numpy as np
import sklearn.metrics

def compute_binary_accuracy(true_vals, pred_vals):
    """
    Computes the accuracy for the given binary labels and binary (thresholded)
    predictions (i.e. this checks for an exact match).
    Arguments:
        `true_vals`: NumPy vector containing binary true predictions
        `pred_vals`: NumPy vector of the same size as `true_vals`, containing
            binarized predictions
    Returns 3 scalar floats: the overall accuracy, the accuracy for only
    positives (i.e. where the true value is 1), and the accuracy for only
    negatives (i.e. where the true value is 0).
    """
    acc = np.sum(pred_vals == true_vals) / len(true_vals)

    pos_mask = true_vals == 1
    pred_vals_pos = pred_vals[pos_mask]
    pos_acc = np.sum(pred_vals_pos == 1) / len(pred_vals_pos)

    neg_mask = true_vals == 0
    pred_vals_neg = pred_vals[neg_mask]
    neg_acc = np.sum(pred_vals_neg == 0) / len(pred_vals_neg)

    return acc, pos_acc, neg_acc


def compute_imbalanced_precision_recall(
    true_vals, pred_vals, neg_upsample_factor=1
):
    """
    Computes the precision and recall for the given predicted probabilities and
    true binary labels. This function will correct for precision inflation due
    to downsampling the negatives during prediction. This will behave just like
    `sklearn.metrics.precision_recall_curve` if `neg_upsample_factor` is 1.
    Arguments:
        `true_vals`: NumPy vector containing true binary values (1 is positive)
        `pred_vals`: NumPy vector of the same size as `true_vals`, containing
            predicted probabilities
        `neg_upsample_factor`: A positive number at least 1, this is the factor
            at which the negatives were downsampled for prediction, and it also
            the factor to upscale the false positive rate
    Returns a NumPy vector of precision values at every possible threshold
    (corrected for downsampling negatives), a NumPy vector of recall values at
    every possible threshold, and a NumPy vector of the thresholds (i.e. the
    sorted prediction values). All NumPy vectors returned are the same size as
    `true_vals` and `pred_vals`.
    """
    # Sort the true values in descending order of prediction values
    sort_inds = np.flip(np.argsort(pred_vals))
    sort_true_vals = true_vals[sort_inds]
    sort_pred_vals = pred_vals[sort_inds]

    num_vals = len(true_vals)
    num_neg_vals = np.sum(true_vals == 0)
    num_pos_vals = num_vals - num_neg_vals

    # Identify the thresholds as the locations where the sorted predicted values
    # differ; these are indices where the _next_ entry is different from the
    # current one
    thresh_inds = np.where(np.diff(sort_pred_vals) != 0)[0]
    # Tack on the last threshold, which is missed by `diff`
    thresh_inds = np.concatenate([thresh_inds, [num_vals - 1]])
    thresh = sort_pred_vals[thresh_inds]

    # Get the number of entries at each threshold point (i.e. at each threshold,
    # there are this many entries with predicted value at least this threshold)
    num_above = np.arange(1, num_vals + 1)[thresh_inds]

    # Compute the true positives at each threshold (i.e. at each threshold, this
    # this many entries with predicted value at least this threshold are truly
    # positives)
    tp = np.cumsum(sort_true_vals)[thresh_inds]

    # Compute the false positives at each threshold (i.e. at each threshold,
    # this many entries with predicted value at least this threshold are truly
    # negatives)
    fp = num_above - tp

    # Compute the false negatives at each threshold (i.e. at each threshold,
    # this many entries with predicted value below this threshold are truly
    # positives)
    fn = num_vals - num_above - (num_neg_vals - fp)

    # The precision is TP / (TP + FP); with `neg_upsample_factor` of 1, FP
    # remains the same; otherwise, there are presumably this many times more
    # true negatives above each threshold
    numer = tp
    denom = tp + (fp * neg_upsample_factor)
    denom[denom == 0] = 1  # When dividing, if there are no positives, keep 0
    precis = numer / denom

    # The recall is TP / (TP + FN); TP + FN is also the total number of true
    # positives
    recall = tp / (tp + fn)
    # Only NaN if no positives at all

    # Cut off the values after which the true positives has reached the maximum
    # (i.e. number of positives total); after this point, recall won't change
    max_ind = np.min(np.where(tp == num_pos_vals)[0])
    precis = precis[:max_ind + 1]
    recall = recall[:max_ind + 1]
    thresh = thresh[:max_ind + 1]

    # Flip the arrays, and concatenate final precision/recall values (i.e. when
    # there are no positives, precision is 1 and recall is 0)
    precis, recall, thresh = np.flip(precis), np.flip(recall), np.flip(thresh)
    precis = np.concatenate([precis, [1]])
    recall = np.concatenate([recall, [0]])
    return precis, recall, thresh


def compute_precision_recall_scores(precis, recall, thresholds, pos_thresh=0.5):
    """
    From parallel NumPy vectors of precision, recall, and their increasing
    thresholds, returns the precision and recall scores, if the threshold to
    call a positive is `pos_thresh`. In practice, the inputs should be the
    output of `compute_imbalanced_precision_recall`, or
    `sklearn.metrics.precision_recall_curve`.
    Arguments:
        `precis`: NumPy vector of precision values for each prediction threshold
        `recall`: NumPy vector of recall values for each prediction threshold
        `thresholds`: NumPy vector of thresholds (in sorted order)
    Returns a scalar precision score and a scalar recall score.
    """
    assert np.all(np.diff(thresholds) >= 0)
    inds = np.where(thresholds >= pos_thresh)[0]
    if not inds.size:
        # If there are no predicted positives, then precision is 0
        return 0, 0
    # Index of the closest threshold at least pos_thresh:
    thresh_ind = np.min(inds)
    return precis[thresh_ind], recall[thresh_ind]


def compute_performance_metrics(
    true_vals, pred_vals, neg_upsample_factor=None, acc_thresh=0.5
):
    """
    Computes varios evaluation metrics and returns them as a dictionary.
    Arguments:
        `true_vals`: NumPy vector of true binary values
        `pred_vals`: parallel NumPy vector of predicted probabilities
        `neg_upsample_factor`: if provided, also compute precision/recall scores
            as if the negatives are inflated by this factor; must be at least 1
        `acc_thresh`: threshold for rounding predictions for computing accuracy
    Returns a dictionary of the following structure:
    {
        "acc": <overall accuracy>,
        "pos_acc": <accuracy on positives>,
        "neg_acc": <accuracy on negatives>,
        "auroc": <auROC>,
        "precis": <precision>,
        "recall": <recall>,
        "auprc": <auPRC>,
        "c_precis": <precision after upsampling correction (optional)>,
        "c_recall": <recall after upsampling correction (optional)>,
        "c_auprc": <auPRC after upsampling correction (optional)>
    }
    """
    assert np.all((true_vals == 0) | (true_vals == 1))

    metrics = {}

    # Overall accuracy, and accuracy for each class
    pred_vals_rounded = np.copy(pred_vals)
    pred_vals_rounded[pred_vals_rounded > acc_thresh] = 1
    pred_vals_rounded[pred_vals_rounded <= acc_thresh] = 0
    acc, pos_acc, neg_acc = compute_binary_accuracy(true_vals, pred_vals_rounded)
    metrics["acc"] = acc
    metrics["pos_acc"] = pos_acc
    metrics["neg_acc"] = neg_acc

    # auROC
    auroc = sklearn.metrics.roc_auc_score(true_vals, pred_vals)
    metrics["auroc"] = auroc

    # Precision, recall, auPRC
    precis, recall, thresh = compute_imbalanced_precision_recall(
        true_vals, pred_vals
    )
    precis_score, recall_score = compute_precision_recall_scores(
        precis, recall, thresh
    )
    auprc = sklearn.metrics.auc(recall, precis)
    metrics["precis"] = precis_score
    metrics["recall"] = recall_score
    metrics["auprc"] = auprc

    # Precision, auPRC, corrected for downsampling
    if neg_upsample_factor:
        c_precis, c_recall, c_thresh = compute_imbalanced_precision_recall(
            true_vals, pred_vals, neg_upsample_factor=neg_upsample_factor
        )
        c_precis_score, c_recall_score = compute_precision_recall_scores(
            c_precis, c_recall, c_thresh
        )
        c_auprc = sklearn.metrics.auc(c_recall, c_precis)
        metrics["c_precis"] = c_precis_score
        metrics["c_recall"] = c_recall_score
        metrics["c_auprc"] = c_auprc

    return metrics


def log_performance_metrics(metrics, _run, log_prefix=None):
    """
    Given the metrics dictionary returned by `compute_performance_metrics`, logs
    them to a Sacred logging object.
    Arguments:
        `metrics`: a dictionary of metrics returned by
            `compute_performance_metrics`
        `_run`: a Sacred logging object
        `log_prefix`: if provided, "{log_prefix}_" is prepended to every output
            key in the log
    """
    prefix = log_prefix + "_" if log_prefix else ""
    _run.log_scalar(prefix + "acc", metrics["acc"])
    _run.log_scalar(prefix + "pos_acc", metrics["pos_acc"])
    _run.log_scalar(prefix + "neg_acc", metrics["neg_acc"])
    _run.log_scalar(prefix + "auroc", metrics["auroc"])
    _run.log_scalar(prefix + "precis_score", metrics["precis"])
    _run.log_scalar(prefix + "recall_score", metrics["recall"])
    _run.log_scalar(prefix + "auprc", metrics["auprc"])
    if "c_auprc" in metrics:
        _run.log_scalar(prefix + "corr_precis_score", metrics["c_precis"])
        _run.log_scalar(prefix + "corr_recall_score", metrics["c_recall"])
        _run.log_scalar(prefix + "corr_auprc", metrics["c_auprc"])


if __name__ == "__main__":
    def test_accuracies():
        np.random.seed(20200218)
        vec_size = 50
        true_vals = np.random.randint(2, size=vec_size)
        pred_vals = np.random.random(vec_size)
    
        print("Testing accuracies...")
        pred_vals_rounded = np.round(pred_vals)
        acc, pos_acc, neg_acc = compute_binary_accuracy(
            true_vals, pred_vals_rounded
        )
    
        num_pos, num_neg = 0, 0
        num_pos_right, num_neg_right = 0, 0
        for i in range(vec_size):
            if true_vals[i] == 1:
                num_pos += 1
                if pred_vals[i] >= 0.5:
                    num_pos_right += 1
            else:
                num_neg += 1
                if pred_vals[i] < 0.5:
                    num_neg_right += 1
    
        print("\tSame result? %s" % all([
            acc == (num_pos_right + num_neg_right) / vec_size,
            pos_acc == num_pos_right / num_pos,
            neg_acc == num_neg_right / num_neg
        ]))
    
    
    def test_corrected_precision_auprc():
        np.random.seed(20200218)
        vec_size = 10000
        neg_upsample_factor = 5
        estimate_tolerance = 0.02
    
        def test_single_result(true_vec, pred_vec):
            """
            Tests similarity of precision/recall/auPRC computation to `sklearn`
            library, without any downsampling of negatives
            """
            precis, recall, thresh = \
                compute_imbalanced_precision_recall(
                    true_vec, pred_vec, neg_upsample_factor=1
                )
            precis_score, recall_score = compute_precision_recall_scores(
                precis, recall, thresh
            )
            auprc = sklearn.metrics.auc(recall, precis)
            sk_precis, sk_recall, sk_thresh = \
                sklearn.metrics.precision_recall_curve(true_vec, pred_vec)
            sk_precis_score = sklearn.metrics.precision_score(
                true_vec, np.round(pred_vec)
            )
            sk_recall_score = sklearn.metrics.recall_score(
                true_vec, np.round(pred_vec)
            )
            sk_auprc = sklearn.metrics.auc(sk_recall, sk_precis)
            print("\tSame result? %s" % all([
                np.allclose(precis, sk_precis),
                np.allclose(recall, sk_recall),
                np.allclose(thresh, sk_thresh),
                precis_score == sk_precis_score,
                recall_score == sk_recall_score,
                auprc == sk_auprc
            ]))
    
        def test_neg_sampling_result(true_vec, pred_vec, neg_upsample_factor):
            """
            Tests that after down-sampling negatives and re-inflating them using
            `neg_upsample_factor`, the precision/recall/auPRC are roughly the same
            as if no down-sampling had occurred.
            """
            # Get results without downsampling negatives
            precis, recall, thresh = \
                compute_imbalanced_precision_recall(
                    true_vec, pred_vec, neg_upsample_factor=1
                )
            precis_score, recall_score = compute_precision_recall_scores(
                precis, recall, thresh
            )
            auprc = sklearn.metrics.auc(recall, precis)
    
            # Subsample negatives
            pos_mask = true_vals == 1
            neg_mask = true_vals == 0
            sub_mask = np.random.choice(
                [True, False], size=vec_size,
                p=[1 / neg_upsample_factor, 1 - (1 / neg_upsample_factor)]
            )
            keep_mask = pos_mask | (neg_mask & sub_mask)
            c_precis, c_recall, c_thresh = \
                compute_imbalanced_precision_recall(
                    true_vec[keep_mask], pred_vec[keep_mask],
                    neg_upsample_factor=neg_upsample_factor
                )
            c_precis_score, c_recall_score = \
                compute_precision_recall_scores(
                    c_precis, c_recall, c_thresh
                )
            c_auprc = sklearn.metrics.auc(c_recall, c_precis)
            print("\tAll within %s? %s" % (
                estimate_tolerance,
                all([
                    abs(precis_score - c_precis_score) < estimate_tolerance,
                    abs(recall_score - c_recall_score) < estimate_tolerance,
                    abs(auprc - c_auprc) < estimate_tolerance
                ])
            ))
    
        true_vals = np.random.choice(2, size=vec_size)
    
        # Random predictions
        pred_vals = np.random.random(vec_size)
    
        print("Testing precision/recall/auPRC on random data without correction...")
        test_single_result(true_vals, pred_vals)
    
        print("Testing precision/recall/auPRC on random data with correction...")
        test_neg_sampling_result(true_vals, pred_vals, 5)
    
        # Predictions are a bit closer to being correct; make predictions Gaussian
        # noise centered as 0 or 1, and cut off anything outside [0, 1]
        pred_vals = np.empty(vec_size)
        rand = np.random.randn(vec_size) / 2
        pos_mask = true_vals == 1
        pred_vals[pos_mask] = rand[pos_mask] + 1
        neg_mask = true_vals == 0
        pred_vals[neg_mask] = rand[neg_mask]
        pred_vals[pred_vals < 0] = 0
        pred_vals[pred_vals > 1] = 1
    
        print("Testing precision/recall/auPRC on good data without correction...")
        test_single_result(true_vals, pred_vals)
    
        print("Testing precision/recall/auPRC on good data with correction...")
        test_neg_sampling_result(true_vals, pred_vals, 5)


    class FakeLogger:
        def log_scalar(self, a, b):
            print("\t%s: %s" % (a, b))


    def test_all_metrics_on_different_predictions():
        np.random.seed(20200218)
        batch_size = 1000
        true_vals = np.random.randint(2, size=batch_size)
    
        _run = FakeLogger()
    
        # Make some "perfect" predictions, which are identical to truth
        print("Testing all metrics on some perfect predictions...")
        pred_vals = true_vals
        metrics = compute_performance_metrics(true_vals, pred_vals, 1)
        log_performance_metrics(metrics, _run, log_prefix="perfect")
    
        # Make some "good" predictions, which are close to truth; make predictions
        # Gaussian noise centered as 0 or 1, and cut off anything outside [0, 1]
        print("Testing all metrics on some good predictions...")
        pred_vals = np.empty(batch_size)
        rand = np.random.randn(batch_size) / 2
        pos_mask = true_vals == 1
        pred_vals[pos_mask] = rand[pos_mask] + 1
        neg_mask = true_vals == 0
        pred_vals[neg_mask] = rand[neg_mask]
        pred_vals[pred_vals < 0] = 0
        pred_vals[pred_vals > 1] = 1
        metrics = compute_performance_metrics(true_vals, pred_vals, 1)
        log_performance_metrics(metrics, _run, log_prefix="good")
    
        # Make some "bad" predictions, which are just random
        print("Testing all metrics on some bad predictions...")
        pred_vals = np.random.random(batch_size)
        metrics = compute_performance_metrics(true_vals, pred_vals, 1)
        log_performance_metrics(metrics, _run, log_prefix="bad")
    
    test_accuracies()
    test_corrected_precision_auprc()
    test_all_metrics_on_different_predictions()
