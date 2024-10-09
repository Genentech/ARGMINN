import torch
import numpy as np
import tqdm
import os
import sacred
import model.util as util
import feature.simulated_dataset as simulated_dataset
import feature.experimental_dataset as experimental_dataset
import model.mechint_net as mechint_net
import model.cnn as cnn
import model.explainn as explainn
import train.performance as performance

MODEL_DIR = os.environ.get(
    "MODEL_DIR",
    "/projects/site/gred/resbioai/tsenga5/mechint_regnet/models/trained_models/misc"
)

train_ex = sacred.Experiment("train", ingredients=[
    simulated_dataset.dataset_ex,
    experimental_dataset.dataset_ex,
    mechint_net.model_ex,
    cnn.model_ex,
    explainn.model_ex
])

train_ex.observers.append(
    sacred.observers.FileStorageObserver.create(MODEL_DIR)
)

# Define device
if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


@train_ex.config
def config():
    # Number of training epochs
    num_epochs = 20

    # Learning rate
    learning_rate = 0.001

    # Whether or not to use early stopping
    early_stopping = True

    # Number of epochs to save validation loss (set to 1 for one step only)
    early_stop_hist_len = 3

    # Minimum improvement in loss at least once over history to not stop early
    early_stop_min_delta = 0.001

    # Training seed
    train_seed = None


def run_epoch(
    data_loader, mode, model, optimizer=None, epoch_num=None,
    return_extras=False
):
    """
    Runs the data from the data loader once through the model, to train or
    simply predict.
    Arguments:
        `data_loader`: an instantiated `DataLoader` instance that gives batches
            of data; each batch must yield the inputs and output labels as the
            first two elements; extra elements are ignored but may be returned
        `mode`: one of "train", "eval"; if "train", run the epoch and perform
            backpropagation; if "eval", only do evaluation
        `model`: the current PyTorch model being trained/evaluated
        `optimizer`: an instantiated PyTorch optimizer, for training mode
        `epoch_num`: integer epoch number {0, 1, ...} passed to loss function if
            needed
        `return_extras`: if specified, also return the individual loss
            components, extra model outputs, and the output of the data loader
    Returns the following:
    1) A NumPy M-vector of overall loss values averaged by batch
    2) A NumPy N-vector of overall loss values for each individual example
    3) A tuple of N x ... NumPy arrays containing the main model outputs, one
        entry per output
    If `return_extras` is True, also return the following:
    4) A tuple of NumPy N-vectors of individual loss components, as determined
        by the model's loss function
    5) A tuple of N x ... NumPy arrays containing secondary model outputs, as
        determined by the model's forward function
    6) A tuple of N x ... NumPy arrays containing the output of the data loader;
        some objects may be lists of length N if the individual objects yielded
        from the data loader are not arrays themselves
    """
    assert mode in ("train", "eval")
    if mode == "train":
        assert optimizer is not None
    else:
        assert optimizer is None 

    data_loader.dataset.on_epoch_start()  # Set-up the epoch
    num_batches = len(data_loader.dataset)
    batch_size = data_loader.dataset.batch_size
    t_iter = tqdm.tqdm(
        enumerate(data_loader), total=num_batches, desc="\tLoss: ---"
    )

    if mode == "train":
        model.train()  # Switch to training mode
        torch.set_grad_enabled(True)
    
    # Allocate empty NumPy arrays to hold losses; other arrays will be allocated
    # once we have an idea of what the sizes will be
    # Real number of samples may be smaller if there is partial last batch
    num_samples_exp = num_batches * batch_size
    all_losses_batched = np.empty(num_batches)
    all_losses = np.empty((num_samples_exp, 1))  # Assume losses are B x 1
    all_model_outputs = None
    num_samples_seen = 0  # Real number of samples seen
    if return_extras:
        all_losses_comps = None
        all_model_outputs_other = None
        all_input_data = None

    for batch_i, batch in t_iter:
        x, y = batch[0], batch[1]
        y = y[:, None]  # Shape: B x 1

        if return_extras:
            x_np, y_np = x, y  # Save NumPy copies
        x = torch.tensor(x).float().to(DEVICE)
        y = torch.tensor(y).float().to(DEVICE)

        # Make predictions
        preds, interims = model(x, return_interims=True)

        # Compute loss
        losses, losses_comps = model.loss(
            preds, y, **interims, epoch_num=epoch_num, return_components=True
        )
        batch_loss = torch.mean(losses)
        batch_loss_val = batch_loss.item()

        if not np.isfinite(batch_loss_val):
            continue

        if mode == "train":
            optimizer.zero_grad()
            batch_loss.backward()  # Compute gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()  # Update weights through backprop

        t_iter.set_description("\tLoss: %6.4f" % batch_loss_val)

        # Allocate needed arrays
        if all_model_outputs is None:
            all_model_outputs = np.empty((num_samples_exp,) + preds.shape[1:])
            if return_extras:
                all_losses_comps = {}
                for key, val in losses_comps.items():
                    all_losses_comps[key] = np.empty(
                        (num_samples_exp,) + val.shape[1:]
                    )
                all_model_outputs_other = {}
                for key, val in interims.items():
                    all_model_outputs_other[key] = np.empty(
                        (num_samples_exp,) + val.shape[1:]
                    )
                all_input_data = [
                    np.empty((num_samples_exp,) + x.shape[1:]),
                    np.empty((num_samples_exp,) + y.shape[1:])
                ]
                for i in range(2, len(batch)):
                    if type(batch[i]) is np.ndarray:
                        all_input_data.append(np.empty(
                            (num_samples_exp,) + batch[i].shape[1:],
                            dtype=batch[i].dtype
                        ))
                    elif type(batch[i]) is list:
                        all_input_data.append([])
                    else:
                        raise ValueError(
                            "Unsupported type of input object: %s" % \
                            type(batch[i])
                        )
        
        # Save info
        num_in_batch = preds.shape[0]
        start, end = num_samples_seen, num_samples_seen + num_in_batch
        all_losses_batched[batch_i] = batch_loss_val
        all_losses[start:end] = losses.detach().cpu().numpy()
        all_model_outputs[start:end] = preds.detach().cpu().numpy()
        if return_extras:
            for key, val in losses_comps.items():
                all_losses_comps[key][start:end] = val.detach().cpu().numpy()
            for key, val in interims.items():
                all_model_outputs_other[key][start:end] = \
                    val.detach().cpu().numpy()
            all_input_data[0][start:end] = x_np
            all_input_data[1][start:end] = y_np
            for i in range(2, len(batch)):
                if type(batch[i]) is np.ndarray:
                    all_input_data[i][start:end] = batch[i]
                elif type(batch[i]) is list:
                    all_input_data[i].extend(batch[i])
                else:
                    raise ValueError(
                        "Unsupported type of input object: %s" % type(batch[i])
                    )
        num_samples_seen += num_in_batch

    # Truncate the saved data to the proper size, based on how many samples
    # actually seen
    all_losses = all_losses[:num_samples_seen]
    all_model_outputs = all_model_outputs[:num_samples_seen]
    if return_extras:
        for key, val in all_losses_comps.items():
            all_losses_comps[key] = all_losses_comps[key][:num_samples_seen]
        for key, val in all_model_outputs_other.items():
            all_model_outputs_other[key] = \
                all_model_outputs_other[key][:num_samples_seen]
        for i in range(len(all_input_data)):
            all_input_data[i] = all_input_data[i][:num_samples_seen]

        return all_losses_batched, all_losses, all_model_outputs, \
            all_losses_comps, all_model_outputs_other, all_input_data
    else:
        return all_losses_batched, all_losses, all_model_outputs


@train_ex.command
def train_model(
    train_loader, val_loader, test_loader, model, num_epochs, learning_rate,
    early_stopping, early_stop_hist_len, early_stop_min_delta, train_seed, _run
):
    """
    Trains the network for the given training and validation data.
    Arguments:
        `train_loader` (DataLoader): a data loader for the training data
        `val_loader` (DataLoader): a data loader for the validation data
        `test_loader` (DataLoader): a data loader for the test data
        `model`: PyTorch model to train
    Note that all data loaders are expected to yield pairs of inputs and labels.
    """
    run_num = _run._id
    output_dir = os.path.join(MODEL_DIR, str(run_num))
    
    if train_seed:
        torch.manual_seed(train_seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if early_stopping:
        val_epoch_loss_hist = []

    best_v_epoch_loss, best_model_state = float("inf"), model.state_dict()

    for epoch_i in range(num_epochs):
        if torch.cuda.is_available:
            torch.cuda.empty_cache()  # Clear GPU memory

        t_losses_batched, t_losses, _ = run_epoch(
            train_loader, "train", model, optimizer=optimizer, epoch_num=epoch_i
        )

        t_epoch_loss = np.nanmean(t_losses_batched)
        print(
            "Train epoch %d: average loss = %6.10f" % (
                epoch_i + 1, t_epoch_loss
            )
        )
        _run.log_scalar("train_epoch_loss", t_epoch_loss)
        _run.log_scalar("train_batch_losses", t_losses_batched.tolist())

        v_losses_batched, v_losses, _, v_losses_comps, _, _ = run_epoch(
            val_loader, "eval", model, epoch_num=epoch_i, return_extras=True
        )
        for key, vals in v_losses_comps.items():
            loss_comp = np.nanmean(vals)
            print("\t\t%s: %s" % (key, loss_comp))
            _run.log_scalar("val_epoch_loss_%s" % key, loss_comp)

        v_epoch_loss = np.nanmean(v_losses_batched)
        print(
            "Valid epoch %d: average loss = %6.10f" % (
                epoch_i + 1, v_epoch_loss
            )
        )
        _run.log_scalar("val_epoch_loss", v_epoch_loss)
        _run.log_scalar("val_batch_losses", v_losses_batched.tolist())

        # Save trained model for the epoch
        model_path = os.path.join(
            output_dir, "epoch_%d_ckpt.pth" % (epoch_i + 1)
        )
        last_link_path = os.path.join(output_dir, "last_ckpt.pth")
        best_link_path = os.path.join(output_dir, "best_ckpt.pth")

        # Save model
        util.save_model(model, model_path)

        # Create symlink to last epoch
        if os.path.islink(last_link_path):
            os.remove(last_link_path)
        os.symlink(os.path.basename(model_path), last_link_path)

        # Save the model state dict of the epoch with the best validation loss
        if v_epoch_loss < best_v_epoch_loss:
            best_v_epoch_loss = v_epoch_loss
            best_model_state = model.state_dict()
            # Update the symlink
            if os.path.islink(best_link_path):
                os.remove(best_link_path)
            os.symlink(os.path.basename(model_path), best_link_path)

        # If losses are both NaN, then stop
        if np.isnan(t_epoch_loss) and np.isnan(v_epoch_loss):
            raise ValueError("Both training/validation losses are NaN")

        # Check for early stopping
        if early_stopping:
            if len(val_epoch_loss_hist) < early_stop_hist_len:
                val_epoch_loss_hist = [v_epoch_loss] + val_epoch_loss_hist
                # Not enough history yet; tack on the loss and stop
            else:
                # Tack on the new validation loss, kicking off the old one if
                # needed
                val_epoch_loss_hist = [v_epoch_loss] + \
                    val_epoch_loss_hist[:early_stop_hist_len] 
                best_delta = np.max(np.diff(val_epoch_loss_hist))
                if best_delta < early_stop_min_delta:
                    print("Stopping early")
                    break  # Not improving enough

    # Compute evaluation metrics and log them
    print("Computing test-set predictions:")
    # Load in the state of the epoch with the best validation loss first
    model.load_state_dict(best_model_state)
    losses_batched, losses, outputs, _, _, input_data = run_epoch(
        test_loader, "eval", model, epoch_num=epoch_i, return_extras=True
    )
    _run.log_scalar("test_epoch_loss", np.nanmean(losses_batched))
    _run.log_scalar("test_batch_losses", losses_batched.tolist())
    
    true_vals = input_data[1]  # Shape: B x 1
    pred_vals = outputs  # Shape: B x 1

    print("Computing test-set performance:")
    metrics = performance.compute_performance_metrics(
        np.squeeze(true_vals, axis=1), np.squeeze(pred_vals, axis=1),
        neg_upsample_factor=test_loader.dataset.negative_ratio
    )
    performance.log_performance_metrics(metrics, _run, log_prefix="test")

    max_key_len = max(len(k) for k in metrics.keys())
    for key, val in metrics.items():
        print("\t%s: %6.4f" % (key.ljust(max_key_len), val))


@train_ex.automain
def test():
    motif_config_path = "/projects/site/gred/resbioai/tsenga5/mechint_regnet/data/simulations/configs/tal_gata_exp_mix_config.json"

    train_loader = simulated_dataset.create_data_loader(
        motif_config_path, num_batches=100
    )
    val_loader = simulated_dataset.create_data_loader(
        motif_config_path, num_batches=10
    )
    test_loader = simulated_dataset.create_data_loader(
        motif_config_path, num_batches=10
    )
    
    # peaks_bed_path = "/home/tsenga5/mechint_regnet/data/encode/chipseq/ENCSR607XFI_CTCF_HepG2/ENCFF664UGR_idrpeaks.bed.gz"
    # full_chrom_set = ["chr" + str(i) for i in range(1, 24)] + ["chrX"]
    # val_chroms = ["chr10", "chr8"]
    # test_chroms = ["chr1"]
    # train_chroms = [
    #     c for c in full_chrom_set if c not in val_chroms + test_chroms
    # ]

    # train_loader = experimental_dataset.create_data_loader(
    #     peaks_bed_path, chrom_set=train_chroms
    # )
    # val_loader = experimental_dataset.create_data_loader(
    #     peaks_bed_path, chrom_set=val_chroms
    # )
    # test_loader = experimental_dataset.create_data_loader(
    #     peaks_bed_path, chrom_set=test_chroms
    # )

    conv_filter_over_loss_weight = np.concatenate([
        np.zeros(10), np.power(10, np.linspace(0.5, 5, 20)), np.tile(1e5, 10)
    ])
    conv_filter_l1_loss_weight = np.concatenate([
        np.zeros(10), np.power(10, np.linspace(-3, -2, 20)), np.tile(1e-2, 10)
    ])
    att_head_sparse_loss_weight = np.concatenate([
        np.zeros(10), np.power(10, np.linspace(-3, -2, 20)), np.tile(1e-2, 10)
    ])

    # model = mechint_net.create_model(
    #     conv_filter_over_loss_weight=conv_filter_over_loss_weight,
    #     conv_filter_l1_loss_weight=conv_filter_l1_loss_weight,
    #     att_head_sparse_loss_weight=att_head_sparse_loss_weight,
    #     num_att_layers=1
    # ).to(DEVICE)
    model = cnn.create_model(
        conv_filter_over_loss_weight=conv_filter_over_loss_weight,
        conv_filter_l1_loss_weight=conv_filter_l1_loss_weight
    ).to(DEVICE)
    
    # import feature.util as feature_util
    # motif_dict = feature_util.import_meme_motifs(
    #     "/projects/site/gred/resbioai/tsenga5/mechint_regnet/data/simulations/motifs/tal_gata.txt"
    # )
    # state_dict = model.state_dict()
    # conv_weights = np.zeros(state_dict["conv_layer.weight"].shape)

    # for i, key in enumerate(motif_dict.keys()):
    #     motif = motif_dict[key]
    #     conv_weights[i * 2, :, :len(motif)] = np.transpose(motif)
    #     conv_weights[(i * 2) + 1, :, :len(motif)] = np.transpose(np.flip(motif, axis=(0, 1)))

    # state_dict["conv_layer.weight"] = torch.tensor(conv_weights).to(DEVICE)
    # model.load_state_dict(state_dict)
    # model.conv_layer.weight.requires_grad = False
 
    train_model(
        train_loader, val_loader, test_loader, model, num_epochs=40,
        early_stopping=False
    )
