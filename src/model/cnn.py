import torch
import sacred
from model.util import sanitize_sacred_arguments, convolution_size
import tqdm
import math

model_ex = sacred.Experiment("cnn_model")

@model_ex.config
def config():
    # Number of convolutional layers to apply
    num_conv_layers = 3

    # Size of convolutional filter to apply
    conv_filter_sizes = [10, 5, 5]

    # Number of filters to use at each convolutional level (i.e. number of
    # channels to output)
    conv_filter_nums = [8, 8, 8]

    # Size max pool filter
    max_pool_size = 40

    # Strides for max pool filter
    max_pool_stride = 40

    # Number of fully-connected layers to apply
    num_fc_layers = 2

    # Number of hidden nodes in each fully-connected layer
    fc_sizes = [10, 5]
    
    # Whether to apply batch normalization
    batch_norm = True

    # Length of input sequence
    input_length = 500

    # Number of channels in input sequence
    input_dim = 4
 
    # Loss weight for overlapping convolutional filter weights
    conv_filter_over_loss_weight = 0
    
    # Loss weight for convolutional filter weights L1
    conv_filter_l1_loss_weight = 0

    # Loss weight for convolutional activation L1
    conv_act_l1_loss_weight = 0
   

class ConvNet(torch.nn.Module):
    def __init__(
        self, num_conv_layers, conv_filter_sizes, conv_filter_nums,
        max_pool_size, max_pool_stride, num_fc_layers, fc_sizes, batch_norm,
        input_length, input_dim, conv_filter_over_loss_weight=0,
        conv_filter_l1_loss_weight=0, conv_act_l1_loss_weight=0
    ):
        """
        Initializes a standard CNN architecture for regulatory genomics.
        Arguments:
            `num_conv_layers`: number of convolutional layers
            `conv_filter_sizes`: size of filters in each layer, a list
            `conv_filter_nums`: number of filters in each layer, a list
            `max_pool_size`: size of max-pool filter
            `max_pool_stride`: stride for max-pool filter
            `num_fc_layers`: number of linear layers at the end
            `fc_sizes`: number of hidden units in each linear layer
            `batch_norm`: whether or not to apply batch normalization
            `input_length`: length of input sequence
            `input_dim`: dimension of input sequence (e.g. 4 for DNA)
            `conv_filter_over_loss_weight`: loss weight for overlapping
                convolutional filter weights
            `conv_filter_l1_loss_weight`: loss weight for convolutional-filter
                L1 penalty
            `conv_act_l1_loss_weight`: loss weight for the convolutional-
                activation L1 penalty
        Note that loss weights can be scalar values, or they can be arrays which
        map epoch index {0, 1, ...} to a scalar loss weight. If the epoch index
        is longer than the length of the array, then the last weight is used.
        """
        super().__init__()

        self.creation_args = locals()
        del self.creation_args["self"]
        del self.creation_args["__class__"]
        self.creation_args = sanitize_sacred_arguments(self.creation_args)
        
        self.conv_filter_over_loss_weight = conv_filter_over_loss_weight
        self.conv_filter_l1_loss_weight = conv_filter_l1_loss_weight
        self.conv_act_l1_loss_weight = conv_act_l1_loss_weight

        assert len(conv_filter_sizes) == num_conv_layers
        assert len(conv_filter_nums) == num_conv_layers
        assert len(fc_sizes) == num_fc_layers

        # Define the convolutional layers
        depths = [input_dim] + conv_filter_nums
        self.conv_layers = torch.nn.ModuleList()
        for i in range(num_conv_layers):
            layer = [
                torch.nn.Conv1d(depths[i], depths[i + 1], conv_filter_sizes[i]),
                torch.nn.ReLU()
            ]
            if batch_norm:
                layer.append(torch.nn.BatchNorm1d(depths[i + 1]))
            self.conv_layers.append(torch.nn.Sequential(*layer))

        # Define the max pooling layer
        self.max_pool_layer = torch.nn.MaxPool1d(
            max_pool_size, stride=max_pool_stride
        )

        # Compute size of the pooling output
        conv_output_size = convolution_size(
            input_length, num_conv_layers, conv_filter_sizes
        )
        pool_output_size = math.floor(
            (conv_output_size - (max_pool_size - 1) - 1) / max_pool_stride
        ) + 1
        pool_output_depth = conv_filter_nums[-1]
        
        # Define the fully connected layers
        dims = [pool_output_size * pool_output_depth] + fc_sizes
        self.fc_layers = torch.nn.ModuleList()
        for i in range(num_fc_layers):
            layer = [
                torch.nn.Linear(dims[i], dims[i + 1]),
                torch.nn.ReLU()
            ]
            if batch_norm:
                layer.append(
                    torch.nn.BatchNorm1d(dims[i + 1])
                )
            self.fc_layers.append(torch.nn.Sequential(*layer))

        # Map last fully connected layer to final outputs
        self.last_fc_layer = torch.nn.Linear(fc_sizes[-1], 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_seq, return_interims=False):
        """
        Runs the forward pass of the model.
        Arguments:
            `input_seq`: a B x L x D tensor of the input sequence, where B is
                the batch dimension, L is the sequence length, and D is the
                feature dimension
            `return_interims`: if True, also return a dictionary of
                intermediates
        Returns a B x 1 tensor containing the predicted probabilities for each
        input sequence. If `return_interims` is True, also returns a dictionary
        containing a B x L' x F tensor of the first-layer convolutional
        activations.
        """
        # Convolutional layers
        conv_out = torch.transpose(input_seq, 1, 2)  # Shape: B x D x L
        for i, conv_layer in enumerate(self.conv_layers):
            if i == 0 and return_interims:
                conv_out = conv_layer[1](conv_layer[0](conv_out))
                conv_acts_cache = conv_out
                for j in range(2, len(conv_layer)):
                    conv_out = conv_layer[j](conv_out)
                interims = {"conv_acts": torch.transpose(conv_acts_cache, 1, 2)}
            else:
                conv_out = conv_layer(conv_out)

        # Max pooling
        pool_out = self.max_pool_layer(conv_out)  # Shape: B x D' x L'
        pool_out = pool_out.view(len(pool_out), -1)  # Shape: B x D'L'

        # Linear layers
        fc_out = pool_out
        for fc_layer in self.fc_layers:
            fc_out = fc_layer(fc_out)

        out = self.sigmoid(self.last_fc_layer(fc_out))  # Shape: B x 1
        
        if return_interims:
            return out, interims
        else:
            return out

    def conv_filter_weight_losses(self, conv_acts):
        """
        Computes losses for the convolutional-filter weights, penalizing filters
        which fire in proximity due to the same part of the input sequence, as
        well as the weights themselves.
        Arguments:
            `conv_acts`: a B x L' x F tensor of the convolutional-filter
                activations (F is the number of filters)
        Returns a B x 1 tensor of loss values for the overlapping filters, and a
        B x 1 tensor of loss values which is simply the L1 norm of the weights,
        tiled.
        """
        conv_weights = torch.transpose(self.conv_layers[0][0].weight, 1, 2)
        # Shape: F x W x D
        assert conv_weights.shape[0] == conv_acts.shape[2]
        f = conv_weights.shape[0]
        w = conv_weights.shape[1]
        l = conv_acts.shape[1]

        # Given a position i, we have a set of positions j in
        # [i - (W - 1) .. i + (W - 1)], for which we want to make sure no other
        # filters have overlapping weights (i.e. non-zero) at j. Because of how
        # i and j are defined, the sum of the weights at each sliding window are
        # the same regardless of what i we are at, and can be computed using
        # cumulative sums.
        conv_weight_abs_total = torch.sum(torch.abs(conv_weights), dim=2)
        conv_weight_cumsum = torch.cumsum(conv_weight_abs_total, dim=1)
        conv_weight_sums_a = torch.cat([
            conv_weight_cumsum,
            torch.sum(conv_weight_abs_total, dim=1, keepdims=True) - \
                conv_weight_cumsum[:, :-1]
        ], dim=1)
        conv_weight_sums_b = torch.flip(conv_weight_sums_a, dims=(1,))
        # Shape: F x S, where S = 2W - 1
        # Note: we use conv_weight_sums_a for the stationary filter, and we use
        # conv_weight_sums_b for the mobile filter, following the notation in
        # the docstring

        # For each i, get the top filter 
        top_act_inds = torch.argmax(conv_acts, dim=2)  # Shape: B x L'
        
        # For each position i, let filter a be the top-activated filter. For all
        # other filters b != a, we penalize the product of the overlapping
        # weights between a and b, scaled by b's activation at each window. This
        # means that if b was not activated at a position, there is no penalty
        # for the weights overlapping. This also takes care of the case where
        # there is no activation at i.

        a_weight_sums = conv_weight_sums_a[top_act_inds]  # Shape: B x L' x S
        # Get the products of the weights for every window between top filter
        # and every filter (including itself, but we'll get rid of that later)
        conv_weight_prods = a_weight_sums[:, :, None, :] * \
            conv_weight_sums_b[None, None]
        # Shape: B x L' x F x S

        # Get the activations of every filter along the sliding windows. We do
        # this by first getting the sliding-window indices for i = 0, then
        # adding all possible i, and then capping off anything that runs over
        # later on
        window_inds = torch.tile(
            torch.arange(-(w - 1), (w - 1) + 1)[None], (l, 1)
        )  # Shape: L' x S
        window_inds = window_inds + torch.arange(l)[:, None]  # Add i
        window_inds[window_inds >= l] = -1  # Set anything over to -1
        window_acts = conv_acts[:, window_inds]  # Shape: B x L' x S x F
        window_acts[:, window_inds < 0] = 0  # Set overruns to 0 activation
        window_acts = torch.transpose(window_acts, 2, 3)
        # Shape: B x L' x F x S

        # Now also set the window activations to 0 for the filter which had the
        # top activation; this is a trick so that we are multiplying the top-
        # activated filter with every _other_ filter only
        window_acts[
            torch.arange(window_acts.shape[0])[:, None].expand_as(top_act_inds),
            torch.arange(window_acts.shape[1])[None].expand_as(top_act_inds),
            top_act_inds
        ] = 0

        # Finally, we weight the product of the windows with the activations
        weighted_prods = conv_weight_prods * window_acts
        # Shape: B x L' x F x S

        # Normalize the products by the sum of all weights (over all filters)
        weighted_prods_norm = weighted_prods / \
            torch.sum(torch.abs(conv_weights))

        conv_weight_l1 = torch.sum(torch.abs(conv_weights))

        final_losses = \
            torch.mean(weighted_prods_norm, dim=(1, 2, 3))[:, None], \
            torch.tile(conv_weight_l1, (len(conv_acts), 1))
        # Shape: B x 1
       
        return final_losses

    def conv_act_sparsity_loss(self, conv_acts):
        """
        Computes a loss for the convolutional-filter activations, rewarding
        sparsity of activations.
        Arguments:
            `conv_acts`: a B x L' x F tensor of the convolutional-filter
                activations (F is the number of filters)
        Returns a B x 1 tensor of loss values, which is simply the L1 norm of
        the activations
        """
        return torch.sum(torch.abs(conv_acts), dim=(1, 2))[:, None]
        # Shape: B x 1
    
    def loss(
        self, pred_probs, true_vals, conv_acts=None, epoch_num=None,
        return_components=False, **kwargs
    ):
        """
        Computes total loss value for the predicted probabilities given the true
        values or probabilities. This loss includes all loss functions weighted
        by the specified weights.
        Arguments:
            `pred_probs`: a B x 1 tensor of predicted probabilities
            `true_vals`: a B x 1 tensor of binary labels or true probabilities
            `conv_acts`: a B x L' x F tensor of the convolutional-filter
                activations (F is the number of filters); if None, the
                associated loss is ignored
            `epoch_num`: integer epoch number {0, 1, ...}, which is used only if
                any of the loss weights are functions which require it
            `return_components`: if True, also return the loss components in a
                dictionary
        Returns a B x 1 tensor of loss values, and optionally also the loss
        components in a dictionary: a B x 1 tensor of the convolutional-filter
        overlap losses, a B x 1 tensor of convolutional-filter L1 losses, a
        B x 1 tensor of convolutional-activation L1 losses, and a B x 1 tensor
        of prediction losses.
        """ 
        weight_func = lambda w: w[min(epoch_num, len(w) - 1)] \
            if hasattr(w, "__getitem__") else w

        conv_filter_over_loss_weight = weight_func(
            self.conv_filter_over_loss_weight
        )
        conv_filter_l1_loss_weight = weight_func(
            self.conv_filter_l1_loss_weight
        )
        conv_act_l1_loss_weight = weight_func(
            self.conv_act_l1_loss_weight
        )

        zero_loss = torch.zeros(
            (pred_probs.shape[0], 1), device=pred_probs.device
        )

        if (conv_acts is not None) and \
            (return_components or conv_filter_over_loss_weight or
            conv_filter_l1_loss_weight):
            conv_filter_losses = self.conv_filter_weight_losses(conv_acts)
        else:
            conv_filter_losses = (zero_loss, zero_loss)

        if (conv_acts is not None) and \
            (return_components or conv_act_l1_loss_weight):
            conv_act_loss = self.conv_act_sparsity_loss(conv_acts)
        else:
            conv_act_loss = zero_loss

        pred_loss = torch.nn.functional.binary_cross_entropy(
            pred_probs, true_vals, reduction="none"
        )

        final_loss = \
            (conv_filter_over_loss_weight * conv_filter_losses[0]) + \
            (conv_filter_l1_loss_weight * conv_filter_losses[1]) + \
            (conv_act_l1_loss_weight * conv_act_loss) + \
            pred_loss

        if return_components:
            return final_loss, {
                "conv_filter_overlap": conv_filter_losses[0],
                "conv_filter_l1": conv_filter_losses[1],
                "conv_act_l1": conv_act_loss,
                "pred": pred_loss
            }
        else:
            return final_loss


@model_ex.command
def create_model(
    num_conv_layers, conv_filter_sizes, conv_filter_nums, max_pool_size,
    max_pool_stride, num_fc_layers, fc_sizes, batch_norm, input_length,
    input_dim, conv_filter_over_loss_weight, conv_filter_l1_loss_weight,
    conv_act_l1_loss_weight
):
    """
    Create a ConvNet with the given parameters.
    """
    return ConvNet(
        num_conv_layers, conv_filter_sizes, conv_filter_nums, max_pool_size,
        max_pool_stride, num_fc_layers, fc_sizes, batch_norm, input_length,
        input_dim, conv_filter_over_loss_weight, conv_filter_l1_loss_weight,
        conv_act_l1_loss_weight
    )


@model_ex.automain
def main():
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"

    model = create_model().to(DEVICE)

    print(
        "Number of parameters: %d" % sum(p.numel() for p in model.parameters())
    )
    
    input_seq = torch.randn(128, 500, 4).to(DEVICE)
    true_probs = torch.randint(2, size=(128, 1)).float().to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters())

    t_iter = tqdm.trange(1000, desc="Loss=----")
    for _ in t_iter:
        optimizer.zero_grad()

        pred_probs = model(input_seq)

        loss = torch.mean(model.loss(pred_probs, true_probs))
        
        t_iter.set_description("Loss=%.3f" % loss.item())

        loss.backward()
        optimizer.step()
