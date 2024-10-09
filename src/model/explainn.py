import torch
import sacred
from model.util import sanitize_sacred_arguments, convolution_size
import tqdm
import math

model_ex = sacred.Experiment("explainn_model")

@model_ex.config
def config():
    # Number of convolutional layers to apply
    num_units = 8

    # Size of convolutional filter to apply
    conv_filter_size = 19

    # Size max pool filter
    max_pool_size = 7

    # Stride for max pool filter
    max_pool_stride = 7

    # Number of hidden nodes in fully-connected layer of each unit
    fc_size = 100

    # Dropout rate for fully-connected layer
    dropout_rate = 0.3
    
    # Length of input sequence
    input_length = 500

    # Number of channels in input sequence
    input_dim = 4


class Exp(torch.nn.Module):
    def forward(self, x):
        return torch.exp(x)


class Unsqueeze(torch.nn.Module):
    def forward(self, x):
        return x.unsqueeze(-1)
 

class ExplaiNN(torch.nn.Module):
    def __init__(
        self, num_units, conv_filter_size, max_pool_size, max_pool_stride,
        fc_size, dropout_rate, input_length, input_dim
    ):
        """
        Initializes the ExplaiNN model architecture for interpretable regulatory
        genomics.
        Arguments:
            `num_units`: number of CNN units to create, U
            `conv_filter_size`: size of convolutional filter in each unit
            `max_pool_size`: size of max-pool filter
            `max_pool_stride`: stride for max-pool filter
            `fc_size`: number of hidden units in the first hidden layer per unit
            `dropout_rate`: dropout rate for this first hidden layer per unit
            `input_length`: length of input sequence
            `input_dim`: dimension of input sequence (e.g. 4 for DNA)
        """
        super().__init__()

        self.creation_args = locals()
        del self.creation_args["self"]
        del self.creation_args["__class__"]
        self.creation_args = sanitize_sacred_arguments(self.creation_args)
        
        self.num_units = num_units

        post_pool_size = int(
            ((input_length - (conv_filter_size - 1)) - (max_pool_size - 1) - 1)
            / max_pool_stride
        ) + 1

        # Define the convolutional units
        self.conv_layer = torch.nn.Conv1d(
            num_units * 4, num_units, conv_filter_size, groups=num_units
        )  # Implement as a single convolution over U groups

        self.post_conv_layer = torch.nn.Sequential(
            torch.nn.BatchNorm1d(num_units),
            Exp(),
            torch.nn.MaxPool1d(max_pool_size, max_pool_stride),
            torch.nn.Flatten(),
            Unsqueeze(),
            torch.nn.Conv1d(
                post_pool_size * num_units, fc_size * num_units, 1,
                groups=num_units
                # Implement fully-connected layers as grouped convolution
            ),
            torch.nn.BatchNorm1d(fc_size * num_units),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Conv1d(
                fc_size * num_units, num_units, 1, groups=num_units
                # One last fully-connected layer to map to 1 scalar per unit
            ),
            torch.nn.BatchNorm1d(num_units),
            torch.nn.ReLU(),
            torch.nn.Flatten()
        )

        self.last_fc_linear = torch.nn.Linear(num_units, 1)
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
        containing a B x L' x U tensor of the convolutional activations.
        """
        # Tile the input sequence `num_units` times
        input_seq_tiled = torch.transpose(input_seq, 1, 2).repeat(
            1, self.num_units, 1
        )  # Shape: B x UD x L

        # Pass through initial convolutions
        conv_acts = self.conv_layer(input_seq_tiled)  # Shape: B x U x L'
        interims = {"conv_acts": torch.transpose(conv_acts, 1, 2)}

        # The rest of the architecture
        post_conv_out = self.post_conv_layer(conv_acts)
        out = self.sigmoid(self.last_fc_linear(post_conv_out))  # Shape: B x 1
        
        if return_interims:
            return out, interims
        else:
            return out

    def loss(self, pred_probs, true_vals, return_components=False, **kwargs):
        """
        Computes total loss value for the predicted probabilities given the true
        values or probabilities.
        Arguments:
            `pred_probs`: a B x 1 tensor of predicted probabilities
            `true_vals`: a B x 1 tensor of binary labels or true probabilities
            `return_components`: if True, also return an empty dictionary (this
                architecture does not have separate losses)
        Returns a B x 1 tensor of loss values.
        """ 
        pred_loss = torch.nn.functional.binary_cross_entropy(
            pred_probs, true_vals, reduction="none"
        )
        if return_components:
            return pred_loss, {}
        else:
            return pred_loss


@model_ex.capture
def create_model(
    num_units, conv_filter_size, max_pool_size, max_pool_stride, fc_size,
    dropout_rate, input_length, input_dim
):
    """
    Create an ExplaiNN model with the given parameters.
    """
    return ExplaiNN(
        num_units, conv_filter_size, max_pool_size, max_pool_stride, fc_size,
        dropout_rate, input_length, input_dim
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
