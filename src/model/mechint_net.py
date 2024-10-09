import torch
import sacred
from model.util import sanitize_sacred_arguments
from model.sparsemax import Sparsemax
import math
import tqdm

model_ex = sacred.Experiment("mechint_model")

@model_ex.config
def config():
    # Number of convolutional filters
    num_conv_filters = 8

    # Size of each convolutional filter
    conv_filter_size = 10

    # Size of positional encoding
    pos_enc_dim = 16

    # Number of attention layers
    num_att_layers = 4

    # Number of attention heads per layer
    att_num_heads = 4

    # Size of hidden dimension in attention layers
    att_hidden_dim = 32
    
    # Size of memory stream in attention layers
    stream_dim = 128

    # Whether or not to use batch norm
    batch_norm = True

    # Whether or not to add dummy token with dummy dimension to activations
    dummy_token = False

    # Number of channels in input sequence
    input_dim = 4
    
    # Loss weight for overlapping convolutional filter weights
    conv_filter_over_loss_weight = 1e-3
    
    # Loss weight for convolutional filter weights L1
    conv_filter_l1_loss_weight = 1e-3

    # Loss weight for sparsity of attention values
    att_head_sparse_loss_weight = 1e-3


class SyntaxBuilder(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def att_head_loss(self, att_vals):
        l = torch.ones((att_vals.shape[0], 1), device=att_vals.device)
        return l


class ManualTransformerLayer(torch.nn.Module):
    def __init__(
        self, input_dim, mlp_hidden_dim, num_heads, dropout_rate=0.1, norm=True,
        resid=True
    ):
        """
        Initializes a single layer of a transformer, manually reimplemented
        based on the PyTorch library's TransformerEncoderLayer module.
        Arguments:
            `input_dim`: the dimension of the input tokens, E
            `mlp_hidden_dim`: dimension of the hidden layer of the MLP
            `num_heads`: number of attention heads, NH; must be a factor of E
            `dropout_rate`: dropout rate
            `norm`: whether or not to perform normalization
            `resid` whether or not to include residual connection
        """
        super().__init__()

        assert input_dim % num_heads == 0

        self.model_dim = input_dim  # This is also the output dimension
        self.num_heads = num_heads
        self.resid = resid
        
        # Multi-headed attention
        self.att_query_linear = torch.nn.Linear(input_dim, input_dim)
        self.att_key_linear = torch.nn.Linear(input_dim, input_dim)
        self.att_value_linear = torch.nn.Linear(input_dim, input_dim)
        self.att_dropout = torch.nn.Dropout(dropout_rate)
        self.att_final_linear = torch.nn.Linear(input_dim, input_dim)

        self.post_att_dropout = torch.nn.Dropout(dropout_rate)
        if norm:
            self.post_att_norm = torch.nn.LayerNorm(input_dim)
        else:
            self.post_att_norm = lambda x: x

        # Feed-forward
        self.feedforward = torch.nn.Sequential(
            torch.nn.Linear(input_dim, mlp_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(mlp_hidden_dim, input_dim)
        )
        
        self.post_ff_dropout = torch.nn.Dropout(dropout_rate)
        if norm:
            self.post_ff_norm = torch.nn.LayerNorm(input_dim)
        else:
            self.post_ff_norm = lambda x: x

    def forward(self, input_tokens):
        """
        Runs the forward pass of the layer.
        Arguments:
            `input_tokens`: a B x L' x E tensor of input tokens
        Returns a B x L' x E tensor of output tokens.
        """
        # Map to query/key/value
        query = self.att_query_linear(input_tokens)
        key = self.att_key_linear(input_tokens)
        value = self.att_value_linear(input_tokens)
        
        # Reshape to extract head dimension
        batch_size, num_tokens = input_tokens.shape[0], input_tokens.shape[1]
        shape = (batch_size, num_tokens, self.num_heads, -1)
        query = query.view(*shape).transpose(1, 2)  # Shape: B x NH x L' x D'
        key = key.view(*shape).transpose(1, 2)  # Shape: B x NH x L' x D'
        value = value.view(*shape).transpose(1, 2)  # Shape: B x NH x L' x D'

        # Compute attention scores
        att_vals = torch.matmul(query, value.transpose(2, 3))
        # Shape: B x NH x L' x L'
        att_vals = att_vals / math.sqrt(query.shape[3])
        att_vals = self.att_dropout(att_vals)
        att_vals = torch.softmax(att_vals, dim=-1)

        # Compute value-vector weighted sums
        value_sums = torch.matmul(att_vals, value)  # Shape: B x NH x L' x D'

        # Reshape to reincorporate heads
        value_sums = value_sums.transpose(1, 2).contiguous().view(
            batch_size, num_tokens, -1
        )  # Shape: B x L' x E

        # Last linear of attention
        att_out = self.att_final_linear(value_sums)

        # Post-attention dropout, residual, norm
        post_att_out = self.post_att_norm(
            (input_tokens if self.resid else 0) + self.post_att_dropout(att_out)
        )
        
        # Feed-forward
        ff_out = self.feedforward(post_att_out)

        # Post-feed-forward dropout, residual, norm
        post_ff_out = self.post_ff_norm(
            (post_att_out if self.resid else 0) + self.post_ff_dropout(ff_out)
        )

        return post_ff_out  # Shape: B x L' x E


class SyntaxBuilderManualTransformer(SyntaxBuilder):
    def __init__(
        self, num_att_layers, input_token_dim, pos_enc_dim, att_hidden_dim,
        att_mlp_hidden_dim, att_num_heads, dropout_rate=0.1, norm=True,
        resid=True
    ):
        """
        Initializes a syntax-building transformer module, which takes in motif
        tokens with the positional embedding, and outputs a vector embedding.
        The transformer is a reimplementation of the PyTorch library's
        TransformerEncoder module.
        Arguments:
            `num_att_layers`: number of attention layers, AL
            `input_token_dim`: dimension of each input token, E
            `pos_enc_dim`: dimension of positional encoding, D'
            `att_hidden_dim`: dimension of query/key vectors in attentions
            `att_mlp_hidden_dim`: dimension of MLP
            `att_num_heads`: number of attention heads, NH
            `dropout_rate`: dropout rate for attention mechanism
            `norm`: whether or not to include normalization in attention
            `resid` whether or not to have residual connections in attention
        """
        super().__init__()
        self.init_dense = torch.nn.Linear(
            input_token_dim + pos_enc_dim, att_hidden_dim
        )
        self.transformer_layers = torch.nn.Sequential(*[
            ManualTransformerLayer(
                att_hidden_dim, att_mlp_hidden_dim, att_num_heads, dropout_rate,
                norm, resid
            ) for _ in range(num_att_layers)
        ])
        if norm:
            self.final_norm = torch.nn.LayerNorm(att_hidden_dim)
        else:
            self.final_norm = lambda x: x

    def forward(self, input_tokens, pos_enc, return_interims=False):
        """
        Runs the forward pass of the model.
        Arguments:
            `input_tokens`: a B x L' x E tensor of the input tokens, where B is
                the batch dimension, L' is the number of tokens, and E is the
                token dimension
            `pos_enc`: a L' x D' tensor of positional encodings to concatenate,
                to be tiled across the batch; note that D' may be 0, in which no
                positional encoding is added
            `return_interims`: if True, also return a dictionary containing the
                attention values
        Returns a B x E' tensor containing the learned representations for the
        input tokens. If `return_interms` is True, also returns a dictionary
        containing the B x AL x L' tensor of attention values.
        """
        # Tile positional encoding
        pos_enc_tiled = torch.tile(pos_enc[None], (input_tokens.shape[0], 1, 1))
        # Shape: B x L' x D'

        input_tokens_with_pos = torch.cat([input_tokens, pos_enc_tiled], dim=2)
        out = self.transformer_layers(self.init_dense(input_tokens_with_pos))
        out = self.final_norm(out)
        out = torch.mean(out, dim=1)

        if return_interims:
            return out, {}
        else:
            return out


class StreamAttentionLayer(torch.nn.Module):
    def __init__(
        self, input_dim, stream_dim, mlp_hidden_dim, num_heads,
        dropout_rate=0.1, norm=True
    ):
        """
        Initializes a single layer of a stream-based attention mechanism, where
        there is a smaller number of queries which are not derived from input
        tokens.
        Arguments:
            `input_dim`: the dimension of the input tokens, E
            `stream_dim`: the dimension of the stream, E'
            `mlp_hidden_dim`: dimension of the hidden layer of the MLP
            `num_heads`: number of attention heads, NH; must be a factor of E
            `dropout_rate`: dropout rate
            `norm`: whether or not to perform normalization
        """
        super().__init__()

        assert input_dim % num_heads == 0

        self.model_dim = input_dim
        self.num_heads = num_heads
        
        # Multi-headed attention
        # self.att_query_linear = torch.nn.Linear(stream_dim, input_dim)
        self.att_query_linear = torch.nn.Sequential(
            torch.nn.Linear(stream_dim, input_dim)
        )
        # self.att_key_linear = torch.nn.Linear(input_dim, input_dim)
        self.att_key_linear = lambda x: x
        self.att_value_linear = torch.nn.Linear(input_dim, input_dim)
        self.att_dropout = torch.nn.Dropout(dropout_rate)
        self.att_final_linear = torch.nn.Linear(input_dim, input_dim)

        self.post_att_dropout = torch.nn.Dropout(dropout_rate)
        if norm:
            self.post_att_norm = torch.nn.LayerNorm(input_dim)
        else:
            self.post_att_norm = lambda x: x

        # Feed-forward
        self.feedforward = torch.nn.Sequential(
            torch.nn.Linear(input_dim, mlp_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(mlp_hidden_dim, stream_dim)
        )
        
        self.post_ff_dropout = torch.nn.Dropout(dropout_rate)
        if norm:
            self.post_ff_norm = torch.nn.LayerNorm(stream_dim)
        else:
            self.post_ff_norm = lambda x: x

        self.sparsemax = Sparsemax()

    def forward(self, input_tokens, stream, return_interims=False):
        """
        Runs the forward pass of the layer.
        Arguments:
            `input_tokens`: a B x L' x E tensor of input tokens
            `stream`: a B x E' tensor of the stream
            `return_interims`: if True, also return a dictionary containing the
                attention values
        Returns a B x E' tensor as an updated stream. If `return_interims` is
        True, also returns a dictionary containing a B x NH x L' tensor of the
        attention values (post-softmax).
        """
        # Map to query/key/value
        query = self.att_query_linear(stream)
        key = self.att_key_linear(input_tokens)
        value = self.att_value_linear(input_tokens)
        
        # Reshape to extract head dimension
        batch_size, num_tokens = input_tokens.shape[0], input_tokens.shape[1]
        shape_1 = (batch_size, 1, self.num_heads, -1)
        shape_2 = (batch_size, num_tokens, self.num_heads, -1)
        query = query.view(*shape_1).transpose(1, 2)  # Shape: B x NH x 1 x D'
        key = key.view(*shape_2).transpose(1, 2)  # Shape: B x NH x L' x D'
        value = value.view(*shape_2).transpose(1, 2)  # Shape: B x NH x L' x D'

        # Compute attention scores
        att_vals = torch.matmul(query, value.transpose(2, 3))
        # Shape: B x NH x 1 x L'
        att_vals = att_vals / math.sqrt(query.shape[3])
        att_vals = self.att_dropout(att_vals)
        att_vals = torch.softmax(att_vals, dim=-1)
        # att_vals = self.sparsemax(att_vals, dim=-1)

        if return_interims:
            interims = {"att_vals": torch.squeeze(att_vals, dim=2)}

        # Compute value-vector weighted sums
        value_sums = torch.matmul(att_vals, value)  # Shape: B x NH x 1 x D'

        # Reshape to reincorporate heads
        value_sums = value_sums.transpose(1, 2).contiguous().view(
            batch_size, -1
        )  # Shape: B x E

        # Last linear of attention
        att_out = self.att_final_linear(value_sums)

        # Post-attention dropout, norm
        post_att_out = self.post_att_norm(self.post_att_dropout(att_out))
        
        # Feed-forward
        ff_out = self.feedforward(post_att_out)

        # Post-feed-forward dropout, residual, norm
        post_ff_out = self.post_ff_norm(self.post_ff_dropout(ff_out))
        # Shape: B x E'

        if return_interims:
            return post_ff_out, interims
        else:
            return post_ff_out


class SyntaxBuilderStreamAttention(SyntaxBuilder):
    def __init__(
        self, num_att_layers, input_token_dim, pos_enc_dim, stream_dim,
        att_hidden_dim, att_mlp_hidden_dim, att_num_heads, dropout_rate=0.1,
        norm=True
    ):
        """
        Initializes a syntax-building stream-attention module, which takes in
        motif tokens with the positional embedding, and outputs a vector
        embedding.
        Arguments:
            `num_att_layers`: number of attention layers, AL
            `input_token_dim`: dimension of each input token, E
            `pos_enc_dim`: dimension of positional encoding, D'
            `stream_dim`: dimension of memory stream, E'
            `att_hidden_dim`: dimension of query/key vectors in attentions
            `att_mlp_hidden_dim`: dimension of MLP
            `att_num_heads`: number of attention heads, NH
            `dropout_rate`: dropout rate for attention mechanism
            `norm`: whether or not to include normalization in attention
        """
        super().__init__()

        self.init_stream = torch.nn.Parameter(
            torch.randn(stream_dim), requires_grad=False
        )

        self.init_dense = torch.nn.Linear(
            input_token_dim + pos_enc_dim, att_hidden_dim
        )

        self.attention_layers = torch.nn.ModuleList([
            StreamAttentionLayer(
                att_hidden_dim, stream_dim, att_mlp_hidden_dim, att_num_heads,
                dropout_rate, norm
            ) for _ in range(num_att_layers)
        ])
        if norm:
            self.final_norm = torch.nn.LayerNorm(stream_dim)
        else:
            self.final_norm = lambda x: x

    def forward(self, input_tokens, pos_enc, return_interims=False):
        """
        Runs the forward pass of the model.
        Arguments:
            `input_tokens`: a B x L' x E tensor of the input tokens, where B is
                the batch dimension, L' is the number of tokens, and E is the
                token dimension
            `pos_enc`: a L' x D' tensor of positional encodings to concatenate,
                to be tiled across the batch; note that D' may be 0, in which no
                positional encoding is added
            `return_interims`: if True, also return a dictionary containing the
                attention values
        Returns a B x E' tensor containing the learned representations for the
        input tokens. If `return_interms` is True, also returns a dictionary
        containing the B x AL x NH x L' tensor of attention values.
        """
        # Tile positional encoding
        pos_enc_tiled = torch.tile(pos_enc[None], (input_tokens.shape[0], 1, 1))
        # Shape: B x L' x D'

        input_tokens_with_pos = torch.cat([input_tokens, pos_enc_tiled], dim=2)
        x = self.init_dense(input_tokens_with_pos)
        
        stream = torch.tile(self.init_stream[None], (x.shape[0], 1))

        if return_interims:
            all_att_vals = []

        for att_layer in self.attention_layers:
            if return_interims:
                att_layer_out, layer_interims = att_layer(x, stream, True)
                all_att_vals.append(layer_interims["att_vals"])
            else:
                att_layer_out = att_layer(x, stream)

            stream = stream + att_layer_out

        out = self.final_norm(stream)

        if return_interims:
            return out, {"att_vals": torch.stack(all_att_vals, dim=1)}
        else:
            return out

    def att_head_loss(self, att_vals, epsilon=1e-6, run_test=False):
        """
        Computes a loss value for the attention weights, thereby encouraging
        sparse attention values.
        Arguments:
            `att_vals`: a B x AL x NH x L' tensor of the attention values
            `epsilon`: small number for numerical stability when computing
                entropy values
            `run_test`: if True, run some unit tests (slow) to check the
                correctness of the vectorized algorithm
        Returns a B x 1 tensor of loss values.
        """
        entropys = -torch.sum(att_vals * torch.log2(att_vals + epsilon), dim=3)
        # Shape: B x AL x NH

        # Subtract the maximum entropy possible to normalize it; doesn't affect
        # training at all, just makes the loss value more interpretable
        max_ent = -torch.log2(torch.tensor(1 / att_vals.shape[3]) + epsilon) / \
            att_vals.shape[3]
        entropys = entropys - max_ent
        final_loss = torch.mean(entropys, dim=(1, 2))[:, None]  # Shape: B x 1

        if not run_test:
            return final_loss

        # Test the entropies are correct
        print("Checking that the entropies are correct...")
        for batch_i in tqdm.trange(att_vals.shape[0]):
            for layer_i in range(att_vals.shape[1]):
                for head_i in range(att_vals.shape[2]):
                    ent = -torch.sum(
                        att_vals[batch_i, layer_i, head_i] * torch.log2(
                            att_vals[batch_i, layer_i, head_i] + epsilon
                        )
                    ) - max_ent
                    assert abs(
                        entropys[batch_i, layer_i, head_i] - ent
                    ) < 1e-6

        return final_loss

    
class MechIntRegNet(torch.nn.Module):
    def __init__(
        self, num_conv_filters, conv_filter_size, pos_enc_dim, num_att_layers,
        att_num_heads, att_hidden_dim, stream_dim, batch_norm, dummy_token,
        input_dim, conv_filter_over_loss_weight, conv_filter_l1_loss_weight,
        att_head_sparse_loss_weight
    ):
        """
        Initializes a mechanistically interpretable regulatory-genome neural
        network.
        Arguments:
            `num_conv_filters`: number of first-layer convolutional filters, F
            `conv_filter_size`: size/width of each convolutional filter, W
            `num_att_layers`: number of subsequent attention layers, AL
            `att_num_heads`: number of attention heads per attention layer, NH
            `att_hidden_dim`: dimension of query/key/value vectors in
                attentions
            `stream_dim`: dimension of memory stream in attention layers
            `batch_norm`: whether or not to use batch norm
            `dummy_token`: if True, add a dummy token and a dummy dimension to
                the activations
            `input_dim`: dimension of input sequence (e.g. 4 for DNA)
            `conv_filter_over_loss_weight`: loss weight for overlapping
                convolutional filter weights
            `conv_filter_l1_loss_weight`: loss weight for convolutional-filter
                L1 penalty
            `att_head_sparse_loss_weight`: loss weight for sparsity of attention
                values
        Note that loss weights can be scalar values, or they can be arrays which
        map epoch index {0, 1, ...} to a scalar loss weight. If the epoch index
        is longer than the length of the array, then the last weight is used.
        """
        super().__init__()

        self.creation_args = locals()
        del self.creation_args["self"]
        del self.creation_args["__class__"]
        self.creation_args = sanitize_sacred_arguments(self.creation_args)
        
        self.pos_enc_dim = pos_enc_dim
        self.batch_norm = batch_norm
        self.conv_filter_over_loss_weight = conv_filter_over_loss_weight
        self.conv_filter_l1_loss_weight = conv_filter_l1_loss_weight
        self.att_head_sparse_loss_weight = att_head_sparse_loss_weight

        # Motif scanners
        self.conv_layer = torch.nn.Conv1d(
            input_dim, num_conv_filters, conv_filter_size
        )
        if batch_norm:
            self.conv_batch_norm = torch.nn.BatchNorm1d(num_conv_filters)

        # Syntax builder
        self.syntax_builder = SyntaxBuilderStreamAttention(
            num_att_layers, num_conv_filters, pos_enc_dim, stream_dim,
            att_hidden_dim, att_hidden_dim, att_num_heads, dropout_rate=0
        )

        # Linear layer
        self.linear_layer = torch.nn.Linear(stream_dim, 1)

        # Activations
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def _get_positional_encoding(self, seq_len, seq_dim):
        """
        Computes a positional encoding for a sequence of tokens.
        Arguments:
            `seq_len`: number of tokens, L'
            `seq_dim`: dimension of each token, D'
        Returns an L' x D'' tensor of encodings, to be concatenated with the
        token representations.
        """
        base = 1e4

        pos_enc = torch.empty((seq_len, seq_dim))

        pos_ran = torch.arange(seq_len)
        dim_ran = torch.arange(0, seq_dim, 2)

        pos_ran_tiled = torch.tile(pos_ran[:, None], (1, len(dim_ran)))
        dim_ran_tiled = torch.tile(dim_ran[None], (len(pos_ran), 1))
        
        trig_arg = pos_ran_tiled / torch.pow(base, dim_ran_tiled / seq_dim)
        
        pos_enc[:, dim_ran] = torch.sin(trig_arg)
        pos_enc[:, dim_ran + 1] = torch.cos(trig_arg)
        return pos_enc

    def forward(self, input_seq, return_interims=False):
        """
        Runs the forward pass of the model.
        Arguments:
            `input_seq`: a B x L x D tensor of the input sequence, where B is
                the batch dimension, L is the sequence length, and D is the
                feature dimension
            `return_interims`: if True, also return the convolutional-layer
                activations and the attention values in a dictionary
        Returns a B x 1 tensor containing the predicted probabilities for each
        input sequence. If `return_interms` is True, also returns a dictionary
        containing the B x L' x F tensor of the convolutional-layer activations
        (F is the number of filters), and a B x AL x L' x L' tensor of attention
        matrices, where AL is the number of attention layers.
        """
        # Motif scanners
        conv_acts = self.relu(self.conv_layer(
            torch.transpose(input_seq, 1, 2)  # Shape: B x D x L
        ))  # Shape: B x F x L'
        
        conv_acts_cache = conv_acts
        if self.batch_norm:
            conv_acts = self.conv_batch_norm(conv_acts)
        conv_acts = torch.transpose(conv_acts, 1, 2)  # Shape: B x L' x F

        # Compute positional encoding
        pos_enc = self._get_positional_encoding(
            conv_acts.shape[1], self.pos_enc_dim
        ).to(conv_acts.device)  # Shape: L' x D'

        # Syntax builder
        if return_interims:
            syntax_out, syntax_interims = self.syntax_builder(
                conv_acts, pos_enc, return_interims=True
            )
        else:
            syntax_out = self.syntax_builder(conv_acts, pos_enc)

        # Final linear layer
        out = self.sigmoid(self.linear_layer(syntax_out))
        
        if return_interims:
            interims = {"conv_acts": torch.transpose(conv_acts_cache, 1, 2)}
            interims.update(syntax_interims)
            return out, interims
        else:
            return out

    def conv_filter_weight_losses(self, conv_acts, run_test=False):
        """
        Computes losses for the convolutional-filter weights, penalizing filters
        which fire in proximity due to the same part of the input sequence, as
        well as the weights themselves.
        Arguments:
            `conv_acts`: a B x L' x F tensor of the convolutional-filter
                activations (F is the number of filters)
            `run_test`: if True, run some unit tests (slow) to check the
                correctness of the vectorized algorithm
        Returns a B x 1 tensor of loss values for the overlapping filters, and a
        B x 1 tensor of loss values which is simply the L1 norm of the weights,
        tiled.
        The requirement we are trying to satisfy is the following:
        Let the filters be width W. At position i in the original sequence s,
        Each filter aggregates values s[i .. i + W]. Indices are 0-indexed and
        intervals are inclusive of endpoints. At each position i, let filter a
        with weights f_a be the most activated filter. For all other filters
        b != a, if b fires at position j for any j in [i .. i + W - 1], then
        we want the following to be true:
        f_a[j - i .. W - 1] = 0 OR f_b[0 .. W - (j - i) - 1] = 0
        Additionally, for all other filters b != a, if b fires at position j
        for any j in [i - (W - 1) .. i], then we want the following to be true:
        f_a[0 .. W - (i - j) - 1] = 0 OR f_b[i - j .. W - 1] = 0
        We can combine these two into a single requirement:
        If filter b fires at position j for any j in
        [i - (W - 1) .. i + (W - 1)], then we require:
        f_a[max(0, j - i) .. W - 1 - max(0, i - j)] = 0 OR
        f_b[max(0, i - j) .. W - 1 - max(0, j - i)] = 0
        """
        conv_weights = torch.transpose(self.conv_layer.weight, 1, 2)
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
       
        if not run_test:
            return final_losses

        # Test that the windows weight sums are the same as using indices
        print("Checking the windowed sums of the convolutional weights...")
        i = 50  # Any i is fine for this test because i - j is constant
        all_j = torch.arange(i - (w - 1), i + (w - 1) + 1)
        zero = torch.zeros_like(all_j)
        a_ind_1 = torch.maximum(all_j - i, zero)
        a_ind_2 = w - 1 - torch.maximum(i - all_j, zero)
        b_ind_1 = torch.maximum(i - all_j, zero)
        b_ind_2 = w - 1 - torch.maximum(all_j - i, zero)
        for a_ind in range(f):
            for b_ind in range(f):
                a, b = conv_weights[a_ind], conv_weights[b_ind]
                a_weight_sums = conv_weight_sums_a[a_ind]
                b_weight_sums = conv_weight_sums_b[b_ind]

                for j in range(len(all_j)):
                    assert abs(
                        torch.sum(torch.abs(a[a_ind_1[j] : a_ind_2[j] + 1])) - \
                        a_weight_sums[j]
                    ) < 1e-5
                    assert abs(
                        torch.sum(torch.abs(b[b_ind_1[j] : b_ind_2[j] + 1])) - \
                        b_weight_sums[j]
                    ) < 1e-5
            
        # Test that the product of windowed sums is the same as using indices
        print("Checking the convolutional activations and each loss value...")
        for batch_i in tqdm.trange(len(conv_acts)):
            for i in range(l):
                a_ind = torch.argmax(conv_acts[batch_i][i])
                for b_ind in range(f):
                    for j in range(len(all_j)):
                        if a_ind == b_ind:
                            assert weighted_prods[batch_i, i, a_ind, j] == 0
                            assert window_acts[batch_i, i, a_ind, j] == 0
                        else:
                            a_weight_sum = torch.sum(torch.abs(
                                conv_weights[a_ind][a_ind_1[j] : a_ind_2[j] + 1]
                            ))
                            b_weight_sum = torch.sum(torch.abs(
                                conv_weights[b_ind][b_ind_1[j] : b_ind_2[j] + 1]
                            ))

                            assert abs(
                                (a_weight_sum * b_weight_sum) - \
                                conv_weight_prods[batch_i, i, b_ind, j]
                            ) < 1e-4

                            if j < (w - 1) - i or j >= l + (w - 1) - i:
                                # Window overruns sequence
                                b_act = 0
                            else:
                                b_act = conv_acts[
                                    batch_i, i - (w - 1) + j, b_ind
                                ]
                            assert b_act == window_acts[batch_i, i, b_ind, j]
                            
                            assert abs(
                                (b_act * a_weight_sum * b_weight_sum) - \
                                weighted_prods[batch_i, i, b_ind, j]
                            ) < 1e-4

        return final_losses
            
    def prediction_loss(self, pred_probs, true_vals):
        """
        Computes a loss value for the predicted probabilities given the true
        values or probabilities.
        Arguments:
            `pred_probs`: a B x 1 tensor of predicted probabilities
            `true_vals`: a B x 1 tensor of binary labels or true probabilities
        Returns a B x 1 tensor of loss values.
        """
        return torch.nn.functional.binary_cross_entropy(
            pred_probs, true_vals, reduction="none"
        )
    
    def loss(
        self, pred_probs, true_vals, conv_acts=None, att_vals=None,
        epoch_num=None, return_components=False, **kwargs
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
            `att_vals`: a B x AL x L' x L' tensor of the attention values (AL is
                the number of attention layers); if None, the associated loss is
                ignored
            `epoch_num`: integer epoch number {0, 1, ...}, which is used only if
                any of the loss weights are functions which require it
            `return_components`: if True, also return the loss components in a
                dictionary
        Returns a B x 1 tensor of loss values, and optionally also the loss
        components in a dictionary: a B x 1 tensor of the convolutional-filter
        overlap losses, a B x 1 tensor of convolutional-filter L1 losses, a
        B x 1 tensor of attention entropy losses, and a B x 1 tensor of
        prediction losses.
        """ 
        weight_func = lambda w: w[min(epoch_num, len(w) - 1)] \
            if hasattr(w, "__getitem__") else w

        conv_filter_over_loss_weight = weight_func(
            self.conv_filter_over_loss_weight
        )
        conv_filter_l1_loss_weight = weight_func(
            self.conv_filter_l1_loss_weight
        )
        att_head_sparse_loss_weight = weight_func(
            self.att_head_sparse_loss_weight
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
        if (att_vals is not None) and \
            (return_components or att_head_sparse_loss_weight):
            att_head_loss = self.syntax_builder.att_head_loss(att_vals)
        else:
            att_head_loss = zero_loss
        pred_loss = self.prediction_loss(pred_probs, true_vals)

        final_loss = \
            (conv_filter_over_loss_weight * conv_filter_losses[0]) + \
            (conv_filter_l1_loss_weight * conv_filter_losses[1]) + \
            (att_head_sparse_loss_weight * att_head_loss) + \
            pred_loss

        if return_components:
            return final_loss, {
                "conv_filter_overlap": conv_filter_losses[0],
                "conv_filter_l1": conv_filter_losses[1],
                "att_head_sparse": att_head_loss,
                "pred": pred_loss
            }
        else:
            return final_loss


@model_ex.command
def create_model(
    num_conv_filters, conv_filter_size, pos_enc_dim, num_att_layers,
    att_num_heads, att_hidden_dim, stream_dim, batch_norm, dummy_token,
    input_dim, conv_filter_over_loss_weight, conv_filter_l1_loss_weight,
    att_head_sparse_loss_weight
):
    """
    Create a MechIntRegNet with the given parameters.
    """
    return MechIntRegNet(
        num_conv_filters, conv_filter_size, pos_enc_dim, num_att_layers,
        att_num_heads, att_hidden_dim, stream_dim, batch_norm, dummy_token,
        input_dim, conv_filter_over_loss_weight, conv_filter_l1_loss_weight,
        att_head_sparse_loss_weight
    )


@model_ex.automain
def main():
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"

    model = create_model(num_att_layers=1).to(DEVICE)
    
    print(
        "Number of parameters: %d" % sum(p.numel() for p in model.parameters())
    )
    
    input_seq = torch.randn(128, 100, 4).to(DEVICE)
    true_probs = torch.randint(2, size=(128, 1)).float().to(DEVICE)

    # Test the correctness of the loss functions
    pred_probs, interims = model(input_seq, return_interims=True)

    conv_filter_losses = model.conv_filter_weight_losses(
        interims["conv_acts"], run_test=True
    )
    att_head_loss = model.syntax_builder.att_head_loss(
        interims["att_vals"], run_test=True
    )

    # Test ability to memorize
    optimizer = torch.optim.Adam(model.parameters())

    t_iter = tqdm.trange(1000, desc="Loss=----")
    for _ in t_iter:
        optimizer.zero_grad()

        pred_probs, interims = model(input_seq, return_interims=True)

        loss = torch.mean(model.loss(pred_probs, true_probs, **interims))
        
        t_iter.set_description("Loss=%.3f" % loss.item())

        loss.backward()
        optimizer.step()
