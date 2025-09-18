import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.TRACK_DRNRL import DRNEncoderBlock, TokenEmbedding, PositionalEncoding
from layers.TRACK_DRNRL import GAT as DRNRLGAT

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def drop_path_func(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_func(x, self.drop_prob, self.training)


class TrajGATLayer(nn.Module):

    head_dim = 1

    def __init__(
        self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
        dropout_prob=0.1, add_skip_connection=True, bias=True, load_trans_prob=True,
    ):
        super().__init__()
        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.concat = concat  # whether we should concatenate or average the attention heads
        self.add_skip_connection = add_skip_connection
        self.load_trans_prob = load_trans_prob

        self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        if self.load_trans_prob:
            self.linear_proj_tran_prob = nn.Linear(1, num_of_heads * num_out_features, bias=False)
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        if self.load_trans_prob:
            self.scoring_trans_prob = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)
        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        self.leakyReLU = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_prob)

        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)
        if self.load_trans_prob:
            nn.init.xavier_uniform_(self.linear_proj_tran_prob.weight)
            nn.init.xavier_uniform_(self.scoring_trans_prob)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def skip_concat_bias(self, in_nodes_features, out_nodes_features):
        if not out_nodes_features.is_contiguous():
            out_nodes_features = out_nodes_features.contiguous()

        if self.add_skip_connection:
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:  # if FIN == FOUT
                # in_nodes_features: (N, FIN), out_nodes_features: (N, NH, FOUT)
                out_nodes_features += in_nodes_features.unsqueeze(1)
            else:
                out_nodes_features += self.skip_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

        if self.concat:
            # (N, NH, FOUT) -> (N, NH * FOUT)
            out_nodes_features = out_nodes_features.view(-1, self.num_of_heads * self.num_out_features)
        else:
            # (N, NH, FOUT) -> (N, FOUT)
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)


class TrajSparseGATLayer(TrajGATLayer):

    src_nodes_dim = 0  # position of source nodes in edge index
    trg_nodes_dim = 1  # position of target nodes in edge index

    nodes_dim = 0      # node dimension/axis
    head_dim = 1       # attention head dimension/axis

    def __init__(
        self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
        dropout_prob=0.1, add_skip_connection=True, bias=True, load_trans_prob=True,
    ):
        super().__init__(
            num_in_features, num_out_features, num_of_heads, concat, activation,
            dropout_prob, add_skip_connection, bias, load_trans_prob,
        )

    def forward(self, data):
        in_nodes_features, edge_index, edge_prob = data
        num_of_nodes = in_nodes_features.shape[self.nodes_dim]
        assert edge_index.shape[0] == 2, f'Expected edge index with shape=(2,E) got {edge_index.shape}'

        in_nodes_features = self.dropout(in_nodes_features)  # (N, FIN)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)
        nodes_features_proj = self.dropout(nodes_features_proj)
        if self.load_trans_prob:
            trans_prob_proj = self.linear_proj_tran_prob(edge_prob).view(-1, self.num_of_heads, self.num_out_features)
            trans_prob_proj = self.dropout(trans_prob_proj)

        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)
        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(scores_source, scores_target, nodes_features_proj, edge_index)
        if self.load_trans_prob:
            scores_trans_prob = (trans_prob_proj * self.scoring_trans_prob).sum(dim=-1)  # (E, NH)
            scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted + scores_trans_prob)
        else:
            scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)

        attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[self.trg_nodes_dim], num_of_nodes)  # (E, NH, 1)
        attentions_per_edge = self.dropout(attentions_per_edge)
        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge
        out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes)  # (N, NH, FOUT)
        out_nodes_features = self.skip_concat_bias(in_nodes_features, out_nodes_features)
        return (out_nodes_features, edge_index, edge_prob)

    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
        scores_per_edge = scores_per_edge - scores_per_edge.max()  # (E, NH)
        exp_scores_per_edge = scores_per_edge.exp()
        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index, num_of_nodes)  # (E, NH)
        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)
        return attentions_per_edge.unsqueeze(-1)

    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):
        trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge)
        size = list(exp_scores_per_edge.shape)
        size[self.nodes_dim] = num_of_nodes
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)
        neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge)
        return neighborhood_sums.index_select(self.nodes_dim, trg_index)

    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
        size = list(nodes_features_proj_lifted_weighted.shape)
        size[self.nodes_dim] = num_of_nodes  # (N, NH, FOUT)
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)
        trg_index_broadcasted = self.explicit_broadcast(edge_index[self.trg_nodes_dim], nodes_features_proj_lifted_weighted)  # (E) -> (E, NH, FOUT)
        out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted_weighted)  # (E, NH, FOUT) -> (N, NH, FOUT)
        return out_nodes_features

    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]
        scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
        scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index)
        return scores_source, scores_target, nodes_features_matrix_proj_lifted

    def explicit_broadcast(self, this, other):
        # E -> (E, NH)
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)
        return this.expand_as(other)


class GAT(nn.Module):

    def __init__(
        self, d_model, in_feature, num_heads_per_layer, num_features_per_layer,
        add_skip_connection=True, bias=True, dropout=0.1, load_trans_prob=True, avg_last=True,
    ):
        super().__init__()
        assert len(num_heads_per_layer) == len(num_features_per_layer), f'Enter valid arch params.'

        num_features_per_layer = [in_feature] + num_features_per_layer
        num_heads_per_layer = [1] + num_heads_per_layer  # trick - so that I can nicely create GAT layers below
        if avg_last:
            assert num_features_per_layer[-1] == d_model
        else:
            assert num_features_per_layer[-1] * num_heads_per_layer[-1] == d_model
        num_of_layers = len(num_heads_per_layer) - 1

        gat_layers = []  # collect GAT layers
        for i in range(num_of_layers):
            concat_input = True
            if i == num_of_layers - 1 and avg_last:
                concat_input = False
            layer = TrajSparseGATLayer(
                num_in_features=num_features_per_layer[i] * num_heads_per_layer[i],  # consequence of concatenation
                num_out_features=num_features_per_layer[i+1], num_of_heads=num_heads_per_layer[i+1],
                concat=concat_input,  # last GAT layer does mean avg, the others do concat
                activation=nn.ELU() if i < num_of_layers - 1 else None,  # last layer just outputs raw scores
                dropout_prob=dropout, add_skip_connection=add_skip_connection, bias=bias, load_trans_prob=load_trans_prob,
            )
            gat_layers.append(layer)

        self.gat_net = nn.Sequential(
            *gat_layers,
        )

    def forward(self, node_features, edge_index_input, edge_prob_input):
        """
        Args:
            node_features: (vocab_size, fea_dim)
            edge_index_input: (2, E)
            edge_prob_input: (E, 1)
        Returns:
            (vocab_size, d_model)
        """
        data = (node_features, edge_index_input, edge_prob_input)
        (node_fea_emb, edge_index, edge_prob) = self.gat_net(data)  # (vocab_size, num_channels[-1])
        return node_fea_emb


class TrajDGATLayer(nn.Module):

    head_dim = 2

    def __init__(
        self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
        dropout_prob=0.1, add_skip_connection=True, bias=True, load_trans_prob=True,
    ):
        super().__init__()
        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.concat = concat  # whether we should concatenate or average the attention heads
        self.add_skip_connection = add_skip_connection
        self.load_trans_prob = load_trans_prob

        self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        if self.load_trans_prob:
            self.linear_proj_tran_prob = nn.Linear(1, num_of_heads * num_out_features, bias=False)
        self.linear_proj_t_tran_prob = nn.Linear(1, num_of_heads * num_out_features, bias=False)
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, 1, num_of_heads, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, 1, num_of_heads, num_out_features))
        if self.load_trans_prob:
            self.scoring_trans_prob = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.scoring_t_trans_prob = nn.Parameter(torch.Tensor(1, 1, num_of_heads, num_out_features))
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)
        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        self.leakyReLU = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_prob)

        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)
        if self.load_trans_prob:
            nn.init.xavier_uniform_(self.linear_proj_tran_prob.weight)
            nn.init.xavier_uniform_(self.scoring_trans_prob)
        nn.init.xavier_uniform_(self.linear_proj_t_tran_prob.weight)
        nn.init.xavier_uniform_(self.scoring_t_trans_prob)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def skip_concat_bias(self, in_nodes_features, out_nodes_features):
        if not out_nodes_features.is_contiguous():
            out_nodes_features = out_nodes_features.contiguous()
        B, N, _, _ = out_nodes_features.shape

        if self.add_skip_connection:
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:  # if FIN == FOUT
                # in_nodes_features: (B, N, FIN), out_nodes_features: (B, N, NH, FOUT)
                out_nodes_features += in_nodes_features.unsqueeze(self.head_dim)
            else:
                out_nodes_features += self.skip_proj(in_nodes_features).view(B, N, self.num_of_heads, self.num_out_features)

        if self.concat:
            # (B, N, NH, FOUT) -> (B, N, NH * FOUT)
            out_nodes_features = out_nodes_features.view(B, N, self.num_of_heads * self.num_out_features)
        else:
            # (B, N, NH, FOUT) -> (B, N, FOUT)
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)


class TrajSparseDGATLayer(TrajDGATLayer):

    src_nodes_dim = 0  # position of source nodes in edge index
    trg_nodes_dim = 1  # position of target nodes in edge index

    nodes_dim = 1      # node dimension/axis
    head_dim = 2       # attention head dimension/axis

    def __init__(
        self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
        dropout_prob=0.1, add_skip_connection=True, bias=True, load_trans_prob=True,
    ):
        super().__init__(
            num_in_features, num_out_features, num_of_heads, concat, activation,
            dropout_prob, add_skip_connection, bias, load_trans_prob,
        )

    def forward(self, data):
        in_nodes_features, edge_index, edge_prob, t_edge_prob = data
        N = in_nodes_features.shape[self.nodes_dim]
        B, E, _ = t_edge_prob.shape
        assert edge_index.shape[0] == 2, f'Expected edge index with shape=(2,E) got {edge_index.shape}'
        in_nodes_features = self.dropout(in_nodes_features)  # (B, N, FIN)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(B, N, self.num_of_heads, self.num_out_features)
        nodes_features_proj = self.dropout(nodes_features_proj)  # (B, N, NH, FOUT)
        if self.load_trans_prob:
            trans_prob_proj = self.linear_proj_tran_prob(edge_prob).view(E, self.num_of_heads, self.num_out_features)
            trans_prob_proj = self.dropout(trans_prob_proj)
        t_trans_prob_proj = self.linear_proj_t_tran_prob(t_edge_prob).view(B, E, self.num_of_heads, self.num_out_features)
        t_trans_prob_proj = self.dropout(t_trans_prob_proj)

        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)  # (B, N, NH)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)  # (B, N, NH)
        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(scores_source, scores_target, nodes_features_proj, edge_index)
        scores_per_edge = scores_source_lifted + scores_target_lifted  # (B, E, NH)
        if self.load_trans_prob:
            scores_trans_prob = (trans_prob_proj * self.scoring_trans_prob).sum(dim=-1).unsqueeze(0)  # (1, E, NH)
            scores_per_edge = scores_per_edge + scores_trans_prob
        scores_t_trans_prob = (t_trans_prob_proj * self.scoring_t_trans_prob).sum(dim=-1)  # (B, E, NH)
        scores_per_edge = scores_per_edge + scores_t_trans_prob
        scores_per_edge = self.leakyReLU(scores_per_edge)

        attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[self.trg_nodes_dim], N)  # (B, E, NH, 1)
        attentions_per_edge = self.dropout(attentions_per_edge)
        # (B, E, NH, FOUT) * (B, E, NH, 1) -> (B, E, NH, FOUT)
        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge
        out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, N)  # (B, N, NH, FOUT)
        out_nodes_features = self.skip_concat_bias(in_nodes_features, out_nodes_features)
        return (out_nodes_features, edge_index, edge_prob, t_edge_prob)

    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
        scores_per_edge = scores_per_edge - scores_per_edge.max()  # (B, E, NH)
        exp_scores_per_edge = scores_per_edge.exp()
        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index, num_of_nodes)  # (B, E, NH)
        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)
        return attentions_per_edge.unsqueeze(-1)  # (B, E, NH) -> (B, E, NH, 1)

    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):
        # (E) -> (B, E, NH)
        trg_index_broadcasted = trg_index.unsqueeze(-1).unsqueeze(0).expand_as(exp_scores_per_edge)
        size = list(exp_scores_per_edge.shape)
        size[self.nodes_dim] = num_of_nodes  # (B, N, NH)
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)
        neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge)
        return neighborhood_sums.index_select(self.nodes_dim, trg_index)  # (B, N, NH) -> (B, E, NH)

    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
        size = list(nodes_features_proj_lifted_weighted.shape)
        size[self.nodes_dim] = num_of_nodes  # (B, N, NH, FOUT)
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)
        # (E) -> (B, E, NH, FOUT)
        trg_index_broadcasted = edge_index[self.trg_nodes_dim].unsqueeze(-1).unsqueeze(-1).unsqueeze(0).expand_as(nodes_features_proj_lifted_weighted)
        out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted_weighted)  # (B, E, NH, FOUT) -> (B, N, NH, FOUT)
        return out_nodes_features

    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        """
        scores_source: (B, N, NH)
        scores_target: (B, N, NH)
        nodes_features_matrix_proj: (B, N, NH, FOUT)
        edge_index: (E, 1)
        """
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]
        scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
        scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index)
        return scores_source, scores_target, nodes_features_matrix_proj_lifted


class DGAT(nn.Module):

    def __init__(
        self, d_model, in_feature, num_heads_per_layer, num_features_per_layer,
        add_skip_connection=True, bias=True, dropout=0.1, load_trans_prob=True, avg_last=True,
    ):
        super().__init__()
        assert len(num_heads_per_layer) == len(num_features_per_layer), f'Enter valid arch params.'

        num_features_per_layer = [in_feature] + num_features_per_layer
        num_heads_per_layer = [1] + num_heads_per_layer  # trick - so that I can nicely create GAT layers below
        if avg_last:
            assert num_features_per_layer[-1] == d_model
        else:
            assert num_features_per_layer[-1] * num_heads_per_layer[-1] == d_model
        num_of_layers = len(num_heads_per_layer) - 1

        gat_layers = []  # collect GAT layers
        for i in range(num_of_layers):
            concat_input = True
            if i == num_of_layers - 1 and avg_last:
                concat_input = False
            layer = TrajSparseDGATLayer(
                num_in_features=num_features_per_layer[i] * num_heads_per_layer[i],  # consequence of concatenation
                num_out_features=num_features_per_layer[i+1], num_of_heads=num_heads_per_layer[i+1],
                concat=concat_input,  # last GAT layer does mean avg, the others do concat
                activation=nn.ELU() if i < num_of_layers - 1 else None,  # last layer just outputs raw scores
                dropout_prob=dropout, add_skip_connection=add_skip_connection, bias=bias, load_trans_prob=load_trans_prob,
            )
            gat_layers.append(layer)

        self.gat_net = nn.Sequential(
            *gat_layers,
        )

    def forward(self, node_features, edge_index_input, edge_prob_input, t_edge_prob_input):
        """
        Args:
            node_features: (vocab_size, fea_dim)
            edge_index_input: (2, E)
            edge_prob_input: (E, 1)
            t_edge_prob_input: (B, E, 1)
        Returns:
            (B, vocab_size, d_model)
        """
        node_features = node_features.unsqueeze(0).repeat(t_edge_prob_input.shape[0], 1, 1)  # (B, vocab_size, fea_dim)
        data = (node_features, edge_index_input, edge_prob_input, t_edge_prob_input)
        (node_fea_emb, edge_index, edge_prob, t_edge_prob) = self.gat_net(data)  # (B, vocab_size, num_channels[-1])
        return node_fea_emb


class TrajPositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=128):
        super().__init__()
        self.d_model = d_model
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)  # (max_len, d_model / 2)
        pe[:, 1::2] = torch.cos(position * div_term)  # (max_len, d_model / 2)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x, position_ids=None):
        """
        Args:
            x: (B * traj_batch_size, seq_len, d_model)
            position_ids: (B * traj_batch_size, seq_len) or None
        Returns:
            (1, seq_len, d_model) / (B * traj_batch_size, seq_len, d_model)
        """
        if position_ids is None:
            return self.pe[:, :x.size(1)].detach()
        batch_size, seq_len = position_ids.shape
        pe = self.pe[:, :seq_len, :]  # (1, seq_len, d_model)
        pe = pe.expand((position_ids.shape[0], -1, -1))  # (B * traj_batch_size, seq_len, d_model)
        pe = pe.reshape(-1, self.d_model)  # (B * traj_batch_size * seq_len, d_model)
        position_ids = position_ids.reshape(-1, 1).squeeze(1)  # (B  * traj_batch_size * seq_len,)
        output_pe = pe[position_ids].reshape(batch_size, seq_len, self.d_model).detach()
        return output_pe


class TimeEmbedding(nn.Module):

    def __init__(
        self, d_model, dropout=0.1, add_pe=True, add_delta_time=True, add_time_in_day=True, add_day_in_week=True,
    ):
        super().__init__()
        self.add_pe = add_pe
        self.add_delta_time = add_delta_time
        self.add_time_in_day = add_time_in_day
        self.add_day_in_week = add_day_in_week

        if add_pe:
            self.position_embedding = TrajPositionalEncoding(d_model=d_model)
        if add_delta_time:
            self.delta_time_embedding = nn.Linear(1, d_model)
        if add_time_in_day:
            self.daytime_embedding = nn.Embedding(1441, d_model, padding_idx=0)
        if add_day_in_week:
            self.weekday_embedding = nn.Embedding(8, d_model, padding_idx=0)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, traj_embs, traj, position_ids=None):
        if self.add_pe:
            traj_embs += self.position_embedding(traj_embs, position_ids)  # (B * traj_batch_size, seq_len, d_model)
        if self.add_delta_time:
            delta_time = torch.clamp((traj[:, :, 1:2]-traj[:, 0:1, 1:2]), min=0).float() / 60.0
            traj_embs += self.delta_time_embedding(delta_time)
        if self.add_time_in_day:
            traj_embs += self.daytime_embedding(traj[:, :, 2])  # (B * traj_batch_size, seq_len, d_model)
        if self.add_day_in_week:
            traj_embs += self.weekday_embedding(traj[:, :, 3])  # (B * traj_batch_size, seq_len, d_model)
        return self.dropout(traj_embs)


class MultiHeadAttention(nn.Module):

    def __init__(
        self, num_heads, d_model, dim_out, attn_drop=0.1, proj_drop=0.1, add_cls=True, device=torch.device('cpu'),
        add_temporal_bias=True, temporal_bias_dim=64, use_mins_interval=False,
    ):
        super().__init__()
        assert d_model % num_heads == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.device = device
        self.add_cls = add_cls
        self.scale = self.d_k ** -0.5  # 1/sqrt(dk)
        self.add_temporal_bias = add_temporal_bias
        self.temporal_bias_dim = temporal_bias_dim
        self.use_mins_interval = use_mins_interval

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.dropout = nn.Dropout(p=attn_drop)

        self.proj = nn.Linear(d_model, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.add_temporal_bias:
            if self.temporal_bias_dim != 0 and self.temporal_bias_dim != -1:
                self.temporal_mat_bias_1 = nn.Linear(1, self.temporal_bias_dim, bias=True)
                self.temporal_mat_bias_2 = nn.Linear(self.temporal_bias_dim, 1, bias=True)
            elif self.temporal_bias_dim == -1:
                self.temporal_mat_bias = nn.Parameter(torch.Tensor(1, 1))
                nn.init.xavier_uniform_(self.temporal_mat_bias)

    def forward(self, x, padding_masks, future_mask=True, output_attentions=False, batch_temporal_mat=None):
        """
        Args:
            x: (B, T, d_model)
            padding_masks: (B, 1, T, T) padding_mask
            future_mask: True/False
            batch_temporal_mat: (B, T, T)
        Returns:
        """
        batch_size, seq_len, d_model = x.shape

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # l(x) --> (B, T, d_model)
        # l(x).view() --> (B, T, head, d_k)
        query, key, value = [
            l(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2) for l, x in zip(self.linear_layers, (x, x, x))
        ]
        # q, k, v --> (B, head, T, d_k)

        # 2) Apply attention on all the projected vectors in batch.
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale  # (B, head, T, T)

        if self.add_temporal_bias:
            if self.use_mins_interval:
                batch_temporal_mat = 1.0 / torch.log(torch.exp(torch.tensor(1.0).to(self.device)) + (batch_temporal_mat / torch.tensor(60.0).to(self.device)))
            else:
                batch_temporal_mat = 1.0 / torch.log(torch.exp(torch.tensor(1.0).to(self.device)) + batch_temporal_mat)
            if self.temporal_bias_dim != 0 and self.temporal_bias_dim != -1:
                batch_temporal_mat = self.temporal_mat_bias_2(F.leaky_relu(self.temporal_mat_bias_1(batch_temporal_mat.unsqueeze(-1)), negative_slope=0.2)).squeeze(-1)  # (B, T, T)
            if self.temporal_bias_dim == -1:
                batch_temporal_mat = batch_temporal_mat * self.temporal_mat_bias.expand((1, seq_len, seq_len))
            batch_temporal_mat = batch_temporal_mat.unsqueeze(1)  # (B, 1, T, T)
            scores += batch_temporal_mat  # (B, 1, T, T)

        if padding_masks is not None:
            scores.masked_fill_(padding_masks == 0, float('-inf'))

        if future_mask:
            mask_postion = torch.triu(torch.ones((1, seq_len, seq_len)), diagonal=1).bool().to(self.device)
            if self.add_cls:
                mask_postion[:, 0, :] = 0
            scores.masked_fill_(mask_postion, float('-inf'))

        p_attn = F.softmax(scores, dim=-1)  # (B, head, T, T)
        p_attn = self.dropout(p_attn)
        out = torch.matmul(p_attn, value)  # (B, head, T, d_k)

        # 3) "Concat" using a view and apply a final linear.
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)  # (B, T, d_model)
        out = self.proj(out)  # (B, T, N, D)
        out = self.proj_drop(out)
        if output_attentions:
            return out, p_attn  # (B, T, dim_out), (B, head, T, T)
        else:
            return out, None  # (B, T, dim_out)


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(
        self, d_model, attn_heads, feed_forward_hidden, drop_path, attn_drop, dropout, type_ln='post', add_cls=True,
        device=torch.device('cpu'), add_temporal_bias=True, temporal_bias_dim=64, use_mins_interval=False,
    ):
        """
        Args:
            d_model: hidden size of transformer
            attn_heads: head sizes of multi-head attention
            feed_forward_hidden: feed_forward_hidden, usually 4*d_model
            drop_path: encoder dropout rate
            attn_drop: attn dropout rate
            dropout: dropout rate
            type_ln:
        """
        super().__init__()
        self.attention = MultiHeadAttention(
            num_heads=attn_heads, d_model=d_model, dim_out=d_model, attn_drop=attn_drop, proj_drop=dropout, add_cls=add_cls,
            device=device, add_temporal_bias=add_temporal_bias, temporal_bias_dim=temporal_bias_dim, use_mins_interval=use_mins_interval,
        )
        self.mlp = Mlp(
            in_features=d_model, hidden_features=feed_forward_hidden, out_features=d_model, act_layer=nn.GELU, drop=dropout,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.type_ln = type_ln

    def forward(self, x, padding_masks, future_mask=True, output_attentions=False, batch_temporal_mat=None):
        """
        Args:
            x: (B, T, d_model)
            padding_masks: (B, 1, T, T)
            future_mask: True/False
            batch_temporal_mat: (B, T, T)
        Returns:
            (B, T, d_model)
        """
        if self.type_ln == 'pre':
            attn_out, attn_score = self.attention(
                self.norm1(x), padding_masks=padding_masks, future_mask=future_mask, output_attentions=output_attentions,
                batch_temporal_mat=batch_temporal_mat,
            )
            x = x + self.drop_path(attn_out)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        elif self.type_ln == 'post':
            attn_out, attn_score = self.attention(
                x, padding_masks=padding_masks, future_mask=future_mask, output_attentions=output_attentions,
                batch_temporal_mat=batch_temporal_mat,
            )
            x = self.norm1(x + self.drop_path(attn_out))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        else:
            raise ValueError('Error type_ln {}'.format(self.type_ln))
        return x, attn_score

'''
class TRL(nn.Module):

    def __init__(self, config, data_feature):
        super().__init__()

        d_model = config.get("d_model", 256)
        n_layers = config.get("n_layers", 6)
        attn_heads = config.get('attn_heads', 8)
        mlp_ratio = config.get('mlp_ratio', 4)
        dropout = config.get('dropout', 0.1)
        drop_path = config.get('drop_path', 0.3)
        attn_drop = config.get('attn_drop', 0.1)
        type_ln = config.get('type_ln', 'post')
        self.future_mask = config.get('future_mask', False)
        add_cls = config.get('add_cls', True)
        device = config.get('device', torch.device('cpu'))
        add_temporal_bias = config.get('add_temporal_bias', True)
        temporal_bias_dim = config.get('temporal_bias_dim', 64)
        use_mins_interval = config.get('use_mins_interval', False)
        feed_forward_hidden = d_model * mlp_ratio

        # multi-layers transformer blocks, deep network
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path, n_layers)]  # stochastic depth decay rule
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model, attn_heads=attn_heads, feed_forward_hidden=feed_forward_hidden, drop_path=enc_dpr[i],
                attn_drop=attn_drop, dropout=dropout, type_ln=type_ln, add_cls=add_cls, device=device,
                add_temporal_bias=add_temporal_bias, temporal_bias_dim=temporal_bias_dim, use_mins_interval=use_mins_interval,
            ) for i in range(n_layers)
        ])

    def forward(
        self, embedding_output, padding_masks, batch_temporal_mat=None,
        output_hidden_states=False, output_attentions=False,
    ):
        """
        Args:
            embedding_output: (B * traj_batch_size, seq_len, feat_dim) torch tensor of masked features (input)
            padding_masks: (B * traj_batch_size, seq_len) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """
        padding_masks_input = padding_masks.unsqueeze(1).repeat(1, embedding_output.size(1), 1).unsqueeze(1)  # (B * traj_batch_size, 1, seq_len, seq_len)
        # running over multiple transformer blocks
        all_hidden_states = [embedding_output] if output_hidden_states else None
        all_self_attentions = [] if output_attentions else None
        for transformer in self.transformer_blocks:
            embedding_output, attn_score = transformer.forward(
                x=embedding_output, padding_masks=padding_masks_input, future_mask=self.future_mask,
                output_attentions=output_attentions, batch_temporal_mat=batch_temporal_mat,
            )  # (B * traj_batch_size, seq_len, d_model)
            if output_hidden_states:
                all_hidden_states.append(embedding_output)
            if output_attentions:
                all_self_attentions.append(attn_score)
        # (B * traj_batch_size, seq_len, d_model), list of (B * traj_batch_size, seq_len, d_model), list of (B * traj_batch_size, head, seq_len, seq_len)
        return embedding_output, all_hidden_states, all_self_attentions
'''

class MLM(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        Args:
            hidden: output size of model
            vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))


class DRNTRLPooler(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.pooling = config.get('pooling', 'cls')
        self.add_cls = config.get('add_cls', True)
        d_model = config.get('d_model', 64)
        self.linear = nn.Linear(d_model, d_model)
        self.activation = nn.Tanh()

    def forward(self, out_traj, padding_masks, hidden_states=None):
        """
        Args:
            out_traj: (B * traj_B, seq_len, d_model) torch tensor of start output
            padding_masks: (B * traj_B, seq_len) boolean tensor, 1 means keep vector at this position, 0 means padding
            hidden_states: list of hidden, (B * traj_B, seq_len, d_model)
        Returns:
            output: (B * traj_B, d_model)
        """
        token_emb = out_traj  # (B * traj_B, seq_len, d_model)
        if self.pooling == 'cls':
            if self.add_cls:
                return self.activation(self.linear(token_emb[:, 0, :]))  # (B * traj_B, d_model)
            else:
                raise ValueError('No use cls!')
        elif self.pooling == 'cls_before_pooler':
            if self.add_cls:
                return token_emb[:, 0, :]  # (B * traj_B, d_model)
            else:
                raise ValueError('No use cls!')
        elif self.pooling == 'mean':
            input_mask_expanded = padding_masks.unsqueeze(-1).expand(token_emb.size()).float()  # (B * traj_B, seq_len, d_model)
            sum_embeddings = torch.sum(token_emb * input_mask_expanded, -2)
            sum_mask = input_mask_expanded.sum(-2)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            return sum_embeddings / sum_mask  # (B * traj_B, feat_dim)
        elif self.pooling == 'max':
            input_mask_expanded = padding_masks.unsqueeze(-1).expand(token_emb.size()).float()  # (B * traj_B, seq_len, d_model)
            token_emb[input_mask_expanded == 0] = float('-inf')  # Set padding tokens to large negative value
            max_over_time = torch.max(token_emb, -2)[0]
            return max_over_time  # (B * traj_B, feat_dim)
        elif self.pooling == "avg_first_last":
            first_hidden = hidden_states[0]  # (B * traj_B, seq_len, d_model)
            last_hidden = hidden_states[-1]  # (B * traj_B, seq_len, d_model)
            avg_emb = (first_hidden + last_hidden) / 2.0
            input_mask_expanded = padding_masks.unsqueeze(-1).expand(avg_emb.size()).float()  # (B * traj_B, seq_len, d_model)
            sum_embeddings = torch.sum(avg_emb * input_mask_expanded, -2)
            sum_mask = input_mask_expanded.sum(-2)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            return sum_embeddings / sum_mask  # (B * traj_B, d_model)
        elif self.pooling == "avg_top2":
            second_last_hidden = hidden_states[-2]  # (B * traj_B, seq_len, d_model)
            last_hidden = hidden_states[-1]  # (B * traj_B, seq_len, d_model)
            avg_emb = (second_last_hidden + last_hidden) / 2.0
            input_mask_expanded = padding_masks.unsqueeze(-1).expand(avg_emb.size()).float()  # (B * traj_B, seq_len, d_model)
            sum_embeddings = torch.sum(avg_emb * input_mask_expanded, -2)
            sum_mask = input_mask_expanded.sum(-2)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            return sum_embeddings / sum_mask  # (B * traj_B, d_model)
        else:
            raise ValueError('Error pooling type {}'.format(self.pooling))


'''
class TokenEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim, norm_layer=None):
        super().__init__()
        self.token_embed = nn.Linear(input_dim, embed_dim, bias=True)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.token_embed(x)
        x = self.norm(x)
        return x


class TrafPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=128):
        super(TrafPositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)].unsqueeze(2).expand_as(x).detach()


class LaplacianPE(nn.Module):
    def __init__(self, lape_dim, embed_dim):
        super().__init__()
        self.embedding_lap_pos_enc = nn.Linear(lape_dim, embed_dim)

    def forward(self, lap_mx):
        lap_pos_enc = self.embedding_lap_pos_enc(lap_mx).unsqueeze(0).unsqueeze(0)
        return lap_pos_enc
'''

class DataEmbedding(nn.Module):
    def __init__(
        self, feature_dim, embed_dim, num_nodes, node_fea_dim,
        gat_heads_per_layer=[8, 8, 1], gat_features_per_layer=[8, 8, 64], gat_dropout=0.1, drop=0.,
        add_time_in_day=False, add_day_in_week=False, time_intervals=1800,
    ):
        super().__init__()
        self.add_time_in_day = add_time_in_day
        self.add_day_in_week = add_day_in_week
        self.feature_dim = feature_dim

        self.value_embedding = TokenEmbedding(feature_dim, embed_dim)
        self.position_encoding = PositionalEncoding(embed_dim)
        if self.add_time_in_day:
            self.minute_size = int(24 * 3600 // time_intervals)
            self.daytime_embedding = nn.Embedding(self.minute_size, embed_dim)
        if self.add_day_in_week:
            self.weekday_embedding = nn.Embedding(7, embed_dim)
        self.ada_spatial_embedding = nn.Parameter(torch.Tensor(num_nodes, embed_dim))
        self.sta_spatial_embedding = nn.Linear(node_fea_dim, embed_dim)
        self.spatial_embedding = DRNRLGAT(
            d_model=embed_dim, in_feature=embed_dim, num_heads_per_layer=gat_heads_per_layer, num_features_per_layer=gat_features_per_layer,
            add_skip_connection=True, bias=True, dropout=gat_dropout, load_trans_prob=False, avg_last=True,
        )
        self.dropout = nn.Dropout(drop)

        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.ada_spatial_embedding)

    def forward(self, x, node_features, edge_index, edge_prob):
        origin_x = x
        x = self.value_embedding(origin_x[:, :, :, :self.feature_dim])
        x += self.position_encoding(x)
        if self.add_time_in_day:
            x += self.daytime_embedding((origin_x[:, :, :, self.feature_dim] * self.minute_size).round().long())
        if self.add_day_in_week:
            x += self.weekday_embedding(origin_x[:, :, :, self.feature_dim + 1].round().long())
        x += self.spatial_embedding(
            node_features=self.sta_spatial_embedding(node_features) + self.ada_spatial_embedding,
            edge_index_input=edge_index, edge_prob_input=edge_prob,
        )
        x = self.dropout(x)
        return x
'''

class TrafGATLayer(nn.Module):

    head_dim = 3

    def __init__(
        self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
        dropout_prob=0.3, add_skip_connection=True, bias=False, load_trans_prob=False,
    ):
        super().__init__()
        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.concat = concat  # whether we should concatenate or average the attention heads
        self.add_skip_connection = add_skip_connection
        self.load_trans_prob = False

        self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        if self.load_trans_prob:
            self.linear_proj_tran_prob = nn.Linear(1, num_of_heads * num_out_features, bias=False)
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, 1, 1, num_of_heads, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, 1, 1, num_of_heads, num_out_features))
        if self.load_trans_prob:
            self.scoring_trans_prob = nn.Parameter(torch.Tensor(1, 1, 1, num_of_heads, num_out_features))
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)
        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        self.leakyReLU = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_prob)

        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)
        if self.load_trans_prob:
            nn.init.xavier_uniform_(self.linear_proj_tran_prob.weight)
            nn.init.xavier_uniform_(self.scoring_trans_prob)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def skip_concat_bias(self, in_nodes_features, out_nodes_features):
        if not out_nodes_features.is_contiguous():
            out_nodes_features = out_nodes_features.contiguous()
        B, T, N, _, _ = out_nodes_features.shape

        if self.add_skip_connection:
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:  # if FIN == FOUT
                # in_nodes_features: (B, T, N, FIN), out_nodes_features: (B, T, N, NH, FOUT)
                out_nodes_features += in_nodes_features.unsqueeze(self.head_dim)
            else:
                out_nodes_features += self.skip_proj(in_nodes_features).view(B, T, N, self.num_of_heads, self.num_out_features)

        if self.concat:
            # (B, T, N, NH, FOUT) -> (B, T, N, NH * FOUT)
            out_nodes_features = out_nodes_features.view(B, T, N, self.num_of_heads * self.num_out_features)
        else:
            # (B, T, N, NH, FOUT) -> (B, T, N, FOUT)
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)


class TrafSparseGATLayer(TrafGATLayer):

    src_nodes_dim = 0  # position of source nodes in edge index
    trg_nodes_dim = 1  # position of target nodes in edge index

    nodes_dim = 2      # node dimension/axis
    head_dim = 3       # attention head dimension/axis

    def __init__(
        self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
        dropout_prob=0.3, add_skip_connection=True, bias=False, load_trans_prob=False,
    ):
        super().__init__(
            num_in_features, num_out_features, num_of_heads, concat, activation,
            dropout_prob, add_skip_connection, bias, load_trans_prob,
        )

    def forward(self, in_nodes_features, edge_index, edge_prob=None):
        B, T, N, _ = in_nodes_features.shape
        assert edge_index.shape[0] == 2, f'Expected edge index with shape=(2,E) got {edge_index.shape}'

        in_nodes_features = self.dropout(in_nodes_features)  # (B, T, N, embed_dim)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(B, T, N, self.num_of_heads, self.num_out_features)  # (B, T, N, NH, FOUT)
        nodes_features_proj = self.dropout(nodes_features_proj)
        if self.load_trans_prob:
            trans_prob_proj = self.linear_proj_tran_prob(edge_prob).view(-1, self.num_of_heads, self.num_out_features)
            trans_prob_proj = self.dropout(trans_prob_proj)

        # (B, T, N, NH, FOUT) * (1, 1, 1, NH, FOUT) -> (B, T, N, NH, FOUT) -> (B, T, N, NH)
        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)
        # scores: (B, T, E, NH), nodes_features_proj_lifted: (B, T, E, NH, FOUT)
        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(scores_source, scores_target, nodes_features_proj, edge_index)
        if self.load_trans_prob:
            scores_trans_prob = (trans_prob_proj * self.scoring_trans_prob).sum(dim=-1)
            scores_trans_prob = scores_trans_prob.unsqueeze(0).unsqueeze(0)
            scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted + scores_trans_prob)
        else:
            scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)

        attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[self.trg_nodes_dim], N)  # (B, T, E, NH, 1)
        attentions_per_edge = self.dropout(attentions_per_edge)
        # (B, T, E, NH, FOUT) * (B, T, E, NH, 1) -> (B, T, E, NH, FOUT)
        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge
        out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, N)  # (B, T, N, NH, FOUT)
        out_nodes_features = self.skip_concat_bias(in_nodes_features, out_nodes_features)
        return out_nodes_features

    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
        scores_per_edge = scores_per_edge - scores_per_edge.max()  # (B, T, E, NH)
        exp_scores_per_edge = scores_per_edge.exp()
        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index, num_of_nodes)  # (B, T, E, NH)
        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)
        return attentions_per_edge.unsqueeze(-1)  # (B, T, E, NH) -> (B, T, E, NH, 1)

    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):
        # (E) -> (B, T, E, NH)
        trg_index_broadcasted = trg_index.unsqueeze(-1).unsqueeze(0).unsqueeze(0).expand_as(exp_scores_per_edge)
        size = list(exp_scores_per_edge.shape)  # (B, T, N, NH)
        size[self.nodes_dim] = num_of_nodes
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)
        neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge)
        return neighborhood_sums.index_select(self.nodes_dim, trg_index)  # (B, T, N, NH) -> (B, T, E, NH)

    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
        size = list(nodes_features_proj_lifted_weighted.shape)
        size[self.nodes_dim] = num_of_nodes  # (B, T, N, NH, FOUT)
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)
        # (E) -> (B, T, E, NH, FOUT)
        trg_index_broadcasted = edge_index[self.trg_nodes_dim].unsqueeze(-1).unsqueeze(-1).unsqueeze(0).unsqueeze(0).expand_as(nodes_features_proj_lifted_weighted)
        out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted_weighted)  # (B, T, E, NH, FOUT) -> (B, T, N, NH, FOUT)
        return out_nodes_features

    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]
        scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
        scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index)
        return scores_source, scores_target, nodes_features_matrix_proj_lifted


class DRNAttention(nn.Module):
    def __init__(
        self, dim, s_num_heads=2, t_num_heads=6, qkv_bias=False,
        attn_drop=0., proj_drop=0., device=torch.device('cpu'), output_dim=1,
    ):
        super().__init__()
        assert dim % (s_num_heads + t_num_heads) == 0
        self.s_num_heads = s_num_heads
        self.t_num_heads = t_num_heads
        self.head_dim = dim // (s_num_heads + t_num_heads)
        self.scale = self.head_dim ** -0.5
        self.device = device
        self.s_ratio = s_num_heads / (s_num_heads + t_num_heads)
        self.t_ratio = 1 - self.s_ratio
        self.output_dim = output_dim

        self.t_q_conv = nn.Conv2d(dim, int(dim * self.t_ratio), kernel_size=1, bias=qkv_bias)
        self.t_k_conv = nn.Conv2d(dim, int(dim * self.t_ratio), kernel_size=1, bias=qkv_bias)
        self.t_v_conv = nn.Conv2d(dim, int(dim * self.t_ratio), kernel_size=1, bias=qkv_bias)
        self.t_attn_drop = nn.Dropout(attn_drop)

        self.gat = TrafSparseGATLayer(dim, self.head_dim, s_num_heads, dropout_prob=attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, edge_index, edge_prob=None):
        B, T, N, D = x.shape

        t_q = self.t_q_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_k = self.t_k_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_v = self.t_v_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_q = t_q.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_k = t_k.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_v = t_v.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_attn = (t_q @ t_k.transpose(-2, -1)) * self.scale
        t_attn = t_attn.softmax(dim=-1)
        t_attn = self.t_attn_drop(t_attn)
        t_x = (t_attn @ t_v).transpose(2, 3).reshape(B, N, T, int(D * self.t_ratio)).transpose(1, 2)

        s_x = self.gat(x, edge_index, edge_prob)

        x = torch.cat([t_x, s_x], dim=-1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DRNEncoderBlock(nn.Module):

    def __init__(
        self, dim, s_num_heads=2, t_num_heads=6, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
        drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, device=torch.device('cpu'), type_ln="post", output_dim=1,
    ):
        super().__init__()
        self.type_ln = type_ln
        self.norm1 = norm_layer(dim)
        self.drn_attn = DRNAttention(
            dim, s_num_heads=s_num_heads, t_num_heads=t_num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop, device=device, output_dim=output_dim,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, edge_index, edge_prob=None):
        if self.type_ln == 'pre':
            x = x + self.drop_path(self.drn_attn(self.norm1(x), edge_index, edge_prob))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        elif self.type_ln == 'post':
            x = self.norm1(x + self.drop_path(self.drn_attn(x, edge_index, edge_prob)))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        return x
'''

class DRNRL(nn.Module):

    def  __init__(self, config, data_feature):
        super().__init__()
        drop_path = config.get("drop_path", 0.3)
        enc_depth = config.get("enc_depth", 6)
        embed_dim = config.get('embed_dim', 64)
        s_num_heads = config.get('s_num_heads', 2)
        t_num_heads = config.get('t_num_heads', 6)
        mlp_ratio = config.get("mlp_ratio", 4)
        qkv_bias = config.get("qkv_bias", True)
        drop = config.get("traf_drop", 0.)
        attn_drop = config.get("traf_attn_drop", 0.)
        device = config.get('device', torch.device('cpu'))
        type_ln = config.get("type_ln", "post")
        output_dim = config.get('output_dim', 1)
        d_model = config.get("d_model", 64)
        self.traf_g_edge_index = data_feature.get("traf_g_edge_index")
        self.traf_g_edge_prob = data_feature.get("traf_g_loc_trans_prob")

        enc_dpr = [x.item() for x in torch.linspace(0, drop_path, enc_depth)]
        self.encoder_blocks = nn.ModuleList([
            DRNEncoderBlock(
                dim=embed_dim, s_num_heads=s_num_heads, t_num_heads=t_num_heads,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, drop_path=enc_dpr[i], act_layer=nn.GELU,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), device=device, type_ln=type_ln, output_dim=output_dim,
            ) for i in range(enc_depth)
        ])
        self.skip_convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=embed_dim, out_channels=d_model, kernel_size=1,
            ) for _ in range(enc_depth)
        ])

    def forward(self, X):
        skip = 0
        for i, encoder_block in enumerate(self.encoder_blocks):
            X = encoder_block(X, self.traf_g_edge_index, self.traf_g_edge_prob)  # (B, T, N, embed_dim)
            skip += self.skip_convs[i](X.permute(0, 3, 2, 1))  # (B, d_model, N, T)
        return skip.permute(0, 3, 2, 1)


'''
class CoAttention(nn.Module):

    def __init__(
        self, num_heads, d_model, dim_out, attn_drop=0.1, proj_drop=0.1, device=torch.device('cpu'),
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.device = device
        self.scale = self.d_k ** -0.5

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.dropout = nn.Dropout(p=attn_drop)

        self.proj = nn.Linear(d_model, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, mask=None, output_attentions=False):
        batch_size = q.shape[0]
        query, key, value = [
            l(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2) for l, x in zip(self.linear_layers, (q, k, k))
        ]
        scores = torch.matmul(query, key.transpose(-2, -1) * self.scale)
        if mask is not None:
            pass
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        out = torch.matmul(p_attn, value)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out, p_attn if output_attentions else None
'''


class CoGATLayer(nn.Module):

    head_dim = 2

    def __init__(
        self, num_in_features, num_out_features, num_of_heads, concat=True, activation=None,
        dropout_prob=0.3, add_skip_connection=False, bias=False, load_trans_prob=False,
    ):
        super().__init__()
        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.concat = concat  # whether we should concatenate or average the attention heads
        self.add_skip_connection = add_skip_connection
        self.load_trans_prob = False

        self.trg_linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        self.src_linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        if self.load_trans_prob:
            self.linear_proj_tran_prob = nn.Linear(1, num_of_heads * num_out_features, bias=False)
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, 1, num_of_heads, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, 1, num_of_heads, num_out_features))
        if self.load_trans_prob:
            self.scoring_trans_prob = nn.Parameter(torch.Tensor(1, 1, 1, num_of_heads, num_out_features))
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)
        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        self.leakyReLU = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_prob)

        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.trg_linear_proj.weight)
        nn.init.xavier_uniform_(self.src_linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)
        if self.load_trans_prob:
            nn.init.xavier_uniform_(self.linear_proj_tran_prob.weight)
            nn.init.xavier_uniform_(self.scoring_trans_prob)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def skip_concat_bias(self, in_nodes_features, out_nodes_features):
        if not out_nodes_features.is_contiguous():
            out_nodes_features = out_nodes_features.contiguous()
        B, N, _, _ = out_nodes_features.shape

        if self.add_skip_connection:
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:  # if FIN == FOUT
                # in_nodes_features: (B, N, FIN), out_nodes_features: (B, N, NH, FOUT)
                out_nodes_features += in_nodes_features.unsqueeze(self.head_dim)
            else:
                out_nodes_features += self.skip_proj(in_nodes_features).view(B, N, self.num_of_heads, self.num_out_features)

        if self.concat:
            # (B, N, NH, FOUT) -> (B, N, NH * FOUT)
            out_nodes_features = out_nodes_features.view(B, N, self.num_of_heads * self.num_out_features)
        else:
            # (B, N, NH, FOUT) -> (B, N, FOUT)
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)


class CoSparseGATLayer(CoGATLayer):

    src_nodes_dim = 0  # position of source nodes in edge index
    trg_nodes_dim = 1  # position of target nodes in edge index

    nodes_dim = 1      # node dimension/axis
    head_dim = 2       # attention head dimension/axis

    def __init__(
        self, num_in_features, num_out_features, num_of_heads, concat=True, activation=None,
        dropout_prob=0.3, add_skip_connection=False, bias=False, load_trans_prob=False,
    ):
        super().__init__(
            num_in_features, num_out_features, num_of_heads, concat, activation,
            dropout_prob, add_skip_connection, bias, load_trans_prob,
        )

    def forward(self, trg, src, edge_index, edge_prob=None):
        B, Nt, _ = trg.shape
        _, Ns, _ = src.shape
        assert edge_index.shape[0] == 2, f'Expected edge index with shape=(2, E) got {edge_index.shape}'

        trg = self.dropout(trg)  # (B, Nt, d_model)
        trg_proj = self.trg_linear_proj(trg).view(B, Nt, self.num_of_heads, self.num_out_features)  # (B, Nt, NH, FOUT)
        trg_proj = self.dropout(trg_proj)
        src = self.dropout(src)  # (B, Ns, d_model)
        src_proj = self.src_linear_proj(src).view(B, Ns, self.num_of_heads, self.num_out_features)  # (B, Ns, NH, FOUT)
        src_proj = self.dropout(src_proj)
        if self.load_trans_prob:
            trans_prob_proj = self.linear_proj_tran_prob(edge_prob).view(-1, self.num_of_heads, self.num_out_features)
            trans_prob_proj = self.dropout(trans_prob_proj)

        # (B, N, NH, FOUT) * (1, 1, NH, FOUT) -> (B, N, NH, FOUT) -> (B, N, NH)
        scores_source = (src_proj * self.scoring_fn_source).sum(dim=-1)  # (B, Ns, NH)
        scores_target = (trg_proj * self.scoring_fn_target).sum(dim=-1)  # (B, Nt, NH)
        # scores: (B, E, NH), src_proj_lifted: (B, E, NH, FOUT)
        scores_source_lifted, scores_target_lifted, src_proj_lifted = self.lift(scores_source, scores_target, src_proj, edge_index)
        if self.load_trans_prob:
            scores_trans_prob = (trans_prob_proj * self.scoring_trans_prob).sum(dim=-1)
            scores_trans_prob = scores_trans_prob.unsqueeze(0).unsqueeze(0)
            scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted + scores_trans_prob)
        else:
            scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)

        attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[self.trg_nodes_dim], Nt)  # (B, E, NH, 1)
        attentions_per_edge = self.dropout(attentions_per_edge)
        src_proj_lifted_weighted = src_proj_lifted * attentions_per_edge  # (B, E, NH, FOUT) * (B, E, NH, 1) -> (B, E, NH, FOUT)
        out_nodes_features = self.aggregate_neighbors(src_proj_lifted_weighted, edge_index, trg, Nt)  # (B, Nt, NH, FOUT)
        out_nodes_features = self.skip_concat_bias(trg, out_nodes_features)
        return out_nodes_features

    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, Nt):
        scores_per_edge = scores_per_edge - scores_per_edge.max()  # (B, E, NH)
        exp_scores_per_edge = scores_per_edge.exp()
        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index, Nt)  # (B, E, NH)
        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)
        return attentions_per_edge.unsqueeze(-1)  # (B, E, NH) -> (B, E, NH, 1)

    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, Nt):
        # (E) -> (B, E, NH)
        trg_index_broadcasted = trg_index.unsqueeze(-1).unsqueeze(0).expand_as(exp_scores_per_edge)
        size = list(exp_scores_per_edge.shape)
        size[self.nodes_dim] = Nt
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)  # (B, Nt, NH)
        neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge)
        return neighborhood_sums.index_select(self.nodes_dim, trg_index)  # (B, Nt, NH) -> (B, E, NH)

    def aggregate_neighbors(self, src_proj_lifted_weighted, edge_index, trg, Nt):
        size = list(src_proj_lifted_weighted.shape)
        size[self.nodes_dim] = Nt  # (B, Nt, NH, FOUT)
        out_nodes_features = torch.zeros(size, dtype=trg.dtype, device=trg.device)
        # (E) -> (B, E, NH, FOUT)
        trg_index_broadcasted = edge_index[self.trg_nodes_dim].unsqueeze(-1).unsqueeze(-1).unsqueeze(0).expand_as(src_proj_lifted_weighted)
        out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted, src_proj_lifted_weighted)  # (B, E, NH, FOUT) -> (B, Nt, NH, FOUT)
        return out_nodes_features

    def lift(self, scores_source, scores_target, src_proj, edge_index):
        """
        scores_source: (B, Ns, NH)
        scores_target: (B, Nt, NH)
        src_proj: (B, Ns, NH, FOUT)
        """
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]
        scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)  # (B, E, NH)
        scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)  # (B, E, NH)
        src_proj_lifted = src_proj.index_select(self.nodes_dim, src_nodes_index)  # (B, E, NH, FOUT)
        return scores_source, scores_target, src_proj_lifted


class CoAttention(nn.Module):

    def __init__(
        self, num_heads, d_model, attn_drop=0.1, proj_drop=0.1, device=torch.device('cpu'),
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.device = device
        self.scale = self.d_k ** -0.5

        self.gat = CoSparseGATLayer(d_model, self.d_k, num_heads, dropout_prob=attn_drop)

        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, edge_index, edge_prob=None):
        x = self.gat(q, k, edge_index, edge_prob)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CoTransformerBlock(nn.Module):

    def __init__(
        self, d_model, coattn_heads, feed_forward_hidden, drop_path, attn_drop, dropout,
        type_ln='post', device=torch.device('cpu'),
    ):
        super().__init__()
        self.type_ln = type_ln

        self.traf_attention = CoAttention(
            num_heads=coattn_heads, d_model=d_model,
            attn_drop=attn_drop, proj_drop=dropout, device=device,
        )
        self.traf_mlp = Mlp(
            in_features=d_model, hidden_features=feed_forward_hidden, out_features=d_model, act_layer=nn.GELU, drop=dropout,
        )
        self.traf_norm1 = nn.LayerNorm(d_model)
        self.traf_norm2 = nn.LayerNorm(d_model)
        self.traf_drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.traj_attention = CoAttention(
            num_heads=coattn_heads, d_model=d_model,
            attn_drop=attn_drop, proj_drop=dropout, device=device,
        )
        self.traj_mlp = Mlp(
            in_features=d_model, hidden_features=feed_forward_hidden, out_features=d_model, act_layer=nn.GELU, drop=dropout,
        )
        self.traj_norm1 = nn.LayerNorm(d_model)
        self.traj_norm2 = nn.LayerNorm(d_model)
        self.traj_drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, raw_traf_rn_emb, raw_traj_rn_emb, co_traf_edge_index, co_traj_edge_index, output_attentions=False):
        if self.type_ln == "pre":
            traf_attn_out = self.traf_attention(
                self.traf_norm1(raw_traf_rn_emb), self.traf_norm1(raw_traj_rn_emb), co_traf_edge_index,
            )
            traf_rn_emb = raw_traf_rn_emb + self.traf_drop_path(traf_attn_out)
            traf_rn_emb += self.traf_drop_path(self.traf_mlp(self.traf_norm2(traf_rn_emb)))

            traj_attn_out = self.traj_attention(
                self.traj_norm1(raw_traj_rn_emb), self.traj_norm1(raw_traf_rn_emb), co_traj_edge_index,
            )
            traj_rn_emb = raw_traj_rn_emb + self.traj_drop_path(traj_attn_out)
            traj_rn_emb += self.traj_drop_path(self.traj_mlp(self.traj_norm2(traj_rn_emb)))
        elif self.type_ln == "post":
            traf_attn_out = self.traf_attention(
                raw_traf_rn_emb, raw_traj_rn_emb, co_traf_edge_index,
            )
            traf_rn_emb = self.traf_norm1(raw_traf_rn_emb + self.traf_drop_path(traf_attn_out))
            traf_rn_emb = self.traf_norm2(traf_rn_emb + self.traf_drop_path(self.traf_mlp(traf_rn_emb)))

            traj_attn_out = self.traj_attention(
                raw_traj_rn_emb, raw_traf_rn_emb, co_traj_edge_index,
            )
            traj_rn_emb = self.traj_norm1(raw_traj_rn_emb + self.traj_drop_path(traj_attn_out))
            traj_rn_emb = self.traj_norm2(traj_rn_emb + self.traj_drop_path(self.traj_mlp(traj_rn_emb)))
        else:
            raise ValueError('Error type_ln {}'.format(self.type_ln))
        return traf_rn_emb, traj_rn_emb


from TRACK_TRLLM_Cont import TRL

class DRNTRLLM(nn.Module):

    def __init__(self, config, data_feature):
        super().__init__()
        input_window = config.get("input_window", 6)
        embed_dim = config.get('embed_dim', 64)
        lape_dim = config.get('lape_dim', 8)
        traf_drop = config.get('traf_drop', 0.)
        d_model = config.get('d_model', 64)
        gat_heads_per_layer = config.get('gat_heads_per_layer', [8, 16, 1])
        gat_features_per_layer = config.get('gat_features_per_layer', [16, 16, 64])
        gat_dropout = config.get('gat_dropout', 0.1)
        load_trans_prob = config.get('load_trans_prob', True)
        self.load_t_trans_prob = config.get('load_t_trans_prob', True)
        gat_avg_last = config.get('gat_avg_last', True)
        traj_drop = config.get('traj_drop', 0.1)
        add_pe = config.get('add_pe', True)
        self.add_delta_time = config.get('add_delta_time', True)
        add_time_in_day = config.get('add_time_in_day', True)
        add_day_in_week = config.get('add_day_in_week', True)
        self.time_intervals = config.get('time_intervals', 1800)
        mlp_ratio = config.get('mlp_ratio', 4)
        type_ln = config.get('type_ln', 'post')
        device = config.get('device', torch.device('cpu'))
        coattn_heads = config.get('coattn_heads', 8)
        co_n_layers = config.get('co_n_layers', 1)
        coattn_drop = config.get('coattn_drop', 0)
        co_dropout = config.get('co_dropout', 0)
        co_drop_path = config.get('co_drop_path', 0)
        self.feature_dim = data_feature.get("feature_dim", 1)
        self.ext_dim = data_feature.get("ext_dim", 0)
        output_dim = data_feature.get("output_dim", 1)
        vocab_size = data_feature.get('vocab_size')
        node_fea_dim = data_feature.get('node_fea_dim')
        self.co_traf_edge_index = data_feature.get('co_traf_edge_index')
        self.co_traj_edge_index = data_feature.get('co_traj_edge_index')
        self.add_aft = config.get('add_aft', True)
        traf_gat_heads_per_layer = config.get('traf_gat_heads_per_layer', [8, 16, 1])
        traf_gat_features_per_layer = config.get('traf_gat_features_per_layer', [16, 16, 64])
        self.node_features = data_feature.get("node_features")
        self.traf_edge_index = data_feature.get("traf_edge_index")
        self.traf_edge_prob = data_feature.get("traf_edge_prob")
        num_nodes = data_feature.get('num_nodes')

        self.enc_embed_layer = DataEmbedding(
            self.feature_dim - self.ext_dim, embed_dim, num_nodes, node_fea_dim, drop=traf_drop,
            gat_heads_per_layer=traf_gat_heads_per_layer, gat_features_per_layer=traf_gat_features_per_layer,
            add_time_in_day=add_time_in_day, add_day_in_week=add_day_in_week,
        )
        self.bef_drnrl = DRNRL(config, data_feature)
        self.traf_emb_conv = nn.Conv2d(
            in_channels=input_window, out_channels=1, kernel_size=1,
        )
        if self.add_aft:
            self.inv_traf_emb_conv = nn.Conv2d(in_channels=1, out_channels=input_window, kernel_size=1)
            self.aft_embedding = nn.Linear(d_model, embed_dim)
            self.aft_drnrl = DRNRL(config, data_feature)
            self.end_traf_emb_conv = nn.Conv2d(in_channels=input_window, out_channels=1, kernel_size=1)
        self.traf_next_linear = nn.Linear(d_model, output_dim)

        self.minute_size = 24 * 3600 // self.time_intervals
        if self.load_t_trans_prob:
            self.s_emb = DGAT(
                d_model=d_model, in_feature=node_fea_dim, num_heads_per_layer=gat_heads_per_layer, num_features_per_layer=gat_features_per_layer,
                add_skip_connection=True, bias=True, dropout=gat_dropout, load_trans_prob=load_trans_prob, avg_last=gat_avg_last,
            )
        else:
            self.s_emb = GAT(
                d_model=d_model, in_feature=node_fea_dim, num_heads_per_layer=gat_heads_per_layer, num_features_per_layer=gat_features_per_layer,
                add_skip_connection=True, bias=True, dropout=gat_dropout, load_trans_prob=load_trans_prob, avg_last=gat_avg_last,
            )
        self.t_emb = TimeEmbedding(
            d_model=d_model, dropout=traj_drop, add_pe=add_pe, add_delta_time=self.add_delta_time, add_time_in_day=add_time_in_day, add_day_in_week=add_day_in_week,
        )
        self.trl = TRL(config, data_feature)
        self.mask_l = MLM(d_model, vocab_size)
        if self.add_delta_time:
            self.mask_t = nn.Linear(d_model, 1)

        self.cotransformer_blocks = nn.ModuleList([
            CoTransformerBlock(
                d_model, coattn_heads, d_model * mlp_ratio, co_drop_path, coattn_drop, co_dropout,
                type_ln=type_ln, device=device,
            ) for _ in range(co_n_layers)
        ])

    def forward(
        self, traf,
        traj, padding_masks, batch_temporal_mat=None,
        graph_dict=None, output_hidden_states=False, output_attentions=False,
    ):
        """
        Args:
            traf: (B, T, N, D)
            traj: (B, traj_B, seq_len, feat_dim)
            padding_masks: (B, traj_B, seq_len) boolean tensor, 1 means keep vector at this position, 0 means padding
            batch_temporal_mat: (B, traj_B, seq_len, seq_len)
        Returns:
            pred_next_traf: (B, N, D)
            pred_masked_traj: (B, traj_B, seq_len, vocab_size)
        """
        traf_enc = self.enc_embed_layer(traf, self.node_features, self.traf_edge_index, self.traf_edge_prob)  # (B, T, N, embed_dim)
        traf_emb = self.bef_drnrl(traf_enc)  # (B, T, N, d_model)
        traf_rn_emb = self.traf_emb_conv(traf_emb).squeeze(1)  # (B, N, d_model)

        batch_size, traj_batch_size, seq_len, feat_dim = traj.shape
        if self.load_t_trans_prob:
            time_ind = (traj[:, 1, 3] - 1) * (24 * 3600 // self.time_intervals) + (traj[:, 1, 2] - 1) // (self.time_intervals // 60)
            t_edge_prob_input = graph_dict['traj_t_loc_trans_prob'][time_ind]
            traj_rn_emb = self.s_emb(
                node_features=graph_dict['node_features'], edge_index_input=graph_dict['traj_edge_index'],
                edge_prob_input=graph_dict['traj_loc_trans_prob'], t_edge_prob_input=t_edge_prob_input,
            )  # (B, vocab_size, d_model)
        else:
            rn_emb = self.s_emb(
                node_features=graph_dict['node_features'], edge_index_input=graph_dict['traj_edge_index'],
                edge_prob_input=graph_dict['traj_loc_trans_prob'],
            )  # (vocab_size, d_model)
            traj_rn_emb = rn_emb.unsqueeze(0).expand(batch_size, -1, -1)  # (B, vocab_size, d_model)

        # rn_emb -> traj_rn_emb & traf_rn_emb by co-attention
        for cotransformer in self.cotransformer_blocks:
            traf_rn_emb, traj_rn_emb = cotransformer.forward(
                traf_rn_emb, traj_rn_emb, self.co_traf_edge_index, self.co_traj_edge_index,
            )

        if self.add_aft:
            """
            traf_rn_emb: (B, N, d_model)
            traf_emb: (B, T, N, d_model)
            """
            inv_traf_rn_emb = self.inv_traf_emb_conv(traf_rn_emb.unsqueeze(1))  # (B, T, N, d_model)
            traf_emb = self.aft_embedding(traf_emb + inv_traf_rn_emb)  # (B, T, N, embed_dim)
            traf_emb = self.aft_drnrl(traf_emb)  # (B, T, N, d_model)
            traf_rn_emb = self.end_traf_emb_conv(traf_emb).squeeze(1)  # (B, N, d_model)

        d_model = traj_rn_emb.shape[-1]
        traj_loc = traj[:, :, :, 0].unsqueeze(-1).expand(-1, -1, -1, d_model)
        traj_embs = traj_rn_emb.unsqueeze(1).expand(-1, traj_batch_size, -1, -1)
        traj_embs = traj_embs.gather(2, traj_loc)  # (B, traj_B, seq_len, d_model)
        '''
        traj_loc_embs = []
        for i in range(batch_size):
            traj_loc = traj[i, :, :, 0].reshape(-1, 1).squeeze(1)  # (traj_batch_size * seq_length,)
            traj_loc_emb = traj_rn_emb[traj_loc]  # (traj_batch_size * seq_length, d_model)
            traj_loc_emb = traj_loc_emb.reshape(traj_batch_size, seq_len, self.d_model)  # (traj_batch_size, seq_length, d_model)
            traj_loc_embs.append(traj_loc_emb)
        traj_loc_embs = torch.cat(traj_loc_embs, dim=0)  # (B, traj_batch_size, seq_length, d_model)
        '''

        traj_embs = traj_embs.reshape(-1, seq_len, d_model)  # (B * traj_B, seq_len, d_model)
        traj = traj.reshape(-1, seq_len, feat_dim)  # (B * traj_B, seq_len, D)
        padding_masks = padding_masks.reshape(-1, seq_len)  # (B * traj_B, seq_len)
        batch_temporal_mat = batch_temporal_mat.reshape(-1, seq_len, seq_len)  # (B * traj_B, seq_len, seq_len)

        traj_embs = self.t_emb(traj_embs, traj, position_ids=None)
        out_masked_traj, _, _ = self.trl(
        # out_masked_traj = self.trl(
            traj_embs, padding_masks, batch_temporal_mat=batch_temporal_mat,
            output_hidden_states=output_hidden_states, output_attentions=output_attentions,
        )  # (B * traj_B, seq_len, d_model)
        out_masked_traj = out_masked_traj.reshape(batch_size, traj_batch_size, seq_len, d_model)

        pred_next_traf = self.traf_next_linear(traf_rn_emb)
        pred_masked_traj = self.mask_l(out_masked_traj)
        pred_masked_time = None
        if self.add_delta_time:
            pred_masked_time = self.mask_t(out_masked_traj).squeeze(-1)

        return pred_next_traf, pred_masked_traj, pred_masked_time


class Model(DRNTRLLM):

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.add_cls = config.get('add_cls', True)
        self.add_match = config.get('add_match', True)
        self.pooling = config.get('pooling', 'cls')
        self.pooler = DRNTRLPooler(config)

    def forward(
        self, traf, out_lap_mx, in_lap_mx,
        traj1, padding_masks1, traj2, padding_masks2,
        traj, padding_masks, batch_temporal_mat=None,
        graph_dict=None, output_hidden_states=False, output_attentions=False,
    ):
        """
        Args:
            traf: (B, T, N, D)
            traj1: (B, traj_B, seq_len, feat_dim)
            padding_masks1: (B, traj_B, seq_len) boolean tensor, 1 means keep vector at this position, 0 means padding
            traj2: (B, traj_B, seq_len, feat_dim)
            padding_masks2: (B, traj_B, seq_len) boolean tensor, 1 means keep vector at this position, 0 means padding
            traj: (B, traj_B, seq_len, feat_dim)
            padding_masks: (B, traj_B, seq_len) boolean tensor, 1 means keep vector at this position, 0 means padding
            batch_temporal_mat: (B, traj_B, seq_len, seq_len)
        Returns:
            pred_next_traf: (B, N, D)
            pred_masked_traj: (B, traj_B, seq_len, vocab_size)
        """
        traf_enc = self.enc_embed_layer(traf, out_lap_mx, in_lap_mx)  # (B, T, N, embed_dim)
        traf_emb = self.bef_drnrl(traf_enc)  # (B, T, N, d_model)
        traf_rn_emb = self.traf_emb_conv(traf_emb).squeeze(1)  # (B, N, d_model)

        batch_size, traj_batch_size, seq_len, feat_dim = traj.shape
        if self.load_t_trans_prob:
            time_ind = traf[:, -1, 0, self.feature_dim - self.ext_dim + 1] * self.minute_size + traf[:, -1, 0, self.feature_dim - self.ext_dim] * self.minute_size + 1
            time_ind = torch.fmod(time_ind.long(), 7 * self.minute_size)
            t_edge_prob_input = graph_dict['traj_t_loc_trans_prob'][time_ind]
            traj_rn_emb = self.s_emb(
                node_features=graph_dict['node_features'], edge_index_input=graph_dict['traj_edge_index'],
                edge_prob_input=graph_dict['traj_loc_trans_prob'], t_edge_prob_input=t_edge_prob_input,
            )  # (B, vocab_size, d_model)
        else:
            rn_emb = self.s_emb(
                node_features=graph_dict['node_features'], edge_index_input=graph_dict['traj_edge_index'],
                edge_prob_input=graph_dict['traj_loc_trans_prob'],
            )  # (vocab_size, d_model)
            traj_rn_emb = rn_emb.unsqueeze(0).expand(batch_size, -1, -1)  # (B, vocab_size, d_model)

        # rn_emb -> traj_rn_emb & traf_rn_emb by co-attention
        for cotransformer in self.cotransformer_blocks:
            traf_rn_emb, traj_rn_emb = cotransformer.forward(
                traf_rn_emb, traj_rn_emb, self.co_traf_edge_index, self.co_traj_edge_index,
            )

        if self.add_aft:
            """
            traf_rn_emb: (B, N, d_model)
            traf_emb: (B, T, N, d_model)
            """
            inv_traf_rn_emb = self.inv_traf_emb_conv(traf_rn_emb.unsqueeze(1))  # (B, T, N, d_model)
            traf_emb = self.aft_embedding(traf_emb + inv_traf_rn_emb)  # (B, T, N, embed_dim)
            traf_emb = self.aft_drnrl(traf_emb)  # (B, T, N, d_model)
            traf_rn_emb = self.end_traf_emb_conv(traf_emb).squeeze(1)  # (B, N, d_model)

        if self.pooling in ['avg_first_last', 'avg_top2']:
            output_hidden_states = True

        d_model = traj_rn_emb.shape[-1]
        traj_loc = traj[:, :, :, 0].unsqueeze(-1).expand(-1, -1, -1, d_model)
        traj_embs = traj_rn_emb.unsqueeze(1).expand(-1, traj_batch_size, -1, -1)
        traj_embs = traj_embs.gather(2, traj_loc)  # (B, traj_B, seq_len, d_model)

        traj_loc1 = traj1[:, :, :, 0].unsqueeze(-1).expand(-1, -1, -1, d_model)
        traj_embs1 = traj_rn_emb.unsqueeze(1).expand(-1, traj_batch_size, -1, -1)
        traj_embs1 = traj_embs1.gather(2, traj_loc1)  # (B, traj_B, seq_len, d_model)

        traj_loc2 = traj2[:, :, :, 0].unsqueeze(-1).expand(-1, -1, -1, d_model)
        traj_embs2 = traj_rn_emb.unsqueeze(1).expand(-1, traj_batch_size, -1, -1)
        traj_embs2 = traj_embs2.gather(2, traj_loc2)  # (B, traj_B, seq_len, d_model)

        traj_embs = traj_embs.reshape(-1, seq_len, d_model)  # (B * traj_B, seq_len, d_model)
        traj = traj.reshape(-1, seq_len, feat_dim)  # (B * traj_B, seq_len, D)
        padding_masks = padding_masks.reshape(-1, seq_len)  # (B * traj_B, seq_len)

        traj_embs1 = traj_embs1.reshape(-1, seq_len, d_model)  # (B * traj_B, seq_len, d_model)
        traj1 = traj1.reshape(-1, seq_len, feat_dim)  # (B * traj_B, seq_len, D)
        padding_masks1 = padding_masks1.reshape(-1, seq_len)  # (B * traj_B, seq_len)

        traj_embs2 = traj_embs2.reshape(-1, seq_len, d_model)  # (B * traj_B, seq_len, d_model)
        traj2 = traj2.reshape(-1, seq_len, feat_dim)  # (B * traj_B, seq_len, D)
        padding_masks2 = padding_masks2.reshape(-1, seq_len)  # (B * traj_B, seq_len)

        traj_embs = self.t_emb(traj_embs, traj, position_ids=None)
        out_masked_traj, hidden_states, _ = self.trl(
            traj_embs, padding_masks, batch_temporal_mat=batch_temporal_mat,
            output_hidden_states=output_hidden_states, output_attentions=output_attentions,
        )  # (B * traj_B, seq_len, d_model)
        out_masked_traj = out_masked_traj.reshape(batch_size, traj_batch_size, seq_len, d_model)

        traj_embs1 = self.t_emb(traj_embs1, traj1, position_ids=None)
        out_traj1, hidden_states1, _ = self.trl(
            traj_embs1, padding_masks1, batch_temporal_mat=None,
            output_hidden_states=output_hidden_states, output_attentions=output_attentions,
        )  # (B * traj_B, seq_len, d_model)

        traj_embs2 = self.t_emb(traj_embs2, traj2, position_ids=None)
        out_traj2, hidden_states2, _ = self.trl(
            traj_embs2, padding_masks2, batch_temporal_mat=None,
            output_hidden_states=output_hidden_states, output_attentions=output_attentions,
        )  # (B * traj_B, seq_len, d_model)

        pred_next_traf = self.traf_next_linear(traf_rn_emb)
        pred_masked_traj = self.mask_l(out_masked_traj)
        pred_masked_time = None
        if self.add_delta_time:
            pred_masked_time = self.mask_t(out_masked_traj).squeeze(-1)
        pool_out_traj1 = self.pooler(out_traj1, padding_masks1, hidden_states=hidden_states1)
        pool_out_traj2 = self.pooler(out_traj2, padding_masks2, hidden_states=hidden_states2)

        pool_out_traj1 = pool_out_traj1.reshape(batch_size, traj_batch_size, d_model)
        pool_out_traj2 = pool_out_traj2.reshape(batch_size, traj_batch_size, d_model)

        pool_drn_out_traj = None
        if self.add_match:
            traj2 = traj2.reshape(batch_size, traj_batch_size, seq_len, feat_dim)
            traj_loc2 = torch.clamp((traj2[:, :, :, 0] - 5), min=0).unsqueeze(-1).expand(-1, -1, -1, d_model)
            if self.add_cls:
                traj_loc2 = traj_loc2[:, :, 1:, :]
                padding_masks2 = padding_masks2[:, 1:]
            drn_traj_embs2 = traf_rn_emb.unsqueeze(1).expand(-1, traj_batch_size, -1, -1)
            drn_traj_embs2 = drn_traj_embs2.gather(2, traj_loc2)  # (B, traj_B, seq_len - 1, d_model)
            drn_traj_embs2 = drn_traj_embs2.reshape(batch_size * traj_batch_size, -1, d_model)
            padding_masks2_expanded = padding_masks2.unsqueeze(-1).expand(drn_traj_embs2.size()).float()  # (B * traj_B, seq_len, d_model)
            sum_embeddings = torch.sum(drn_traj_embs2 * padding_masks2_expanded, -2)
            sum_mask = padding_masks2_expanded.sum(-2)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            pool_drn_out_traj = sum_embeddings / sum_mask  # (B * traj_B, d_model)
            pool_drn_out_traj = pool_drn_out_traj.reshape(batch_size, traj_batch_size, d_model)

        return pred_next_traf, pred_masked_traj, pred_masked_time, pool_out_traj1, pool_out_traj2, pool_drn_out_traj
