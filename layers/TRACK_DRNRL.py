import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from utils.losses import loss


class GATLayer(nn.Module):

    head_dim = 1

    def __init__(
        self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
        dropout_prob=0.1, add_skip_connection=True, bias=True, load_trans_prob=False,
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


class SparseGATLayer(GATLayer):

    src_nodes_dim = 0  # position of source nodes in edge index
    trg_nodes_dim = 1  # position of target nodes in edge index

    nodes_dim = 0      # node dimension/axis
    head_dim = 1       # attention head dimension/axis

    def __init__(
        self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
        dropout_prob=0.1, add_skip_connection=True, bias=True, load_trans_prob=False,
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
        add_skip_connection=True, bias=True, dropout=0.1, load_trans_prob=False, avg_last=True,
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
            layer = SparseGATLayer(
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
            node_features: (N, fea_dim)
            edge_index_input: (2, E)
            edge_prob_input: (E, 1)
        Returns:
            (N, d_model)
        """
        data = (node_features, edge_index_input, edge_prob_input)
        (node_fea_emb, edge_index, edge_prob) = self.gat_net(data)  # (N, num_channels[-1])
        return node_fea_emb


class TokenEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim, norm_layer=None):
        super().__init__()
        self.token_embed = nn.Linear(input_dim, embed_dim, bias=True)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.token_embed(x)
        x = self.norm(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=100):
        super(PositionalEncoding, self).__init__()
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


class DataEmbedding(nn.Module):
    def __init__(
        self, feature_dim, embed_dim, num_nodes, node_fea_dim,
        gat_heads_per_layer=[8, 8, 1], gat_features_per_layer=[8, 8, 64], gat_dropout=0.1, drop=0.,
        add_time_in_day=False, add_day_in_week=False,
    ):
        super().__init__()

        self.add_time_in_day = add_time_in_day
        self.add_day_in_week = add_day_in_week
        self.feature_dim = feature_dim

        self.value_embedding = TokenEmbedding(feature_dim, embed_dim)
        self.position_encoding = PositionalEncoding(embed_dim)
        if self.add_time_in_day:
            self.daytime_embedding = nn.Embedding(48, embed_dim)
        if self.add_day_in_week:
            self.weekday_embedding = nn.Embedding(7, embed_dim)
        self.ada_spatial_embedding = nn.Parameter(torch.Tensor(num_nodes, embed_dim))
        self.sta_spatial_embedding = nn.Linear(node_fea_dim, embed_dim)
        self.spatial_embedding = GAT(
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
            x += self.daytime_embedding((origin_x[:, :, :, self.feature_dim] * 48).round().long())
        if self.add_day_in_week:
            x += self.weekday_embedding(origin_x[:, :, :, self.feature_dim + 1: self.feature_dim + 8].argmax(dim=3))
        x += self.spatial_embedding(
            node_features=self.sta_spatial_embedding(node_features) + self.ada_spatial_embedding,
            edge_index_input=edge_index, edge_prob_input=edge_prob,
        )
        x = self.dropout(x)
        return x


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

        if self.t_ratio != 0:
            self.t_q_conv = nn.Conv2d(dim, int(dim * self.t_ratio), kernel_size=1, bias=qkv_bias)
            self.t_k_conv = nn.Conv2d(dim, int(dim * self.t_ratio), kernel_size=1, bias=qkv_bias)
            self.t_v_conv = nn.Conv2d(dim, int(dim * self.t_ratio), kernel_size=1, bias=qkv_bias)
            self.t_attn_drop = nn.Dropout(attn_drop)

        if self.s_ratio != 0:
            self.gat = TrafSparseGATLayer(dim, self.head_dim, s_num_heads, dropout_prob=attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, edge_index, edge_prob=None):
        B, T, N, D = x.shape

        if self.t_ratio != 0:
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

        if self.s_ratio != 0:
            s_x = self.gat(x, edge_index, edge_prob)

        if self.t_ratio == 0:
            x = s_x
        elif self.s_ratio == 0:
            x = t_x
        else:
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


class DRNRL(nn.Module):
    def __init__(self, config, data_feature):
        super().__init__()

        feature_dim = data_feature.get("feature_dim", 1)
        ext_dim = data_feature.get("ext_dim", 0)
        self.node_features = data_feature.get("node_features")
        self.edge_index = data_feature.get("edge_index")  # 地理邻接矩阵 (2, E)
        self.edge_prob = data_feature.get("edge_prob")  # 全为1 (E, 1)
        self.g_edge_index = data_feature.get("g_edge_index")  # 全局邻接矩阵 (2, gE)
        self.g_edge_prob = data_feature.get("g_edge_prob")  # 全为1 (gE, 1)
        num_nodes = data_feature.get('num_nodes')
        node_fea_dim = data_feature.get('node_fea_dim')

        embed_dim = config.get('embed_dim', 64)
        skip_dim = config.get("skip_dim", 256)
        s_num_heads = config.get('s_num_heads', 6)
        t_num_heads = config.get('t_num_heads', 2)
        mlp_ratio = config.get("mlp_ratio", 4)
        qkv_bias = config.get("qkv_bias", True)
        drop = config.get("drop", 0.)
        attn_drop = config.get("attn_drop", 0.)
        drop_path = config.get("drop_path", 0.3)
        enc_depth = config.get("enc_depth", 6)
        type_ln = config.get("type_ln", "pre")
        gat_heads_per_layer = config.get("gat_heads_per_layer", [8, 8, 1])
        gat_features_per_layer = config.get("gat_features_per_layer", [8, 8, 64])

        output_dim = config.get('output_dim', 1)
        add_time_in_day = config.get("add_time_in_day", True)
        add_day_in_week = config.get("add_day_in_week", True)
        device = config.get('device', torch.device('cpu'))

        self.enc_embed_layer = DataEmbedding(
            feature_dim - ext_dim, embed_dim, num_nodes, node_fea_dim, drop=drop,
            gat_heads_per_layer=gat_heads_per_layer, gat_features_per_layer=gat_features_per_layer,
            add_time_in_day=add_time_in_day, add_day_in_week=add_day_in_week,
        )  # (B, T, N, embed_dim)

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
                in_channels=embed_dim, out_channels=skip_dim, kernel_size=1,
            ) for _ in range(enc_depth)
        ])

    def forward(self, X):
        enc = self.enc_embed_layer(X, self.node_features, self.edge_index, self.edge_prob)
        skip = 0
        for i, encoder_block in enumerate(self.encoder_blocks):
            enc = encoder_block(enc, self.g_edge_index, self.g_edge_prob)
            skip += self.skip_convs[i](enc.permute(0, 3, 2, 1))
        return skip


class DRNRLLM(nn.Module):

    def __init__(self, config, data_feature):
        super().__init__()
        self.data_feature = data_feature
        skip_dim = config.get('skip_dim', 128)
        input_window = config.get('input_window', 6)
        self.output_dim = config.get('output_dim', 1)
        self.mask_lambda = config.get("mask_lambda", 1)
        self.next_lambda = config.get("next_lambda", 1)
        self._scaler = data_feature.get('scaler')

        self.drnrl = DRNRL(config, data_feature)
        self.mask_end_conv = nn.Conv2d(in_channels=skip_dim, out_channels=self.output_dim, kernel_size=1, bias=True)
        self.next_end_conv1 = nn.Conv2d(in_channels=input_window, out_channels=1, kernel_size=1, bias=True)
        self.next_end_conv2 = nn.Conv2d(in_channels=skip_dim, out_channels=self.output_dim, kernel_size=1, bias=True)

    def forward(self, X_mask, X):
        mask_skip = self.drnrl(X_mask)
        mask_skip = self.mask_end_conv(F.relu(mask_skip))
        next_skip = self.drnrl(X)
        next_skip = self.next_end_conv1(F.relu(next_skip.permute(0, 3, 2, 1)))
        next_skip = self.next_end_conv2(F.relu(next_skip.permute(0, 3, 2, 1)))
        return mask_skip.permute(0, 3, 2, 1), next_skip.permute(0, 3, 2, 1)

    def get_loss_func(self, set_loss):
        if set_loss.lower() not in ['mae', 'mse', 'rmse', 'mape', 'logcosh', 'huber', 'quantile', 'masked_mae',
                                           'masked_mse', 'masked_rmse', 'masked_mape', 'masked_huber', 'r2', 'evar']:
            self._logger.warning('Received unrecognized train loss function, set default mae loss func.')
        if set_loss.lower() == 'mae':
            lf = loss.masked_mae_torch
        elif set_loss.lower() == 'mse':
            lf = loss.masked_mse_torch
        elif set_loss.lower() == 'rmse':
            lf = loss.masked_rmse_torch
        elif set_loss.lower() == 'mape':
            lf = loss.masked_mape_torch
        elif set_loss.lower() == 'logcosh':
            lf = loss.log_cosh_loss
        elif set_loss.lower() == 'huber':
            lf = partial(loss.huber_loss, delta=self.huber_delta)
        elif set_loss.lower() == 'quantile':
            lf = partial(loss.quantile_loss, delta=self.quan_delta)
        elif set_loss.lower() == 'masked_mae':
            lf = partial(loss.masked_mae_torch, null_val=0)
        elif set_loss.lower() == 'masked_mse':
            lf = partial(loss.masked_mse_torch, null_val=0)
        elif set_loss.lower() == 'masked_rmse':
            lf = partial(loss.masked_rmse_torch, null_val=0)
        elif set_loss.lower() == 'masked_mape':
            lf = partial(loss.masked_mape_torch, null_val=0)
        elif set_loss.lower() == 'masked_huber':
            lf = partial(loss.masked_huber_loss, delta=self.huber_delta, null_val=0)
        elif set_loss.lower() == 'r2':
            lf = loss.r2_score_torch
        elif set_loss.lower() == 'evar':
            lf = loss.explained_variance_score_torch
        else:
            lf = loss.masked_mae_torch
        return lf

    def calculate_loss_without_predict(self, X_unmask, y_true, y_mask, y_next, batches_seen=None, set_loss='masked_mae'):
        lf = self.get_loss_func(set_loss=set_loss)
        X_unmask = self._scaler.inverse_transform(X_unmask[..., :self.output_dim])
        y_mask = self._scaler.inverse_transform(y_mask[..., :self.output_dim])
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_next = self._scaler.inverse_transform(y_next[..., :self.output_dim])
        return self.mask_lambda * lf(y_mask, X_unmask) + self.next_lambda * lf(y_next, y_true)

    def predict(self, X_mask, X):
        return self.forward(X_mask, X)

    def get_data_feature(self):
        return self.data_feature
