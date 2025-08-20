import torch
import torch.nn as nn

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

class DRNRLGAT(nn.Module):

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
    
