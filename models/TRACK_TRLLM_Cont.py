import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.GAT import DGAT, GAT
from layers.embedding import TimeEmbedding
from layers.MLPwithDROPOUT import Mlp, DropPath

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

class ScaleOffset(nn.Module):
    def __init__(self, qk_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(qk_dim))
        self.bias = nn.Parameter(torch.zeros(qk_dim))

    def forward(self, x):
        return x * self.weight + self.bias

class Norm(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        variance = torch.mean(torch.square(x), dim=-1, keepdim=True)
        return x / torch.sqrt(variance + self.eps)

class GatedAttentionUnit(nn.Module):
    def __init__(self, d_model, vg_dim, qk_dim):
        super().__init__()
        self.qk_dim = qk_dim
        self.vg_dim = vg_dim
        self.i_dense = nn.Linear(d_model, 2 * vg_dim + qk_dim)
        self.o_dense = nn.Linear(vg_dim, d_model)
        self.q_scaleoffset = ScaleOffset(qk_dim)
        self.k_scaleoffest = ScaleOffset(qk_dim)
        self.drop = nn.Dropout(0.1)

    @staticmethod
    def apply_rotary(x, sinusoidal_pos=None):
        if sinusoidal_pos is None:
            return x
        sin, cos = sinusoidal_pos
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    def forward(self, x, padding_masks, sinusoidal_pos=None):
        x = self.i_dense(x)
        u, v, qk = torch.split(x * torch.sigmoid(x), [self.vg_dim, self.vg_dim, self.qk_dim], dim=-1)
        q, k = self.q_scaleoffset(qk), self.k_scaleoffest(qk)
        # q, k = self.apply_rotary(q, sinusoidal_pos), self.apply_rotary(k, sinusoidal_pos)
        a = torch.einsum("bmd,bnd->bmn", q, k)
        a = a / (self.qk_dim ** 0.5)
        if padding_masks is not None:
            a = a.masked_fill(padding_masks == 0, float('-inf'))
        A = torch.softmax(a, dim=-1)
        A = self.drop(A)
        o = self.o_dense(u * torch.einsum("bmn,bnd->bmd", A, v))
        return o

class GAULayer(nn.Module):
    def __init__(self, d_model, vg_dim, qk_dim):
        super().__init__()
        self.gau = GatedAttentionUnit(d_model, vg_dim, qk_dim)
        self.norm = Norm(eps=1e-6)
        self.drop = nn.Dropout(0.1)

    def forward(self, x, padding_masks, sinusoidal_pos=None):
        gau_output = self.gau(x, padding_masks, sinusoidal_pos)
        o = self.drop(gau_output)
        o = self.norm(x + o)
        return o

class TRL(nn.Module):

    def __init__(self, config, data_feature):
        super().__init__()

        d_model = config.get("d_model", 128)
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

        '''
        vg_dim = d_model * 2
        qk_dim = d_model // 2
        self.gau_block = nn.ModuleList([
            GAULayer(
                d_model, vg_dim, qk_dim,
            ) for i in range(n_layers)
        ])
        sinusoidal_id = self._get_sinusoidal_id(qk_dim)
        self.register_buffer("sin_pos", sinusoidal_id.sin(), persistent=False)
        self.register_buffer("cos_pos", sinusoidal_id.cos(), persistent=False)
        '''

    def _get_sinusoidal_id(self, qk_dim, max_length=128):
        position_ids = torch.arange(0, max_length, dtype=torch.float32)
        indices = torch.arange(0, qk_dim // 2, dtype=torch.float32)
        indices = torch.pow(10000.0, -2 * indices / qk_dim)
        sinusoidal_id = torch.einsum("n,d->nd", position_ids, indices)
        return sinusoidal_id[None, :, :]

    '''
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
        seq_len = embedding_output.shape[1]
        sinusoidal_pos = self.sin_pos[:, :seq_len, :], self.cos_pos[:, :seq_len, :]
        padding_masks_input = padding_masks.unsqueeze(1).repeat(1, seq_len, 1)  # (B * traj_batch_size, seq_len, seq_len)
        for gau in self.gau_block:
            embedding_output = gau.forward(
                x=embedding_output, padding_masks=padding_masks_input, sinusoidal_pos=sinusoidal_pos,
            )
        return embedding_output
    '''
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

class Model(nn.Module):
    '''TRLLM_Cont'''
    def __init__(self, config, data_feature):
        super().__init__()
        d_model = config.get('d_model', 64)
        gat_heads_per_layer = config.get('gat_heads_per_layer', [8, 16, 1])
        gat_features_per_layer = config.get('gat_features_per_layer', [16, 16, 64])
        gat_dropout = config.get('gat_dropout', 0.1)
        load_trans_prob = config.get('load_trans_prob', True)
        self.load_t_trans_prob = config.get('load_t_trans_prob', True)
        gat_avg_last = config.get('gat_avg_last', True)
        traj_drop = config.get('traj_drop', 0.1)
        self.pooling = config.get('pooling', 'cls')
        add_pe = config.get('add_pe', True)
        self.add_delta_time = config.get('add_delta_time', True)
        add_time_in_day = config.get('add_time_in_day', True)
        add_day_in_week = config.get('add_day_in_week', True)
        self.time_intervals = config.get('time_intervals', 1800)
        self.add_cls = config.get('add_cls', True)
        vocab_size = data_feature.get('vocab_size')
        node_fea_dim = data_feature.get('node_fea_dim')
        self.t_loc_trans_prob = data_feature.get('t_loc_trans_prob')
        self.node_features = data_feature.get('node_features')
        self.edge_index = data_feature.get('edge_index')
        self.loc_trans_prob = data_feature.get('loc_trans_prob')

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
        self.pooler = DRNTRLPooler(config)

    def forward(
        self, traj1, padding_masks1, traj2, padding_masks2,
        traj, padding_masks, batch_temporal_mat=None,
        output_hidden_states=False, output_attentions=False,
    ):
        if self.pooling in ['avg_first_last', 'avg_top2']:
            output_hidden_states = True
        if self.load_t_trans_prob:
            if self.add_cls:
                time_ind = (traj2[:, 1, 3] - 1) * (24 * 3600 // self.time_intervals) + (traj2[:, 1, 2] - 1) // (self.time_intervals // 60)
            else:
                time_ind = (traj2[:, 0, 3] - 1) * (24 * 3600 // self.time_intervals) + (traj2[:, 0, 2] - 1) // (self.time_intervals // 60)
            t_edge_prob_input = self.t_loc_trans_prob[time_ind]
            traj_rn_emb = self.s_emb(
                node_features=self.node_features, edge_index_input=self.edge_index,
                edge_prob_input=self.loc_trans_prob, t_edge_prob_input=t_edge_prob_input,
            )  # (B, vocab_size, d_model)
            d_model = traj_rn_emb.shape[-1]
            traj_loc = traj[:, :, 0:1].expand(-1, -1, d_model)
            traj_embs = traj_rn_emb.gather(1, traj_loc)
            traj_loc1 = traj1[:, :, 0:1].expand(-1, -1, d_model)
            traj_embs1 = traj_rn_emb.gather(1, traj_loc1)
            traj_loc2 = traj2[:, :, 0:1].expand(-1, -1, d_model)
            traj_embs2 = traj_rn_emb.gather(1, traj_loc2)
        else:
            rn_emb = self.s_emb(
                node_features=self.node_features, edge_index_input=self.edge_index,
                edge_prob_input=self.loc_trans_prob,
            )  # (vocab_size, d_model)
            traj_embs = rn_emb[traj[..., 0]]
            traj_embs1 = rn_emb[traj1[..., 0]]
            traj_embs2 = rn_emb[traj2[..., 0]]

        traj_embs = self.t_emb(traj_embs, traj, position_ids=None)
        out_masked_traj, hidden_states, _ = self.trl(
        # out_masked_traj = self.trl(
            traj_embs, padding_masks, batch_temporal_mat=batch_temporal_mat,
            output_hidden_states=output_hidden_states, output_attentions=output_attentions,
        )  # (B, seq_len, d_model)

        traj_embs1 = self.t_emb(traj_embs1, traj1, position_ids=None)
        out_traj1, hidden_states1, _ = self.trl(
        # out_traj1 = self.trl(
            traj_embs1, padding_masks1, batch_temporal_mat=None,
            output_hidden_states=output_hidden_states, output_attentions=output_attentions,
        )  # (B, seq_len, d_model)

        traj_embs2 = self.t_emb(traj_embs2, traj2, position_ids=None)
        out_traj2, hidden_states2, _ = self.trl(
        # out_traj2 = self.trl(
            traj_embs2, padding_masks2, batch_temporal_mat=None,
            output_hidden_states=output_hidden_states, output_attentions=output_attentions,
        )  # (B, seq_len, d_model)

        pred_masked_traj = self.mask_l(out_masked_traj)
        pred_masked_time = None
        if self.add_delta_time:
            pred_masked_time = self.mask_t(out_masked_traj).squeeze(-1)
        pool_out_traj1 = self.pooler(out_traj1, padding_masks1, hidden_states=hidden_states1)
        pool_out_traj2 = self.pooler(out_traj2, padding_masks2, hidden_states=hidden_states2)
        # pool_out_traj1 = self.pooler(out_traj1, padding_masks1, hidden_states=None)
        # pool_out_traj2 = self.pooler(out_traj2, padding_masks2, hidden_states=None)
        return pred_masked_traj, pred_masked_time, pool_out_traj1, pool_out_traj2
