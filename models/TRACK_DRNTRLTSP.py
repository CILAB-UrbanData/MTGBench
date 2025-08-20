import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.MLPwithDROPOUT import Mlp, DropPath
from layers.GAT import GAT, DGAT, DRNRLGAT, TrafSparseGATLayer, CoAttention

''' DRNRL对应原文框架图的左半边，即Dynamic Road Network Representation Learning，
    为所有节点的每个时刻的交通状态编码出隐变量
    TRL对应的是原文框架图的右半边，即Trajectory Representation Learning，
    为每条轨迹编码出隐变量'''

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

class DRNRL(nn.Module):
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
            self.drn_attn = DRNRL.DRNAttention(
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

    def  __init__(self, args, data_feature):
        super().__init__()
        drop_path = getattr(args, "drop_path", 0.3)
        enc_depth = getattr(args, "enc_depth", 6)
        embed_dim = getattr(args, 'embed_dim', 64)
        s_num_heads = getattr(args, 's_num_heads', 2)
        t_num_heads = getattr(args, 't_num_heads', 6)
        mlp_ratio = getattr(args, "mlp_ratio", 4)
        qkv_bias = getattr(args, "qkv_bias", True)
        drop = getattr(args, "traf_drop", 0.)
        attn_drop = getattr(args, "traf_attn_drop", 0.)
        device = getattr(args, 'device', torch.device('cpu'))
        type_ln = getattr(args, "type_ln", "post")
        output_dim = getattr(args, 'output_dim', 1)
        d_model = getattr(args, "d_model", 64)
        self.traf_g_edge_index = data_feature.get("traf_g_edge_index")
        self.traf_g_edge_prob = data_feature.get("traf_g_loc_trans_prob")

        enc_dpr = [x.item() for x in torch.linspace(0, drop_path, enc_depth)]
        self.encoder_blocks = nn.ModuleList([
            DRNRL.DRNEncoderBlock(
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

class Model(nn.Module):
    '''DRN + TRL 两个上游任务  TSP这一下游任务(Traffic State Prediction)'''
    def __init__(self, args, data_feature):
        super().__init__()
        embed_dim = getattr(args, 'embed_dim', 64)
        lape_dim = getattr(args, 'lape_dim', 8)
        traf_drop = getattr(args, 'traf_drop', 0.)
        add_time_in_day = getattr(args, 'add_time_in_day', True)
        add_day_in_week = getattr(args, 'add_day_in_week', True)
        time_intervals = getattr(args, 'time_intervals', 1800)
        input_window = getattr(args, "input_window", 6)
        self.add_aft = getattr(args, 'add_aft', True)
        d_model = getattr(args, 'd_model', 64)
        gat_heads_per_layer = getattr(args, 'gat_heads_per_layer', [8, 16, 1])
        gat_features_per_layer = getattr(args, 'gat_features_per_layer', [16, 16, 64])
        gat_dropout = getattr(args, 'gat_dropout', 0.1)
        load_trans_prob = getattr(args, 'load_trans_prob', True)
        self.load_t_trans_prob = getattr(args, 'load_t_trans_prob', True)
        gat_avg_last = getattr(args, 'gat_avg_last', True)
        output_window = getattr(args, 'output_window', 6)
        coattn_heads = getattr(args, 'coattn_heads', 8)
        co_n_layers = getattr(args, 'co_n_layers', 1)
        coattn_drop = getattr(args, 'coattn_drop', 0)
        co_dropout = getattr(args, 'co_dropout', 0)
        co_drop_path = getattr(args, 'co_drop_path', 0)
        mlp_ratio = getattr(args, 'mlp_ratio', 4)
        type_ln = getattr(args, 'type_ln', 'post')
        device = getattr(args, 'device', torch.device('cpu'))
        self.feature_dim = data_feature.get("feature_dim", 1)
        self.ext_dim = data_feature.get("ext_dim", 0)
        output_dim = data_feature.get("output_dim", 1)
        node_fea_dim = data_feature.get('node_fea_dim')
        self.co_traf_edge_index = data_feature.get('co_traf_edge_index')
        self.co_traj_edge_index = data_feature.get('co_traj_edge_index')
        traf_gat_heads_per_layer = getattr(args, 'traf_gat_heads_per_layer', [8, 16, 1])
        traf_gat_features_per_layer = getattr(args, 'traf_gat_features_per_layer', [16, 16, 64])
        self.node_features = data_feature.get("node_features")
        self.traf_edge_index = data_feature.get("traf_edge_index")
        self.traf_edge_prob = data_feature.get("traf_edge_prob")
        num_nodes = data_feature.get('num_nodes')

        self.enc_embed_layer = DataEmbedding(
            self.feature_dim - self.ext_dim, embed_dim, num_nodes, node_fea_dim, drop=traf_drop,
            gat_heads_per_layer=traf_gat_heads_per_layer, gat_features_per_layer=traf_gat_features_per_layer,
            add_time_in_day=add_time_in_day, add_day_in_week=add_day_in_week,
        )
        self.bef_drnrl = DRNRL(args, data_feature)
        self.traf_emb_conv = nn.Conv2d(
            in_channels=input_window, out_channels=1, kernel_size=1,
        )
        if self.add_aft:
            self.inv_traf_emb_conv = nn.Conv2d(in_channels=1, out_channels=input_window, kernel_size=1)
            self.aft_embedding = nn.Linear(d_model, embed_dim)
            self.aft_drnrl = DRNRL(args, data_feature)
            self.end_traf_emb_conv = nn.Conv2d(in_channels=input_window, out_channels=1, kernel_size=1)

        self.minute_size = 24 * 3600 // time_intervals
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

        self.end_conv1 = nn.Conv2d(in_channels=1, out_channels=output_window, kernel_size=1)
        self.end_conv2 = nn.Linear(d_model, output_dim)

        self.cotransformer_blocks = nn.ModuleList([
            CoTransformerBlock(
                d_model, coattn_heads, d_model * mlp_ratio, co_drop_path, coattn_drop, co_dropout,
                type_ln=type_ln, device=device,
            ) for _ in range(co_n_layers)
        ])

    def predict(self, traf, out_lap_mx, in_lap_mx, graph_dict=None):
        '''这一方法直接输出道路网络节点的嵌入式表示，forward方法则直接输出道路网络的交通预测'''
        traf_enc = self.enc_embed_layer(traf, out_lap_mx, in_lap_mx)  # (B, T, N, embed_dim)
        traf_emb = self.bef_drnrl(traf_enc)  # (B, T, N, d_model)
        traf_rn_emb = self.traf_emb_conv(traf_emb).squeeze(1)  # (B, N, d_model)

        batch_size = traf.shape[0]
        if self.load_t_trans_prob:
            time_ind = traf[:, -1, 0, self.feature_dim - self.ext_dim + 1] * self.minute_size + traf[:, -1, 0, self.feature_dim - self.ext_dim] * self.minute_size + 1
            time_ind = time_ind.long() % (7 * self.minute_size)
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
            traf_rn_emb = self.end_traf_emb_conv(traf_emb)  # (B, 1, N, d_model)
        return traf_rn_emb.squeeze(1)

    def forward(self, traf, graph_dict=None):
        traf_enc = self.enc_embed_layer(traf, self.node_features[5:, :], self.traf_edge_index, self.traf_edge_prob)  # (B, T, N, embed_dim)
        traf_emb = self.bef_drnrl(traf_enc)  # (B, T, N, d_model)
        traf_rn_emb = self.traf_emb_conv(traf_emb).squeeze(1)  # (B, N, d_model)

        batch_size = traf.shape[0]
        if self.load_t_trans_prob:
            time_ind = traf[:, -1, 0, self.feature_dim - self.ext_dim + 1] * self.minute_size + traf[:, -1, 0, self.feature_dim - self.ext_dim] * self.minute_size + 1
            time_ind = time_ind.long() % (7 * self.minute_size)
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
            traf_rn_emb = self.end_traf_emb_conv(traf_emb)  # (B, 1, N, d_model)

        pred_traf = self.end_conv1(traf_rn_emb)
        pred_traf = self.end_conv2(pred_traf)

        return pred_traf