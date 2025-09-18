import torch
import torch.nn as nn
from TRACK_DRNTRLLM_Cont import DataEmbedding, DGAT, DRNRL, GAT, CoTransformerBlock


class Model(nn.Module):

    def __init__(self, config, data_feature):
        super().__init__()
        embed_dim = config.get('embed_dim', 64)
        lape_dim = config.get('lape_dim', 8)
        traf_drop = config.get('traf_drop', 0.)
        add_time_in_day = config.get('add_time_in_day', True)
        add_day_in_week = config.get('add_day_in_week', True)
        time_intervals = config.get('time_intervals', 1800)
        input_window = config.get("input_window", 6)
        self.add_aft = config.get('add_aft', True)
        d_model = config.get('d_model', 64)
        gat_heads_per_layer = config.get('gat_heads_per_layer', [8, 16, 1])
        gat_features_per_layer = config.get('gat_features_per_layer', [16, 16, 64])
        gat_dropout =config.get('gat_dropout', 0.1)
        load_trans_prob = config.get('load_trans_prob', True)
        self.load_t_trans_prob = config.get('load_t_trans_prob', True)
        gat_avg_last = config.get('gat_avg_last', True)
        output_window = config.get('output_window', 6)
        coattn_heads = config.get('coattn_heads', 8)
        co_n_layers = config.get('co_n_layers', 1)
        coattn_drop = config.get('coattn_drop', 0)
        co_dropout = config.get('co_dropout', 0)
        co_drop_path = config.get('co_drop_path', 0)
        mlp_ratio = config.get('mlp_ratio', 4)
        type_ln = config.get('type_ln', 'post')
        device = config.get('device', torch.device('cpu'))
        self.feature_dim = data_feature.get("feature_dim", 1)
        self.ext_dim = data_feature.get("ext_dim", 0)
        output_dim = data_feature.get("output_dim", 1)
        node_fea_dim = data_feature.get('node_fea_dim')
        self.co_traf_edge_index = data_feature.get('co_traf_edge_index')
        self.co_traj_edge_index = data_feature.get('co_traj_edge_index')
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

    '''
    def predict(self, traf, out_lap_mx, in_lap_mx, graph_dict=None):
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
    '''
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
