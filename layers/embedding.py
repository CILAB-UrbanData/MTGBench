import torch, math
import torch.nn as nn

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