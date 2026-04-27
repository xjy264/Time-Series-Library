import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_inverted
from layers.SelfAttention_Family import FullAttention, AttentionLayer


class DLinearGlobalToken(nn.Module):
    def __init__(self, seq_len, pred_len, d_model, moving_avg, dropout=0.0):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.decomposition = series_decomp(moving_avg)
        self.linear_seasonal = nn.Linear(seq_len, pred_len)
        self.linear_trend = nn.Linear(seq_len, pred_len)
        self.token_projection = nn.Linear(pred_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.linear_seasonal.weight = nn.Parameter((1 / seq_len) * torch.ones([pred_len, seq_len]))
        self.linear_trend.weight = nn.Parameter((1 / seq_len) * torch.ones([pred_len, seq_len]))

    def forward(self, target_history):
        seasonal_init, trend_init = self.decomposition(target_history)
        seasonal_init = seasonal_init.squeeze(-1)
        trend_init = trend_init.squeeze(-1)
        dlinear_forecast = self.linear_seasonal(seasonal_init) + self.linear_trend(trend_init)
        global_token = self.token_projection(dlinear_forecast).unsqueeze(1)
        return self.dropout(global_token)


class FusionLayer(nn.Module):
    def __init__(self, cross_attention, d_model, d_ff=None, dropout=0.1, activation="gelu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, global_token, variable_tokens, cross_mask=None, tau=None, delta=None):
        attn_out = self.cross_attention(
            global_token,
            variable_tokens,
            variable_tokens,
            attn_mask=cross_mask,
            tau=tau,
            delta=delta,
        )[0]
        x = self.norm1(global_token + self.dropout(attn_out))
        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.features = configs.features
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = getattr(configs, "use_norm", 0)

        self.global_token_generator = DLinearGlobalToken(
            seq_len=configs.seq_len,
            pred_len=configs.pred_len,
            d_model=configs.d_model,
            moving_avg=configs.moving_avg,
            dropout=configs.dropout,
        )
        self.ex_embedding = DataEmbedding_inverted(
            configs.seq_len,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )
        self.fusion_layers = nn.ModuleList([
            FusionLayer(
                AttentionLayer(
                    FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                    configs.d_model,
                    configs.n_heads,
                ),
                configs.d_model,
                configs.d_ff,
                dropout=configs.dropout,
                activation=configs.activation,
            )
            for _ in range(configs.e_layers)
        ])
        self.output_projection = nn.Linear(configs.d_model, configs.pred_len)

    def dlinear_global_token(self, x_enc):
        target_history = x_enc[:, :, -1:].contiguous()
        return self.global_token_generator(target_history)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        global_token = self.dlinear_global_token(x_enc)
        variable_tokens = self.ex_embedding(x_enc, x_mark_enc)

        for layer in self.fusion_layers:
            global_token = layer(global_token, variable_tokens)

        dec_out = self.output_projection(global_token.squeeze(1)).unsqueeze(-1)

        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1)

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None
