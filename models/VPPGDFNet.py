import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_inverted
from layers.SelfAttention_Family import FullAttention, AttentionLayer


class ComponentToken(nn.Module):
    def __init__(self, seq_len, pred_len, d_model, dropout=0.0):
        super().__init__()
        self.linear_forecast = nn.Linear(seq_len, pred_len)
        self.token_projection = nn.Linear(pred_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.linear_forecast.weight = nn.Parameter((1 / seq_len) * torch.ones([pred_len, seq_len]))

    def forward(self, component):
        component = component.squeeze(-1)
        forecast = self.linear_forecast(component)
        query = self.token_projection(forecast).unsqueeze(1)
        return self.dropout(query), forecast.unsqueeze(-1)


class CrossAttentionFusionLayer(nn.Module):
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

    def forward(self, query_token, exog_tokens, cross_mask=None, tau=None, delta=None):
        attn_out = self.cross_attention(
            query_token,
            exog_tokens,
            exog_tokens,
            attn_mask=cross_mask,
            tau=tau,
            delta=delta,
        )[0]
        x = self.norm1(query_token + self.dropout(attn_out))
        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class Model(nn.Module):
    SUPPORTED_ABLATIONS = {"full", "no_exog", "unified_exog"}

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.features = configs.features
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out
        self.use_norm = getattr(configs, "use_norm", 0)
        self.vpp_ablation = getattr(configs, "vpp_ablation", "full")
        if self.vpp_ablation not in self.SUPPORTED_ABLATIONS:
            raise ValueError(f"Unsupported VPPGDFNet ablation mode: {self.vpp_ablation}")
        self.decomposition = series_decomp(configs.moving_avg)

        self.trend_token = ComponentToken(
            seq_len=configs.seq_len,
            pred_len=configs.pred_len,
            d_model=configs.d_model,
            dropout=configs.dropout,
        )
        self.seasonal_token = ComponentToken(
            seq_len=configs.seq_len,
            pred_len=configs.pred_len,
            d_model=configs.d_model,
            dropout=configs.dropout,
        )
        self.ex_embedding = DataEmbedding_inverted(
            configs.seq_len,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )

        self.trend_fusion_layers = nn.ModuleList([
            CrossAttentionFusionLayer(
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
        self.seasonal_fusion_layers = nn.ModuleList([
            CrossAttentionFusionLayer(
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
        self.unified_query_projection = nn.Linear(configs.d_model * 2, configs.d_model)
        self.unified_fusion_layers = nn.ModuleList([
            CrossAttentionFusionLayer(
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

        self.trend_projection = nn.Linear(configs.d_model, configs.pred_len * configs.c_out)
        self.seasonal_projection = nn.Linear(configs.d_model, configs.pred_len * configs.c_out)
        self.unified_projection = nn.Linear(configs.d_model, configs.pred_len * configs.c_out)

    def decompose_target(self, x_enc):
        target_history = x_enc[:, :, -1:].contiguous()
        seasonal_component, trend_component = self.decomposition(target_history)
        return trend_component, seasonal_component

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        trend_component, seasonal_component = self.decompose_target(x_enc)
        trend_query, _ = self.trend_token(trend_component)
        seasonal_query, _ = self.seasonal_token(seasonal_component)

        if self.vpp_ablation == "unified_exog":
            exog_tokens = self.ex_embedding(x_enc[:, :, :-1], x_mark_enc)
            unified_query = self.unified_query_projection(
                torch.cat([trend_query.squeeze(1), seasonal_query.squeeze(1)], dim=-1)
            ).unsqueeze(1)
            for layer in self.unified_fusion_layers:
                unified_query = layer(unified_query, exog_tokens)
            dec_out = self.unified_projection(unified_query.squeeze(1)).view(-1, self.pred_len, self.c_out)
        else:
            if self.vpp_ablation != "no_exog":
                exog_tokens = self.ex_embedding(x_enc[:, :, :-1], x_mark_enc)
                for layer in self.trend_fusion_layers:
                    trend_query = layer(trend_query, exog_tokens)
                for layer in self.seasonal_fusion_layers:
                    seasonal_query = layer(seasonal_query, exog_tokens)

            trend_pred = self.trend_projection(trend_query.squeeze(1)).view(-1, self.pred_len, self.c_out)
            seasonal_pred = self.seasonal_projection(seasonal_query.squeeze(1)).view(-1, self.pred_len, self.c_out)
            dec_out = trend_pred + seasonal_pred

        if self.use_norm:
            if self.c_out == x_enc.shape[-1]:
                dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
                dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            else:
                dec_out = dec_out * stdev[:, 0, -self.c_out:].unsqueeze(1).repeat(1, self.pred_len, 1)
                dec_out = dec_out + means[:, 0, -self.c_out:].unsqueeze(1).repeat(1, self.pred_len, 1)

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None
