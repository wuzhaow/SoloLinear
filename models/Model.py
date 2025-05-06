import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, DataEmbedding_wo_pos
import numpy as np
from einops import rearrange
from torch.nn.functional import gumbel_softmax
from layers.RevIN import RevIN_P,RevIN


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x
class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    """
    Multiple Series decomposition block from FEDformer
    """

    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.kernel_size = kernel_size
        self.series_decomp = [series_decomp(kernel) for kernel in kernel_size]

    def forward(self, x):
        moving_mean = []
        res = []
        for func in self.series_decomp:
            sea, moving_avg = func(x)
            moving_mean.append(moving_avg)
            res.append(sea)

        sea = sum(res) / len(res)
        moving_mean = sum(moving_mean) / len(moving_mean)
        return sea, moving_mean

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.prediction_layer = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)
        self.head = nn.Linear(n_vars, 1)
        #self.weight = 1
    def forward(self, x):  # x: [bs x nvars x d_model]
        x = self.dropout(self.prediction_layer(x.permute(0, 2, 1)).permute(0, 2, 1))  # x: [bs x pred_len x d_model]
        x = self.head(x)
        return x

class Flatten(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.prediction_layer = nn.Linear(nf, target_window)
        self.res_layer = nn.Linear(n_vars, 1)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model]
        x = self.res_layer(self.prediction_layer(x.permute(0, 2, 1)).permute(0, 2, 1))
        return x

class TransposeLayer(nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.cycle_len = configs.cycle_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        self.use_inorm = configs.use_inorm
        self.data = configs.data
        self.k = configs.k
        self.init_type = configs.init_type
        self.mean = configs.mean
        self.std = configs.std
        # Embedding
        # self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
        #                                             configs.dropout)
        self.demo = series_decomp(configs.moving_avg)
        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'without_point'or self.task_name == 'm_ms_forecast' :
            self.head_nf = configs.d_model * configs.seq_len
            self.head = nn.Sequential(
                TransposeLayer(dim0=1, dim1=2),
                nn.Linear(self.seq_len, self.pred_len),
                TransposeLayer(dim0=1, dim1=2),
            )
        #if self.task_name == 'without_point'or self.task_name == 'm_ms_forecast' :
        # self.FLinear1 = nn.Linear(configs.enc_in, configs.enc_in).to(torch.cfloat)
        # self.FLinear2 = nn.Linear(configs.enc_in, configs.enc_in).to(torch.cfloat)
        # self.Linear1 = nn.Linear(configs.cycle_len * configs.enc_in, configs.cycle_len* configs.enc_in)

        # else:
        self.FLinear1 = nn.Linear(configs.enc_in, 1).to(torch.cfloat)
        self.FLinear2 = nn.Linear(configs.enc_in, 1).to(torch.cfloat)
        self.Linear1 = nn.Linear(configs.cycle_len * configs.enc_in, configs.cycle_len)

        self.channel_independence = False
        if self.channel_independence:
            self.Linear2 = nn.ModuleList([
            nn.Linear(self.seq_len//configs.cycle_len,self.seq_len//configs.cycle_len)
            for _ in range(self.cycle_len)
        ])
        else:
            self.Linear2 = nn.Linear(self.seq_len//configs.cycle_len, self.seq_len//configs.cycle_len)

       # self.linear_mean = nn.Linear(configs.enc_in, 1)
        #self.linear_std = nn.Linear(configs.enc_in, 1)
        #self.revin = RevIN_P(configs.enc_in)
        #self.A = nn.Parameter(torch.randn(1, 1, configs.enc_in),requires_grad=True)
        init_type = self.init_type  # 可选：'zero'、'normal'、'uniform

        self.patch = False
        if self.patch:
            self.step = int(self.cycle_len * 0.5)

            self.num_patch = int((self.seq_len - self.cycle_len)//self.step + 1)
            self.Linear2 = nn.Linear(self.num_patch,
                                     self.seq_len // configs.cycle_len)
        else:
            pass
            #self.Linear2 = nn.Linear(self.seq_len//configs.cycle_len, self.seq_len//configs.cycle_len)

        self.init_linear_weights(init_type)




    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, batch_point_y):
        x_enc_pre = x_enc
        # batchsize, Seq, d = x_enc_pre.shape
        # list23 = x_enc[:, [23,47,71,95],:].mean(dim=1,keepdim=True) # B 1 D
        # dist = list23 * self.A
        # dist = torch.abs(dist - batch_point_y) # B 1 D
        #
        #
        # exp_dist = 1 / (dist + 1e-10)
        # exp_max, _ = torch.max(exp_dist, dim=-1, keepdim=True)
        # exp_max = exp_max
        # #
        # p = exp_dist / exp_max * 0.99
        # index = torch.topk(p, int(d * 0.95), dim=2)
        # resample_matrix = self.bernoulli_gumbel_rsample(p) # B S D * B 1 D
        # resample_matrix.scatter_(2, index.indices, 1)
        if self.use_norm:
            #x_enc = self.revin(x_enc, 'norm')
            means = x_enc.mean(1, keepdim=True).detach() # B 1 D
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5) # B 1 D
            x_enc /= stdev
            #input = torch.complex(means,stdev)
            last_means = means.mean(2, keepdim=True).detach()
            last_std = stdev.mean(2,keepdim=True).detach()
        B, S, N = x_enc.shape

        # Embedding
        # if self.data == 'battery':
        # 	enc_out = self.enc_embedding(x_enc, None)
        # else:
        # 	enc_out = self.enc_embedding(x_enc, x_mark_enc)
        season, trend = self.demo(x_enc)

        trend_out = torch.fft.rfft(trend, dim=1, norm ="ortho")  # b s/2 + 1 d

        k_values = torch.topk(trend_out.abs(), self.k, dim=1)
        indices = k_values.indices

        mask = torch.zeros_like(trend_out)
        mask.scatter_(1, indices, 1)

        trend_hi = trend_out * mask
        trend_low = trend_out - trend_hi

        trend_hi = self.FLinear1(trend_hi)
        trend_low = self.FLinear2(trend_low)
        trend_out =  trend_hi + trend_low
        temp =trend_out
        trend_out = torch.fft.irfft(trend_out, dim=1)
        if self.patch:
            season = season.unfold(dimension=1, size=self.cycle_len, step=self.step)
            season = season.reshape(B, self.num_patch, self.cycle_len * N)

        else:
            season = season.reshape(B, S//self.cycle_len, self.cycle_len, N)
            season = season.reshape(B, S//self.cycle_len, self.cycle_len*N)
        season = self.Linear1(season) # B S//24 24 C -> B S//24 24 1
        season = season.permute(0, 2, 1) # B 24 S//24

        season_out = []
        if self.channel_independence:
            for i, linear in enumerate(self.Linear2):
                channel_output = linear(season[:,i,:])  # 形状: (B, out_features)
                season_out.append(channel_output)
            season_out = torch.stack(season_out, dim=1).permute(0, 2, 1)
        else:
            season_out = self.Linear2(season).permute(0, 2, 1)

        season_out = season_out.reshape(B, S, -1)

        dec_out = trend_out + season_out
        # self.check_layer_params(self.FLinear1, "FLinear1")
        # self.check_layer_params(self.FLinear2, "FLinear2")
        dec_out = self.head(dec_out)
        if self.use_inorm:
            # means = self.revin.mean.mean(2, keepdim=True).detach()
            # stdev = self.revin.stdev.mean(2, keepdim=True).detach()
            dec_out = (dec_out * last_std) + last_means

        # if self.use_norm:
        #     # De-Normalization from Non-stationary Transformer
        #     dec_out = dec_out * (stdev[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))
        #     dec_out = dec_out + (means[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out,temp

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, batch_point_y,mask=None):
        if self.task_name == 'long_term_forecast':
            dec_out,temp = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec,batch_point_y)
        elif self.task_name == 'without_point' or self.task_name == 'm_ms_forecast':
            dec_out = self.forecast_withoutp(x_enc, x_mark_enc, x_dec, x_mark_dec,batch_point_y)
        return dec_out[:, -self.pred_len:, :],temp # [B, L, D]


    def forecast_withoutp(self, x_enc, x_mark_enc, x_dec, x_mark_dec, batch_point_y):
        if self.use_norm:
            #x_enc = self.revin(x_enc, 'norm')
            means = x_enc.mean(1, keepdim=True).detach() # B 1 D
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5) # B 1 D
            x_enc /= stdev
            #input = torch.complex(means,stdev)
            last_means = means.mean(2, keepdim=True).detach()
            last_std = stdev.mean(2,keepdim=True).detach()
        B, S, N = x_enc.shape
        season, trend = self.demo(x_enc)

        trend_out = torch.fft.rfft(trend, dim=1)  # b s/2 + 1 d

        k_values = torch.topk(trend_out.abs(), self.k, dim=1)
        indices = k_values.indices

        mask = torch.zeros_like(trend_out)
        mask.scatter_(1, indices, 1)

        trend_hi = trend_out * mask
        trend_low = trend_out - trend_hi

        trend_hi = self.FLinear1(trend_hi)
        trend_low = self.FLinear2(trend_low)
        trend_out = trend_hi + trend_low
        trend_out = torch.fft.irfft(trend_out, dim=1)

        season = season.reshape(B, S // self.cycle_len, self.cycle_len, N)
        season = season.reshape(B, S // self.cycle_len, self.cycle_len * N)
        season = self.Linear1(season).reshape(B, S // self.cycle_len, self.cycle_len,  N)
        season = season.permute(0, 2, 1, 3).reshape(B, self.cycle_len, S // self.cycle_len * N)  # B 24 S//24
        season_out = self.Linear2(season).reshape(B, self.cycle_len, S // self.cycle_len,  N).permute(0, 2, 1, 3)
        season_out = season_out.reshape(B, S, -1)

        dec_out = trend_out + season_out

        dec_out = self.head(dec_out)
        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def init_linear_weights(self, init_type='zero'):
        """统一初始化所有线性层权重
        Args:
            init_type: 'zero'（全0）, 'normal_0.1'（均值0标准差0.1的正态分布）,
                     'uniform_0.1'（均匀分布-0.1~0.1）
        """
        for m in self.modules():  # 递归遍历所有子模块
            if isinstance(m, nn.Linear):
                # 初始化权重
                if init_type == 'zero':
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)
                elif init_type == 'normal':
                    nn.init.normal_(m.weight,mean=self.mean,std=self.std)
                   # nn.init.normal_(m.bias, mean=self.mean, std=self.std)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

                elif init_type == 'uniform':
                    nn.init.uniform_(m.weight, a=-0.01, b=0.01)
                    nn.init.zeros_(m.bias)
                else:
                    pass
    def bernoulli_gumbel_rsample(self, distribution_matrix):
        b, c, d = distribution_matrix.shape

        flatten_matrix = rearrange(distribution_matrix, 'b c d -> (b c d) 1')
        r_flatten_matrix = 1 - flatten_matrix

        log_flatten_matrix = torch.log(flatten_matrix / r_flatten_matrix)
        log_r_flatten_matrix = torch.log(r_flatten_matrix / flatten_matrix)

        new_matrix = torch.concat([log_flatten_matrix, log_r_flatten_matrix], dim=-1)
        resample_matrix = gumbel_softmax(new_matrix, hard=True)

        resample_matrix = rearrange(resample_matrix[..., 0], '(b c d) -> b c d', b=b, c=c, d=d)
        return resample_matrix
    def compute_masked_statistics(self, x_enc, resample_matrix):
        # 假设 resample_matrix 是二值矩阵，1表示保留，0表示屏蔽
        # x_enc 形状为 [B, S, D]
        # resample_matrix 形状为 [B, S, D]

        # 计算每个特征维度上有效的(未被屏蔽的)样本数量
        valid_counts = torch.sum(resample_matrix, dim=2, keepdim=True)  # [B, 1, D]
        #print(valid_counts[1,:,:])
        # 防止除零
        valid_counts = torch.clamp(valid_counts, min=1.0)

        # 计算屏蔽后的均值 (只考虑未屏蔽的值)
        masked_sum = torch.sum(x_enc * resample_matrix, dim=2, keepdim=True)  # [B, 1, D]
        masked_mean = masked_sum / valid_counts  # [B, 1, D]

        # 计算屏蔽后的方差
        squared_diff = (x_enc - masked_mean) ** 2
        masked_squared_diff = squared_diff * resample_matrix  # 只考虑未屏蔽的值
        masked_var = torch.sum(masked_squared_diff, dim=2, keepdim=True) / valid_counts  # [B, 1, D]

        # 计算屏蔽后的标准差
        masked_std = torch.sqrt(masked_var)  # 添加小常数防止数值不稳定

        # 如果你需要计算所有批次的最终均值和标准差
        # last_masked_mean = masked_mean.mean(2, keepdim=True)  # [B, 1, 1]
        # last_masked_std = masked_std.mean(2, keepdim=True)  # [B, 1, 1]

        return masked_mean, masked_std#, last_masked_mean, last_masked_std




