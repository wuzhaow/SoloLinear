import torch
import torch.nn as nn
from layers.RevIN import RevIN


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.scale = 0.02
        self.revin_layer = RevIN(configs.enc_in, affine=True, subtract_last=False)

        self.embed_size = self.seq_len
        self.hidden_size = configs.d_model

        self.w = nn.Parameter(self.scale * torch.randn(1, self.embed_size))

        self.fc = nn.Sequential(
            nn.Linear(self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pred_len)
        )
        self.last_head = nn.Linear(configs.enc_in, configs.c_out)

    def circular_convolution(self, x, w):
        x = torch.fft.rfft(x, dim=2, norm='ortho')
        w = torch.fft.rfft(w, dim=1, norm='ortho')
        y = x * w
        out = torch.fft.irfft(y, n=self.embed_size, dim=2, norm="ortho")
        return out

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        z = x
        z = self.revin_layer(z, 'norm')
        x = z

        x = x.permute(0, 2, 1)

        x = self.circular_convolution(x, self.w.to(x.device))  # B, N, D

        x = self.fc(x)
        x = x.permute(0, 2, 1)

        z = x
        z = self.revin_layer(z, 'denorm')
        x = z
        if self.task_name == 'm_ms_forecast':
            return x
        else:
            x = self.last_head(x)

        return x