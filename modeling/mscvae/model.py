import torch
import torch.nn as nn
from utils import attribute_matrix


class ConvVAE(nn.Module):
    def __init__(self, dim, latent_dim=64):
        super().__init__()
        self.dim = dim
        # Encoder: (dim, dim) attribute matrix를 처리
        self.enc_conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.enc_conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.enc_conv3 = nn.Conv2d(64, 128, 3, padding=1)
        
        # Latent: dim x dim -> latent_dim
        self.fc_mu = nn.Linear(128 * dim * dim, latent_dim)
        self.fc_logvar = nn.Linear(128 * dim * dim, latent_dim)

        # Decoder
        self.dec_fc = nn.Linear(latent_dim, 128 * dim * dim)
        self.dec_conv1 = nn.ConvTranspose2d(128, 64, 3, padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(64, 32, 3, padding=1)
        self.dec_conv3 = nn.ConvTranspose2d(32, 1, 3, padding=1)

    def encode(self, x):
        # x: (batch, 1, dim, dim)
        h = nn.functional.relu(self.enc_conv1(x))
        h = nn.functional.relu(self.enc_conv2(h))
        h = nn.functional.relu(self.enc_conv3(h))
        h = h.flatten(start_dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.dec_fc(z)
        batch = z.size(0)
        h = h.view(batch, 128, self.dim, self.dim)
        h = nn.functional.relu(self.dec_conv1(h))
        h = nn.functional.relu(self.dec_conv2(h))
        return torch.sigmoid(self.dec_conv3(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim,
                              kernel_size, padding=padding)

    def forward(self, x, h_cur, c_cur):
        # x/h_cur/c_cur: (batch, channel, H, W)
        combined = torch.cat([x, h_cur], dim=1)
        conv_out = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_out, conv_out.size(1)//4, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class TemporalConvLSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim, dim_size):
        super().__init__()
        self.dim_size = dim_size
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.cell = ConvLSTMCell(in_dim, hidden_dim, kernel_size=3)
        # hidden_dim -> in_dim projection
        self.projection = nn.Conv2d(hidden_dim, in_dim, kernel_size=1)

    def forward(self, x_seq):
        # x_seq: (batch, seq, channel, H, W)
        b, seq, c, h, w = x_seq.shape
        device = x_seq.device
        # hidden state는 hidden_dim 채널로 초기화해야 함
        h_cur = torch.zeros(b, self.hidden_dim, h, w, device=device)
        c_cur = torch.zeros(b, self.hidden_dim, h, w, device=device)
        outputs = []
        for t in range(seq):
            h_cur, c_cur = self.cell(x_seq[:, t], h_cur, c_cur)
            # hidden_dim -> in_dim으로 projection
            h_proj = self.projection(h_cur)  # (batch, in_dim, h, w)
            outputs.append(h_proj)
        return torch.stack(outputs, dim=1)  # (batch, seq, in_dim, h, w)


class MSCVAE(nn.Module):
    def __init__(self, dim, latent_dim=64, lstm_hidden=64):
        super().__init__()
        self.dim = dim

        # ConvVAE: attribute matrix (dim x dim)를 처리
        self.convvae = ConvVAE(dim=dim, latent_dim=latent_dim)

        # Temporal ConvLSTM
        self.temporal = TemporalConvLSTM(in_dim=1, hidden_dim=lstm_hidden, dim_size=dim)

    def forward(self, x):
        """
        x: (batch, seq, dim)
        """
        # compute attribute matrices
        M = attribute_matrix(x)  # (batch, seq, dim, dim)
        # add channel dimension
        M = M.unsqueeze(2)  # (batch, seq, 1, dim, dim)

        # reconstruct each time attribute matrix
        recon_seq = []
        mus, logvars = [], []
        for t in range(M.size(1)):
            recon, mu, logvar = self.convvae(M[:, t])  # M[:, t]: (batch, 1, dim, dim)
            recon_seq.append(recon)
            mus.append(mu)
            logvars.append(logvar)
        recon_seq = torch.stack(recon_seq, dim=1)  # (batch, seq, 1, dim, dim)

        # Temporal modeling
        recon_temporal = self.temporal(recon_seq)  # (batch, seq, 1, dim, dim)

        return recon_temporal, mus, logvars
