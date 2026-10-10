"""Small convolutional auto-encoder for bathymetric relief patches (PyTorch). Used only offline, never in live playback."""
import torch
from torch import nn


class PatchAutoEncoder(nn.Module):
    def __init__(self, size=32, latent=16):
        super().__init__()
        self.size, self.latent = size, latent
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.ReLU(), nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, 1), nn.ReLU(), nn.Flatten(), nn.Linear(32 * (size // 8) ** 2, latent))
        self.dec_fc = nn.Linear(latent, 32 * (size // 8) ** 2)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.ReLU(), nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, 2, 1))

    def encode(self, x):
        return self.enc(x)

    def decode(self, z):
        h = self.dec_fc(z).view(-1, 32, self.size // 8, self.size // 8)
        return self.dec(h)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z
