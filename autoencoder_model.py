'''
Description: This script provides the main autoencoder model structure that is used for GFS forecast products bias correction task.
    
Author: Sadegh Sadeghi Tabas (sadegh.tabas@noaa.gov)
Revision history:
    -20231029: Sadegh Tabas, initial code
'''

import torch
import torch.nn as nn

class CompactAutoencoder(nn.Module):
    def __init__(self):
        super(CompactAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(True),
            nn.Conv3d(8, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(True),
        )

        # Bottleneck (no further reduction of dimensions)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(True),
            nn.ConvTranspose3d(8, 1, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def get_autoencoder(device):
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        autoencoder = nn.DataParallel(CompactAutoencoder()).to(device)
    else:
        autoencoder = CompactAutoencoder().to(device)
    return autoencoder
        
