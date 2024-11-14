
import torch
import torch.nn as nn
from modules import FeatureAttentionLayer, TemporalAttentionLayer

import numpy as np


class CustomModel(nn.Module):
    def __init__(self, window, input_size):
        super(CustomModel, self).__init__()
        self.window = window
        self.input_size = input_size
        
        self.model = nn.Sequential(
            nn.Linear(window * input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_size),
        )

    def forward(self, x):
        # Reshape input from (batch_size, window, input_size) to (batch_size, window * input_size)
        x = x.view(x.size(0), -1)
        return self.model(x)



class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, num_layers=1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True) #, dropout=0.1)
        
        # Layers for mean and log variance
        self.hidden_to_mean = nn.Linear(hidden_size, latent_size)
        self.hidden_to_log_var = nn.Linear(hidden_size, latent_size)
        
    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        _, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Use the final hidden state to compute mean and log variance
        hn = hn[-1]  # Take the last layer's hidden state
        mean = self.hidden_to_mean(hn)
        log_var = self.hidden_to_log_var(hn)
        
        return mean, log_var

class Decoder(nn.Module):
    def __init__(self, latent_size, hidden_size, output_size, num_layers=1):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.latent_to_hidden = nn.Linear(latent_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers,batch_first=True)#, dropout=0.1)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, z):
        # Transform the latent vector into the initial hidden state
        hidden = self.latent_to_hidden(z).unsqueeze(0).repeat(self.num_layers, 1, 1)
        cell = torch.zeros(self.num_layers, z.size(0), self.hidden_size).to(z.device)
        
        # Initialize decoder input with zeros
        decoder_input = torch.zeros(z.size(0), 1, self.hidden_size).to(z.device)
        
        # Decode the latent vector to produce the output sequence
        out, _ = self.lstm(decoder_input, (hidden, cell))
        
        # Pass the last time step through a fully connected layer
        out = self.fc(out[:, -1, :])  # (batch_size, output_size)
        
        return out

class LSTM_VAE(nn.Module):
    def __init__(self, window_size, input_size, hidden_size, latent_size, num_layers=1):
        super(LSTM_VAE, self).__init__()
        
        self.window_size = window_size
        self.input_size = input_size
        self.encoder = Encoder(input_size*3, hidden_size, latent_size, num_layers)
        self.decoder = Decoder(latent_size, hidden_size, input_size, num_layers)
        self.feature_gat = FeatureAttentionLayer(input_size, window_size, dropout=0.2, alpha=0.2, embed_dim=None, use_gatv2=True)
        self.temporal_gat = TemporalAttentionLayer(input_size, window_size, dropout=0.2, alpha=0.2, embed_dim=None, use_gatv2=True)
        

        
    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        # x shape: (batch_size, window_size, input_size)
        batch_size, window_size, input_size = x.shape 
                
        h_feat = self.feature_gat(x)
        h_temp = self.temporal_gat(x)
        h_cat = torch.cat([x, h_feat, h_temp], dim=2)
        
        # Encode
        mean, log_var = self.encoder(h_cat)
 
        # Reparameterize
        z = self.reparameterize(mean, log_var)
        # Decode
        out = self.decoder(z)
        
        return out, mean, log_var