# autoencoder to encode mnist image data

import torch
from torch import nn

class Autoencoder(nn.Module):
    def __init__(self, layers=3, input_dim=28*28, target_dim=1):
        super(Autoencoder, self).__init__()
        self.layers = layers
        self.target_dim = target_dim
        self.input_dim = input_dim
        
        # create weights
        self.weights = []
        for i in range(self.layers):
            if i == 0:
                self.weights.append(nn.Linear(self.input_dim, 128))
            elif i == self.layers-1:
                self.weights.append(nn.Linear(128, self.target_dim))
            else:
                self.weights.append(nn.Linear(128, 128))

        # create activations
        self.activations = []
        for i in range(self.layers):
            if i == self.layers-1:
                self.activations.append(nn.Sigmoid())
            else:
                self.activations.append(nn.ReLU())

        # create encoder
        self.encoder = nn.Sequential()
        for i in range(self.layers):
            self.encoder.add_module("encoder_layer_"+str(i), self.weights[i])
            self.encoder.add_module("encoder_activation_"+str(i), self.activations[i])

        # create decoder
        self.decoder = nn.Sequential()
        for i in range(self.layers-1, -1, -1):
            self.decoder.add_module("decoder_layer_"+str(i), self.weights[i])
            self.decoder.add_module("decoder_activation_"+str(i), self.activations[i])

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def encode(self, x):
        x = self.encoder(x)
        return x
    
    def decode(self, x):
        x = self.decoder(x)
        return x
        
        