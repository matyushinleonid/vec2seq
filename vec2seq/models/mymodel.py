import pandas as pd
import numpy as np
import torch.utils.data
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from addict import Dict


class Encoder(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_dim, self.hidden_dim, self.output_dim = input_dim, hidden_dim, output_dim
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):

        return self.net(x)


class Decoder(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, n_layers):
        super().__init__()

        self.net = nn.GRU(input_dim, hidden_dim, n_layers)

    def forward(self, x, h):

        return self.net(x, h)


class Model(pl.LightningModule):
    def __init__(self,
                 encoder_input_dim,
                 encoder_hidden_dim,
                 encoder_output_dim,
                 decoder_input_dim,
                 decoder_hidden_dim,
                 decoder_n_layers,
                 decoder_output_len,
                 lr):
        super().__init__()
        assert encoder_output_dim == decoder_input_dim
        assert decoder_input_dim == decoder_hidden_dim

        self.encoder_input_dim = encoder_input_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_output_dim = encoder_output_dim

        self.decoder_input_dim = decoder_input_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.decoder_n_layers = decoder_n_layers
        self.decoder_output_len = decoder_output_len

        self.lr = lr

        self.encoder = Encoder(encoder_input_dim, encoder_hidden_dim, encoder_output_dim)
        self.decoder = Decoder(decoder_input_dim, decoder_hidden_dim, decoder_n_layers)

        self.test_labels = []
        self.test_preds = []

    @staticmethod
    def one_hot_embedding(labels, num_classes):
        y = torch.eye(num_classes, device=labels.device)

        return y[labels]

    def forward(self, x, y=None):
        encoded_vector = self.encoder(x)
        encoded_vector = encoded_vector.view(1, x.shape[0], self.decoder_input_dim)

        x = encoded_vector
        hidden_vector = None

        outputs = []
        for i in range(self.decoder_output_len):
            x, hidden_vector = self.decoder(x, hidden_vector)
            outputs.append(x)

            if y is not None:
                x = self.one_hot_embedding(y[:, i], self.decoder_hidden_dim)
                x = x.unsqueeze(0)

        outputs = torch.stack(outputs)
        outputs = outputs.squeeze(1)
        outputs = torch.movedim(outputs, [0, 1, 2], [1, 0, 2])

        return outputs

    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y = torch.from_numpy(x).to(self.device), torch.from_numpy(y).to(self.device)
        pred = self(x, y=y)

        loss = F.cross_entropy(
            pred.reshape(-1, self.decoder_hidden_dim),
            y.reshape(-1),
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x, y = torch.from_numpy(x).to(self.device), torch.from_numpy(y).to(self.device)
        pred = self(x, y=None)

        loss = F.cross_entropy(
            pred.reshape(-1, self.decoder_hidden_dim),
            y.reshape(-1),
        )
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('avg_val_loss', avg_loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        x, y = torch.from_numpy(x).to(self.device), torch.from_numpy(y).to(self.device)
        pred = self(x, y=None)

        self.test_labels = self.test_labels + y.tolist()
        self.test_preds = self.test_preds + pred.argmax(2).tolist()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return optimizer
