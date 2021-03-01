import pandas as pd
import numpy as np
import torch.utils.data
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from addict import Dict


class Encoder(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_p):
        super().__init__()
        self.input_dim, self.hidden_dim, self.output_dim = input_dim, hidden_dim, output_dim
        self.net = nn.Sequential(
            # nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(dropout_p),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(dropout_p),

            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(dropout_p),

            nn.Linear(hidden_dim // 4, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(negative_slope=0.1),
        )

    def forward(self, x):

        return self.net(x)


class Decoder(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout_p):
        super().__init__()

        self.net = nn.GRU(input_dim, hidden_dim, n_layers, dropout=dropout_p)

    def forward(self, x, h):

        return self.net(x, h)


class Model(pl.LightningModule):
    def __init__(self, cfg, abc_len, decoder_output_len):
        super().__init__()
        self.cfg = cfg

        # assert encoder_output_dim == decoder_input_dim
        # assert decoder_input_dim == decoder_hidden_dim

        self.encoder_input_dim = cfg.model.encoder_input_dim
        self.encoder_hidden_dim = cfg.model.encoder_hidden_dim
        self.encoder_output_dim = abc_len
        self.encoder_dropout_p = cfg.model.encoder_dropout_p

        self.decoder_input_dim = cfg.model.decoder_input_dim
        decoder_hidden_dim = cfg.model.decoder_input_dim # TODO
        self.decoder_hidden_dim = decoder_hidden_dim
        self.decoder_n_layers = cfg.model.decoder_n_layers
        self.decoder_output_len = decoder_output_len
        self.decoder_dropout_p = cfg.model.decoder_dropout_p

        self.lr = cfg.optimization.lr

        self.encoder = Encoder(self.encoder_input_dim, self.encoder_hidden_dim, self.encoder_output_dim,
                               self.encoder_dropout_p)
        self.decoder = Decoder(self.decoder_input_dim, self.decoder_hidden_dim, self.decoder_n_layers,
                               self.decoder_dropout_p)

        self.nn1 = nn.Sequential(
            nn.Linear(self.encoder_output_dim, self.decoder_input_dim),
            nn.Tanh()
        )

        self.nn2 = nn.Sequential(
            nn.Linear(self.decoder_hidden_dim, self.encoder_output_dim),
            nn.LeakyReLU()
        )

        self.test_labels = []
        self.test_preds = []

        self.save_hyperparameters()

    @staticmethod
    def one_hot_embedding(labels, num_classes):
        y = torch.eye(num_classes, device=labels.device)

        return y[labels]

    def on_after_backward(self) -> None:
        valid_gradients = True
        for param in self.parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break

        if not valid_gradients:
            print('detected inf or nan values in gradients. not updating model parameters')
            self.zero_grad()

    def forward(self, x, y=None):
        encoded_vector = self.encoder(x)
        # encoded_vector = encoded_vector.view(1, x.shape[0], self.decoder_input_dim)

        bs = x.shape[0]

        x = encoded_vector
        hidden_vector = None

        outputs = []
        for i in range(self.decoder_output_len):

            x = x.view(bs, self.encoder_output_dim)
            x = self.nn1(x)
            x = x.view(1, bs, self.decoder_input_dim)

            x, hidden_vector = self.decoder(x, hidden_vector)

            x = x.view(bs, self.decoder_input_dim)
            x = self.nn2(x)
            x = x.view(1, bs, self.encoder_output_dim)

            outputs.append(x)

            if (y is not None) and (np.random.binomial(1, 0.5) == 1):
                x = self.one_hot_embedding(y[:, i], self.encoder_output_dim)
                x = x.unsqueeze(0).detach()
            else:
                x = x.squeeze(0).argmax(1)
                x = self.one_hot_embedding(x, self.encoder_output_dim)
                x = x.unsqueeze(0).detach()

        outputs = torch.stack(outputs)
        outputs = outputs.squeeze(1)
        outputs = torch.movedim(outputs, [0, 1, 2], [1, 0, 2])

        return outputs

    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y = torch.from_numpy(x).to(self.device), torch.from_numpy(y).to(self.device)
        pred = self(x, y=y)

        loss = F.cross_entropy(
            pred.reshape(-1, self.encoder_output_dim),
            y.reshape(-1),
        )

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x, y = torch.from_numpy(x).to(self.device), torch.from_numpy(y).to(self.device)
        pred = self(x, y=None)

        loss = F.cross_entropy(
            pred.reshape(-1, self.encoder_output_dim),
            y.reshape(-1),
        )
        acc = (pred.reshape(-1, self.encoder_output_dim).argmax(1) == y.reshape(-1)).float().mean()

        return {'val_loss': loss, 'val_acc': acc}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('avg_train_loss', avg_loss)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        self.log('avg_val_loss', avg_loss)
        self.log('avg_val_acc', avg_acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        x, y = torch.from_numpy(x).to(self.device), torch.from_numpy(y).to(self.device)
        pred = self(x, y=None)

        self.test_labels = self.test_labels + y.tolist()
        self.test_preds = self.test_preds + pred.argmax(2).tolist()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return optimizer
