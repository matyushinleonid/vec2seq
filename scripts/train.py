from vec2seq.models.mymodel import Model
from vec2seq.utils.data import Vec2SeqDataset
import torch.utils.data
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


def main():
    train_gp_csv_path = '../data/train_gp.txt'
    train_formulae_csv_path = '../data/train_formulae.txt'
    train_dataset = Vec2SeqDataset(train_gp_csv_path, train_formulae_csv_path)
    val_gp_csv_path = '../data/train_gp.txt'
    val_formulae_csv_path = '../data/val_formulae.txt'
    val_dataset = Vec2SeqDataset(val_gp_csv_path, val_formulae_csv_path)
    val_dataset.set_field_from_different_dataset(train_dataset)

    def collate_fn(data):
        batch_size = len(data)
        return np.stack([data[i][0] for i in range(batch_size)]), np.stack([data[i][1] for i in range(batch_size)])

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=1000,
        collate_fn=collate_fn,
        num_workers=2
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=1000,
        collate_fn=collate_fn,
        num_workers=2
    )

    model = Model(encoder_input_dim=200,
                  encoder_hidden_dim=1024,
                  encoder_output_dim=train_dataset.abc_len,
                  decoder_input_dim=train_dataset.abc_len,
                  decoder_hidden_dim=train_dataset.abc_len,
                  decoder_n_layers=5,
                  decoder_output_len=train_dataset.max_formula_plus_tech_symbols_len,
                  lr=1e-4)

    early_stop_callback = EarlyStopping(
        monitor='avg_val_loss',
        min_delta=0.0,
        patience=30,
        verbose=True,
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='avg_val_loss',
        mode='min',
        save_top_k=1,
    )

    tb_logger = pl.loggers.TensorBoardLogger('logs/test')

    trainer = pl.Trainer(gpus=[3],
                         min_epochs=30,
                         max_epochs=10000,
                         callbacks=[early_stop_callback, checkpoint_callback],
                         logger=tb_logger,
                         check_val_every_n_epoch=100)

    trainer.fit(
        model,
        train_dataloader,
        val_dataloader
    )


if __name__ == '__main__':
    main()