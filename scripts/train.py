from vec2seq.models.mymodel import Model
from vec2seq.utils.data import Vec2SeqDataset
import torch.utils.data
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from fire import Fire
from addict import Dict
import yaml


def main(cfg):
    train_gp_csv_path = f'../data/{cfg.task}/train_gp.txt'
    train_formulae_csv_path = f'../data/{cfg.task}/train_formulae.txt'
    train_dataset = Vec2SeqDataset(train_gp_csv_path, train_formulae_csv_path)
    val_gp_csv_path = f'../data/{cfg.task}/val_gp.txt'
    val_formulae_csv_path = f'../data/{cfg.task}/val_formulae.txt'
    val_dataset = Vec2SeqDataset(val_gp_csv_path, val_formulae_csv_path)
    val_dataset.set_field_from_different_dataset(train_dataset)

    def collate_fn(data):
        batch_size = len(data)
        return np.stack([data[i][0] for i in range(batch_size)]), np.stack([data[i][1] for i in range(batch_size)])

    train_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset([train_dataset]), # * 100000
        shuffle=True,
        batch_size=cfg.data.batch_size, # * 3
        collate_fn=collate_fn,
        num_workers=cfg.data.num_workers
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=cfg.data.batch_size,
        collate_fn=collate_fn,
        num_workers=cfg.data.num_workers
    )

    # decoder_hidden_dim = cfg.model.decoder_input_dim
    # model = Model(encoder_input_dim=cfg.model.encoder_input_dim,
    #               encoder_hidden_dim=cfg.model.encoder_hidden_dim,
    #               encoder_output_dim=train_dataset.abc_len,
    #               decoder_input_dim=cfg.model.decoder_input_dim,#train_dataset.abc_len,
    #               decoder_hidden_dim=decoder_hidden_dim,#train_dataset.abc_len,
    #               decoder_n_layers=cfg.model.decoder_n_layers,
    #               decoder_output_len=train_dataset.max_formula_plus_tech_symbols_len,
    #               lr=cfg.optimization.lr,
    #               encoder_dropout_p=cfg.model.encoder_dropout_p,
    #               decoder_dropout_p=cfg.model.decoder_dropout_p)

    model = Model(cfg=cfg,
                  abc_len=train_dataset.abc_len,
                  decoder_output_len=train_dataset.max_formula_plus_tech_symbols_len)


    early_stop_callback = EarlyStopping(
        monitor='avg_val_acc',
        min_delta=cfg.early_stopping.min_delta,
        patience=cfg.early_stopping.patience,
        verbose=True,
        mode='max',
        strict=False
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='avg_val_acc',
        mode='max',
        save_top_k=cfg.checkpointing.save_top_k,
    )

    tb_logger = pl.loggers.TensorBoardLogger(f'logs/{cfg.task}')


    if cfg.trainer.run_on_gpus:
        gpus = -1
        distributed_backend = 'ddp'
    else:
        gpus = None
        distributed_backend = None
    trainer = pl.Trainer(gpus=gpus,
                         distributed_backend=distributed_backend,
                         callbacks=[early_stop_callback, checkpoint_callback],
                         logger=tb_logger,
                         log_gpu_memory='all',
                         min_epochs=cfg.trainer.min_epochs,
                         max_epochs=cfg.trainer.max_epochs)

    trainer.fit(
        model,
        train_dataloader,
        val_dataloader
    )

def get_cfg(**kwargs):
    with open(kwargs['config'], 'r') as stream:
        cfg = yaml.safe_load(stream)
    kwargs.pop('config')

    return cfg

if __name__ == '__main__':
    cfg = Dict(Fire(get_cfg))
    main(cfg)
