from vec2seq.models.mymodel import Model
from vec2seq.utils.data import Vec2SeqDataset
import torch.utils.data
import pytorch_lightning as pl
import numpy as np
from argparse import ArgumentParser
import pandas as pd
from pathlib import Path


def main(checkpoint_path):
    model = Model.load_from_checkpoint(checkpoint_path)
    cfg = model.cfg

    train_gp_csv_path = f'../data/{cfg.task}/train_gp.txt'
    train_formulae_csv_path = f'../data/{cfg.task}/train_formulae.txt'
    train_dataset = Vec2SeqDataset(train_gp_csv_path, train_formulae_csv_path)

    test_gp_csv_path = f'../data/{cfg.task}/test_gp.txt'
    test_formulae_csv_path = f'../data/{cfg.task}/test_formulae.txt'
    test_dataset = Vec2SeqDataset(test_gp_csv_path, test_formulae_csv_path)
    test_dataset.set_field_from_different_dataset(train_dataset)

    def collate_fn(data):
        batch_size = len(data)
        return np.stack([data[i][0] for i in range(batch_size)]), np.stack([data[i][1] for i in range(batch_size)])


    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=cfg.data.batch_size,
        collate_fn=collate_fn,
        num_workers=cfg.data.num_workers
    )

    if cfg.trainer.run_on_gpus:
        gpus = -1
        distributed_backend = 'ddp'
    else:
        gpus = None
        distributed_backend = None
    trainer = pl.Trainer(gpus=gpus,
                         distributed_backend=distributed_backend,
                         logger=False)

    trainer.test(model, test_dataloader)

    df = pd.DataFrame(zip(model.test_labels, model.test_preds), columns=['labels', 'preds'])
    print(df)
    for col in ['labels', 'preds']:
        df[col] = df[col].apply(lambda v: ' '.join(test_dataset.vec_to_formula(v, drop_tech_symbols=False)))
    print(df)

    output_name = str(Path(checkpoint_path).with_suffix('').name) + '_output.csv'
    output_path = Path(checkpoint_path).parent.parent / output_name
    df.to_csv(output_path, index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--checkpoint_path')
    args = parser.parse_args()
    main(args.checkpoint_path)
