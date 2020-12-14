from vec2seq.models.mymodel import Model
from vec2seq.utils.data import Vec2SeqDataset
import torch.utils.data
import pytorch_lightning as pl
import numpy as np
from argparse import ArgumentParser
import pandas as pd


def main(checkpoint_path):
    print(checkpoint_path)
    train_gp_csv_path = '../data/train_gp.txt'
    train_formulae_csv_path = '../data/train_formulae.txt'
    train_dataset = Vec2SeqDataset(train_gp_csv_path, train_formulae_csv_path)

    test_gp_csv_path = '../data/test_gp.txt'
    test_formulae_csv_path = '../data/test_formulae.txt'
    test_dataset = Vec2SeqDataset(test_gp_csv_path, test_formulae_csv_path)
    test_dataset.set_field_from_different_dataset(train_dataset)

    def collate_fn(data):
        batch_size = len(data)
        return np.stack([data[i][0] for i in range(batch_size)]), np.stack([data[i][1] for i in range(batch_size)])


    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=1000,
        collate_fn=collate_fn,
        num_workers=2
    )

    model = Model.load_from_checkpoint(
                  checkpoint_path,
                  encoder_input_dim=200,
                  encoder_hidden_dim=1024,
                  encoder_output_dim=train_dataset.abc_len,
                  decoder_input_dim=train_dataset.abc_len,
                  decoder_hidden_dim=train_dataset.abc_len,
                  decoder_n_layers=5,
                  decoder_output_len=train_dataset.max_formula_plus_tech_symbols_len,
                  lr=1e-4)

    trainer = pl.Trainer(gpus=[3],
                         logger=False)

    trainer.test(model, test_dataloader)

    df = pd.DataFrame(zip(model.test_labels, model.test_preds), columns=['labels', 'preds'])
    print(df)
    for col in ['labels', 'preds']:
        df[col] = df[col].apply(lambda v: ' '.join(test_dataset.vec_to_formula(v, drop_tech_symbols=False)))
    print(df)
    df.to_csv('output.csv', index=False)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--checkpoint_path')
    args = parser.parse_args()
    main(args.checkpoint_path)