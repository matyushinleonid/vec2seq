import pandas as pd
import numpy as np
import torch.utils.data
import torch
import pickle


class Vec2SeqDataset(torch.utils.data.Dataset):
    def __init__(self, gps_path, formulae_path):
        self.tech_symbols = ['<init>', '<pad>', '<eos>']

        with open(gps_path, 'rb') as f:
            gp_csv = pd.DataFrame(pickle.load(f))['ys_eval']
            self.gp = gp_csv.values.tolist()
        with open(formulae_path, 'rb') as f:
            formulae_csv = pd.DataFrame(pickle.load(f))['prefix_expr']
            self.formulae = formulae_csv.values.tolist()

        self.abc = np.unique(np.concatenate(self.formulae)).tolist()
        self.abc += self.tech_symbols
        self.abc_len = len(self.abc)
        print(f'abc :{self.abc}')

        self.s_to_n = {s: n for n, s in enumerate(self.abc)}
        self.n_to_s = {n: s for s, n in self.s_to_n.items()}

        self.max_formula_len = np.max([len(f) for f in self.formulae])
        self.max_formula_plus_tech_symbols_len = self.max_formula_len + 2
        print(f'max formula len = {self.max_formula_len}')
        print(f'max formula plus tech symbols len = {self.max_formula_plus_tech_symbols_len}')

    def set_field_from_different_dataset(self, dataset):
        # if set(self.abc) != set(dataset.abc):
        #     raise Exception
        # if self.max_formula_plus_tech_symbols_len != dataset.max_formula_plus_tech_symbols_len:
        #     raise Exception

        self.max_formula_len = dataset.max_formula_len
        self.max_formula_plus_tech_symbols_len = dataset.max_formula_plus_tech_symbols_len
        self.s_to_n = dataset.s_to_n
        self.n_to_s = dataset.n_to_s

    def formulae_to_vec(self, formulae):
        vec = [self.s_to_n['<init>']] + \
              [self.s_to_n[s] for s in formulae] + \
              [self.s_to_n['<eos>']]

        vec = vec + [self.s_to_n['<pad>']] * (self.max_formula_plus_tech_symbols_len - len(vec))

        return vec

    def vec_to_formula(self, vec, drop_tech_symbols=False):
        formula = [self.n_to_s[n] for n in vec]
        if drop_tech_symbols:
            formula = [s for s in formula if s not in self.tech_symbols]

        return formula

    def __len__(self):

        return len(self.formulae)

    def __getitem__(self, i):
        gp = self.gp[i]
        formula = self.formulae_to_vec(self.formulae[i])

        return np.asarray(gp, dtype=np.float32), np.asarray(formula, dtype=np.long)
