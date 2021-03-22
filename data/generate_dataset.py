from addict import Dict
import yaml
import numpy as np
import pandas as pd
import warnings

from sympy.utilities.lambdify import lambdify
from sympy.abc import x

import os
from pathlib import Path
import pickle
import sys
import argparse
from tqdm.auto import tqdm

from sklearn.gaussian_process import GaussianProcessRegressor


def main(symbolic_mathematics_path, dataset_config_path):

    sys.path.append(symbolic_mathematics_path)
    from src.envs import build_env
#     dataset_dir = Path('./data/toy_example')
#     configs_dir = dataset_dir / 'configs'
    
    config_path = Path(dataset_config_path)
    dataset_dir = config_path.parent
    

    with open(config_path) as stream:
        cfg = yaml.safe_load(stream)
        cfg = Dict(cfg)

    meta_cfg = cfg.meta
    formulae_env_cfg = cfg.formulae_env
    formulae_other_cfg = cfg.formulae_other
    numeric_cfg = cfg.numeric
    gp_cfg = cfg.gp

    env = build_env(formulae_env_cfg)

    def generate_formula(min_ops, random_state):

        n_ops = np.random.randint(min_ops, env.max_ops)
        prefix_expr = env._generate_expr(n_ops, env.max_int, np.random.RandomState(random_state))
        infix_expr = env.prefix_to_infix(prefix_expr)
        sympy_expr = env.infix_to_sympy(infix_expr)

        formulae = {'prefix_expr': prefix_expr,
             'infix_expr': infix_expr,
             'sympy_expr': sympy_expr}
        
        return formulae

    def generate_numeric(formula, x_min, x_max, n_points_min, n_points_max, seed=0):
        np.random.seed(seed)
        sympy_expr = formula['sympy_expr']
        lambd_expr = lambdify(x, sympy_expr)
        xs = np.random.uniform(
            low = x_min,
            high = x_max,
            size = np.random.randint(n_points_min, n_points_max + 1)
        )
        ys = np.vectorize(lambd_expr)(xs)
        numeric = {'xs':xs,
             'ys': ys}

        return numeric

    def generate_gp(numeric, x_min, x_max, n_points, seed=0):
        xs_eval = np.linspace(x_min, x_max, n_points)

        gp = GaussianProcessRegressor(random_state=seed)
        xs = numeric['xs']
        ys = numeric['ys']
        gp.fit(np.expand_dims(xs, 1), ys)
        ys_eval = gp.predict(np.expand_dims(xs_eval, 1))
        gp = {'ys_eval': ys_eval}

        return gp

    for kind in ['train', 'val', 'test']:

        formulae, numerics, gps = [], [], []

        pbar = tqdm(total=meta_cfg.size[kind], desc='generate formulae')
        random_state = 0
        while len(formulae) !=  meta_cfg.size[kind]:
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    formula = generate_formula(formulae_other_cfg.min_ops, random_state)
                    numeric = generate_numeric(formula, numeric_cfg.x_min, numeric_cfg.x_max, numeric_cfg.n_points_min[kind], numeric_cfg.n_points_max[kind])
                    gp = generate_gp(numeric, gp_cfg.x_min, gp_cfg.x_max, gp_cfg.n_points)
                    assert len(w) == 0

                    formulae.append(formula)
                    numerics.append(numeric)
                    gps.append(gp)

                    pbar.update(1)

            except Exception:
                pass
            random_state += 1
        pbar.close()

        os.makedirs(dataset_dir / kind, exist_ok=True)
        d = {'formulae.pickle': formulae, 'numerics.pickle': numerics, 'gps.pickle': gps}
        for name, obj in d.items():
            with open(dataset_dir / kind / name, 'wb') as f:
                pickle.dump(obj, f)
                
        print('success')
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbolic_mathematics_path') # '../../../seq2seq_math/SymbolicMathematics/'
    parser.add_argument('--dataset_config_path')
    args = parser.parse_args()
    
    main(args.symbolic_mathematics_path, args.dataset_config_path)