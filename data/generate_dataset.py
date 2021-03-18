from addict import Dict
import yaml
import numpy as np
import pandas as pd

from sympy.utilities.lambdify import lambdify
from sympy.abc import x

import os
from pathlib import Path
import pickle
import sys
import argparse
from tqdm.auto import tqdm

from sklearn.gaussian_process import GaussianProcessRegressor


def main(symbolic_mathematics_path, dataset_configs_dir_path):

    sys.path.append(symbolic_mathematics_path)
    from src.envs import build_env
#     dataset_dir = Path('./data/toy_example')
#     configs_dir = dataset_dir / 'configs'
    
    configs_dir = Path(dataset_configs_dir_path)
    dataset_dir = configs_dir.parent
    

    def parse_cfg(path):
        with open(path) as stream:
            cfg = yaml.safe_load(stream)
            cfg = Dict(cfg)

        return cfg

    meta_cfg = parse_cfg(configs_dir / 'meta_cfg.yaml')
    formulae_env_cfg = parse_cfg(configs_dir / 'formulae_env_cfg.yaml')
    formulae_other_cfg = parse_cfg(configs_dir / 'formulae_other_cfg.yaml')
    numeric_cfg = parse_cfg(configs_dir / 'numeric_cfg.yaml')
    gp_cfg = parse_cfg(configs_dir / 'gp_cfg.yaml')

    env = build_env(formulae_env_cfg)

    def generate_formulae(n_formulae, min_ops):
        i = 0
        formulae = []
        
        pbar = tqdm(total=n_formulae, desc='generate formulae')
        while len(formulae) != n_formulae:
            try:
                n_ops = np.random.randint(min_ops, env.max_ops)
                prefix_expr = env._generate_expr(n_ops, env.max_int, np.random.RandomState(i))
                infix_expr = env.prefix_to_infix(prefix_expr)
                sympy_expr = env.infix_to_sympy(infix_expr)

                formulae.append(
                    {'prefix_expr': prefix_expr,
                     'infix_expr': infix_expr,
                     'sympy_expr': sympy_expr}
                )
                pbar.update(1)
            except Exception:
                pass
            i += 1
        pbar.close()
        
        return formulae

    def generate_numerics(formulae, x_min, x_max, n_points_min, n_points_max, seed=0):
        np.random.seed(seed)
        numerics = []
        for formula in tqdm(formulae, desc='generate numerics'):
            sympy_expr = formula['sympy_expr']
            lambd_expr = lambdify(x, sympy_expr)
            xs = np.random.uniform(
                low = x_min,
                high = x_max,
                size = np.random.randint(n_points_min, n_points_max + 1)
            )
            ys = np.vectorize(lambd_expr)(xs)
            numerics.append(
                {'xs':xs,
                 'ys': ys}
            )

        return numerics

    def generate_gps(numerics, x_min, x_max, n_points, seed=0):
        xs_eval = np.linspace(x_min, x_max, n_points)
        gps = []
        for numeric in tqdm(numerics, desc='generate GPs'):
            gp = GaussianProcessRegressor(random_state=seed)
            xs = numeric['xs']
            ys = numeric['ys']
            gp.fit(np.expand_dims(xs, 1), ys)
            ys_eval = gp.predict(np.expand_dims(xs_eval, 1))
            gps.append({'ys_eval': ys})

        return gps

    for kind in ['train', 'val', 'test']:
        formulae = generate_formulae(meta_cfg.size[kind], formulae_other_cfg.min_ops)
        numerics = generate_numerics(formulae, numeric_cfg.x_min, numeric_cfg.x_max, numeric_cfg.n_points_min[kind], numeric_cfg.n_points_max[kind])
        gps = generate_gps(numerics, gp_cfg.x_min, gp_cfg.x_max, gp_cfg.n_points)

        os.makedirs(dataset_dir / kind, exist_ok=True)
        d = {'formulae.pickle': formulae, 'numerics.pickle': numerics, 'gps.pickle': gps}
        for name, obj in d.items():
            with open(dataset_dir / kind / name, 'wb') as f:
                pickle.dump(obj, f)
                
        print('success')
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbolic_mathematics_path') # '../../../seq2seq_math/SymbolicMathematics/'
    parser.add_argument('--dataset_configs_dir_path')
    args = parser.parse_args()
    
    main(args.symbolic_mathematics_path, args.dataset_configs_dir_path)