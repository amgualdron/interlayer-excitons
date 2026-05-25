import jax 
import jax.numpy as jnp
import numpy as np
import h5py
import yaml
from pathlib import Path
import os

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
MATERIALS_DIR = PROJECT_ROOT / 'materials'
RUN_CONFIG = PROJECT_ROOT / 'config.yaml'

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_run(file_name, potential, basis_states, bands, config_name, params):
    with h5py.File(file_name, 'a') as f:

        if config_name in f:
            del f[config_name]
        grp = f.create_group(config_name)
        
        for key, value in params.items():
            grp.attrs[key] = value
        grp.create_dataset('potential', data=np.array(potential), compression='gzip')
        grp.create_dataset('basis_states', data=np.array(basis_states), compression='gzip')
        grp.create_dataset('bands', data=np.array(bands), compression='gzip') 

    print(f"Saved '{config_name}' to {file_name}!")

print("project directory" , PROJECT_ROOT)
print("materials directory" , MATERIALS_DIR)
print("run config" , RUN_CONFIG)