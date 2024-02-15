from omegaconf import OmegaConf
from typing import Dict

from collections.abc import MutableMapping

def flatten_dictionary(dictionary, parent_key='', separator='_'):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten_dictionary(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)

def flatten_omegaconf(cfg, parent_key='', sep='.') -> Dict[str, any]:
    """
    Flatten an OmegaConf configuration object into a flat dictionary.

    Parameters:
    - cfg: The OmegaConf configuration object to flatten.
    - parent_key: The base key to use for the current level of the hierarchy.
    - sep: The separator to use between nested keys.

    Returns:
    - A flat dictionary representation of the OmegaConf configuration.
    """
    items = []
    for k, v in cfg.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if OmegaConf.is_config(v):
            items.extend(flatten_omegaconf(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

if __name__=="__main__":
    # Example usage
    cfg = OmegaConf.create({
        'database': {
            'host': 'localhost',
            'port': 3306
        },
        'logging': {
            'level': 'INFO',
            'file': 'app.log'
        }
    })

    flat_cfg = flatten_omegaconf(cfg)
    print(flat_cfg)