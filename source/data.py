from pathlib import Path
import numpy as np
import math

import selfies as sf

import torch
import torch.nn as nn
import torch.nn.functional as F

from pysmilesutils.augment import SMILESAugmenter

class RSmilesUspto50(torch.utils.data.Dataset):
    def __init__(self, data_path, split='train', aug_prob=0.0, forward=True, use_canonical=False, use_selfies=False):
        self.path = Path(data_path)
        reactants, products = self.read_data_dir(self.path, 'train')

        if len(reactants) != len(products):
            raise ValueError(f"There must be an equal number of reactants and products")

        self.reactants = reactants
        self.products = products
        self.forward = forward
        self.use_canonical = use_canonical
        self.use_selfies = use_selfies
        self.aug_prob = aug_prob
        self.aug = SMILESAugmenter()

    def __len__(self):
        return len(self.reactants)

    def __getitem__(self, item):
        reactant = self.reactants[item]
        product = self.products[item]
        output = (reactant, product)
        output = self.transform(*output) if self.transform is not None else output
        return output

    def transform(self, react, prod, type_token=None):
        react_str, prod_str = react.replace(' ', ''), prod.replace(' ', '')
        react_str = self.aug(react_str)[0]
        prod_str = self.aug(prod_str)[0]

        if self.use_selfies:
            react_str = sf.encoder(react_str)
            prod_str = sf.encoder(prod_str)

        if self.forward:
            react_str = f"{str(type_token)}{react_str}" if type_token else react_str
        else:
            prod_str = f"{str(type_token)}{prod_str}" if type_token else prod_str

        return react_str, prod_str

    def read_data_dir(self, path, split, subsample_interval=None):
        product_path = path / split / f'src-{split}.txt'
        reactant_path = path / split / f'tgt-{split}.txt'

        with open(product_path) as fp:
            products = list(map(str.strip, fp.readlines()))

        with open(reactant_path) as fp:
            reactants = list(map(str.strip, fp.readlines()))

        if subsample_interval is not None:
            idxs = np.arange(0, len(reactants), subsample_interval)
            subsample_idxs = np.random.randint(0, subsample_interval, size=len(idxs))
            idxs = idxs + subsample_idxs
            reactants = [reactants[i] for i in idxs]
            products = [products[i] for i in idxs]

        return reactants, products
