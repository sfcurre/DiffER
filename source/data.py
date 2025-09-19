from pathlib import Path
import numpy as np

import torch
import torch.nn.functional as F

import selfies as sf

class RSmilesUspto50(torch.utils.data.Dataset):
    def __init__(self, tokeniser, data_path, split='train', forward=False, pad_limit=None, max_seq_len=256, selfies=False):
        self.path = Path(data_path)
        reactants, products = self.read_data_dir(self.path, split)

        if len(reactants) != len(products):
            raise ValueError(f"There must be an equal number of reactants and products")

        self.tokeniser = tokeniser
        self.reactants = reactants
        self.products = products
        self.forward = forward
        self.pad_limit = pad_limit
        self.max_seq_len = max_seq_len
        self.selfies = selfies

    def __len__(self):
        return len(self.reactants)

    def __getitem__(self, item):
        reactant = self.reactants[item]
        product = self.products[item]
        output = self.transform(reactant, product)
        return output

    def transform(self, reactant, product):
        react_str, prod_str = reactant.replace(' ', ''), product.replace(' ', '')
        
        if self.selfies:
            react_str = sf.encoder(react_str)
            prod_str = sf.encoder(prod_str)
        
        if self.forward:
            encoder_smiles, decoder_smiles = react_str, prod_str
        else:
            encoder_smiles, decoder_smiles = prod_str, react_str

        if self.pad_limit is not None:
            if self.pad_limit > 0: 
                decoder_smiles = decoder_smiles + self.tokeniser.unk_token * np.random.randint(1, self.pad_limit)
            elif self.pad_limit == 0:
                decoder_smiles = decoder_smiles
            elif self.pad_limit == -1:
                decoder_smiles = decoder_smiles + self.tokeniser.unk_token * self.max_seq_len

        return encoder_smiles, decoder_smiles
    
    def apply_tokeniser(self, smiles):
        tokeniser_output = self.tokeniser.tokenise(smiles, mask=False, pad=True)
        input_tokens = tokeniser_output["original_tokens"]
        input_pad_mask = np.array(tokeniser_output["original_pad_masks"])

        input_token_ids = np.array(self.tokeniser.convert_tokens_to_ids(input_tokens))
        one_hots = np.eye(len(self.tokeniser))
        input_one_hots = one_hots[input_token_ids]

        input_one_hots = torch.tensor(input_one_hots[:, :self.max_seq_len], dtype=torch.float)
        input_pad_mask = torch.tensor(input_pad_mask[:, :self.max_seq_len], dtype=torch.bool)

        return input_one_hots, input_pad_mask

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

    def collate_fn(self, batch):
        encode_strs, decode_strs = zip(*batch)
        y_0, y_mask = self.apply_tokeniser(encode_strs)
        x_0, x_mask = self.apply_tokeniser(decode_strs)

        return {
            'y_0': y_0,
            'y_mask': y_mask,
            'x_0': x_0,
            'x_mask': x_mask,
            'encoder_smiles': encode_strs,
            'decoder_smiles': decode_strs,
            'target_smiles': [s.rstrip(self.tokeniser.unk_token) for s in decode_strs]
        }