from pathlib import Path
import numpy as np

import torch
import torch.functional as F

class RSmilesUspto50(torch.utils.data.Dataset):
    def __init__(self, tokeniser, data_path, split='train', forward=False, randomize_padding=None, max_seq_len=256):
        self.path = Path(data_path)
        reactants, products = self.read_data_dir(self.path, split)

        if len(reactants) != len(products):
            raise ValueError(f"There must be an equal number of reactants and products")

        self.tokeniser = tokeniser
        self.reactants = reactants
        self.products = products
        self.forward = forward
        self.randomize_padding = randomize_padding
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.reactants)

    def __getitem__(self, item):
        reactant = self.reactants[item]
        product = self.products[item]
        output = self.transform(reactant, product)
        return output

    def transform(self, product, reactant):
        react_str, prod_str = reactant.replace(' ', ''), product.replace(' ', '')
        
        if self.forward:
            encoder_smiles, decoder_smiles = react_str, prod_str
        else:
            encoder_smiles, decoder_smiles = prod_str, react_str

        if self.randomize_padding is not None:
            if self.pad_limit > 0: 
                decoder_smiles = tuple(smi + self.tokeniser.unk_token * np.random.randint(1, self.pad_limit) for smi in decoder_smiles)
            elif self.pad_limit == 0:
                decoder_smiles = decoder_smiles
            elif self.pad_limit == -1:
                decoder_smiles = tuple(smi + self.tokeniser.unk_token * self.max_seq_len for smi in decoder_smiles)

        encoder_input, encoder_mask = self.apply_tokeniser(encoder_smiles)
        decoder_input, decoder_mask = self.apply_tokeniser(decoder_smiles)

        return {
            'y_0': encoder_input,
            'y_mask': encoder_mask,
            'x_0': decoder_input,
            'x_mask': decoder_mask,
            'encoder_smiles': encoder_smiles,
            'decoder_smiles': decoder_smiles
        }
    
    def apply_tokeniser(self, smiles):
        tokeniser_output = self.tokeniser.tokenise(smiles, mask=False, pad=True)
        input_tokens = tokeniser_output["original_tokens"]
        input_mask = tokeniser_output["original_pad_masks"]

        input_token_ids = self.tokeniser.convert_tokens_to_ids(input_tokens)
        input_token_ids = torch.tensor(input_token_ids)
        input_token_ids = F.one_hot(input_token_ids, len(self.tokeniser))

        input_pad_mask = torch.tensor(input_mask, dtype=torch.bool)

        input_token_ids = input_token_ids[:, :self.max_seq_len]
        input_pad_mask = input_pad_mask[:, :self.max_seq_len]

        return input_token_ids, input_pad_mask

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
