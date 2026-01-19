import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import sys, json, zlib, os, random, pickle

from collections import defaultdict, Counter

from rdkit.Chem import rdMolDescriptors, AllChem, Draw, rdFMCS
from rdkit import Chem, RDLogger, DataStructs

RDLogger.DisableLog("rdApp.*")

import json

def canonicalize(smi, iso=False):
    smi = smi.replace('?', '')
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    else:
        return Chem.MolToSmiles(m, isomericSmiles=iso)
    
def compute_rank(prediction, alpha=1.0):
    valid_score = [[k for k in range(len(prediction[j]))] for j in range(len(prediction))]
    valid_rates = [0 for k in range(len(prediction[0]))]
    rank = {}
    highest = {}
    
    for j in range(len(prediction)):
        for k in range(len(prediction[j])):
            # predictions[i][j][k] = canonicalize_smiles_clear_map(predictions[i][j][k])
            if prediction[j][k] is None:
                valid_score[j][k] = 11
            valid_rates[k] += 1
        # error detection and deduplication
        de_error = [i[0] for i in sorted(list(zip(prediction[j], valid_score[j])), key=lambda x: x[1]) if i[0] is not None]
        prediction[j] = list(set(de_error))
        prediction[j].sort(key=de_error.index)
        for k, data in enumerate(prediction[j]):
            if data in rank:
                rank[data] += 1 / (alpha * k + 1)
            else:
                rank[data] = 1 / (alpha * k + 1)
            if data in highest:
                highest[data] = min(k,highest[data])
            else:
                highest[data] = k
    for key in rank.keys():
        rank[key] += highest[key] * -1e8
    return rank,valid_rates

def process_samples(samples, topk_fname, rsmiles_fname):
    loading_bar = st.progress(0.0, "Loading tokeniser...")
    
    from source.tokeniser import load_tokeniser_from_rsmiles
    print("Building tokeniser...")
    tokeniser = load_tokeniser_from_rsmiles("data/USPTO_50K_PtoR_aug20")
    print(f"Finished tokeniser with {len(tokeniser)} tokens.")

    for i, source in enumerate(samples):
        if (i+1) % 10 == 0:
            loading_bar.progress((i+1) / len(samples), f"Processing samples... {i+1}/{len(samples)}")
        canon_source = canonicalize(source, iso=True)
        target = samples[source]['target']
        smis = samples[source]['samples']
        mol = Chem.MolFromSmiles(target.rstrip('?'))
        canon_target = Chem.MolToSmiles(mol)

        rankings, _ = compute_rank(smis)

        with open(rsmiles_fname, 'a') as f:
            with open(topk_fname, 'a') as f2:
                for k, smi in enumerate(sorted(rankings, key = lambda s: rankings[s], reverse=True)):
                    if k > 10:
                        break
                    tokenised_smi = tokeniser.tokenise([smi])['original_tokens'][0]
                    f.write(' '.join(tokenised_smi) + '\n')
                    f2.write(f"{smi}\t{canon_source}\t{k+1}\n")
            
    loading_bar.empty()


st.title('Categorical Diffusion for Retrosynthesis - Evaluation')
"""Sean Current"""

with open('st_rsmiles_samples.tmp') as fp:
        samples = json.load(fp)

file_base = 'RSMILES'
if os.path.exists(f'{file_base}_topk.txt') and os.path.exists(f'{file_base}_rsmiles.txt'):
    st.write("Files already processed. Skipping...")
else:
    process_samples(samples, f'{file_base}_topk.txt', f'{file_base}_rsmiles.txt')


predictions = st.file_uploader('label="Upload Predictions File"')
if not predictions:
    st.stop()

data = {'rank': [], 'source': [], 'smi': [], 'pred_source': [], 'rt_accuracy': []}

with open(f'RSMILES_topk.txt') as f:
    last_rank = 0
    last_accurate = False
    for line in f:
        cleaned = line.strip().split()
        if len(cleaned) != 3:
            predictions.readline()
            continue
        smi, source, rank = cleaned
        rank = int(rank)
        pred_source = ''.join(predictions.readline().decode().strip().split())
        data['rank'].append(int(rank))
        data['source'].append(source)
        data['smi'].append(smi)
        data['pred_source'].append(pred_source)
        if rank <= last_rank:
            last_accurate = int(canonicalize(pred_source, iso=False) == canonicalize(source, iso=False))
        elif not last_accurate:
            last_accurate = int(canonicalize(pred_source, iso=False) == canonicalize(source, iso=False))
        data['rt_accuracy'].append(last_accurate)
        # data['rt_accuracy'].append(int(canonicalize(pred_source, iso=False) == canonicalize(source, iso=False)))
        last_rank = rank

data = pd.DataFrame(data)

st.write('Top-k round-trip accuracy:')
col1, col2, col3, col4 = st.columns(4)
col1.metric("k=1", f"{data[data['rank'] <= 1]['rt_accuracy'].mean():2.3%}")
col2.metric("k=3", f"{data[data['rank'] <= 3]['rt_accuracy'].mean():2.3%}")
col3.metric("k=5", f"{data[data['rank'] <= 5]['rt_accuracy'].mean():2.3%}")
col4.metric("k=10", f"{data[data['rank'] <= 10]['rt_accuracy'].mean():2.3%}")

