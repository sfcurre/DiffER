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

def download_upload_session():
    # 1. Download Settings Button
    col1, col2 = st.columns([6, 5])
    settings_to_download = {k: v for k, v in st.session_state.items() if k in ['samples', 'data', 'files']}

    button_download = col1.download_button(label="Download Session",
                                           data=json.dumps(settings_to_download),
                                           file_name=f"session.json",
                                           help="Click to Download Current Session")

    # 2. Select Settings to be uploaded
    uploaded_file = st.file_uploader(label="Select the Session File to be uploaded",
                                     help="Select the Session File (Downloaded in a previous run) that you want"
                                          " to be uploaded and then load (by clicking 'load Session' above)")
    if uploaded_file is not None:
        uploaded_settings = json.load(uploaded_file)
    else:
        return False

    # 3. Apply Settings
    def upload_json_settings(json_settings):
        """Set session state values to what specified in the json_settings."""
        for k in json_settings.keys():
            st.session_state[k] = json_settings[k]
        
    button_apply_settings = col2.button(label="Load Session",
                                        on_click=upload_json_settings,
                                        args=(uploaded_settings,),
                                        help="Click to Load the Session of the Uploaded file.\\\n"
                                             "Please start by uploading a Session File below")
    return uploaded_file

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

        valid, accurate = 0, 0
        rankings_by_plurality = defaultdict(int)
        rankings_by_model = defaultdict(lambda: defaultdict(int))
        for i, smi in enumerate(smis):
            smi = canonicalize(smi, iso=True)
            if smi is None:
                continue

            valid += 1
            if smi == canon_source:
                continue
            
            if smi == canon_target:
                accurate += 1
            
            rankings_by_plurality[smi] += 1
            rankings_by_model[i // 20][smi] += 1    

        # get rankings by model
        mod_ranks = {}
        ranked_choice = {}
        for mod in rankings_by_model:
            mod_ranks[mod], curr_rating, curr_rank = {}, np.inf, 0
            for smi, rating in sorted(rankings_by_model[mod].items(), key=lambda x: x[1]):
                ranked_choice[smi] = 0
                if rating < curr_rating:
                    curr_rating = rating
                    curr_rank += 1
                mod_ranks[mod][smi] = curr_rank

        for smi in ranked_choice:
            for mod in mod_ranks:
                if smi in mod_ranks[mod]:
                    ranked_choice[smi] += mod_ranks[mod][smi]

        rankings = {smi: (rankings_by_plurality[smi], ranked_choice[smi]) for smi in ranked_choice}
        
        with open(rsmiles_fname, 'a') as f:
            with open(topk_fname, 'a') as f2:
                for k, smi in enumerate(sorted(rankings, key = lambda s: rankings[s], reverse=True)):
                    tokenised_smi = tokeniser.tokenise([smi])['original_tokens'][0]
                    f.write(' '.join(tokenised_smi) + '\n')
                    f2.write(f"{smi}\t{canon_source}\t{k+1}\n")
            
    loading_bar.empty()


st.title('Categorical Diffusion for Retrosynthesis - Evaluation')
"""Sean Current"""

file = download_upload_session()
if 'samples' not in st.session_state or 'data' not in st.session_state or 'files' not in st.session_state:
    st.stop()

samples = st.session_state['samples']
files = st.session_state['files']

st.write('Uploaded files:')
for f in files:
    st.write(f)

if file:
    file_base = os.path.basename(file.name).split('.')[0]
else:
    st.stop()

if os.path.exists(f'{file_base}_topk.txt') and os.path.exists(f'{file_base}_rsmiles.txt'):
    st.write("Files already processed. Skipping...")
else:
    process_samples(samples, f'{file_base}_topk.txt', f'{file_base}_rsmiles.txt')

predictions = st.file_uploader('label="Upload Predictions File"')
if not predictions:
    st.stop()

data = {'rank': [], 'source': [], 'smi': [], 'pred_source': [], 'rt_accuracy': []}

with open(f'{file_base}_topk.txt') as f:
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

