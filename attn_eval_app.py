import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import sys, json, zlib, os, random, pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import torch

from difflib import SequenceMatcher
from collections import defaultdict, Counter
from altair import datum

from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, Crippen, AllChem, Draw
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

rxn_types = {'<RX_1>': 'Heteroatom alkylation and arylation',
                     '<RX_2>': 'acyclation and related processes',
                     '<RX_3>': 'C-C bond formation',
                     '<RX_4>': 'heterocycle formation',
                     '<RX_5>': 'protections',
                     '<RX_6>': 'deprotections',
                     '<RX_7>': 'reductions',
                     '<RX_8>': 'oxidations',
                     '<RX_9>': 'functional group interconversion',
                     '<RX_10>': 'functional group addition'}

def canonicalize(smi):
    smi = smi.replace('?', '')
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    else:
        return Chem.MolToSmiles(m)

def clear_tmp_data():
    if os.path.exists('st_attn_data.tmp'):
        os.remove('st_attn_data.tmp')

st.title('Categorical Diffusion for Retrosynthesis - Attention Evaluation')
"""Sean Current"""

DATAFILE = st.file_uploader('Upload JSON source/target/samples attention dataset:', on_change=clear_tmp_data, accept_multiple_files=False)
if not DATAFILE:
    RECOVERED = st.checkbox('Recover previous session?', disabled=bool(DATAFILE))
    if not RECOVERED:
        st.stop()

if not os.path.exists('st_attn_data.tmp'):
    data = torch.load(DATAFILE)

    loading_bar = st.progress(0.0, "Loading tokeniser...")
    
    from source.tokeniser import load_tokeniser_from_rsmiles
    print("Building tokeniser...")
    tokeniser = load_tokeniser_from_rsmiles("data/USPTO_50K_PtoR_aug20")
    print(f"Finished tokeniser with {len(tokeniser)} tokens.")

    with open('data/uspto_50.pickle', 'rb') as fp:
        rxn_type_df = pickle.load(fp)
        rxn_type_df['products_smiles'] = rxn_type_df['products_mol'].map(Chem.MolToSmiles)

        rxn_type_df['reaction_type'] = rxn_type_df['reaction_type'].map(rxn_types.get)
        rxn_type_map = dict(zip(rxn_type_df['products_smiles'], rxn_type_df['reaction_type']))

    descriptors = ['MolWt', 'NumAromaticRings', 'NumAliphaticRings', 'RingCount', 'NumHeteroatoms']
    for i, source in enumerate(data):
        if (i+1) % 10 == 0:
            loading_bar.progress((i+1) / len(data), f"Calculating sample statistics... {i+1}/{len(data)}")
        canon_source = canonicalize(source)
        target = data[source]['target']
        sample = data[source]['sample']
        mol = Chem.MolFromSmiles(target)
        
        data[source]['ReactionType'] = rxn_type_map[canon_source]

        change_in_num_rings = rdMolDescriptors.CalcNumRings(Chem.MolFromSmiles(source)) - rdMolDescriptors.CalcNumRings(Chem.MolFromSmiles(target))
        data[source]['RingForming'] = change_in_num_rings > 0
        data[source]['RingOpening'] = change_in_num_rings < 0
        data[source]['NonRing'] = change_in_num_rings == 0

        synthesis = '.' in target
        data[source]['Synthesis'] = synthesis
        data[source]['Accurate'] = canonicalize(target) == canonicalize(sample)
        data[source]['Valid'] = canonicalize(sample) is not None

    torch.save(data, 'st_attn_data.tmp')

    loading_bar.empty()

else:
    # Load DataFrame
    data = torch.load('st_attn_data.tmp')


st.header('Sampling Statistics')

# col1_, col2_, _, _ = st.columns(4)
# col1_.metric('Single Prediction Validity', f"{data['SampleValidity'].mean():2.3%}")
# col2_.metric('Single Prediction Accuracy', f"{data['SampleAccuracy'].mean():2.3%}")

st.write(f'Attention was collected for {len(data)} samples.')

col1, col2, col3, col4 = st.columns(4)
col1.metric("Has Valid Sample", f"{np.mean([data[source]['Valid'] for source in data]):2.3%}")
col2.metric("Has Accurate Sample", f"{np.mean([data[source]['Accurate'] for source in data]):2.3%}")

st.write('Sample reaction statistics:')

has_rank = data[data['RankOfAccurate'] != 0]
col1, col2, col3, col4 = st.columns(4)
col1.metric("RingForming", f"{np.mean([data[source]['RingForming'] for source in data]):2.3%}")
col2.metric("RingOpening", f"{np.mean([data[source]['RingOpening'] for source in data]):2.3%}")
col3.metric("NonRing", f"{np.mean([data[source]['NonRing'] for source in data]):2.3%}")
col4.metric("Synthesis", f"{np.mean([data[source]['Synthesis'] for source in data]):2.3%}")

st.write(' ')
st.header('Metrics by reaction type:')

col_names = ['Reaction Type', 'Validity', 'Accuracy', 'Support']
table = []
for rxn_type in rxn_types.values():
    row=[]
    has_rxn_type = [source for source in data if source['ReactionType'] == rxn_type]
    row.append(rxn_type)
    row.append(f"{np.mean([data[source]['Valid'] for source in has_rxn_type]):2.3%}")
    row.append(f"{np.mean([data[source]['Accurate'] for source in has_rxn_type]):2.3%}")
    row.append(len(has_rxn_type))
    table.append(row)

st.dataframe(pd.DataFrame(table, columns=col_names).sort_values(by='Support', ascending=False))

st.header('Attention Visualization:')

st.write("Filter reactions:")
filters = []
st.write("Not implemented!")
filtered_data = data

st.write("Select reaction:")
reaction = None

if st.button('Generate'):
    sources = sorted(np.random.choice(list(filtered_data), size=max(5, len(filtered_data)), replace=False))
    for source in sources:
        sm = Chem.MolFromSmiles(source)
        AllChem.Compute2DCoords(sm)
        
        target = data[source]['target']
        tm = Chem.MolFromSmiles(target)
        AllChem.Compute2DCoords(tm)

        sample = data[source]['sample']
        pm = Chem.MolFromSmiles(target)
        AllChem.Compute2DCoords(pm)
        
        mols, legs = [], []
        mols.append(sm)
        legs.append('Source')
        mols.append(tm)
        legs.append('Target')
        mols.append(pm)
        legs.append('Sample')

        img=Draw.MolsToGridImage(mols, molsPerRow=len(mols),subImgSize=(300,300),legends=legs, returnPNG=True)
        st.image(img)
        if st.button('Use Reaction?'):
            reaction = source

if reaction is None:
    st.stop()

st.write('Here are the encoder attention maps:')
attns = data[reaction]['in_attns']

for m, attn_block in attns.items():
    fig = plt.figure(figsize=(10, 4))
    sns.heatmap(attn_block, cmap='viridis')
    plt.xticks(np.arange(len(reaction)), reaction)
    plt.title(f'Layer {m}')
    st.pyplot(fig)

st.write('Here are the decoder attention maps:')
step = st.selectbox('Diffusion step:', [1, 10, 50, 100, 200], index=None, key='step', placeholder=1)
attns = data[reaction['out_attns']][step]
target = data[reaction]['target']

for m, (smiles, attn_block) in attns.items():
    fig = plt.figure(figsize=(10, 4))
    sns.heatmap(attn_block, cmap='viridis')
    plt.xticks(np.arange(len(smiles)), smiles)
    plt.title(f'Layer {m}')
    st.pyplot(fig)

