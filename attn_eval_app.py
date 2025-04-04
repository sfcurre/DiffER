import streamlit as st
import pandas as pd
import numpy as np
import os, pickle
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from streamlit_image_select import image_select

from rdkit.Chem import rdMolDescriptors, AllChem, Draw
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

# DATAFILE = st.file_uploader('Upload JSON source/target/samples attention dataset:', on_change=clear_tmp_data, accept_multiple_files=False)
DATAFILE = 'BackwardUnifiedContinuous_NoPadLimit_T100_100Samples-2_attns.json'

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

st.header('Sampling Statistics')

# col1_, col2_, _, _ = st.columns(4)
# col1_.metric('Single Prediction Validity', f"{data['SampleValidity'].mean():2.3%}")
# col2_.metric('Single Prediction Accuracy', f"{data['SampleAccuracy'].mean():2.3%}")

st.write(f'Attention was collected for {len(data)} samples.')

col1, col2, col3, col4 = st.columns(4)
col1.metric("Has Valid Sample", f"{np.mean([data[source]['Valid'] for source in data]):2.3%}")
col2.metric("Has Accurate Sample", f"{np.mean([data[source]['Accurate'] for source in data]):2.3%}")

st.write('Sample reaction statistics:')

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
    has_rxn_type = [source for source in data if data[source]['ReactionType'] == rxn_type]
    row.append(rxn_type)
    row.append(f"{np.mean([data[source]['Valid'] for source in has_rxn_type]):2.3%}")
    row.append(f"{np.mean([data[source]['Accurate'] for source in has_rxn_type]):2.3%}")
    row.append(len(has_rxn_type))
    table.append(row)

st.dataframe(pd.DataFrame(table, columns=col_names).sort_values(by='Support', ascending=False))

st.header('Attention Visualization:')


st.write("Select reaction to view:")
slice_id = st.selectbox('Reaction slice:', list(range(len(data) // 4)), index=0, key='slice')

sorted_sources = sorted(data)
sources = sorted_sources[slice_id:slice_id+4]

source_mols = []
target_mols = []
sample_mols = []
for source in sources:
    sm = Chem.MolFromSmiles(source)
    AllChem.Compute2DCoords(sm)
    source_mols.append(sm)

    target = data[source]['target']
    tm = Chem.MolFromSmiles(target)
    AllChem.Compute2DCoords(tm)
    target_mols.append(tm)

    sample = canonicalize(data[source]['sample'])
    if sample is None:
        pm = Chem.Mol()
    else:
        pm = Chem.MolFromSmiles(sample)
    AllChem.Compute2DCoords(pm)
    sample_mols.append(pm)

legends=list(map(str, range(0, len(source_mols))))
st.write('Source:')
img=Draw.MolsToGridImage(source_mols, molsPerRow=len(source_mols),subImgSize=(300,300),legends=legends, returnPNG=True)
st.image(img)
st.write('Target:')
img=Draw.MolsToGridImage(target_mols, molsPerRow=len(source_mols),subImgSize=(300,300),legends=legends, returnPNG=True)
st.image(img)
st.write('Sample:')
img=Draw.MolsToGridImage(sample_mols, molsPerRow=len(source_mols),subImgSize=(300,300),legends=legends, returnPNG=True)
st.image(img)

reaction_id = st.selectbox('Select reaction:', list(range(0, len(source_mols))), index=0, key='reaction')
if reaction_id is None:
    st.stop()

reaction = sources[reaction_id]
tokenised_reaction = ['<L>'] + tokeniser.tokenise([reaction])['original_tokens'][0]

st.write('Here are the encoder attention maps:')
attns = data[reaction]['in_attns']

for m, attn_block in attns.items():
    x_ticks_i = {
        0: (np.arange(len(tokenised_reaction)) + 0.5, tokenised_reaction),
        1: (np.arange(len(tokenised_reaction)) + 0.5, tokenised_reaction),
        2: (np.arange(len(tokenised_reaction)) + 0.5, tokenised_reaction)
    }
    y_ticks_i = {
        0: (np.arange(len(tokenised_reaction)) + 0.5, tokenised_reaction),
        1: (np.arange(len(attn_block)) + 0.5, range(len(attn_block))),
        2: (np.arange(len(attn_block)) + 0.5, range(len(attn_block)))
    }
    for i in range(3):
        fig = plt.figure(figsize=(10, 10))
        sns.heatmap(attn_block.mean(axis=i)[:len(tokenised_reaction), :len(tokenised_reaction)], cmap='viridis', square=True)
        plt.xticks(*x_ticks_i[i], fontsize=8, rotation='horizontal')
        plt.yticks(*y_ticks_i[i], fontsize=8, rotation='horizontal')
        plt.title(f'Layer {m}')
        st.pyplot(fig)

st.write('Here are the decoder multihead attention maps:')
step = st.selectbox('Diffusion step:', [1, 10, 50, 100, 150, 199], index=0, key='step')
if step is None:
    st.stop()

attns = data[reaction]['out_attns'][step]
smiles = data[reaction]['x_t'][step]
tokenised_smiles = tokeniser.tokenise([smiles])['original_tokens'][0]

cutoff_len = len(tokenised_smiles) - smiles.count('?') + 5
tokenised_smiles = tokenised_smiles[:cutoff_len]

for m, attn_block in attns.items():
    if not m.startswith('m'):
        continue
    
    x_ticks_i = {
        0: (np.arange(len(tokenised_smiles)) + 0.5, tokenised_smiles),
        1: (np.arange(len(tokenised_smiles)) + 0.5, tokenised_smiles),
        2: (np.arange(len(tokenised_smiles)) + 0.5, tokenised_smiles)
    }
    y_ticks_i = {
        0: (np.arange(len(tokenised_smiles)) + 0.5, tokenised_smiles),
        1: (np.arange(len(attn_block)) + 0.5, range(len(attn_block))),
        2: (np.arange(len(attn_block)) + 0.5, range(len(attn_block)))
    }
    for i in range(3):
        fig = plt.figure(figsize=(10, 10))
        sns.heatmap(attn_block.mean(axis=i)[:cutoff_len, :cutoff_len], cmap='viridis', square=True)
        plt.xticks(*x_ticks_i[i], fontsize=8, rotation='horizontal')
        plt.yticks(*y_ticks_i[i], fontsize=8, rotation='horizontal')
        plt.title(f'Layer {m}')
        st.pyplot(fig)

st.write('Here are the decoder self attention maps:')
for m, attn_block in attns.items():
    if not m.startswith('s'):
        continue
    
    x_ticks_i = {
        0: (np.arange(len(tokenised_smiles)) + 0.5, tokenised_smiles),
        1: (np.arange(len(tokenised_smiles)) + 0.5, tokenised_smiles),
        2: (np.arange(len(tokenised_smiles)) + 0.5, tokenised_smiles)
    }
    y_ticks_i = {
        0: (np.arange(len(tokenised_smiles)) + 0.5, tokenised_smiles),
        1: (np.arange(len(attn_block)) + 0.5, range(len(attn_block))),
        2: (np.arange(len(attn_block)) + 0.5, range(len(attn_block)))
    }
    for i in range(3):
        fig = plt.figure(figsize=(10, 10))
        sns.heatmap(attn_block.mean(axis=i)[:cutoff_len, :cutoff_len], cmap='viridis', square=True)
        plt.xticks(*x_ticks_i[i], fontsize=8, rotation='horizontal')
        plt.yticks(*y_ticks_i[i], fontsize=8, rotation='horizontal')
        plt.title(f'Layer {m}')
        st.pyplot(fig)

