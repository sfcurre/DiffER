import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import sys, json, zlib, os, random, pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from scipy.stats import pointbiserialr, pearsonr, linregress
import scipy.stats as stats
from difflib import SequenceMatcher
from collections import defaultdict, Counter
from altair import datum
from nltk import edit_distance

from rdkit.Chem import rdMolDescriptors, AllChem, Draw
from rdkit import Chem, RDLogger, DataStructs
drawOptions = Draw.rdMolDraw2D.MolDrawOptions()
drawOptions.prepareMolsBeforeDrawing = False

from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.drawOptions.addAtomIndices = True
IPythonConsole.drawOptions.useBWAtomPalette()
IPythonConsole.molSize = 300,300

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
    return True


st.title('Categorical Diffusion for Retrosynthesis - DiffER vs RSMILES Evaluation')
"""Sean Current"""

download_upload_session()

if 'samples' not in st.session_state or 'data' not in st.session_state or 'files' not in st.session_state:
    st.stop()

samples_differ = st.session_state['samples']
data_differ = pd.read_json(st.session_state['data'])

with open('st_rsmiles_samples.tmp') as fp:
    samples_rsmiles = json.load(fp)
data_rsmiles = pd.read_pickle('st_rsmiles_data.tmp')

# align and merge the two dataframes
#rename columns
data_differ.columns = [f'{col}_differ' for col in data_differ.columns]
data_rsmiles.columns = [f'{col}_rsmiles' for col in data_rsmiles.columns]
data_differ.sort_values('SourceSmiles_differ', inplace=True)
data_rsmiles['SourceSmiles_rsmiles'] = data_rsmiles.index
data_rsmiles.sort_values('SourceSmiles_rsmiles', inplace=True)

data = pd.merge(data_differ, data_rsmiles, left_on='SourceSmiles_differ', right_on='SourceSmiles_rsmiles', how='inner')

st.header('Sampling Statistics - Combined')

# col1_, col2_, _, _ = st.columns(4)
# col1_.metric('Single Prediction Validity', f"{data['SampleValidity'].mean():2.3%}")
# col2_.metric('Single Prediction Accuracy', f"{data['SampleAccuracy'].mean():2.3%}")

st.write(f'Sampling was run for {len(data)} molecules.')

st.write('Top-k accuracy:')

col1, col2, col3, col4 = st.columns(4)
col1.metric("k=1", f"{((data['K=1_differ'] + data['K=1_rsmiles']) / 2).mean():2.3%}")
col2.metric("k=1", f"{((data['K=3_differ'] + data['K=3_rsmiles']) / 2).mean():2.3%}")
col3.metric("k=1", f"{((data['K=5_differ'] + data['K=5_rsmiles']) / 2).mean():2.3%}")
col4.metric("k=1", f"{((data['K=10_differ'] + data['K=10_rsmiles']) / 2).mean():2.3%}")


st.header('Lines of best fit for Dataset Statistics')
statistics = ['TargetLengthIncrease', 'EditDistance', 'TanimotoSimilarity', 'RingForming', 'RingOpening', 'RingCount', 'BranchCount', 'NumAtoms']
ks = ['K=1', 'K=3', 'K=5', 'K=10']
dat_mat = []
for stat in statistics:
    dat_mat.append([])
    for k in ks:
        # build data
        res_differ = linregress(data[stat + '_differ'], data[k + '_differ'])
        res_rsmiles = linregress(data[stat + '_rsmiles'], data[k + '_rsmiles'])
        
        z = (res_differ.slope - res_rsmiles.slope) / np.sqrt(res_differ.stderr**2 + res_rsmiles.stderr**2)
        p = 2 * stats.norm.cdf(-abs(z))
        
        significance = ''.join(['*' for t in [.05, .01, .001] if p<=t])
        dat_mat[-1].append(f'{res_differ.slope:.4f} ({res_differ.slope - res_rsmiles.slope:.4f}){significance}')

st.dataframe(pd.DataFrame(dat_mat, index=statistics, columns=ks))


st.header('Randomly Generated Example Reactions')
rxn_type = st.selectbox("Specify a Reaction Type?", ['None'] + list(rxn_types.values()), key='rxn_type', index=0, placeholder='None')

sub_samples_d = samples_rsmiles
if rxn_type != 'None':
    print(data.columns)
    has_rxn_type = set(data[data[rxn_type + '_differ']]['SourceSmiles_differ'])
    sub_samples_d = {k: v for k, v in samples_differ.items() if k in has_rxn_type}
    

if st.button('Generate'):
    for i in range(5):        
        source = np.random.choice(list(sub_samples_d))
        canon_source = canonicalize(source)
        sm = Chem.MolFromSmiles(source)
        AllChem.Compute2DCoords(sm)
        
        target = sub_samples_d[source]['target'].rstrip('?')
        tm = Chem.MolFromSmiles(target)
        AllChem.Compute2DCoords(tm)
        canon_target = Chem.MolToSmiles(tm)
        
        mols, legs = [], []
        mols.append(sm)
        legs.append('Source')
        mols.append(tm)
        legs.append('Target')

        img=Draw.MolsToGridImage(mols, molsPerRow=2,subImgSize=(300,300),legends=legs, useSVG=True)
        st.image(img._repr_svg_())

        ### differ
        mols, legs = [], []
        rankings = defaultdict(int)
        valid = 0
        for smi in samples_differ[source]['samples']:
            num_pad = smi.count('?')
            smi = canonicalize(smi)
            if smi is None or smi == canon_source:
                # print(f'\t{smi}')
                continue
            valid += 1
            rankings[smi] += 1

        for smi, rating in sorted(rankings.items(), key = lambda x: x[1], reverse=True):
            if smi is None:
                continue
            m = Chem.MolFromSmiles(smi)
            AllChem.Compute2DCoords(m)
            mols.append(m)
            legs.append(f'{"*" if smi == canon_target else ""}Rating: {rating} ({rating  / valid:2.3%})')
        
        max_mols = 5
        img=Draw.MolsToGridImage(mols[:max_mols], molsPerRow=min(len(mols), max_mols),subImgSize=(300,300),legends=legs, useSVG=True)
        st.image(img._repr_svg_())

        ### rsmiles
        mols, legs = [], []
        smis = samples_rsmiles[source]['samples']
        rankings, valid_rates = compute_rank(smis)

        for k, (smi, rating) in enumerate(sorted(rankings.items(), key = lambda x: x[1], reverse=True)):
            if smi is None:
                continue
            m = Chem.MolFromSmiles(smi)
            AllChem.Compute2DCoords(m)
            mols.append(m)
            legs.append(f'{"*" if smi == canon_target else ""}Rating: {k+1}')
        
        max_mols = 5
        img=Draw.MolsToGridImage(mols[:max_mols], molsPerRow=min(len(mols), max_mols),subImgSize=(300,300),legends=legs, useSVG=True)
        st.image(img._repr_svg_())
        
st.header("Reaction Search")
source_smiles = st.text_input('Source (Product) SMILES')
if source_smiles:
    source = canonicalize(source_smiles)
    if source in samples_rsmiles:
        sm = Chem.MolFromSmiles(source)
        AllChem.Compute2DCoords(sm)
        
        target = samples_differ[source]['target'].rstrip('?')
        tm = Chem.MolFromSmiles(target)
        AllChem.Compute2DCoords(tm)
        canon_target = Chem.MolToSmiles(tm)
        
        mols, legs = [], []
        mols.append(sm)
        legs.append('Source')
        mols.append(tm)
        legs.append('Target')

        img=Draw.MolsToGridImage(mols, molsPerRow=2,subImgSize=(300,300),legends=legs, useSVG=True)
        st.image(img._repr_svg_())

        ### differ
        mols, legs = [], []
        mols.append(sm)
        legs.append('Source')
        mols.append(tm)
        legs.append('Target')

        mols, legs = [], []
        rankings = defaultdict(int)
        valid = 0
        for smi in samples_differ[source]['samples']:
            num_pad = smi.count('?')
            smi = canonicalize(smi)
            if smi is None or smi == source:
                # print(f'\t{smi}')
                continue
            valid += 1
            rankings[smi] += 1

        for smi, rating in sorted(rankings.items(), key = lambda x: x[1], reverse=True):
            if smi is None:
                continue
            m = Chem.MolFromSmiles(smi)
            AllChem.Compute2DCoords(m)
            mols.append(m)
            legs.append(f'{"*" if smi == canon_target else ""}Rating: {rating} ({rating  / valid:2.3%})')
        
        max_mols = 5
        img=Draw.MolsToGridImage(mols[:max_mols], molsPerRow=min(len(mols), max_mols),subImgSize=(300,300),legends=legs, useSVG=True)
        st.image(img._repr_svg_())

        ### rsmiles
        mols, legs = [], []
        smis = samples_rsmiles[source]['samples']
        rankings, valid_rates = compute_rank(smis)

        for k, (smi, rating) in enumerate(sorted(rankings.items(), key = lambda x: x[1], reverse=True)):
            if smi is None:
                continue
            m = Chem.MolFromSmiles(smi)
            AllChem.Compute2DCoords(m)
            mols.append(m)
            legs.append(f'{"*" if smi == canon_target else ""}Rating: {k+1}')
        
        max_mols = 5
        img=Draw.MolsToGridImage(mols[:max_mols], molsPerRow=min(len(mols), max_mols),subImgSize=(300,300),legends=legs, useSVG=True)
        st.image(img._repr_svg_())

else:
    st.write('Not a valid SMILES string.')

st.header("Images for Publication")

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

no_particle = np.load('sample_count_20to80_no_particle.npy', allow_pickle=True)
particle_only = np.load('sample_count_20to80_particle_only.npy', allow_pickle=True)
both_particle = np.load('sample_count_20to80_both_particle.npy', allow_pickle=True)

labels = ['DiffER$^2$'] * len(no_particle) + ['DiffER$^2$PG'] * len(particle_only) + ['DiffER$^2$PG+'] * len(both_particle)
print(no_particle.shape, particle_only.shape, both_particle.shape)
data = np.concatenate([no_particle, particle_only, both_particle], axis=0).astype(int)
df1 = pd.DataFrame({'labels': labels, 'count': data[:, 0], 'accuracy': data[:, 1]})

df2 = df1.copy()
df2['accuracy'] = 2

df = pd.concat([df1, df2], axis=0)

f = plt.figure(figsize=(12,3.5))
ax = f.add_subplot(1,1,1)

ax = sns.boxplot(data=df, x='accuracy', y='count', hue='labels', palette='pastel', orient='v', showfliers=False)
legend = ax.legend(title='DiffER Ensemble', fontsize=14, ncol=3, loc='upper right')
plt.setp(legend.get_title(), fontsize=14)
ax.set_title("Number of Molecules Produced by Diffusion Ensembles", fontsize=16)
ax.set_ylabel("Number of Unique Molecules", fontsize=16)
ax.set_xlabel("")
ax.set_xticklabels(['K=1 is False', 'K=1 is True', 'Overall'], fontsize=16)
ax.set_ylim(0, 50)

st.pyplot(f)
