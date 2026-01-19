import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import sys, json, zlib, os, random, pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import io, base64, PIL
import selfies as sf

from scipy.stats import pointbiserialr, pearsonr, linregress
from scipy.stats.contingency import crosstab, odds_ratio, chi2_contingency
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.metrics import matthews_corrcoef, jaccard_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from difflib import SequenceMatcher
from collections import defaultdict, Counter
from altair import datum
from nltk import edit_distance

from rdkit.Chem import rdMolDescriptors, AllChem, Draw, rdFMCS
from rdkit import Chem, RDLogger, DataStructs
drawOptions = Draw.rdMolDraw2D.MolDrawOptions()
drawOptions.prepareMolsBeforeDrawing = False
AllChem.ConstrainedDepictionParams.alignOnly=True

from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.drawOptions.addAtomIndices = False
IPythonConsole.drawOptions.useBWAtomPalette()
IPythonConsole.drawOptions.continuousHighlight = False
IPythonConsole.drawOptions.circleAtoms = False
IPythonConsole.drawOptions.setHighlightColour((.9,0,0,.8))
IPythonConsole.drawOptions.legendFontSize = 24
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

import streamlit as st
import json

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


def wiener_index(m):
    res = 0
    amat = Chem.GetDistanceMatrix(m)
    amat[amat > 1e6] = 0
    num_atoms = m.GetNumAtoms()
    for i in range(num_atoms):
        for j in range(i+1,num_atoms):
            res += amat[i][j]
    return res

def canonicalize(smi):
    smi = smi.replace('?', '')
    if smi.startswith('['):
        smi = sf.decoder(smi)
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    else:
        return Chem.MolToSmiles(m)
    
def shannon(string):
    counts = Counter(string)
    frequencies = ((i / len(string)) for i in counts.values())
    return -sum(f * np.log2(f) for f in frequencies)

def clear_tmp_data():
    if os.path.exists('st_samples.tmp'):
        os.remove('st_samples.tmp')
    if os.path.exists('st_data.tmp'):
        os.remove('st_data.tmp')

def get_tokenised_length(tokeniser, smiles):
    smiles = smiles.replace('?', '')
    return len(tokeniser.tokenise([smiles])['original_tokens'][0])

def calc_topk_interval(p, n):
    z = stats.norm.ppf(1 - (1 - .95) / 2)
    pm = z * np.sqrt(p * (1 - p) / n)
    return pm

def find_mol_differences(sm, tm, mols):
    mcs = rdFMCS.FindMCS([sm, tm, *mols])
    mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
    AllChem.Compute2DCoords(mcs_mol)

    mol_atoms = []
    mcs_ = rdFMCS.FindMCS([sm, tm])
    mcs_mol_ = Chem.MolFromSmarts(mcs_.smartsString)
    AllChem.Compute2DCoords(sm)
    AllChem.Compute2DCoords(tm)

    sm_match = sm.GetSubstructMatch(mcs_mol_)
    atoms = []
    for atom in sm.GetAtoms():
        if atom.GetIdx() not in sm_match:
            atoms.append(atom.GetIdx())
    mol_atoms.append(atoms)

    tm_match = tm.GetSubstructMatch(mcs_mol_)
    atoms = []
    for atom in tm.GetAtoms():
        if atom.GetIdx() not in tm_match:
            atoms.append(atom.GetIdx())
    mol_atoms.append(atoms)

    for mol in mols:
        AllChem.Compute2DCoords(mol)
    
        mcs_ = rdFMCS.FindMCS([sm, mol])
        mcs_mol_ = Chem.MolFromSmarts(mcs_.smartsString)
        match = mol.GetSubstructMatch(mcs_mol_)
        atoms = []
        for atom in mol.GetAtoms():
            if atom.GetIdx() not in match:
                atoms.append(atom.GetIdx())
        mol_atoms.append(atoms)

    return mol_atoms

def read_datafiles(datafiles):
    loading_bar = st.progress(0.0, "Loading data...")
    i = 0
    samples = {}
    for file in datafiles:
        sub_samples = json.load(file)
        for source in sub_samples:
            i += 1
            if i % 1000 == 0:
                loading_bar.progress(i / (len(sub_samples) * len(datafiles)), f"Loading data... {i}/{len(sub_samples) * len(datafiles)}")
            canon_source = canonicalize(source)
            if canon_source in samples:
                samples[canon_source]['samples'].extend(sub_samples[source]['samples'])
                samples[canon_source]['edit_distance'].append(edit_distance(source, sub_samples[source]['target']))
            else:
                samples[canon_source] = sub_samples[source]
                samples[canon_source]['edit_distance'] = [edit_distance(source, sub_samples[source]['target'])]

    loading_bar.empty()
    return samples

def process_samples(samples):
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

    data = defaultdict(list)
    for i, source in enumerate(samples):
        if (i+1) % 10 == 0:
            loading_bar.progress((i+1) / len(samples), f"Calculating sample statistics... {i+1}/{len(samples)}")
        canon_source = source #= canonicalize(source)
        data['SourceSmiles'].append(canon_source)
        target = samples[source]['target']
        smis = samples[source]['samples']
        if target.startswith('['):
            target = sf.decoder(target)
        mol = target_mol = Chem.MolFromSmiles(target.rstrip('?'))
        canon_target = Chem.MolToSmiles(mol)

        for rxn_type in rxn_types.values():
            data[rxn_type].append(False)
        
        if canon_source in rxn_type_map:
            data[rxn_type_map[canon_source]][-1] = True

        source_length = get_tokenised_length(tokeniser, source)
        target_length = get_tokenised_length(tokeniser, target)
        sample_lengths = [get_tokenised_length(tokeniser, smi) for smi in smis]

        data['TargetLength'].append(target_length)
        data['SourceLength'].append(source_length)
        data['TargetLengthIncrease'].append(target_length - source_length)
        
        avg_sample_length = np.mean(sample_lengths)
        data['SampleLength'].append(avg_sample_length)
        data['SampleLengthIncrease'].append(avg_sample_length - source_length)
        data['SampleLengthMinusTargetLength'].append(avg_sample_length - target_length)
        data['SampleLengthVariance'].append(np.std(sample_lengths))

        data['MaxAccurateLengthDifference'].append(max([0, *(abs(s - target_length) for s, smi in zip(sample_lengths, smis)
                                                                if canonicalize(smi) == canon_target)]))

        data['P2RSimilarity'].append(SequenceMatcher(None, canon_source, canon_target).ratio()) 
        data['EditDistance'].append(np.mean(samples[source]['edit_distance']))

        source_mol = Chem.MolFromSmiles(source)
        change_in_num_rings = rdMolDescriptors.CalcNumRings(source_mol) - rdMolDescriptors.CalcNumRings(target_mol)
        data['RingForming'].append(change_in_num_rings > 0)
        data['RingOpening'].append(change_in_num_rings < 0)
        data['NonRing'].append(change_in_num_rings == 0)
        data['RingCount'].append(rdMolDescriptors.CalcNumRings(target_mol))
        data['BranchCount'].append(canon_target.count('('))
        data['NumAtoms'].append(mol.GetNumAtoms())

        source_fingerprints = np.array(rdMolDescriptors.GetMorganFingerprintAsBitVect(source_mol, radius=2, nBits=2024))
        target_fingerprints = np.array(rdMolDescriptors.GetMorganFingerprintAsBitVect(target_mol, radius=2, nBits=2024))
        intersection = np.sum(source_fingerprints * target_fingerprints)
        union = np.sum(source_fingerprints) + np.sum(target_fingerprints) - intersection
        data['TanimotoSimilarity'].append(intersection/union if union > 0 else 0)

        data['SourceMF'].append(list(source_fingerprints))
        data['TargetMF'].append(list(target_fingerprints))

        synthesis = '.' in target
        data['Synthesis'].append(synthesis)
        
        valid, accurate = 0, 0
        rankings_by_plurality = defaultdict(int)
        rankings_by_model = defaultdict(lambda: defaultdict(int))
        for i, smi in enumerate(smis):
            smi = canonicalize(smi)
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
            mod_ranks[mod], curr_rating, curr_rank = {}, np.infty, 0
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
        max_smis, max_smi_rating = [], (0, 0)
        accurate_rank = 11
        for k, smi in enumerate(sorted(rankings, key = lambda s: rankings[s], reverse=True)):

            if smi == canon_target:
                accurate_rank = k + 1

            if rankings[smi] > max_smi_rating:
                max_smis = [smi]
                max_smi_rating = rankings[smi]
            elif rankings[smi] == max_smi_rating:
                max_smis.append(smi)

        assert canon_target == Chem.MolToSmiles(target_mol)

        data['TargetSmiles'].append(canon_target)
        data['SampleValidity'].append(valid / len(smis))
        data['SampleAccuracy'].append(accurate / len(smis))
        data['SampleCount'].append(len(rankings))
        data['Generations'].append(len(smis))
        data['HasValid'].append(valid > 0)
        data['HasAccurate'].append(accurate > 0)
        data['AccuracyOfValid'].append(accurate / valid if valid != 0 else 0)
        data['MaxIsAccurate'].append([canon_target] == max_smis)
        data['MaxHasAccurate'].append(canon_target in max_smis)
        data['RankOfAccurate'].append(accurate_rank)
        data['K=1'].append(accurate_rank <= 1)
        data['K=3'].append(accurate_rank <= 3)
        data['K=5'].append(accurate_rank <= 5)
        data['K=10'].append(accurate_rank <= 10)

    loading_bar.empty()
    data = pd.DataFrame(data)
    return data


st.title('Categorical Diffusion for Retrosynthesis - Evaluation')
"""Sean Current"""

DATAFILE = st.file_uploader('Upload JSON source/target/samples dataset:', on_change=clear_tmp_data, accept_multiple_files=True)
if DATAFILE and st.button('Read Datafiles?'):
    samples = read_datafiles(DATAFILE)
    data = process_samples(samples)
    
    st.session_state['samples'] = samples
    st.session_state['data'] = data.to_json()
    st.session_state['files'] = [file.name for file in DATAFILE]

with st.sidebar:
    # Create a container to put the download/upload settings at the top
    container_upload_session_data = st.container()
    with container_upload_session_data:
        with st.expander(label="UPLOAD SESSION", expanded=False):
            download_upload_session()

if 'samples' not in st.session_state or 'data' not in st.session_state or 'files' not in st.session_state:
    st.stop()

samples = st.session_state['samples']
data = pd.read_json(st.session_state['data'])
files = st.session_state['files']

st.write('Uploaded files:')
for file in files:
    st.write(file)

st.header('Sampling Statistics')

# col1_, col2_, _, _ = st.columns(4)
# col1_.metric('Single Prediction Validity', f"{data['SampleValidity'].mean():2.3%}")
# col2_.metric('Single Prediction Accuracy', f"{data['SampleAccuracy'].mean():2.3%}")

st.write(f'Sampling was run for {len(data)} molecules with {int(data["Generations"].mean())} samples taken for each molecule.')

col1, col2, col3, col4 = st.columns(4)
col1.metric("Has Valid Sample", f"{data['HasValid'].mean():2.3%}")
col2.metric("Has Accurate Sample", f"{data['HasAccurate'].mean():2.3%}")
col3.metric("Max Sample is Accurate", f"{data['MaxIsAccurate'].mean():2.3%}")
col4.metric("Max Sample has Accurate", f"{data['MaxHasAccurate'].mean():2.3%}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Per Sample Validity", f"{data['SampleValidity'].mean():2.3%}")
col2.metric("Per Sample Accuracy", f"{data['SampleAccuracy'].mean():2.3%}")
# col3.metric("Max Sample is Accurate", f"{data['MaxIsAccurate'].mean():2.3%}")
# col4.metric("Max Sample has Accurate", f"{data['MaxHasAccurate'].mean():2.3%}")

st.write('Top-k accuracy:')

col1, col2 = st.columns(2)#, col3, col4 = st.columns(4)
col1.metric("k=1", f"{data['K=1'].mean():2.1%} $\\pm$ {calc_topk_interval(data['K=1'].mean(), len(data)):.2%}")
col2.metric("k=3", f"{data['K=3'].mean():2.1%} $\\pm$ {calc_topk_interval(data['K=3'].mean(), len(data)):.2%}")

col1, col2 = st.columns(2)#, col3, col4 = st.columns(4)
col1.metric("k=5", f"{data['K=5'].mean():2.1%} $\\pm$ {calc_topk_interval(data['K=5'].mean(), len(data)):.2%}")
col2.metric("k=10", f"{data['K=10'].mean():2.1%} $\\pm$ {calc_topk_interval(data['K=10'].mean(), len(data)):.2%}")


st.write('Statistics on number of samples produced:')
col1, col2, col3, col4 = st.columns(4)
col1.metric("Mean Sample Count", f"{data['SampleCount'].mean():.1f}")
col2.metric("Median Sample Count", f"{data['SampleCount'].median():.1f}")
col3.metric("Less than 5 Samples", f"{len(data[data['SampleCount'] < 5]) / len(data):2.3%}")
col4.metric("Less than 10 Samples", f"{len(data[data['SampleCount'] < 10]) / len(data):2.3%}")


col1, col2, col3, col4 = st.columns(4)
col1.metric("Mean Sample Count (Top-1 Accurate)", f"{data[data['MaxIsAccurate']]['SampleCount'].mean():.1f}")
col2.metric("Median Sample Count (Top-1 Accurate)", f"{data[data['MaxIsAccurate']]['SampleCount'].median():.1f}")
col3.metric("Mean Sample Count (Top-1 Not Accurate)", f"{data[~data['MaxIsAccurate']]['SampleCount'].mean():.1f}")
col4.metric("Median Sample Count (Top-1 Not Accurate)", f"{data[~data['MaxIsAccurate']]['SampleCount'].median():.1f}")

st.write(' ')
st.write(f'Metrics conditioned on samples with at least one valid output:')

valid_data = data[data['HasValid']]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Has Valid Sample", f"{valid_data['HasValid'].mean():2.3%}")
col2.metric("Has Accurate Sample", f"{valid_data['HasAccurate'].mean():2.3%}")
col3.metric("Max Sample is Accurate", f"{valid_data['MaxIsAccurate'].mean():2.3%}")
col4.metric("Max Sample has Accurate", f"{valid_data['MaxHasAccurate'].mean():2.3%}")

st.write('Top-k accuracy:')

col1, col2, col3, col4 = st.columns(4)
col1.metric("k=1", f"{valid_data['K=1'].mean():2.3%}")
col2.metric("k=3", f"{valid_data['K=3'].mean():2.3%}")
col3.metric("k=5", f"{valid_data['K=5'].mean():2.3%}")
col4.metric("k=10", f"{valid_data['K=10'].mean():2.3%}")

st.header('Metrics by reaction type:')

col_names = ['Reaction Type', 'k=1', 'k=3', 'k=5', 'k=10', 'Support']
features = ['TargetLengthIncrease', 'EditDistance', 'TanimotoSimilarity', 'RingForming', 'RingOpening', 'RingCount', 'BranchCount', 'NumAtoms']
col_names += features
table = []
for rxn_type in rxn_types.values():
    row=[]
    has_rxn_type = data[data[rxn_type]]
    row.append(rxn_type.lower().capitalize())
    row.append(f"{has_rxn_type['K=1'].mean()*100:.1f}")# $\pm$ {calc_topk_interval(has_rxn_type['K=1'].mean(), len(data)):.2%}")
    row.append(f"{has_rxn_type['K=3'].mean()*100:.1f}")# $\pm$ {calc_topk_interval(has_rxn_type['K=3'].mean(), len(data)):.2%}")
    row.append(f"{has_rxn_type['K=5'].mean()*100:.1f}")# $\pm$ {calc_topk_interval(has_rxn_type['K=5'].mean(), len(data)):.2%}")
    row.append(f"{has_rxn_type['K=10'].mean()*100:.1f}")# $\pm$ {calc_topk_interval(has_rxn_type['K=10'].mean(), len(data)):.2%}")
    row.append(len(has_rxn_type))
    for feature in features:
        if feature == 'RingForming':
            row.append(has_rxn_type[feature].mean() * 100)
        elif feature == 'RingOpening':
            row.append(has_rxn_type[feature].mean() * 100)
        else:
            row.append(has_rxn_type[feature].mean())
    table.append(row)

df = pd.DataFrame(table, columns=col_names).sort_values(by='Support', ascending=False)
st.dataframe(df)
# print(df.to_latex(index=False, float_format="%.1f"))


st.header('Correlation of Dataset Statistics')

df = data.select_dtypes(include=['bool', 'number'])
rho = df.corr()
pval = df.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*rho.shape)
p = pval.map(lambda x: ''.join(['*' for t in [.05, .01, .001] if x<=t]))
st.dataframe(rho.round(4).astype(str) + p)


st.header('Lines of best fit for Dataset Statistics')

data_rsmiles = pd.read_pickle('st_rsmiles_data.tmp')
statistics = ['TargetLengthIncrease', 'EditDistance', 'TanimotoSimilarity', 'RingForming', 'RingOpening', 'RingCount', 'BranchCount', 'NumAtoms', 'TargetLength']
statistics = ['TargetLengthIncrease', 'EditDistance', 'BranchCount']
ks = ['K=1', 'K=3', 'K=10']
dat_mat = []
for stat in statistics:
    dat_mat.append([])
    print(stat)
    for k in ks:
        res_differ = sm.Logit(data[k], sm.add_constant(data[stat])).fit(disp=0)
        res_rsmiles = sm.Logit(data_rsmiles[k], sm.add_constant(data_rsmiles[stat])).fit(disp=0)
        
        res_differ.slope = res_differ.params[stat]
        res_differ.stderr = res_differ.bse[stat]
        res_differ.pvalue = res_differ.pvalues[stat]
        res_rsmiles.slope = res_rsmiles.params[stat]
        res_rsmiles.stderr = res_rsmiles.bse[stat]
        res_rsmiles.pvalue = res_rsmiles.pvalues[stat]

        # res_differ = linregress(data[stat], data[k])
        # res_rsmiles = linregress(data_rsmiles[stat], data_rsmiles[k])
        
        z = (res_differ.slope - res_rsmiles.slope) / np.sqrt(res_differ.stderr**2 + res_rsmiles.stderr**2)
        p = 2 * stats.norm.cdf(-abs(z))
        
        significance = ''.join(['*' for t in [.1, .05, .01, .001] if res_differ.pvalue<=t])
        rsmiles_significance = ''.join(['+' for t in [.1, .05, .01, .001] if p<=t])
        difference = res_differ.slope - res_rsmiles.slope
        dat_mat[-1].append(f'{res_differ.slope:.4f}{significance} ({difference:.4f}){rsmiles_significance}')
        
        part1 = f'\\textbf{{{res_differ.slope:.4f}}}' if res_differ.pvalue <= .05 else (f'\\underline{{{res_differ.slope:.4f}}}' if res_differ.pvalue <= .1 else f'{res_differ.slope:.4f}')
        part2 = f'\\textbf{{({difference:+.4f})}}' if p <= .05 else (f'(\\underline{{{difference:+.4f}}})' if p <= .1 else f'({difference:+.4f})')
        print(f'{part1}~{part2} ' + ('&' if k != ks[-1] else '\\\\'))

st.dataframe(pd.DataFrame(dat_mat, index=statistics, columns=ks))

st.header('Comparison of Molecular Properties for Molecules with and without Valid Samples')

mode = st.selectbox("Mode:", sorted(data.select_dtypes(include=['bool']).columns), index=4, key='mode', placeholder='K=1')
var1 = st.selectbox("Choose X Property:", sorted(data.select_dtypes(include=['bool', 'number']).columns), key='var1', index=None, placeholder='TargetLengthIncrease')
var2 = st.selectbox("Choose Y Property:", ['None'] + sorted(data.select_dtypes(include=['bool', 'number']).columns), key='var2', index=None, placeholder='SampleLengthIncrease')

if var1 is None:
    var1 = 'TargetLengthIncrease'
if var2 is None:
    var2 = 'SampleLengthIncrease'

domain = [True, False]
range_ = ['steelblue', 'orange']

if var2 == 'None':
    joint_chart = alt.Chart(data).mark_bar(
        opacity=0.3,
        binSpacing=0
    ).encode(
        alt.X(var1).bin(maxbins=40),
        alt.Y('count():Q'),
        alt.Color(mode + ':N').scale(domain=domain, range=range_)
    )

else:
    joint_chart = alt.Chart(data).mark_point(size=60).encode(
        x=var1,
        y=var2,
        color=alt.Color(mode + ':N').scale(domain=domain, range=range_),
    )

st.altair_chart(joint_chart, use_container_width=True)


st.header('Morgan Fingerprint Analysis')
fvar = st.selectbox("Choose Property:", sorted(data.select_dtypes(include=['bool', 'number']).columns), key='fvar', index=None)

if fvar is None:
    fvar='K=10'

fps = np.array(list(data['TargetMF']))

# tfps = np.array(list(data['TargetMF']))
# sfps = np.array(list(data['SourceMF']))
# fps = (tfps - sfps).astype(int)

stat = [pearsonr(data[fvar], fps[:, i]) for i in range(fps.shape[-1])]
corrs = np.array([s.statistic for s in stat])
pvals = np.array([s.pvalue for s in stat])
counts = fps.sum(axis=0)
arg_corrs = np.argsort(corrs)
arg_corrs = arg_corrs[counts[arg_corrs] > 0]
top_neg, top_pos = arg_corrs[:10], arg_corrs[-10:]

mols = []
for fpbit in top_neg:
    i = np.random.choice(np.where(fps[:, fpbit] == 1)[0])
    mol = Chem.MolFromSmiles(data['TargetSmiles'][i])
    # mol = Chem.MolFromSmiles(data['SourceSmiles'][i])
    info={}
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2024, bitInfo=info)
    mols.append((mol, fpbit, info))

labels = ['C1', 'C2', 'C3', 'C4', 'C5', 'D1', 'D2', 'D3', 'D4', 'D5']
formatter = lambda s: f'{s[3]}: r={s[0]:.2f}, N={s[1]}'#, p={s[2]:.2f}'
img=Draw.DrawMorganBits(mols, molsPerRow=5, legends=list(map(formatter, zip(corrs[top_neg], counts[top_neg], pvals[top_neg], labels))), drawOptions=drawOptions)
st.image(img)

mols = []
for fpbit in top_pos:
    i = np.random.choice(np.where(fps[:, fpbit] == 1)[0])
    mol = Chem.MolFromSmiles(data['TargetSmiles'][i])
    # mol = Chem.MolFromSmiles(data['SourceSmiles'][i])
    info={}
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2024, bitInfo=info)
    mols.append((mol, fpbit, info))

labels = ['B5', 'B4', 'B3', 'B2', 'B1', 'A5', 'A4', 'A3', 'A2', 'A1']
img=Draw.DrawMorganBits(mols, molsPerRow=5, legends=reversed(list(map(formatter, zip(corrs[top_pos], counts[top_pos], pvals[top_pos], labels)))), drawOptions=drawOptions)
st.image(img)

st.header('Randomly Generated Example Reactions')
rxn_type = st.selectbox("Specify a Reaction Type?", ['None'] + list(rxn_types.values()), key='rxn_type', index=0, placeholder='None')

sub_samples = samples
if rxn_type != 'None':
    has_rxn_type = set(data[data[rxn_type]]['SourceSmiles'])
    sub_samples = {k: v for k, v in samples.items() if k in has_rxn_type}

# has_length = set(data[data['TargetLengthIncrease'] > 16]['SourceSmiles'])
# sub_samples = {k: v for k, v in samples.items() if k in has_length}

if st.button('Generate'):
    for i in range(5):
        source = np.random.choice(list(sub_samples))
        canon_source = canonicalize(source)
        sm = Chem.MolFromSmiles(source)
        # AllChem.Compute2DCoords(sm)
        
        target = samples[source]['target'].rstrip('?')
        tm = Chem.MolFromSmiles(target)
        # AllChem.Compute2DCoords(tm)
        canon_target = Chem.MolToSmiles(tm)
        
        mols, legs = [], []
        mols.append(sm)
        legs.append('Source')
        mols.append(tm)
        legs.append('Target')

        rankings = defaultdict(int)
        valid = 0
        for smi in samples[source]['samples']:
            num_pad = smi.count('?')
            smi = canonicalize(smi)
            if smi is None or smi == source:
                continue
            valid += 1
            rankings[smi] += 1

        for i, (smi, rating) in enumerate(sorted(rankings.items(), key = lambda x: x[1], reverse=True)):
            if smi is None:
                continue
            m = Chem.MolFromSmiles(smi)
            mols.append(m)
            legs.append(f'{"*" if smi == canon_target else ""}Rank: {i + 1} ({rating  / valid:2.1%})')

        max_mols = 5
        mols = mols[:max_mols] + [m for m, l in zip(mols[max_mols:], legs[max_mols:]) if l.startswith('*')]
        legs = legs[:max_mols] + [l for l in legs[max_mols:] if l.startswith('*')]
        highlight = find_mol_differences(sm, tm, mols[2:])
        img=Draw.MolsToGridImage(mols, molsPerRow=len(mols),subImgSize=(300,300), legends=legs, highlightAtomLists=highlight, returnPNG=True)
        png_bytes = base64.b64decode(img._repr_png_())
        image_file = io.BytesIO(png_bytes)
        img = PIL.Image.open(image_file)
        st.image(img)

st.header("Reaction Search")
source_smiles = st.text_input('Source (Product) SMILES')
if source_smiles:
    source = canonicalize(source_smiles)
    if source in samples:
        sm = Chem.MolFromSmiles(source)
        
        target = samples[source]['target'].rstrip('?')
        tm = Chem.MolFromSmiles(target)
        canon_target = Chem.MolToSmiles(tm)
        
        mols, legs = [], []
        mols.append(sm)
        legs.append('Source')
        mols.append(tm)
        legs.append('Target')

        rankings = defaultdict(int)
        valid = 0
        for smi in samples[source]['samples']:
            num_pad = smi.count('?')
            smi = canonicalize(smi)
            if smi is None or smi == source:
                continue
            valid += 1
            rankings[smi] += 1

        for i, (smi, rating) in enumerate(sorted(rankings.items(), key = lambda x: x[1], reverse=True)):
            if smi is None:
                continue
            m = Chem.MolFromSmiles(smi)
            mols.append(m)
            legs.append(f'{"*" if smi == canon_target else ""}Rank: {i + 1} ({rating  / valid:2.1%})')

        max_mols = 10
        mols = mols[:max_mols]# + [m for m, l in zip(mols[max_mols:], legs[max_mols:]) if l.startswith('*')]
        legs = legs[:max_mols]# + [l for l in legs[max_mols:] if l.startswith('*')]
        highlight = find_mol_differences(sm, tm, mols[2:])
        img=Draw.MolsToGridImage(mols, molsPerRow=len(mols),subImgSize=(300,300), legends=legs, highlightAtomLists=highlight, returnPNG=True)
        png_bytes = base64.b64decode(img._repr_png_())
        image_file = io.BytesIO(png_bytes)
        img = PIL.Image.open(image_file)
        st.image(img)


b_count = 0
lengths = defaultdict(int)
partial = Chem.MolFromSmarts('CC1(C)OBOC1(C)C')
for source in samples:
    target = samples[source]['target']
    mol = Chem.MolFromSmiles(target.rstrip('?'))
    source_mol = Chem.MolFromSmiles(source)
    if mol is not None and mol.HasSubstructMatch(partial) and not source_mol.HasSubstructMatch(partial):
        b_count += 1
        l = data[data['SourceSmiles'] == source]['TargetLengthIncrease'].values[0]
        if l != 25:
            b_count -= 1
            continue
        lengths[l] += 1
st.write(f'Found {b_count} molecules with the substructure CC1(C)OBOC1(C)C in the target molecule.')
st.write(f'Lengths of these molecules: {lengths}')

st.write('Total reactions with these length changes: {}'.format(len(data[data['TargetLengthIncrease'].isin(lengths)])))


st.header("Images for Publication")

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

# stacked histogram
f = plt.figure(figsize=(8,3.5))
ax = f.add_subplot(1,1,1)

# plot
hist = sns.histplot(data=data, ax=ax, stat="percent", multiple="stack",
                    x="SampleCount", kde=False,
                    palette='pastel', hue="MaxIsAccurate",
                    element="bars", legend=True, binwidth=2, discrete=True, hue_order=[True, False])
patches=[]
for bar, hatch in zip(hist.patches, [''] * (len(hist.patches) // 2) + ['//'] * (len(hist.patches) // 2)):
    bar.set_hatch(hatch)

palette=sns.color_palette('pastel')
patch1 = mpatches.Patch(facecolor=palette[0], hatch='//', label=True)
patch2 = mpatches.Patch(facecolor=palette[1], hatch='', label=False)
legend = ax.legend(handles = [patch1, patch2], title='K=1 is Accurate', fontsize=14)
plt.setp(legend.get_title(), fontsize=14)
ax.set_title("Number of Molecules Produced by Diffusion Ensemble", fontsize=16)
ax.set_xlabel("Number of Molecules", fontsize=16)
ax.set_ylabel("Percent", fontsize=16)

st.pyplot(f)

# np.save('sample_count.npy', data[['SampleCount', 'K=1']].values)

st.header('Lines of best fit for Dataset Statistics')

data['RingReaction'] = data['RingForming'].astype(int) - data['RingOpening'].astype(int)
data_rsmiles['RingReaction'] = data_rsmiles['RingForming'].astype(int) - data_rsmiles['RingOpening'].astype(int)

statistics = ['TargetLengthIncrease', 'BranchCount', 'EditDistance', 'RingReaction']
titles = ['Target Length Diff.', 'Branch Count', 'Edit Distance', 'Ring Reactions']
ks = ['K=1', 'K=3', 'K=10']
fig, axes = plt.subplots(len(ks), len(statistics), figsize=(13, 2.2*len(ks)), sharex='col', sharey=False, squeeze=False, layout='constrained')
for i, stat in enumerate(statistics):
    print(stat)
    for j, k in enumerate(ks):
        if stat == 'RingReaction':
            rdata_differ = data[data[stat] != 0][[stat, k]]
            rdata_differ['Model'] = 'DiffER$^2$PG+'
            rdata_rsmiles = data_rsmiles[data_rsmiles[stat] != 0][[stat, k]]
            rdata_rsmiles['Model'] = 'R-SMILES'
            rdata = pd.concat([rdata_differ, rdata_rsmiles], axis=0)
            # sns.barplot(x=stat, y=k, data=rdata, width=0.5, ax=axes[j, i], hue='Model', palette=['tab:blue', 'tab:red'], errorbar=None)
            # dat = rdata_differ.groupby(stat).count()
            sns.barplot(x=stat, y=k, data=rdata_differ, color='gray', alpha=0.3, fill=True, ax=axes[j, i], width=0.5, estimator=len, edgecolor='black')
            sns.barplot(x=stat, y=k, data=rdata, width=0.5, ax=axes[j, i], hue='Model', palette=['tab:blue', 'tab:red'], errorbar=None, estimator=np.sum, edgecolor='black')
            axes[j, i].set_xlabel('')
            axes[j, i].set_ylabel('Count')
            axes[j, i].set_xticks([0, 1], ['Ring Removing', 'Ring Adding'], fontsize=12)
            axes[j, i].legend_.remove()
        else:
            # res_differ = linregress(data[stat], data[k])
            # res_rsmiles = linregress(data_rsmiles[stat], data_rsmiles[k])

            # vals = np.linspace(data[stat].min(), data[stat].max(), 30)#len(set(data[stat])))
            # y_differ = res_differ.intercept + res_differ.slope * vals
            # y_rsmiles = res_rsmiles.intercept + res_rsmiles.slope * vals
            # intersect = -(res_differ.intercept - res_rsmiles.intercept) / (res_differ.slope - res_rsmiles.slope)

            regressor = sm.Logit(data[k], sm.add_constant(data[stat]))
            res = regressor.fit(disp=0)
            # print(res.summary())

            regressor_rsmiles = sm.Logit(data_rsmiles[k], sm.add_constant(data_rsmiles[stat]))
            res_rsmiles = regressor_rsmiles.fit(disp=0)
            # print(res_rsmiles.summary())

            def sigmoid(x):
                return 1 / (1 + np.exp(-x))
            
            vals = np.linspace(data[stat].min(), data[stat].max(), 30)#len(set(data[stat])))
            y_differ = sigmoid(res.params['const'] + res.params[stat] * vals)
            y_rsmiles = sigmoid(res_rsmiles.params['const'] + res_rsmiles.params[stat] * vals)

            # data[stat] = pd.cut(data[stat], bins=vals, labels=(vals[1:] + vals[:-1])/2, include_lowest=True).astype(float)
            # y_differ = data.groupby(stat)[k].mean()
            # data_rsmiles[stat] = pd.cut(data_rsmiles[stat], bins=vals, labels=(vals[1:] + vals[:-1])/2, include_lowest=True).astype(float)
            # y_rsmiles = data_rsmiles.groupby(stat)[k].mean()
            # axes[j, i].plot(vals, y_differ, color='tab:blue', marker='o', markersize=4, linestyle='')
            # axes[j, i].plot(vals, y_rsmiles, color='tab:red', marker='*', markersize=4, linestyle='')

            # difference = y_differ - y_rsmiles
            # axes[j, i].stem(difference.index, difference.values, linefmt='k-', markerfmt='.', basefmt='k-')#, markercolor='tab:blue', linecolor='k')
            # axes[j, i].plot(vals, difference, color='tab:blue')
            # axes[j, i].axhline(0, color='black', linestyle='--', linewidth=1)
            # axes[j, i].axvline(np.median(data[stat]), color='red', linestyle='--', linewidth=1)
            # axes[j, i].axvline(intersect, color='black', linestyle='--', linewidth=1)
            # axes[j, i].set_ylim(-0.1, 0.3)
            
            axes[j, i].plot(vals, y_differ, label='DiffER$^2$PG+', color='tab:blue')
            axes[j, i].plot(vals, y_rsmiles, label='R-SMILES', color='tab:red')
            ax2 = axes[j, i].twinx()
            sns.histplot(data[stat], color='gray', alpha=0.3, fill=True, ax=ax2, bins=vals)
            ax2.set_ylabel('', fontsize=12)

            # sns.histplot(data[data[k] == 1][stat], color='tab:blue', alpha=0.5, fill=True, ax=ax2, bins=vals)
            # ax2.set_ylabel('', fontsize=12)
            # sns.histplot(data_rsmiles[data_rsmiles[k] == 1][stat], color='tab:red', alpha=0.5, fill=True, ax=ax2, bins=vals)
            ax2.set_ylabel('', fontsize=12)
            ax2.set_yticks([])
            axes[j, i].set_xlim(vals.min(), vals.max())
            if i == 0:
                axes[j, i].set_xlim(vals.min(), 25)
            if i == 1:
                axes[j, i].set_xlim(vals.min(), 15)
            if i == 2:
                axes[j, i].set_xlim(vals.min(), 40)
            axes[j, i].set_ylim(0.2, 1.0)
            # axes[j, i].set_ylim(-0.3, 0.3)

        if i == 0:
            axes[j, i].set_ylabel(f'Top-{k.split('=')[-1]} Accuracy', fontsize=16)
        if j == 0:
            axes[j, i].set_title(titles[i], fontsize=16)

axes[0, -1].legend(fontsize=12, loc='upper right')

st.pyplot(fig)