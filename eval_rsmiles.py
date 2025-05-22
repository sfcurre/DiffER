import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import sys, json, zlib, os, random, pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from scipy.stats import pointbiserialr, pearsonr, linregress
from difflib import SequenceMatcher
from collections import defaultdict, Counter
from altair import datum
from nltk import edit_distance

from rdkit.Chem import rdMolDescriptors, AllChem, Draw
from rdkit import Chem, RDLogger, DataStructs
drawOptions = Draw.rdMolDraw2D.MolDrawOptions()
drawOptions.prepareMolsBeforeDrawing = False
from rdkit.Chem.Draw import IPythonConsole

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

def format_rsmiles_output(input_file, target_file, output_file, beam_size=10, augmentation=20, loading_bar=None):
    samples = {}
    clean_line = lambda s: canonicalize(''.join(s.split()))
    with open(input_file, 'r') as f:
        with open(target_file, 'r') as f2:
            input_lines = f.readlines()
            target_lines = f2.readlines()

            i = 0
            for source_line, target_line in zip(input_lines, target_lines):  
                i += 1
                if loading_bar is not None:
                    if i % 100 == 0:
                        loading_bar.progress(i / len(input_lines), f"Loading data... {i}/{len(input_lines)}")

                source = clean_line(source_line)
                target = clean_line(target_line)
                
                if source not in samples:
                    samples[source] = {}
                    samples[source]['target'] = target
                    samples[source]['samples'] = []
                    samples[source]['edit_distance'] = []
                    
                samples[source]['samples'].append([clean_line(output_file.readline().decode()) for _ in range(beam_size)])
                samples[source]['edit_distance'].append(edit_distance(''.join(source_line.split()), ''.join(target_line.split())))


    return samples

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

def canonicalize(smi):
    smi = smi.replace('?', '')
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    else:
        return Chem.MolToSmiles(m)

def clear_tmp_data():
    if os.path.exists('st_rsmiles_samples.tmp'):
        os.remove('st_rsmiles_samples.tmp')
    if os.path.exists('st_rsmiles_data.tmp'):
        os.remove('st_rsmiles_data.tmp')

def get_tokenised_length(tokeniser, smiles):
    smiles = smiles.replace('?', '')
    return len(tokeniser.tokenise([smiles])['original_tokens'][0])

def rating_func(smi, canon_source, pred_confidence, length_confidence, num_pad):
    # rating = 20 - (len(smi) - len(canon_source))
    rating = 1 #length_confidence
    return round(rating, 2)

st.title('Categorical Diffusion for Retrosynthesis - Evaluation')
"""Sean Current"""

DATAFILE = st.file_uploader('Upload JSON source/target/samples dataset:', on_change=clear_tmp_data, accept_multiple_files=False)
if not DATAFILE:
    RECOVERED = st.checkbox('Recover previous session?', disabled=bool(DATAFILE))
    if not RECOVERED:
        st.stop()

if not os.path.exists('st_rsmiles_samples.tmp'):
    if isinstance(DATAFILE, str):
        DATAFILE = [DATAFILE]
    
    loading_bar = st.progress(0.0, "Loading data...")
    samples = format_rsmiles_output('data/USPTO_50K_PtoR_aug20/test/src-test.txt', 'data/USPTO_50K_PtoR_aug20/test/tgt-test.txt', DATAFILE, loading_bar=loading_bar)

    with open('st_rsmiles_samples.tmp', 'w') as fp:
        json.dump(samples, fp)
    loading_bar.empty()

else:
    with open('st_rsmiles_samples.tmp') as fp:
        samples = json.load(fp)

if not os.path.exists('st_rsmiles_data.tmp'):

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
    descriptors = ['MolWt', 'NumAromaticRings', 'NumAliphaticRings', 'RingCount', 'NumHeteroatoms']
    clean = True
    all_reactants, all_products = set(), set()
    reactant_collisions, product_collisions = 0, 0
    for i, source in enumerate(samples):
        if (i+1) % 10 == 0:
            loading_bar.progress((i+1) / len(samples), f"Calculating sample statistics... {i+1}/{len(samples)}")
        canon_source = canonicalize(source)
        data['SourceSmiles'].append(canon_source)
        target = samples[source]['target']
        smis = samples[source]['samples']
        mol = target_mol = Chem.MolFromSmiles(target.rstrip('?'))
        canon_target = Chem.MolToSmiles(mol)

        for rxn_type in rxn_types.values():
            data[rxn_type].append(False)
        
        data[rxn_type_map[canon_source]][-1] = True

        source_length = get_tokenised_length(tokeniser, source)
        target_length = get_tokenised_length(tokeniser, target)
        sample_lengths = [get_tokenised_length(tokeniser, smi) for aug_smis in smis for smi in aug_smis if smi is not None]

        data['TargetLength'].append(target_length)
        data['SourceLength'].append(source_length)
        data['TargetLengthIncrease'].append(target_length - source_length)
        
        avg_sample_length = np.mean(sample_lengths)
        data['SampleLength'].append(avg_sample_length)
        data['SampleLengthIncrease'].append(avg_sample_length - source_length)
        data['SampleLengthMinusTargetLength'].append(avg_sample_length - target_length)
        data['SampleLengthVariance'].append(np.std(sample_lengths))

        # data['ShannonEntropy'].append(shannon(canon_target))
        # compressed = zlib.compress(canon_target.encode())
        # data['CompressionRate'] = sys.getsizeof(canon_target.encode()) / sys.getsizeof(compressed)
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

        synthesis = '.' in target
        data['Synthesis'].append(synthesis)
  
        rankings, valid_rates = compute_rank(smis)
        
        accurate_rank = 11
        for k, smi in enumerate(sorted(rankings, key = lambda s: rankings[s], reverse=True)):
            if smi == canon_target:
                accurate_rank = k + 1

        assert canon_target == Chem.MolToSmiles(target_mol)

        data['TargetSmiles'].append(canon_target)
        data['SampleValidity'].append(np.mean(valid_rates) / (len(smis) * len(smis[0])))
        data['SampleAccuracy'].append(np.mean([smi==canon_target for aug_smis in smis for smi in aug_smis]))
        # data['SampleMaxFrag'].append(max_frag_accurate / len(smis))
        data['SampleCount'].append(len(set(rankings)))
        data['Generations'].append(len(smis))
        data['HasValid'].append(np.mean(valid_rates) > 0)
        data['HasAccurate'].append(accurate_rank < 11)
        data['MaxIsAccurate'].append(accurate_rank == 1)
        data['K=1'].append(accurate_rank <= 1)
        data['K=3'].append(accurate_rank <= 3)
        data['K=5'].append(accurate_rank <= 5)
        data['K=10'].append(accurate_rank <= 10)

        # source_size = Chem.MolFromSmiles(source).GetNumAtoms()
        # target_size = Chem.MolFromSmiles(target).GetNumAtoms()
        
        # data['TargetSize'].append(target_size)
        # data['SourceSize'].append(source_size)
        # data['TargetSizeIncrease'].append(target_size - source_size)
        
        # avg_sample_size = np.mean(sample_sizes)
        # data['SampleSize'].append(avg_sample_size)
        # data['SampleSizeIncrease'].append(avg_sample_size - source_size)
        # data['SampleSizeMinusTargetSize'].append(avg_sample_size - target_size)
        # data['SampleSizeVariance'].append(np.std(sample_sizes))
        

    data = pd.DataFrame(data)
    data = data.groupby('SourceSmiles').mean(numeric_only=True)
    # data = data.set_index('Target')
    data.to_pickle('st_rsmiles_data.tmp')

    loading_bar.empty()

else:
    # Load DataFrame
    data = pd.read_pickle('st_rsmiles_data.tmp')


st.header('Sampling Statistics')

# col1_, col2_, _, _ = st.columns(4)
# col1_.metric('Single Prediction Validity', f"{data['SampleValidity'].mean():2.3%}")
# col2_.metric('Single Prediction Accuracy', f"{data['SampleAccuracy'].mean():2.3%}")

st.write(f'Sampling was run for {len(data)} molecules with {int(data["Generations"].mean())} samples taken for each molecule.')

col1, col2, col3, col4 = st.columns(4)
col1.metric("Has Valid Sample", f"{data['HasValid'].mean():2.3%}")
col2.metric("Has Accurate Sample", f"{data['HasAccurate'].mean():2.3%}")
col3.metric("Max Sample is Accurate", f"{data['MaxIsAccurate'].mean():2.3%}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Per Sample Validity", f"{data['SampleValidity'].mean():2.3%}")
col2.metric("Per Sample Accuracy", f"{data['SampleAccuracy'].mean():2.3%}")

st.write('Top-k accuracy:')

col1, col2, col3, col4 = st.columns(4)
col1.metric("k=1", f"{data['K=1'].mean():2.3%}")
col2.metric("k=3", f"{data['K=3'].mean():2.3%}")
col3.metric("k=5", f"{data['K=5'].mean():2.3%}")
col4.metric("k=10", f"{data['K=10'].mean():2.3%}")


st.write('Statistics on number of samples produced:')
col1, col2, col3, col4 = st.columns(4)
col1.metric("Mean Sample Count", f"{data['SampleCount'].mean():.1f}")
col2.metric("Median Sample Count", f"{data['SampleCount'].median():.1f}")
col3.metric("Less than 5 Samples", f"{len(data[data['SampleCount'] < 5]) / len(data):2.3%}")
col4.metric("Less than 10 Samples", f"{len(data[data['SampleCount'] < 10]) / len(data):2.3%}")


col1, col2, col3, col4 = st.columns(4)
col1.metric("Mean Sample Count (Top-1 Accurate)", f"{data[data['MaxIsAccurate'] > 0]['SampleCount'].mean():.1f}")
col2.metric("Median Sample Count (Top-1 Accurate)", f"{data[data['MaxIsAccurate'] > 0]['SampleCount'].median():.1f}")
col3.metric("Mean Sample Count (Top-1 Not Accurate)", f"{data[~(data['MaxIsAccurate'] > 0)]['SampleCount'].mean():.1f}")
col4.metric("Median Sample Count (Top-1 Not Accurate)", f"{data[~(data['MaxIsAccurate'] > 0)]['SampleCount'].median():.1f}")

# st.write('Accuracy of Reactions with Rings:')
# col1, col2, col3 = st.columns(3)
# col1.metric("Non-Ring Reaction", f"{data[data['NonRing']]['MaxHasAccurate'].mean():2.3%}")
# col2.metric("Ring-Opening Reaction", f"{data[data['RingOpening']]['MaxHasAccurate'].mean():2.3%}")
# col3.metric("Ring-Closing Reaction", f"{data[data['RingForming']]['MaxHasAccurate'].mean():2.3%}")

# st.write('Accuracy of Reaction Types:')
# col1, col2 = st.columns(2)
# col1.metric("Synthesis Reaction", f"{data[data['Synthesis']]['MaxHasAccurate'].mean():2.3%}")
# col2.metric("Elimination Reaction", f"{data[~data['Synthesis'].astype(bool)]['MaxHasAccurate'].mean():2.3%}")

st.header('Metrics by reaction type:')

col_names = ['Reaction Type', 'k=1', 'k=3', 'k=5', 'k=10', 'Support']
table = []
for rxn_type in rxn_types.values():
    row=[]
    has_rxn_type = data[data[rxn_type].astype(bool)]
    row.append(rxn_type)
    row.append(f"{has_rxn_type['K=1'].mean():2.3%}")
    row.append(f"{has_rxn_type['K=3'].mean():2.3%}")
    row.append(f"{has_rxn_type['K=5'].mean():2.3%}")
    row.append(f"{has_rxn_type['K=10'].mean():2.3%}")
    row.append(len(has_rxn_type))
    table.append(row)

st.dataframe(pd.DataFrame(table, columns=col_names).sort_values(by='Support', ascending=False))

st.header('Correlation of Dataset Statistics')

df = data.select_dtypes(include=['bool', 'number'])
rho = df.corr()
pval = df.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*rho.shape)
p = pval.map(lambda x: ''.join(['*' for t in [.05, .01, .001] if x<=t]))
st.dataframe(rho.round(4).astype(str) + p)

st.header('Lines of best fit for Dataset Statistics')
statistics = ['TargetLengthIncrease', 'EditDistance', 'TanimotoSimilarity', 'RingForming', 'RingOpening', 'RingCount', 'BranchCount', 'NumAtoms']
ks = ['K=1', 'K=3', 'K=5', 'K=10']
dat_mat = []
for stat in statistics:
    dat_mat.append([])
    for k in ks:
        res = linregress(data[stat], data[k])
        significance = ''.join(['*' for t in [.05, .01, .001] if res.pvalue<=t])
        dat_mat[-1].append(f'{res.intercept:.4f} + {res.slope:.4f}x{significance}  ({res.stderr:.4f})')
st.dataframe(pd.DataFrame(dat_mat, index=statistics, columns=ks))

st.header('Comparison of Molecular Properties for Molecules with and without Valid Samples')

mode = st.selectbox("Mode:", sorted(data.select_dtypes(include=['bool', 'number']).columns), index=6, key='mode', placeholder='K=1')
var1 = st.selectbox("Choose X Property:", sorted(data.select_dtypes(include=['bool', 'number']).columns), key='var1', index=None, placeholder='TargetLengthIncrease')
var2 = st.selectbox("Choose Y Property:", ['None'] + sorted(data.select_dtypes(include=['bool', 'number']).columns), key='var2', index=None, placeholder='SampleLengthIncrease')

if var1 is None:
    var1 = 'TargetLengthIncrease'
if var2 is None:
    var2 = 'SampleLengthIncrease'

domain = [1, 0]
range_ = ['steelblue', 'orange']

if var2 == 'None':
    joint_chart = alt.Chart(data).mark_bar(
        opacity=0.3,
        binSpacing=0
    ).encode(
        alt.X(var1).bin(maxbins=40),
        alt.Y('count():Q'),
        alt.Color(mode + ':Q').scale(domain=domain, range=range_)
    )

else:
    joint_chart = alt.Chart(data).mark_point(size=60).encode(
        x=var1,
        y=var2,
        color=alt.Color(mode + ':Q').scale(domain=domain, range=range_),
        # tooltip=['Name', 'Origin', 'Horsepower', 'Miles_per_Gallon']
    )

st.altair_chart(joint_chart, use_container_width=True)

st.header('Randomly Generated Example Reactions')
rxn_type = st.selectbox("Specify a Reaction Type?", ['None'] + list(rxn_types.values()), key='rxn_type', index=0, placeholder='None')

sub_samples = samples
if rxn_type != 'None':
    has_rxn_type = set(data[data['ReactionType'] == rxn_type]['SourceSmiles'])
    sub_samples = {k: v for k, v in samples.items() if k in has_rxn_type}

if st.button('Generate'):
    for i in range(5):
        source = np.random.choice(list(sub_samples))
        canon_source = canonicalize(source)
        sm = Chem.MolFromSmiles(source)
        AllChem.Compute2DCoords(sm)
        
        target = sub_samples[source]['target'].rstrip('?')
        tm = Chem.MolFromSmiles(target)
        AllChem.Compute2DCoords(tm)
        canon_target = Chem.MolToSmiles(tm)
        
        mols, legs = [], []
        mols.append(sm)
        legs.append('Source')
        mols.append(tm)
        legs.append('Target')

        smis = samples[source]['samples']
        rankings, valid_rates = compute_rank(smis)

        for k, (smi, rating) in enumerate(sorted(rankings.items(), key = lambda x: x[1], reverse=True)):
            if smi is None:
                continue
            m = Chem.MolFromSmiles(smi)
            AllChem.Compute2DCoords(m)
            mols.append(m)
            legs.append(f'{"*" if smi == canon_target else ""}Rating: {k+1}')
        
        max_mols = 7
        img=Draw.MolsToGridImage(mols[:max_mols], molsPerRow=min(len(mols), max_mols),subImgSize=(300,300),legends=legs, useSVG=True)
        st.image(img._repr_svg_())

st.header("Reaction Search")
source_smiles = st.text_input('Source (Product) SMILES')
if source_smiles:
    source = canonicalize(source_smiles)
    if source in samples:
        sm = Chem.MolFromSmiles(source)
        AllChem.Compute2DCoords(sm)
        
        target = samples[source]['target']
        tm = Chem.MolFromSmiles(target)
        AllChem.Compute2DCoords(tm)
        canon_target = Chem.MolToSmiles(tm)
        
        mols, legs = [], []
        mols.append(sm)
        legs.append('Source')
        mols.append(tm)
        legs.append('Target')

        smis = samples[source]['samples']
        rankings, valid_rates = compute_rank(smis)

        for k, (smi, rating) in enumerate(sorted(rankings.items(), key = lambda x: x[1], reverse=True)):
            if smi is None:
                continue
            m = Chem.MolFromSmiles(smi)
            AllChem.Compute2DCoords(m)
            mols.append(m)
            legs.append(f'{"*" if smi == canon_target else ""}Rating: {k+1}')
        
        max_mols = 7
        img=Draw.MolsToGridImage(mols[:max_mols], molsPerRow=min(len(mols), max_mols),subImgSize=(300,300),legends=legs, useSVG=True)
        st.image(img._repr_svg_())

else:
    st.write('Not a valid SMILES string.')

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