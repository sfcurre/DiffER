import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import sys, json, zlib, os, random, pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

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

def rating_func(smi, canon_source, pred_confidence, length_confidence, num_pad):
    # rating = 20 - (len(smi) - len(canon_source))
    rating = 1 #length_confidence
    return round(rating, 2)

st.title('Categorical Diffusion for Retrosynthesis - Evaluation')
"""Sean Current"""

DATAFILE = st.file_uploader('Upload JSON source/target/samples dataset:', on_change=clear_tmp_data, accept_multiple_files=True)
if not DATAFILE:
    RECOVERED = st.checkbox('Recover previous session?', disabled=bool(DATAFILE))
    if not RECOVERED:
        st.stop()

if not os.path.exists('st_samples.tmp'):
    if isinstance(DATAFILE, str):
        DATAFILE = [DATAFILE]
    
    loading_bar = st.progress(0.0, "Loading data...")
    i = 0
    samples = {}
    for file in DATAFILE:
        sub_samples = json.load(file)
        for source in sub_samples:
            i += 1
            if i % 1000 == 0:
                loading_bar.progress(i / (len(sub_samples) * len(DATAFILE)), f"Loading data... {i}/{len(sub_samples) * len(DATAFILE)}")
            canon_source = canonicalize(source)
            if canon_source in samples:
                samples[canon_source]['samples'].extend(sub_samples[source]['samples'])
            else:
                samples[canon_source] = sub_samples[source]

    with open('st_samples.tmp', 'w') as fp:
        json.dump(samples, fp)
    loading_bar.empty()

else:
    with open('st_samples.tmp') as fp:
        samples = json.load(fp)

if not os.path.exists('st_data.tmp'):

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
        mol = Chem.MolFromSmiles(target)
        canon_target = Chem.MolToSmiles(mol)

        # mol_descriptors = Descriptors.CalcMolDescriptors(mol)
        # for d in descriptors:
        #     data[d].append(mol_descriptors[d])
        
        # data['NumAtoms'].append(mol.GetNumAtoms())
        # data['NumBonds'].append(mol.GetNumBonds())
        # data['WienerIndex'].append(wiener_index(mol))

        # dmat = Chem.GetDistanceMatrix(mol)
        # data['GraphDistance'].append(dmat[dmat < 1e6].max())
        # data['NumStereocenters'].append(rdMolDescriptors.CalcNumAtomStereoCenters(mol))
        # data['NumBranches'].append(canon_target.count('('))

        data['ReactionType'].append(rxn_type_map[canon_source])

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

        # data['ShannonEntropy'].append(shannon(canon_target))
        # compressed = zlib.compress(canon_target.encode())
        # data['CompressionRate'] = sys.getsizeof(canon_target.encode()) / sys.getsizeof(compressed)
        data['P2RSimilarity'].append(SequenceMatcher(None, canon_source, canon_target).ratio()) 

        change_in_num_rings = rdMolDescriptors.CalcNumRings(Chem.MolFromSmiles(source)) - rdMolDescriptors.CalcNumRings(Chem.MolFromSmiles(target))
        data['RingForming'].append(change_in_num_rings > 0)
        data['RingOpening'].append(change_in_num_rings < 0)
        data['NonRing'].append(change_in_num_rings == 0)

        synthesis = '.' in target
        data['Synthesis'] = synthesis
        
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
                    # print(smi, mod_ranks[mod][smi])
                    ranked_choice[smi] += mod_ranks[mod][smi]

        # rankings = ranked_choice
        # rankings = rankings_by_plurality
        # rankings = {smi: (ranked_choice[smi], rankings_by_plurality[smi]) for smi in ranked_choice}
        rankings = {smi: (rankings_by_plurality[smi], ranked_choice[smi]) for smi in ranked_choice}
        max_smis, max_smi_rating = [], (0, 0)
        accurate_rank = 0
        for k, smi in enumerate(sorted(rankings, key = lambda s: rankings[s], reverse=True)):

            if smi == canon_target:
                accurate_rank = k + 1

            if rankings[smi] > max_smi_rating:
                max_smis = [smi]
                max_smi_rating = rankings[smi]
            elif rankings[smi] == max_smi_rating:
                max_smis.append(smi)

        data['TargetSmiles'].append(canon_target)
        data['SampleValidity'].append(valid / len(smis))
        data['SampleAccuracy'].append(accurate / len(smis))
        # data['SampleMaxFrag'].append(max_frag_accurate / len(smis))
        data['SampleCount'].append(len(rankings))
        data['HasValid'].append(valid > 0)
        data['HasAccurate'].append(accurate > 0)
        data['AccuracyOfValid'].append(accurate / valid if valid != 0 else 0)
        data['MaxIsAccurate'].append([canon_target] == max_smis)
        data['MaxHasAccurate'].append(canon_target in max_smis)
        data['RankOfAccurate'].append(accurate_rank)

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
    data = data.set_index('TargetSmiles')
    data.to_pickle('st_data.tmp')

    loading_bar.empty()

else:
    # Load DataFrame
    data = pd.read_pickle('st_data.tmp')


st.header('Sampling Statistics')

# col1_, col2_, _, _ = st.columns(4)
# col1_.metric('Single Prediction Validity', f"{data['SampleValidity'].mean():2.3%}")
# col2_.metric('Single Prediction Accuracy', f"{data['SampleAccuracy'].mean():2.3%}")

st.write(f'Sampling was run for {len(data)} molecules with {int(data["SampleCount"].mean())} samples taken for each molecule.')

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

has_rank = data[data['RankOfAccurate'] != 0]
col1, col2, col3, col4 = st.columns(4)
col1.metric("k=1", f"{len(has_rank[has_rank['RankOfAccurate'] <= 1]) / len(data):2.3%}")
col2.metric("k=3", f"{len(has_rank[has_rank['RankOfAccurate'] <= 3]) / len(data):2.3%}")
col3.metric("k=5", f"{len(has_rank[has_rank['RankOfAccurate'] <= 5]) / len(data):2.3%}")
col4.metric("k=10", f"{len(has_rank[has_rank['RankOfAccurate'] <= 10]) / len(data):2.3%}")


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

# st.write('Accuracy of Reactions with Rings:')
# col1, col2, col3 = st.columns(3)
# col1.metric("Non-Ring Reaction", f"{data[data['NonRing']]['MaxHasAccurate'].mean():2.3%}")
# col2.metric("Ring-Opening Reaction", f"{data[data['RingOpening']]['MaxHasAccurate'].mean():2.3%}")
# col3.metric("Ring-Closing Reaction", f"{data[data['RingForming']]['MaxHasAccurate'].mean():2.3%}")

# st.write('Accuracy of Reaction Types:')
# col1, col2 = st.columns(2)
# col1.metric("Synthesis Reaction", f"{data[data['Synthesis']]['MaxHasAccurate'].mean():2.3%}")
# col2.metric("Elimination Reaction", f"{data[~data['Synthesis'].astype(bool)]['MaxHasAccurate'].mean():2.3%}")

st.write(' ')
st.write(f'Metrics conditioned on samples with at least one valid output:')

valid_data = data[data['HasValid']]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Has Valid Sample", f"{valid_data['HasValid'].mean():2.3%}")
col2.metric("Has Accurate Sample", f"{valid_data['HasAccurate'].mean():2.3%}")
col3.metric("Max Sample is Accurate", f"{valid_data['MaxIsAccurate'].mean():2.3%}")
col4.metric("Max Sample has Accurate", f"{valid_data['MaxHasAccurate'].mean():2.3%}")

st.write('Top-k accuracy:')

has_rank = valid_data[valid_data['RankOfAccurate'] != 0]
col1, col2, col3, col4 = st.columns(4)
col1.metric("k=1", f"{len(has_rank[has_rank['RankOfAccurate'] <= 1]) / len(valid_data):2.3%}")
col2.metric("k=3", f"{len(has_rank[has_rank['RankOfAccurate'] <= 3]) / len(valid_data):2.3%}")
col3.metric("k=5", f"{len(has_rank[has_rank['RankOfAccurate'] <= 5]) / len(valid_data):2.3%}")
col4.metric("k=10", f"{len(has_rank[has_rank['RankOfAccurate'] <= 10]) / len(valid_data):2.3%}")

st.header('Metrics by reaction type:')

col_names = ['Reaction Type', 'k=1', 'k=3', 'k=5', 'k=10', 'Support']
table = []
for rxn_type in rxn_types.values():
    row=[]
    has_rxn_type = data[data['ReactionType'] == rxn_type]
    has_rank = has_rxn_type[has_rxn_type['RankOfAccurate'] != 0]
    row.append(rxn_type)
    row.append(f"{len(has_rank[has_rank['RankOfAccurate'] <= 1]) / len(has_rxn_type):2.3%}")
    row.append(f"{len(has_rank[has_rank['RankOfAccurate'] <= 3]) / len(has_rxn_type):2.3%}")
    row.append(f"{len(has_rank[has_rank['RankOfAccurate'] <= 5]) / len(has_rxn_type):2.3%}")
    row.append(f"{len(has_rank[has_rank['RankOfAccurate'] <= 10]) / len(has_rxn_type):2.3%}")
    row.append(len(has_rxn_type))
    table.append(row)

st.dataframe(pd.DataFrame(table, columns=col_names).sort_values(by='Support', ascending=False))

st.header('Comparison of Molecular Properties for Molecules with and without Valid Samples')

mode = st.selectbox("Mode:", ['HasValid', 'HasAccurate', 'MaxIsAccurate', 'MaxHasAccurate'], index=2, key='mode', placeholder='HasValid')
var1 = st.selectbox("Choose X Property:", sorted(filter(lambda s: 'Smiles' not in s, data.columns)), key='var1', index=None, placeholder='TargetLengthIncrease')
var2 = st.selectbox("Choose Y Property:", ['None'] + sorted(filter(lambda s: 'Smiles' not in s, data.columns)), key='var2', index=None, placeholder='SampleLengthIncrease')

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
        alt.Y('count():Q').stack(None),
        alt.Color(mode + ':N').scale(domain=domain, range=range_)
    )

else:
    joint_chart = alt.Chart(data).mark_point(size=60).encode(
        x=var1,
        y=var2,
        color=alt.Color(mode + ':N').scale(domain=domain, range=range_),
        # tooltip=['Name', 'Origin', 'Horsepower', 'Miles_per_Gallon']
    )

st.altair_chart(joint_chart, use_container_width=True)

st.header('Accuracy and Validity rates of Diffusion Samples Compared to Molecular Properties of the Target')

var = st.selectbox("Choose Property:", filter(lambda s: 'Smiles' not in s, sorted(data.columns)), key='var', index=None, placeholder='SampleLengthMinusTargetLength')

if var is None:
    var = 'SampleLengthMinusTargetLength'

v_chart = alt.Chart(data).mark_rect().encode(
    alt.X(var).bin(maxbins=20),
    alt.Y('SampleValidity').bin(maxbins=11),
    alt.Color('count():Q').scale(scheme='greenblue')
    ).properties(
    width=200,
    height=200        
)

a_chart = alt.Chart(data).mark_rect().encode(
    alt.X(var).bin(maxbins=20),
    alt.Y('SampleAccuracy').bin(maxbins=11),
    alt.Color('count():Q').scale(scheme='greenblue')
    ).properties(
    width=200,
    height=200        
)

st.altair_chart(v_chart | a_chart, use_container_width=True)

st.header('Randomly Generated Example Reactions')

if st.button('Generate'):
    for i in range(5):
        source = np.random.choice(list(samples))
        canon_source = canonicalize(source)
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

        rankings = defaultdict(int)
        valid = 0
        for smi in samples[source]['samples']:
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
        
        max_mols = 7
        img=Draw.MolsToGridImage(mols[:max_mols], molsPerRow=min(len(mols), max_mols),subImgSize=(300,300),legends=legs, returnPNG=True)
        st.image(img)

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

        rankings = defaultdict(int)
        valid = 0
        for smi in samples[source]['samples']:
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
        
        max_mols = 7
        img=Draw.MolsToGridImage(mols[:max_mols], molsPerRow=min(len(mols), max_mols),subImgSize=(300,300),legends=legs, returnPNG=True)
        st.image(img)

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