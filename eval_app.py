import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import sys, json, zlib, os, random

from collections import defaultdict, Counter
from altair import datum

from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, Crippen, AllChem, Draw
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

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
    if os.path.exists('st_data.tmp'):
        os.remove('st_data.tmp')    

st.title('Categorical Diffusion for Retrosynthesis - Evaluation')
"""Sean Current"""

DATAFILE = st.file_uploader('Upload JSON source/target/samples dataset:', on_change=clear_tmp_data)
if not DATAFILE:
    st.stop()
samples = json.load(DATAFILE)

if not os.path.exists('st_data.tmp'):
    with st.spinner('Loading...'):
        data = defaultdict(list)
        descriptors = ['MolWt', 'NumAromaticRings', 'NumAliphaticRings', 'RingCount', 'NumHeteroatoms']
        for source in samples:
            canon_source = canonicalize(source)
            data['SourceSmiles'].append(canon_source)
            target = samples[source]['target']
            smis = samples[source]['samples']
            mol = Chem.MolFromSmiles(target)
            canon_target = Chem.MolToSmiles(mol)
            mol_descriptors = Descriptors.CalcMolDescriptors(mol)
            for d in descriptors:
                data[d].append(mol_descriptors[d])
            
            data['NumAtoms'].append(mol.GetNumAtoms())
            data['NumBonds'].append(mol.GetNumBonds())
            data['WienerIndex'].append(wiener_index(mol))

            dmat = Chem.GetDistanceMatrix(mol)
            data['GraphDistance'].append(dmat[dmat < 1e6].max())
            data['NumStereocenters'].append(rdMolDescriptors.CalcNumAtomStereoCenters(mol))

            data['NumBranches'].append(canon_target.count('('))
            data['StringLength'].append(len(canon_target))
            data['ShannonEntropy'].append(shannon(canon_target))
            
            compressed = zlib.compress(canon_target.encode())
            data['CompressionRate'] = sys.getsizeof(canon_target.encode()) / sys.getsizeof(compressed)

            valid, accurate = 0, 0
            max_smis, max_smi_count = [], 0
            accurate_index, accurate_count, k = 0, 0, 0
            # print(target)
            for smi, count in sorted(Counter(map(canonicalize, smis)).items(), key=lambda x: x[1], reverse=True):
                if smi is None:
                    # print(f'\t{smi}')
                    continue
                k += 1
                valid += count
                # print(f'\t{canon_smi} / {smi}' + (' (match)' if canon_smi == canon_target else ''))
                if smi == canon_target:
                    accurate += count
                    accurate_count = count
                if count == accurate_count:
                    accurate_index = k

                if count > max_smi_count:
                    max_smis = [smi]
                    max_smi_count = count
                elif count == max_smi_count:
                    max_smis.append(smi)

            data['TargetSmiles'].append(canon_target)
            data['SampleValidity'].append(valid / len(smis))
            data['SampleAccuracy'].append(accurate / len(smis))
            data['SampleCount'].append(len(smis))
            data['HasValid'].append(valid > 0)
            data['HasAccurate'].append(accurate > 0)
            data['AccuracyOfValid'].append(accurate / valid if valid != 0 else 0)
            data['MaxIsAccurate'].append([canon_target] == max_smis)
            data['MaxHasAccurate'].append(canon_target in max_smis)
            data['RankOfAccurate'].append(accurate_index)

        data = pd.DataFrame(data)
        data = data.set_index('TargetSmiles')
        data.to_pickle('st_data.tmp')

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

st.write('Top-k accuracy:')

has_rank = data[data['RankOfAccurate'] != 0]
col1, col2, col3, col4 = st.columns(4)
col1.metric("k=1", f"{len(has_rank[has_rank['RankOfAccurate'] <= 1]) / len(data):2.3%}")
col2.metric("k=3", f"{len(has_rank[has_rank['RankOfAccurate'] <= 3]) / len(data):2.3%}")
col3.metric("k=5", f"{len(has_rank[has_rank['RankOfAccurate'] <= 5]) / len(data):2.3%}")
col4.metric("k=10", f"{len(has_rank[has_rank['RankOfAccurate'] <= 10]) / len(data):2.3%}")

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


st.header('Comparison of Molecular Properties for Molecules with and without Valid Samples')

mode = st.selectbox("Mode:", ['HasValid', 'HasAccurate', 'MaxIsAccurate', 'MaxHasAccurate'], index=2, key='mode', placeholder='HasValid')
var1 = st.selectbox("Choose X Property:", sorted(filter(lambda s: 'Smiles' not in s, data.columns)), key='var1', index=None, placeholder='NumAtoms')
var2 = st.selectbox("Choose Y Property:", ['None'] + sorted(filter(lambda s: 'Smiles' not in s, data.columns)), key='var2')

if var1 is None:
    var1 = 'NumAtoms'

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

var = st.selectbox("Choose Property:", filter(lambda s: 'Smiles' not in s, sorted(data.columns)), key='var', index=None, placeholder='NumAtoms')

if var is None:
    var = 'NumAtoms'

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

        counts = Counter(map(canonicalize, samples[source]['samples']))
        for smi, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            if smi is None:
                continue
            m = Chem.MolFromSmiles(smi)
            AllChem.Compute2DCoords(m)
            mols.append(m)
            legs.append(f'{"*" if smi == canon_target else ""}Count: {count}')
        
        img=Draw.MolsToGridImage(mols, molsPerRow=len(mols),subImgSize=(300,300),legends=legs, returnPNG=True)
        st.image(img)
