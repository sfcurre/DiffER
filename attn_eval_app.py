import streamlit as st
import pandas as pd
import numpy as np
import os, pickle, re
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import PIL, io
# PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True


from source.tokeniser import REGEX

from rdkit.Chem import rdMolDescriptors, AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D, SimilarityMaps

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
    
def tokenise(smi):
    regex = re.compile(REGEX)
    return ['^'] + re.findall(regex, smi) + ['&']

def clear_tmp_data():
    if os.path.exists('st_attn_data.tmp'):
        os.remove('st_attn_data.tmp')

st.title('Categorical Diffusion for Retrosynthesis - Attention Evaluation')
"""Sean Current"""

# DATAFILE = st.file_uploader('Upload JSON source/target/samples attention dataset:', on_change=clear_tmp_data, accept_multiple_files=False)
DATAFILE = 'BackwardUnifiedContinuous_NoPadLimit_T100-3_attns.json'

data = torch.load(DATAFILE)

loading_bar = st.progress(0.0, "Loading...")

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
    target = data[source]['target'].rstrip('?')
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
sources = sorted_sources[slice_id*4:slice_id*4+4]

source_mols = []
target_mols = []
sample_mols = []
for source in sources:
    sm = Chem.MolFromSmiles(source)
    AllChem.Compute2DCoords(sm)
    source_mols.append(sm)

    target = data[source]['target'].rstrip('?')
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

st.text(r"Source: {}".format(sources[reaction_id]))
st.text(r"Target: {}".format(data[sources[reaction_id]]['target'].rstrip('?')))
st.text(r"Sample: {}".format(data[sources[reaction_id]]['sample'].rstrip('?')))

step = st.selectbox('Diffusion step:', [1, 10, 50, 100, 150, 199], index=0, key='mh_step')

reaction = sources[reaction_id]
tokenised_reaction = ['<L>'] + tokenise(reaction)

if st.checkbox('View encoder attention maps?'):
    st.write('Here are the encoder attention maps:')
    attns = data[reaction]['in_attns']

    for m, attn_block in attns.items():
        x_ticks_i = {
            0: (np.arange(len(tokenised_reaction)) + 0.5, tokenised_reaction),
            1: (np.arange(len(tokenised_reaction)) + 0.5, tokenised_reaction),
        }
        y_ticks_i = {
            0: (np.arange(len(tokenised_reaction)) + 0.5, tokenised_reaction),
            1: (np.arange(len(attn_block)) + 0.5, range(len(attn_block))),
        }
        for i in range(2):
            fig = plt.figure(figsize=(10, 10))
            sns.heatmap(attn_block.mean(axis=i)[:len(tokenised_reaction), :len(tokenised_reaction)], cmap='viridis', square=True, cbar=False)
            plt.xticks(*x_ticks_i[i], fontsize=8, rotation='horizontal')
            plt.yticks(*y_ticks_i[i], fontsize=8, rotation='horizontal')
            plt.title(f'Layer {m}')
            st.pyplot(fig)

        avg_source_attn = np.mean(attn_block.numpy(), axis=(0,1))
        attn_by_token = zip(tokenised_reaction, avg_source_attn)
        mol = Chem.MolFromSmiles(reaction)
        atom_colors = {}
        atom_finder = re.compile(r"(Cl?|Br?|[NOSPFIbcnosp*]|\[[^]]+\])", re.X)
        atoms = atom_finder.findall(reaction)
        first = next(attn_by_token)
        atom_ids = []
        atom_weights = []
        for atom, atom_str in zip(mol.GetAtoms(), atoms):
            while first[0] != atom_str:
                first = next(attn_by_token)
            atom_ids.append(atom.GetIdx())
            atom_weights.append(float(first[1]))    
            first = next(attn_by_token)    
        
        max_weight = max(atom_weights)
        atom_weights = [(w / max_weight) ** 2 for w in atom_weights]

        d = rdMolDraw2D.MolDraw2DCairo(500, 500)
        SimilarityMaps.GetSimilarityMapFromWeights(mol, atom_weights, draw2d=d)
        st.image(d.GetDrawingText())


if st.checkbox('View joint decoder attention maps?'):
    st.write('Here are the joint decoder attention maps:')

    attns = data[reaction]['out_attns'][step]
    smiles = data[reaction]['x_t'][step]
    tokenised_smiles = tokenise(smiles)

    cutoff_len = len(tokenised_smiles) - smiles.count('?') + 5
    tokenised_smiles = tokenised_smiles[:cutoff_len]

    sample = data[reaction]['sample']
    tokenised_sample = tokenise(sample.rstrip('?'))[:-1]
    tokenised_sample += ['?'] * (len(tokenised_smiles) - len(tokenised_sample))

    for j in range(len(attns) // 2):
        m_attn_block = attns[f'm{j}']
        s_attn_block = attns[f's{j}']
        
        m_x_ticks_i = {
            0: (np.arange(len(tokenised_reaction)) + 0.5, tokenised_reaction),
            1: (np.arange(len(tokenised_reaction)) + 0.5, tokenised_reaction),
        }
        m_y_ticks_i = {
            0: (np.arange(len(tokenised_sample)) + 0.5, tokenised_sample),
            1: (np.arange(len(m_attn_block)) + 0.5, range(len(m_attn_block))),
        }

        s_x_ticks_i = {
            0: (np.arange(len(tokenised_smiles)) + 0.5, tokenised_smiles),
            1: (np.arange(len(tokenised_smiles)) + 0.5, tokenised_smiles),
        }
        s_y_ticks_i = {
            0: (np.arange(len(tokenised_sample)) + 0.5, tokenised_sample),
            1: (np.arange(len(s_attn_block)) + 0.5, range(len(s_attn_block))),
        }
        for i in range(2):
            fig, axes = plt.subplots(1, 2, figsize=(20, 10))
            sns.heatmap(m_attn_block.mean(axis=i)[:cutoff_len, :cutoff_len], cmap='viridis', square=True, cbar=False, ax=axes[0])
            axes[0].set_xticks(*m_x_ticks_i[i], fontsize=8, rotation='horizontal')
            axes[0].set_yticks(*m_y_ticks_i[i], fontsize=8, rotation='horizontal')
            axes[0].set_title(f'Layer m{j}')
            axes[0].set_xlabel('Input Sequence', fontsize=12)
            axes[0].set_ylabel('Output Sequence', fontsize=12)
            sns.heatmap(s_attn_block.mean(axis=i)[:cutoff_len, :cutoff_len], cmap='viridis', square=True, cbar=False, ax=axes[1])
            axes[1].set_xticks(*s_x_ticks_i[i], fontsize=8, rotation='horizontal')
            axes[1].set_yticks(*s_y_ticks_i[i], fontsize=8, rotation='horizontal')
            axes[1].set_title(f'Layer s{j}')
            axes[1].set_xlabel('Input Sequence', fontsize=12)
            axes[1].set_ylabel('Output Sequence', fontsize=12)
            st.pyplot(fig)

        images, captions = [], []
        avg_source_attn = np.mean(m_attn_block.numpy(), axis=(0,1))
        attn_by_token = zip(tokenised_reaction, avg_source_attn)
        mol = Chem.MolFromSmiles(reaction)
        atom_colors = {}
        atom_finder = re.compile(r"(Cl?|Br?|[NOSPFIbcnosp*]|\[[^]]+\])", re.X)
        atoms = atom_finder.findall(reaction)
        first = next(attn_by_token)
        atom_ids = []
        atom_weights = []
        for atom, atom_str in zip(mol.GetAtoms(), atoms):
            while first[0] != atom_str:
                first = next(attn_by_token)
            atom_ids.append(atom.GetIdx())
            atom_weights.append(float(first[1]))
            first = next(attn_by_token)

        max_weight = max(atom_weights)
        atom_weights = [(w / max_weight) ** 2 for w in atom_weights]

        d = rdMolDraw2D.MolDraw2DCairo(500, 500)
        SimilarityMaps.GetSimilarityMapFromWeights(mol, atom_weights, draw2d=d)
        images.append(d.GetDrawingText())
        captions.append(f'm{j}')

        avg_source_attn = np.mean(s_attn_block.numpy(), axis=(0,1))
        sample = data[reaction]['sample'].rstrip('?')
        canon_sample = canonicalize(sample)
        if canon_sample is not None:
            tokenised_sample_ = tokenise(sample)
            attn_by_token = zip(tokenised_sample_, avg_source_attn)
            mol = Chem.MolFromSmiles(sample)
            atom_colors = {}
            atom_finder = re.compile(r"(Cl?|Br?|[NOSPFIbcnosp*]|\[[^]]+\])", re.X)
            atoms = atom_finder.findall(sample)
            first = next(attn_by_token)
            atom_ids = []
            atom_weights = []
            for atom, atom_str in zip(mol.GetAtoms(), atoms):
                while first[0] != atom_str:
                    first = next(attn_by_token)
                atom_ids.append(atom.GetIdx())
                atom_weights.append(float(first[1]))            
                first = next(attn_by_token)

            max_weight = max(atom_weights)
            atom_weights = [(w / max_weight) ** 2 for w in atom_weights]

            d = rdMolDraw2D.MolDraw2DCairo(500, 500)
            SimilarityMaps.GetSimilarityMapFromWeights(mol, atom_weights, draw2d=d)
            images.append(d.GetDrawingText())
            captions.append(f's{j}')

        st.image(images, caption=captions, width=350)


if st.checkbox('View decoder attention maps for publication?'):
    st.write('Here are the decoder attention maps for publication:')

    attns = data[reaction]['out_attns'][step]
    smiles = data[reaction]['x_t'][step]
    tokenised_smiles = tokenise(smiles)

    cutoff_len = len(tokenised_smiles) // 2
    tokenised_smiles = tokenised_smiles[:cutoff_len]

    sample = data[reaction]['sample']
    tokenised_sample = tokenise(sample.rstrip('?'))[:-1]
    tokenised_sample += ['?'] * (len(tokenised_smiles) - len(tokenised_sample))

    all_images, all_captions = [], []
    plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "figure.dpi": 500, 
    })
    
    for j in range(len(attns) // 2):
        m_attn_block = attns[f'm{j}']
        s_attn_block = attns[f's{j}']
        
        m_x_ticks_i = {
            0: (np.arange(len(tokenised_reaction)) + 0.5, tokenised_reaction),
            1: (np.arange(len(tokenised_reaction)) + 0.5, tokenised_reaction),
        }
        m_y_ticks_i = {
            0: (np.arange(len(tokenised_sample)) + 0.5, tokenised_sample),
            1: (np.arange(len(m_attn_block)) + 0.5, range(len(m_attn_block))),
        }

        s_x_ticks_i = {
            0: (np.arange(len(tokenised_smiles)) + 0.5, tokenised_smiles),
            1: (np.arange(len(tokenised_smiles)) + 0.5, tokenised_smiles),
        }
        s_y_ticks_i = {
            0: (np.arange(len(tokenised_sample)) + 0.5, tokenised_sample),
            1: (np.arange(len(s_attn_block)) + 0.5, range(len(s_attn_block))),
        }

        images, captions = [], []
        for i in range(1):
            fig = plt.figure(figsize=(10, 10))
            fig.tight_layout(pad=0)
            ax = fig.add_subplot(1, 1, 1)
            fig.subplots_adjust(0.0,0.05,1,1)
            ax.margins(0)
            sns.heatmap(m_attn_block.mean(axis=i)[:cutoff_len, :len(tokenised_reaction)], cmap='viridis', square=True, cbar=False, ax=ax)
            ax.set_xticks(*m_x_ticks_i[i], fontsize=8, rotation='horizontal')
            ax.set_yticks(*m_y_ticks_i[i], fontsize=8, rotation='horizontal')
            # ax.set_title(f'Layer m{j}')
            # ax.set_xlabel('Input Sequence', fontsize=12)
            # ax.set_ylabel('Output Sequence', fontsize=12)
            
            fig.canvas.draw()
            # plt.savefig('img.png', dpi=500, bbox_inches='tight')
            rgba = np.asarray(fig.canvas.buffer_rgba())
            image = PIL.Image.fromarray(rgba)
            # image = PIL.Image.open('img.png')
            images.append(image)
            captions.append(f'Source Attention')
            
            fig = plt.figure(figsize=(10, 10))
            fig.tight_layout(pad=0)
            ax = fig.add_subplot(1, 1, 1)
            fig.subplots_adjust(0.0,0.05,1,1)
            ax.margins(0)
            sns.heatmap(s_attn_block.mean(axis=i)[:cutoff_len, :cutoff_len], cmap='viridis', square=True, cbar=False, ax=ax)
            ax.set_xticks(*s_x_ticks_i[i], fontsize=8, rotation='horizontal')
            ax.set_yticks(*s_y_ticks_i[i], fontsize=8, rotation='horizontal')
            # ax.set_title(f'Layer s{j}')
            # ax.set_xlabel('Input Sequence', fontsize=12)
            # ax.set_ylabel('Output Sequence', fontsize=12)
            
            fig.canvas.draw()
            # plt.savefig('img.png', dpi=500, bbox_inches='tight')
            rgba = np.asarray(fig.canvas.buffer_rgba())
            image = PIL.Image.fromarray(rgba)
            # image = PIL.Image.open('img.png')
            images.append(image)
            captions.append(f'Self Attention')
            
        avg_source_attn = np.mean(m_attn_block.numpy(), axis=(0,1))
        attn_by_token = zip(tokenised_reaction, avg_source_attn)
        mol = Chem.MolFromSmiles(reaction)
        atom_colors = {}
        atom_finder = re.compile(r"(Cl?|Br?|[NOSPFIbcnosp*]|\[[^]]+\])", re.X)
        atoms = atom_finder.findall(reaction)
        first = next(attn_by_token)
        atom_ids = []
        atom_weights = []
        for atom, atom_str in zip(mol.GetAtoms(), atoms):
            while first[0] != atom_str:
                first = next(attn_by_token)
            atom_ids.append(atom.GetIdx())
            atom_weights.append(float(first[1]))
            first = next(attn_by_token)

        max_weight = max(atom_weights)
        atom_weights = [(w / max_weight) ** 2 for w in atom_weights]

        d = rdMolDraw2D.MolDraw2DCairo(500, 500)
        SimilarityMaps.GetSimilarityMapFromWeights(mol, atom_weights, draw2d=d)
        images.append(d.GetDrawingText())
        captions.append(f'Attention on\nSource Molecule')

        avg_source_attn = np.mean(s_attn_block.numpy(), axis=(0,1))
        sample = data[reaction]['sample'].rstrip('?')
        canon_sample = canonicalize(sample)
        if canon_sample is not None:
            tokenised_sample_ = tokenise(sample)
            attn_by_token = zip(tokenised_sample_, avg_source_attn)
            mol = Chem.MolFromSmiles(sample)
            atom_colors = {}
            atom_finder = re.compile(r"(Cl?|Br?|[NOSPFIbcnosp*]|\[[^]]+\])", re.X)
            atoms = atom_finder.findall(sample)
            first = next(attn_by_token)
            atom_ids = []
            atom_weights = []
            for atom, atom_str in zip(mol.GetAtoms(), atoms):
                while first[0] != atom_str:
                    first = next(attn_by_token)
                atom_ids.append(atom.GetIdx())
                atom_weights.append(float(first[1]))            
                first = next(attn_by_token)

            max_weight = max(atom_weights)
            atom_weights = [(w / max_weight) ** 2 for w in atom_weights]

            d = rdMolDraw2D.MolDraw2DCairo(500, 500)
            SimilarityMaps.GetSimilarityMapFromWeights(mol, atom_weights, draw2d=d)
            images.append(d.GetDrawingText())
            captions.append(f'Attention on\nOutput Molecule')

        all_images.append(images)
        all_captions.append(captions)

        # container = st.container(border=True)
        # with container:
        #     columns = st.columns(4)
        #     for i, (image, caption) in enumerate(zip(images, captions)):
        #         with columns[i]:
        #             st.image(image, caption=caption)
    

    for k in range(2):        
        overfig = plt.figure(constrained_layout=True, figsize=(15, len(all_images) // 2 * 5.5))
        figs = overfig.subfigures(len(all_images) // 2)
        for i, fig in enumerate(figs):
            fig.suptitle(f'Layer {k*len(all_images)//2+i+1}', fontsize=24)
            axes = fig.subplots(1, 5, width_ratios=[1, 1, 0.2, 1, 1])
            images, captions = all_images[k*len(all_images)//2+i], all_captions[k*len(all_images)//2+i]
            images.insert(2, None)
            captions.insert(2, None)
            for j, (image, caption) in enumerate(zip(images, captions)):
                if j == 2:
                    axes[j].axis('off')
                    continue
        
                if isinstance(image, PIL.Image.Image):
                    image = np.asarray(image)
                    axes[j].imshow(image)
                else:
                    image = PIL.Image.open(io.BytesIO(image))
                    axes[j].imshow(image)
                axes[j].set_title(caption, fontsize=16)

                if j < 2:
                    axes[j].spines['top'].set_visible(False)
                    axes[j].spines['right'].set_visible(False)
                    axes[j].spines['bottom'].set_visible(False)
                    axes[j].spines['left'].set_visible(False)
                    axes[j].set_yticks([])
                    axes[j].set_xticks([])
                    axes[j].set_xlabel('Input Sequence', fontsize=14)
                    axes[j].set_ylabel('Output Sequence', fontsize=14)
                    axes[j].set_title(caption, fontsize=16, pad=20)
                else:
                    axes[j].axis('off')

                if j == 0:
                    axes[j].set_ylabel('Output Sequence', fontsize=14, labelpad=-70)

        plt.savefig(f'attention_maps_{k}.png', dpi=300, bbox_inches='tight')
        st.pyplot(overfig)
    