from transformers import BertModel, BertTokenizer
import re
import torch.nn as nn
import torch
import argparse
import streamlit as st
import pandas as pd
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
# 1DCNN definition
class CNNOD(nn.Module):
    def __init__(self):
        super(CNNOD, self).__init__()
        self.conv1 = nn.Conv1d(1024, 1024, kernel_size=7, stride=1, padding=3)
        self.norm1 = nn.BatchNorm1d(1024)
        self.conv2 = nn.Conv1d(1024, 128, kernel_size=5, stride=1, padding=2)
        self.norm2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1)
        self.norm3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 2, kernel_size=5, stride=1, padding=2)
        self.head = nn.Softmax(-1)
        self.act = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.act(x)
        x = self.conv4(x)
        x = x.permute(0, 2, 1)
        return self.head(x)


# load instructions
parse = argparse.ArgumentParser()
parse.add_argument('--ligand', '-l', type=str, help='Ligand type, including DNA, RNA, and antibody',
                   default='DNA', choices=['DNA', 'RNA', 'AB'])
parse.add_argument('--threshold', '-t', type=float, help='Threshold of classification score', default=0.5)
parse.add_argument('--input', '-i', help='Input protein sequences in FASTA format', required=True)
parse.add_argument('--output', '-o', help='Output file path, default clape_result.txt',
                   default='clape_result.txt')
parse.add_argument('--cache', '-c', help='Path for saving cached pre-trained model', default='protbert')

args = parse.parse_args()

# parameter judge
if args.threshold > 1 or args.threshold < 0:
    raise ValueError("Threshold is out of range.")

# input sequences
input_file = open(args.input, 'r').readlines()
seq_ids = []
seqs = []
for line in input_file:
    if line.startswith('>'):
        seq_ids.append(line.strip())
    else:
        seqs.append(line.strip())
if len(seq_ids) != len(seqs):
    raise ValueError("FASTA file is not valid.")

# feature generation
print("=====Loading pre-trained protein language model=====")
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False, cache_dir=args.cache)
pretrain_model = BertModel.from_pretrained("Rostlab/prot_bert", cache_dir=args.cache)
print("Done!")


def get_protein_features(seq):
    sequence_Example = ' '.join(seq)
    sequence_Example = re.sub(r"[UZOB]", "X", sequence_Example)
    encoded_input = tokenizer(sequence_Example, return_tensors='pt')
    last_hidden = pretrain_model(**encoded_input).last_hidden_state.squeeze(0)[1:-1, :]
    return last_hidden.detach()


# generate sequence feature
features = []
print("=====Generating protein sequence feature=====")
for s in seqs:
    features.append(get_protein_features(s).unsqueeze(0))
print("Done!")

# load CNN model
print("=====Loading classification model=====")
predictor = CNNOD()
if args.ligand == 'DNA':
    predictor.load_state_dict(torch.load("./weights/DNA.pth"))
elif args.ligand == 'RNA':
    predictor.load_state_dict(torch.load("./weights/RNA.pth"))
elif args.ligand == 'AB':
    predictor.load_state_dict(torch.load("./weights/AB.pth"))
else:
    raise ValueError(args.ligand)
print("Done!")

# prediction process
results = []
print(f"=====Predicting {args.ligand}-binding sites=====")
predictor.eval()

### FIND SEQ:


for f in features:
    out = predictor(f).squeeze(0).detach().numpy()[:, 1]
    score = ''.join([str(1) if x > args.threshold else str(0) for x in out])
    results.append(score)
print("Done!")
s1=""
s2=""
s3=""
for i in range(len(seq_ids)):
    # s1=seq_ids[i]
    s2 = seqs[i]
    s3 =results[i]
    # st.header(s1)
    # st.header(s2)
    # st.header(s3)

#### DATAFRAME


import streamlit as st
import pandas as pd

maindf = pd.DataFrame()
# Input sequences and annotations
st.title("🧬 Protein Binding Site Viewer")
# Mapping of single-letter amino acid codes to full names
amino_acid_names = {
    'A': 'Alanine',
    'R': 'Arginine',
    'N': 'Asparagine',
    'D': 'Aspartic acid',
    'C': 'Cysteine',
    'E': 'Glutamic acid',
    'Q': 'Glutamine',
    'G': 'Glycine',
    'H': 'Histidine',
    'I': 'Isoleucine',
    'L': 'Leucine',
    'K': 'Lysine',
    'M': 'Methionine',
    'F': 'Phenylalanine',
    'P': 'Proline',
    'S': 'Serine',
    'T': 'Threonine',
    'W': 'Tryptophan',
    'Y': 'Tyrosine',
    'V': 'Valine'
}

for i in range(len(seq_ids)):
    sequences = {}
    seq_l=[]
    seq_l.append(seq_ids[i])
    s1 = seq_ids[i]      # this is your sequence ID
    s2 = seqs[i]         # this is the sequence
    s3 = results[i]      # this is the annotation

    # build dictionary entry
    sequences[s1] = {
        "sequence": s2,
        "annotation": s3
    }
    # Create combined DataFrame
    combined_df = pd.DataFrame()

    for seq_id, data in sequences.items():
        sequence = list(data["sequence"])
        annotation = list(data["annotation"])
        df = pd.DataFrame({
            "Sequence ID": [seq_id] * len(sequence),
            "Position": list(range(1, len(sequence) + 1)),
            "Amino Acid": sequence,
            "Binding Site": annotation
        })
    combined_df = pd.concat([combined_df, df], ignore_index=True)
    maindf = pd.concat([maindf, df], ignore_index=True)

# st.dataframe(maindf)
maindf["Amino Acid Name"] = maindf["Amino Acid"].map(amino_acid_names)



########### FILTER AND GRAPH OF RESIDUES




st.sidebar.header("🔍 Filter")
selected_seq = st.sidebar.selectbox("Choose sequence", ["All"] + [">seq1",">seq2"])

if selected_seq == "All":
    st.dataframe(maindf)
    if st.button("Check Residues"):
        sequence_df = maindf

        # Streamlit UI
        st.title("🧬 ProBERT Binding Site Visualizer")

        # Show full DataFrame
        st.markdown("### Full Protein Sequence Data")
        st.dataframe(sequence_df)

        # Filter only binding residues for graph
        binding_df = sequence_df[sequence_df["Binding Site"] == '1'].reset_index(drop=True)

        # Plotting only binding residues
        st.markdown("### 🔴 Binding Residues Visualization")
        # st.header(binding_df.shape)
        if binding_df.shape[0]!=0:
            colors = ['red'] * len(binding_df)

            fig, ax = plt.subplots(figsize=(len(binding_df) * 0.6, 2))  # Dynamic width based on residue count

            for idx, (pos, aa) in enumerate(zip(binding_df["Position"], binding_df["Amino Acid"])):
                ax.text(idx, 0, f'{aa}\n{pos}', ha='center', va='center', fontsize=14, color='white',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='red', edgecolor='black'))

            ax.set_xlim(-1, len(binding_df))
            ax.set_ylim(-1, 1)
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.success("NO RESIDUES FIND")

    # Get sequence from DataFrame
    sequence = "".join(maindf["Amino Acid"].tolist())
    positions = maindf["Position"]

    # Get features and predictions
    features = get_protein_features(sequence).unsqueeze(0)
    scores = predictor(features).squeeze(0).detach().numpy()[:, 1]  # Confidence for class 1

    # Apply threshold to classify binding (1) and non-binding (0) residues
    binding_predictions = [1 if score > args.threshold else 0 for score in scores]

    # Count binding and non-binding residues
    binding_count = sum(binding_predictions)
    non_binding_count = len(binding_predictions) - binding_count

    # Create Pie Chart
    labels = ['Binding Residues', 'Non-Binding Residues']
    sizes = [binding_count, non_binding_count]
    colors = ['#66b3ff', '#ff6666']


    # Function to format the labels with both count and percentage
    def func(pct, allsizes):
        absolute = int(pct / 100. * sum(allsizes))
        return f"{absolute} ({pct:.1f}%)"


    # Exploding the binding slice
    explode = (0.1, 0)  # "explode" the binding slice a little

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(sizes, labels=labels, autopct=lambda pct: func(pct, sizes), colors=colors, startangle=90, explode=explode)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Title for the Pie Chart
    ax.set_title("Binding vs Non-Binding Residues in Protein Sequence")

    # Show the plot
    st.pyplot(fig)

    ##score graph

    st.markdown("### 📈 Prediction Scores for All Residues")
    # Get the scores again from the model
    score_array = \
        predictor(
            torch.tensor([get_protein_features("".join(maindf["Amino Acid"].tolist())).numpy()])).detach().numpy()[
            0]
    scores = score_array[:, 1]  # Class 1 confidence
    maindf["Binding Score"] = scores
    fig_score, ax_score = plt.subplots(figsize=(12, 3))
    ax_score.plot(maindf["Position"], scores, marker='o', color='blue', label='Binding Score (Class 1)')
    ax_score.axhline(y=args.threshold, color='red', linestyle='--', label=f'Threshold ({args.threshold})')

    ax_score.set_xlabel("Residue Position")
    ax_score.set_ylabel("Score")
    ax_score.set_title("Binding Site Prediction Scores")
    ax_score.legend()
    st.pyplot(fig_score)

    csv = maindf.to_csv(index=False)

    # Add download button
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name='protein_binding_sites.csv',
        mime='text/csv'
    )
else:
    filtered_df = maindf[maindf["Sequence ID"] == selected_seq]
    st.dataframe(filtered_df)
    if st.button("Check Residues"):
        sequence_df = filtered_df

        # Streamlit UI
        st.title("🧬 ProBERT Binding Site Visualizer")

        # Show full DataFrame
        st.markdown("### Full Protein Sequence Data")
        st.dataframe(sequence_df)

        # Filter only binding residues for graph
        binding_df = sequence_df[sequence_df["Binding Site"] == '1'].reset_index(drop=True)

        # Plotting only binding residues
        st.markdown("### 🔴 Binding Residues Visualization")
        if binding_df.shape[0] != 0:
            colors = ['red'] * len(binding_df)

            fig, ax = plt.subplots(figsize=(len(binding_df) * 0.6, 2))  # Dynamic width based on residue count

            for idx, (pos, aa) in enumerate(zip(binding_df["Position"], binding_df["Amino Acid"])):
                ax.text(idx, 0, f'{aa}\n{pos}', ha='center', va='center', fontsize=14, color='white',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='red', edgecolor='black'))

            ax.set_xlim(-1, len(binding_df))
            ax.set_ylim(-1, 1)
            ax.axis('off')

            st.pyplot(fig)
        else:
            st.success("NO RESIDUES FIND")
    sequence = "".join(filtered_df["Amino Acid"].tolist())
    positions = filtered_df["Position"]

    # Get features and predictions
    features = get_protein_features(sequence).unsqueeze(0)
    scores = predictor(features).squeeze(0).detach().numpy()[:, 1]  # Confidence for class 1

    # Apply threshold to classify binding (1) and non-binding (0) residues
    binding_predictions = [1 if score > args.threshold else 0 for score in scores]

    # Count binding and non-binding residues
    binding_count = sum(binding_predictions)
    non_binding_count = len(binding_predictions) - binding_count

    # Create Pie Chart
    labels = ['Binding Residues', 'Non-Binding Residues']
    sizes = [binding_count, non_binding_count]
    colors = ['#66b3ff', '#ff6666']


    # Function to format the labels with both count and percentage
    def func(pct, allsizes):
        absolute = int(pct / 100. * sum(allsizes))
        return f"{absolute} ({pct:.1f}%)"


    # Exploding the binding slice
    explode = (0.1, 0)  # "explode" the binding slice a little

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(sizes, labels=labels, autopct=lambda pct: func(pct, sizes), colors=colors, startangle=90, explode=explode)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Title for the Pie Chart
    ax.set_title("Binding vs Non-Binding Residues in Protein Sequence")

    # Show the plot
    st.pyplot(fig)
    ##score graph

    st.markdown("### 📈 Prediction Scores for All Residues")
    # Get the scores again from the model
    score_array = \
        predictor(
            torch.tensor([get_protein_features("".join(filtered_df["Amino Acid"].tolist())).numpy()])).detach().numpy()[0]
    scores = score_array[:, 1]  # Class 1 confidence
    filtered_df["Binding Score"] = scores
    fig_score, ax_score = plt.subplots(figsize=(12, 3))
    ax_score.plot(filtered_df["Position"], scores, marker='o', color='blue', label='Binding Score (Class 1)')
    ax_score.axhline(y=args.threshold, color='red', linestyle='--', label=f'Threshold ({args.threshold})')

    ax_score.set_xlabel("Residue Position")
    ax_score.set_ylabel("Score")
    ax_score.set_title("Binding Site Prediction Scores")
    ax_score.legend()
    st.pyplot(fig_score)

    csv = filtered_df.to_csv(index=False)

    # Add download button
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name='protein_binding_sites.csv',
        mime='text/csv'
    )