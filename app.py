from flask import Flask, request, render_template, send_file
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense, Input
from tensorflow.keras import regularizers, Model
from itertools import product
import re
#%%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=INFO, 1=WARN, 2=ERROR, 3=FATAL
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations (optional)
app = Flask(__name__)

# Ensure 'static' directory exists for storing DDS plot and promoter subsequences file
if not os.path.exists("static"):
    os.makedirs("static")

# Function to generate k-mers
def generate_kmers(sequence, k):
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

# Function to encode sequences using k-mer frequency
def kmer_encode(sequences, k):
    possible_kmers = [''.join(p) for p in product('ACGT', repeat=k)]
    kmer_vectors = []

    for sequence in sequences:
        kmers = generate_kmers(sequence, k)
        kmer_freq = {kmer: 0 for kmer in possible_kmers}
        for kmer in kmers:
            if kmer in kmer_freq:
                kmer_freq[kmer] += 1
        total_kmers = len(kmers)
        kmer_vector = [count / total_kmers for count in kmer_freq.values()]
        kmer_vectors.append(kmer_vector)
    return np.array(kmer_vectors)

# Function to build CNN model
def get_model(input_shape):
    inputs = Input(shape=input_shape)
    convLayer = Conv1D(filters=16, kernel_size=2, activation='relu', kernel_regularizer=regularizers.l2(1e-5))(inputs)
    poolingLayer1 = MaxPooling1D(pool_size=2, strides=2)(convLayer)
    convLayer2 = Conv1D(filters=16, kernel_size=2, activation='relu', kernel_regularizer=regularizers.l2(1e-5))(poolingLayer1)
    poolingLayer2 = MaxPooling1D(pool_size=4, strides=2)(convLayer2)
    flattenLayer = Flatten()(poolingLayer2)
    dropoutLayer = Dropout(0.22)(flattenLayer)
    denseLayer2 = Dense(100, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(dropoutLayer)
    dropoutLayer1 = Dropout(0.25)(denseLayer2)
    outLayer = Dense(1, activation='sigmoid')(dropoutLayer1)
    model = Model(inputs=inputs, outputs=[outLayer])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Load the model
sequence_length = 100
k = 6
input_shape = (4**k, 1)
model_weights_path = 'CNN_model_3_19_mar.h5'#'D:/Archea Models/render_v6/CNN_model_3_19_mar.h5'
model = get_model(input_shape)
model.load_weights(model_weights_path)

# DDS Encoding Dictionary
dds_dict = {
    'AA': -1.0, 'AC': -1.45, 'AG': -1.3, 'AT': -0.88,
    'CA': -1.45, 'CC': -1.28, 'CG': -2.24, 'CT': -1.28,
    'GA': -1.3, 'GC': -2.24, 'GG': -1.84, 'GT': -1.44,
    'TA': -0.58, 'TC': -1.28, 'TG': -1.44, 'TT': -1.0,
    'NN': 0
}

# Function to compute DDS encoding
def compute_dds(sequence):
    """Compute DDS encoding for a given DNA sequence."""
    dds_values = []
    for i in range(len(sequence) - 1):
        dinucleotide = sequence[i:i+2]
        dds_values.append(dds_dict.get(dinucleotide, 0))  # Default to 0 if not found
    return dds_values

# Function to read sequences from file (FASTA or text)
def read_sequences(file_path):
    sequences = []
    if not os.path.exists(file_path):
        return sequences
    with open(file_path, "r") as file:
        lines = file.readlines()
        sequence = ""
        for line in lines:
            if line.startswith(">"):  # FASTA header
                if sequence:
                    sequences.append(sequence)
                    sequence = ""
            else:
                sequence += line.strip().upper()
        if sequence:
            sequences.append(sequence)
    return sequences

def read_sequences2(file_path):
    """Reads sequences from a text file and computes DDS encoding."""
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return []
    
    dds_profiles = []
    with open(file_path, "r") as file:
        for line in file:
            seq = line.strip().upper()
            if len(seq) < 60:
                print(f" \n Warning: Skipping short sequence in {file_path}")
                continue
            dds_profiles.append(compute_dds(seq))
    
    print(f"Total sequences read from {file_path}: {len(dds_profiles)}")
    return dds_profiles

# Function to split sequence into 80bp subsequences
def split_sequence(sequence):
    subsequences = []
    for i in range(0, len(sequence), 100):
        subsequence = sequence[i:i+100]
        if len(subsequence) < 80:
            continue
        if len(subsequence) < 80:
            subsequence += 'N' * (80 - len(subsequence))
        subsequences.append(subsequence)
    return subsequences

# Function to predict subsequences and highlight promoters
def predict_and_highlight(sequence):
    subsequences = split_sequence(sequence)
    highlighted_sequence = ""
    promoter_indices = []
    for i, subseq in enumerate(subsequences):
        prediction = model.predict(np.expand_dims(kmer_encode([subseq], k), axis=2))
        prediction = np.where(prediction > 0.75, np.round(prediction), prediction)
        if prediction == 1:
            highlighted_sequence += f"<span class='highlight'>{subseq}</span>"
            promoter_indices.append(i)
        else:
            highlighted_sequence += subseq
    return highlighted_sequence, promoter_indices

# Function to extract promoter subsequences from highlighted_sequence
def extract_promoter_subsequences(highlighted_sequence):
    # Use regex to find all subsequences wrapped in <span class='highlight'> tags
    promoter_subsequences = re.findall(r"<span class='highlight'>(.*?)</span>", highlighted_sequence)
    return promoter_subsequences

def generate_dds_plot(default_dds_profiles, user_dds_profiles, plot_path):
    # Read sequences from both files
 
    # Handle empty cases
    if not default_dds_profiles or not user_dds_profiles:
        plt.figure(figsize=(10, 5))
        if not default_dds_profiles and not user_dds_profiles:
            plt.text(0.5, 0.5, 'No sequences found in either dataset', 
                    ha='center', va='center', fontsize=12)
        elif not default_dds_profiles:
            plt.text(0.5, 0.5, 'No sequences found in default dataset', 
                    ha='center', va='center', fontsize=12)
        else:
            plt.text(0.5, 0.5, 'No promoter sequences found in user input', 
                    ha='center', va='center', fontsize=12)
        plt.axis('off')
        plt.savefig(plot_path)
        plt.close()
        return
    
    # Calculate max length for alignment
    max_length = max(
        max(len(profile) for profile in default_dds_profiles),
        max(len(profile) for profile in user_dds_profiles)
    )
    max2=max(len(profile) for profile in default_dds_profiles)
    max3=max(len(profile) for profile in user_dds_profiles)
    # Pad sequences to match the max length
    padded_default = [profile + [np.nan] * (max_length - len(profile)) for profile in default_dds_profiles]
    padded_user = [profile + [np.nan] * (max_length - len(profile)) for profile in user_dds_profiles]
    

    # Create matrices
    default_matrix = np.array(padded_default)
    user_matrix = np.array(padded_user)
    
    # Compute mean DDS values
    mean_default = np.nanmean(default_matrix, axis=0)
    mean_user = np.nanmean(user_matrix, axis=0)
    
    # Compute Pearson's correlation coefficient
    valid_indices = ~np.isnan(mean_default) & ~np.isnan(mean_user)
    if np.any(valid_indices):
        pcc_value, _ = pearsonr(mean_default[valid_indices], mean_user[valid_indices])
    else:
        pcc_value = np.nan
    
    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(mean_default), ), mean_default, color='blue', label='Validation Data')
    plt.plot(range(len(mean_user), ), mean_user, color='red', label='Predicted Data')
    plt.axhline(0, linestyle='dashed', color='black', alpha=0.5)
    
    plt.xlabel("Nucleotide Position", fontsize=12)
    plt.ylabel("DDS Value (kcal/mol-bp^-1)", fontsize=12)
    plt.title("DDS Profile Comparison", fontsize=14)

    # Add text annotations
    if not np.isnan(pcc_value):
        plt.text(0.05, 0.95, f"Pearson's r = {pcc_value:.2f}", 
                transform=plt.gca().transAxes, fontsize=12,
                bbox=dict(facecolor='white', alpha=0.7))
    
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/load_example', methods=['GET'])
def load_example():
    # Path to your example sequences file
    example_file_path = 'example_sequence.txt'
    try:
        with open(example_file_path, 'r') as file:
            example_sequences = file.read()
        return example_sequences
    except FileNotFoundError:
        return "Error: Example sequences file not found.", 404

@app.route('/predict', methods=['POST'])
def predict():
    sequences = []
    uploaded_file_path = None

    # Handle textbox input (FASTA or plain text)
    if 'sequence' in request.form and request.form['sequence'].strip():
        input_text = request.form['sequence'].strip()
        if input_text.startswith(">"):  # FASTA format
            sequences.extend([seq.strip().upper() for seq in input_text.split(">") if seq.strip()])
        else:  # Plain text (one sequence per line)
            sequences.extend([seq.strip().upper() for seq in input_text.splitlines() if seq.strip()])

    # Handle file upload (FASTA or plain text)
    if 'file' in request.files:
        file = request.files['file']
        if file and (file.filename.endswith('.txt') or file.filename.endswith('.fasta')):
            uploaded_file_path = os.path.join("static", "user_sequences.txt")
            file.save(uploaded_file_path)
            if file.filename.endswith('.fasta'):  # FASTA format
                sequences.extend(read_sequences(uploaded_file_path))
            else:  # Plain text (one sequence per line)
                with open(uploaded_file_path, "r") as f:
                    sequences.extend([line.strip().upper() for line in f if line.strip()])

    if not sequences:
        return render_template('index.html', error="Please enter a sequence or upload a file.")

    results = []
    highlighted_sequences = []  # List to store highlighted sequences
    promoter_subsequences = []  # List to store promoter subsequences

    for seq in sequences:
        highlighted_sequence, promoter_indices = predict_and_highlight(seq)
        results.append((seq, highlighted_sequence, promoter_indices))
        highlighted_sequences.append(highlighted_sequence)

        # Extract promoter subsequences from highlighted_sequence
        promoter_subsequences.extend(extract_promoter_subsequences(highlighted_sequence))

    promoters_count = sum(len(indices) for _, _, indices in results)
    non_promoters_count = sum(len(split_sequence(seq)) - len(indices) for seq, _, indices in results)

    # Save promoter_subsequences to a file
    # promoter_subsequences_path = os.path.join("static", "promoter_subsequences.txt")
    # with open(promoter_subsequences_path, "w") as f:
    #     for subseq in promoter_subsequences:
    #         f.write(subseq + "\n")
    promoter_subsequences_path = os.path.join("static", "promoter_subsequences.fasta")  # Changed extension to .fasta
    with open(promoter_subsequences_path, "w") as f:
        for i, subseq in enumerate(promoter_subsequences, start=1):
            f.write(f">Promoter_{i}\n")  # FASTA header with sequential numbering
            f.write(subseq + "\n")
    # Generate DDS plot for default dataset and promoter subsequences
    default_dataset_path = "all_promoters_v2.txt"
    default_dds_profiles = read_sequences2(default_dataset_path)
    user_dds_profiles = read_sequences2(promoter_subsequences_path)
    dds_plot_path = os.path.join("static", "dds_plot.png")
    generate_dds_plot(default_dds_profiles, user_dds_profiles, dds_plot_path)

    return render_template('index.html', 
                           results=results, 
                           promoters_count=promoters_count, 
                           non_promoters_count=non_promoters_count,
                           dds_plot=dds_plot_path,
                           promoter_subsequences_file=promoter_subsequences_path)

@app.route('/download_promoter_subsequences')
def download_promoter_subsequences():
    promoter_subsequences_path = request.args.get('file')
    return send_file(promoter_subsequences_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
