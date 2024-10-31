from flask import Flask, request, render_template
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense, Input
from tensorflow.keras import regularizers, Model
from itertools import product

# Print versions
#print("Flask version:", flask.__version__)
#print("NumPy version:", np.__version__)
#print("TensorFlow version:", tf.__version__)
#import seaborn as sns
#%%
# Initialize Flask app
app = Flask(__name__)

# Load the CNN model and its weights
#model = load_model('D:/Archea Models/Project/model/cnn_model.h5')



# Function to read sequences from a file and generate labels
def read_sequences(file_path, sequence_length):
    sequences = []
    with open(file_path, 'r') as file:
        for line in file:
            sequence = line.strip().upper().ljust(sequence_length, 'N')  # Pad sequences if needed
            sequences.append(sequence)
    #labels = [label] * len(sequences)
    return sequences#, labels


# Function to build the CNN model
def get_model(input_shape):
    inputs = Input(shape=input_shape)
      # Add Gaussian noise with standard deviation 0.1
    convLayer = Conv1D(filters=8, kernel_size=2, activation='relu', kernel_regularizer=regularizers.l2(1e-5))(inputs)#(noisy_inputs)
    poolingLayer1 = MaxPooling1D(pool_size=2, strides=2)(convLayer)
    convLayer2 = Conv1D(filters=16, kernel_size=2, activation='relu', kernel_regularizer=regularizers.l2(1e-5))(poolingLayer1)
    poolingLayer2 = MaxPooling1D(pool_size=4, strides=2)(convLayer2)
    flattenLayer = Flatten()(poolingLayer2)
    dropoutLayer = Dropout(0.20)(flattenLayer)
    denseLayer2 = Dense(99, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(dropoutLayer)
    dropoutLayer1 = Dropout(0.22)(denseLayer2)
    outLayer = Dense(1, activation='sigmoid')(dropoutLayer1)
    model = Model(inputs=inputs, outputs=[outLayer])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model



# Function to generate k-mers from sequences
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

def predict_promoter(sequence, model, k):
    encoded_sequence = kmer_encode([sequence], k)
    encoded_sequence = np.expand_dims(encoded_sequence, axis=2)  # Reshape for CNN
    prediction = model.predict(encoded_sequence).round()
    
    return 'Promoter' if prediction == 1 else 'NON-Promoter'
sequence_length = 100
k=6
input_shape = (4**k, 1)
model_weights_path = 'D:/Archea Models/Project/model/model_cnn.weights.h5'
    
# Load model and weights
model = get_model(input_shape)
model.load_weights(model_weights_path)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sequences = []

    # Check if a sequence was entered directly
    if 'sequence' in request.form and request.form['sequence'].strip():
        sequences.append(request.form['sequence'].strip().upper())

    # Check if a file was uploaded
    if 'file' in request.files:
        file = request.files['file']
        if file and file.filename.endswith('.txt'):
            # Read each line in the file, strip whitespace, and convert to uppercase
            #sequences.extend([line.strip().upper() for line in file if line.strip()])
            sequences.extend([line.decode('utf-8').strip().upper() for line in file if line.strip()])


    # Make predictions for each sequence
    results = [(seq, predict_promoter(seq,model, k)) for seq in sequences]

    #results = []  # This would be your prediction results
    promoters_count = 0  # Initialize the count
    non_promoters_count = 0  # Initialize the count
    
    # Logic to count promoters and non-promoters based on results
    for seq, result in results:
        if result == 'Promoter':
            promoters_count += 1
        else:
            non_promoters_count += 1
            
    return render_template('index.html', 
                           results=results, 
                           promoters_count=promoters_count, 
                           non_promoters_count=non_promoters_count)

if __name__ == '__main__':
    app.run(debug=True)