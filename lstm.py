import numpy as np
import sys
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import LSTM
import utils
from keras.preprocessing.sequence import pad_sequences



FREQ_DIST_FILE = '../train-processed-freqdist.pkl'
BI_FREQ_DIST_FILE = '../train-processed-freqdist-bi.pkl'
TRAIN_PROCESSED_FILE = '../train-processed.csv'
TEST_PROCESSED_FILE = '../test-processed.csv'
GLOVE_FILE = 'glove-seeds.txt'
dim = 200


def get_glove_vectors(vocab):
    glove_vectors = {}
    found = 0
    with open(GLOVE_FILE, 'r') as glove_file:
        for i, line in enumerate(glove_file):
            tokens = line.split()
            word = tokens[0]
            if vocab.get(word):
                vector = [float(e) for e in tokens[1:]]
                glove_vectors[word] = np.array(vector)
                found += 1
    return glove_vectors


def get_feature_vector(comment):
    words = comment.split()
    feature_vector = []
    for i in range(len(words) - 1):
        word = words[i]
        if vocab.get(word) is not None:
            feature_vector.append(vocab.get(word))
    if len(words) >= 1:
        if vocab.get(words[-1]) is not None:
            feature_vector.append(vocab.get(words[-1]))
    return feature_vector


def process_comments(csv_file, test_file=True,run=False):
    comments = []
    labels = []
    
    if(run):
        for comment in csv_file:

            feature_vector=get_feature_vector(comment)
            comments.append(feature_vector)
    with open(csv_file, 'r') as csv:
        lines = csv.readlines()
        total = len(lines)
        for i, line in enumerate(lines):
            if test_file:
                id, comment = line.split(',')
            else:
                id, comment, sentiment = line.split(',')
            feature_vector = get_feature_vector(comment)
            if test_file:
                comments.append(feature_vector)
            else:
                comments.append(feature_vector)
                labels.append(int(1 if sentiment=="N" else 0))
            
    
    return comments, np.array(labels)



np.random.seed(1337)
vocab_size = 90000
batch_size = 500
max_length = 40
filters = 600
kernel_size = 3
vocab = utils.top_n_words(FREQ_DIST_FILE, vocab_size, shift=1)
glove_vectors = get_glove_vectors(vocab)
comments, labels = process_comments(TRAIN_PROCESSED_FILE, test_file=False)
embedding_matrix = np.random.randn(vocab_size + 1, dim) * 0.01
for word, i in vocab.items():
    glove_vector = glove_vectors.get(word)
    if glove_vector is not None:
        embedding_matrix[i] = glove_vector
comments = pad_sequences(comments, maxlen=max_length, padding='post')
shuffled_indices = np.random.permutation(comments.shape[0])
comments = comments[shuffled_indices]
labels = labels[shuffled_indices]
    
model = Sequential()
model.add(Embedding(vocab_size + 1, dim, weights=[embedding_matrix], input_length=max_length))
model.add(Dropout(0.4))
model.add(LSTM(128))
model.add(Dense(64))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.000001)
model.fit(comments, labels, batch_size=128, epochs=5, validation_split=0.1, shuffle=True, callbacks=[reduce_lr])