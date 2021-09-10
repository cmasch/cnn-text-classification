# -*- coding: utf-8 -*-
"""
Utils

@author: Christopher Masch
"""

import urllib3
import os
import re
import string
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from io import BytesIO
from zipfile import ZipFile
from nltk.corpus import stopwords


def clean_doc(doc):
    """
    Cleaning a document by several methods:
        - Lowercase
        - Removing whitespaces
        - Removing numbers
        - Removing stopwords
        - Removing punctuations
        - Removing short words

    Arguments:
        doc : Text

    Returns:
        str : Cleaned text
    """
    
    #stop_words = set(stopwords.words('english'))
    
    # Lowercase
    doc = doc.lower()
    # Remove numbers
    #doc = re.sub(r"[0-9]+", "", doc)
    # Split in tokens
    tokens = doc.split()
    # Remove Stopwords
    #tokens = [w for w in tokens if not w in stop_words]
    # Remove punctuation
    #tokens = [w.translate(str.maketrans('', '', string.punctuation)) for w in tokens]
    # Tokens with less then two characters will be ignored
    #tokens = [word for word in tokens if len(word) > 1]
    return ' '.join(tokens)


def read_files(path):
    """
    Read in files of a given path.
    This can be a directory including many files or just one file.
    
    Arguments:
        path : Filepath to file(s)

    Returns:
        documents : Return a list of cleaned documents
    """
    
    documents = list()
    # Read in all files in directory
    if os.path.isdir(path):
        for filename in os.listdir(path):
            with open(f"{path}/{filename}") as f:
                doc = f.read()
                doc = clean_doc(doc)
                documents.append(doc)
    
    # Read in all lines in one file
    if os.path.isfile(path):        
        with open(path, encoding='iso-8859-1') as f:
            doc = f.readlines()
            for line in doc:
                documents.append(clean_doc(line))
                
    return documents


def char_vectorizer(X, char_max_length, char2idx_dict):
    """
    Vectorize an array of word sequences to char vector.
    Example (length 15): [test entry] --> [[1,2,3,1,4,2,5,1,6,7,0,0,0,0,0]]

    Arguments:
        X               : Array of word sequences
        char_max_length : Maximum length of vector
        char2idx_dict   : Dictionary of indices for converting char to integer

    Returns:
        str2idx : Array of vectorized char sequences
    """
    
    str2idx = np.zeros((len(X), char_max_length), dtype='int64')
    for idx, doc in enumerate(X):
        max_length = min(len(doc), char_max_length)
        for i in range(0, max_length):
            c = doc[i]
            if c in char2idx_dict:
                str2idx[idx, i] = char2idx_dict[c]
    return str2idx


def create_glove_embeddings(embedding_dim, max_num_words, max_seq_length, tokenizer):
    """
    Load and create GloVe embeddings.

    Arguments:
        embedding_dim : Dimension of embeddings (e.g. 50,100,200,300)
        max_num_words : Maximum count of words in vocabulary
        max_seq_length: Maximum length of vector
        tokenizer     : Tokenizer for converting words to integer

    Returns:
        tf.keras.layers.Embedding : Glove embeddings initialized in Keras Embedding-Layer
    """
    
    print("Pretrained GloVe embedding is loading...")
    
    if not os.path.exists("data"):
        os.makedirs("data")
    
    if not os.path.exists("data/glove"):
        print("No previous embeddings found. Will be download required files...")
        os.makedirs("data/glove")
        http = urllib3.PoolManager()
        response = http.request(
            url     = "http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip",
            method  = "GET",
            retries = False
        )

        with ZipFile(BytesIO(response.data)) as myzip:
            for f in myzip.infolist():
                with open(f"data/glove/{f.filename}", "wb") as outfile:
                    outfile.write(myzip.open(f.filename).read())
                    
        print("Download of GloVe embeddings finished.")

    embeddings_index = {}
    with open(f"data/glove/glove.6B.{embedding_dim}d.txt") as glove_embedding:
        for line in glove_embedding.readlines():
            values = line.split()
            word   = values[0]
            coefs  = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
    
    print(f"Found {len(embeddings_index)} word vectors in GloVe embedding\n")

    embedding_matrix = np.zeros((max_num_words, embedding_dim))

    for word, i in tokenizer.word_index.items():
        if i >= max_num_words:
            continue
        
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return tf.keras.layers.Embedding(
        input_dim    = max_num_words,
        output_dim   = embedding_dim,
        input_length = max_seq_length,
        weights      = [embedding_matrix],
        trainable    = True,
        name         = "word_embedding"
    )


def plot_acc_loss(title, histories, key_acc, key_loss):
    """
    Generate a plot for visualizing accuracy and loss

    Arguments:
        title     : Title of visualization
        histories : Array of Keras metrics per run and epoch
        key_acc   : Key of accuracy (accuracy, val_accuracy)
        key_loss  : Key of loss (loss, val_loss)
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    # Accuracy
    ax1.set_title(f"Model accuracy ({title})")
    names = []
    for i, model in enumerate(histories):
        ax1.plot(model[key_acc])
        ax1.set_xlabel("epoch")
        names.append(f"Model {i+1}")
        ax1.set_ylabel("accuracy")
    ax1.legend(names, loc="lower right")
    
    # Loss
    ax2.set_title(f"Model loss ({title})")
    for model in histories:
        ax2.plot(model[key_loss])
        ax2.set_xlabel('epoch')
        ax2.set_ylabel('loss')
    ax2.legend(names, loc='upper right')
    fig.set_size_inches(20, 5)
    plt.show()

    
def visualize_features(ml_classifier, nb_neg_features=15, nb_pos_features=15):
    """
    Visualize trained coefficient of log regression in respect to vectorizer.

    Arguments:
        ml_classifier   : ML-Pipeline including vectorizer as well as trained model
        nb_neg_features : Number of features to visualize
        nb_pos_features : Number of features to visualize
    """

    feature_names = ml_classifier.get_params()['vectorizer'].get_feature_names()
    coef = ml_classifier.get_params()['classifier'].coef_.ravel()

    print('Extracted features: {}'.format(len(feature_names)))

    pos_coef = np.argsort(coef)[-nb_pos_features:]
    neg_coef = np.argsort(coef)[:nb_neg_features]
    interesting_coefs = np.hstack([neg_coef, pos_coef])

    # Plot
    plt.figure(figsize=(20, 5))
    colors = ['red' if c < 0 else 'green' for c in coef[interesting_coefs]]
    plt.bar(np.arange(nb_neg_features + nb_pos_features), coef[interesting_coefs], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(
        np.arange(nb_neg_features+nb_pos_features),
        feature_names[interesting_coefs],
        size     = 15,
        rotation = 75,
        ha       = 'center'
    );
    plt.show()
        
