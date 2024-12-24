import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re

def read_glove_vecs(glove_file):
    """
    read pretrained embeddings
    returns three dicts
    words_to_index: mapping words to indices
    index_to_words: mapping indices to words
    word_to_vec_map: mapping words to vectors
    """
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map



def pre_process(x, Tx, word_to_index):
    """
    pre-process data and return X of shape (m, Tx) padded and contain indices
    x: shape (m,) -> m samples with a string of text for each sample
    Tx: longest sequence in all samples
    word_to_index: dict mapping from word to index from the pretrained embeddings
    """
    # from sentences to words
    all_words = np.empty((x.shape[0], Tx), dtype='<U15')
    for i, sentence in enumerate(x):
        words = sentence.split()
        for j, word in enumerate(words):
            all_words[i, j] = word

    # remove punctuation and 'br' tags
    for i, sentence in enumerate(all_words):
        for j, word in enumerate(sentence):
            w = re.sub(r"(?:[^\w]|<br/?\s*|<br)+", "", word)
            all_words[i, j] = re.sub(r"^br|br$", "", w)

    # words to indices
    X = np.zeros((x.shape[0], Tx))
    for i, sentence in enumerate(all_words):
        for j, word in enumerate(sentence):

            if word.lower() in word_to_index:
                X[i, j] = word_to_index[word.lower()]

    # shifts all padding to the right only
    for i in range(X.shape[0]):
        non_zero = X[i][X[i] != 0]
        num_zeros = Tx - len(non_zero)
        X[i] = np.concatenate([non_zero, np.zeros(num_zeros, dtype=X.dtype)])

    return X