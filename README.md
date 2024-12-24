# Movie-Reviews-Sentiment-Analysis

A Tensorflow implementation of a multi-layer LSTM sentiment classifier.

## Architecture

This model is designed to classify movie reviews as either positive or negative. It consists of two stacked LSTM layers. The input embeddings are derived from GloVe pre-trained vectors. Sequences are padded to ensure uniform length and a masking layer is applied to inform the model to ignore padding tokens during computation. The architecture includes a fully connected dense layer with a sigmoid activation function at the output, ensuring the predictions are in the range of 0 to 1, where values closer to 1 indicate positive sentiment.

## Training
The model takes as input a 2D tensor of shape (batch_size, sequence_length) representing tokenized and padded movie reviews. The sequence_length is fixed during preprocessing. The output is a 1D tensor of shape (batch_size,), where each value represents the probability of the corresponding review being positive.

