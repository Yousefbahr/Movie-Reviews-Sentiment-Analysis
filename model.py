from tensorflow.keras.layers import Input, Embedding, Dense, Dropout ,LSTM
from tensorflow.keras.models import Sequential
import tensorflow as tf
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load embeddings
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('glove.6B.50d.txt')

dataset = pd.read_csv("IMDB Dataset.csv")

x = dataset["review"].values # ->  shape(m, Tx of longest sentence in dataset)
y = np.where(dataset["sentiment"].values == "positive", 1, 0)# 1 for positive, 0 for negative

# longest sequence
Tx = 2470

X = pre_process(x, Tx, word_to_index)

X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# fill the embedding matrix with pretrained embeddings
vocab_size = len(word_to_index) + 1
embedding_matrix = np.zeros((vocab_size, 50))
for word, index in word_to_index.items():
  embedding_matrix[index, :] = word_to_vec_map[word]


embedding_layer = Embedding(vocab_size, 50, weights= [embedding_matrix], trainable=False, mask_zero=True )

sentence_indices = Input((Tx,), dtype='int32')

model = Sequential([
    Input((Tx,), dtype='int32'), # indices
    embedding_layer, # indices to embeddings
    LSTM(64, return_sequences=True),
    Dropout(0.5),
    LSTM(64, return_sequences=False),
    Dense(1, activation='sigmoid')
])

model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])

with tf.device('/GPU:0'):
  history = model.fit(X_train, y_train, epochs=50, shuffle=True, verbose=1)

preds = np.where(model.predict(X_test) > 0.5, 1, 0)

print(f"Accuracy on test set: {accuracy_score(y_test, preds)}")




