from ptic import pmi_tfidf_classifier as ptah
import numpy as np
import pandas as pd
from sklearn import metrics
import torch
from ptic import neuro_ptic

np.random.seed(42)
path = "doc/datasets/"

# Data is taken from here
# https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
data_raw = pd.read_csv(path + 'IMDB_Dataset.csv')
indices = np.random.permutation(data_raw.index)
data = data_raw.loc[indices]
data = data_raw.sample(frac=1)
labels_names = ['negative', 'positive']
nl = len(labels_names)

data = data.replace(to_replace=labels_names, value=[0, 1])

idx = int(data.shape[0] * 0.1)
test_data = data.iloc[:idx]
train_data = data.iloc[idx:]
targets_train = train_data["sentiment"].values
targets_test = test_data["sentiment"].values

tokenized_texts = ptah.tokenization(train_data, 'review')
tokenized_test_texts = ptah.tokenization(test_data, 'review')
N = len(tokenized_texts)

word2text_count = ptah.get_word_stat(tokenized_texts)
words_pmis = ptah.create_pmi_dict(tokenized_texts, targets_train, min_count=5)

words = set()
for wpid in words_pmis:
    words.update(words_pmis[wpid].keys())
word2id = {w:id for id, w in enumerate(words)}
lpw = len(words)

X = neuro_ptic.get_pmi_vectors(words_pmis, word2text_count, word2id, tokenized_texts, N, lpw, nl)
X_test = neuro_ptic.get_pmi_vectors(words_pmis, word2text_count, word2id, tokenized_test_texts, N, lpw, nl)
Y = torch.from_numpy(targets_train)
Y_test = torch.from_numpy(targets_test)
dili_net = neuro_ptic.train(X=X, Y=Y, X_test=X_test, Y_test=Y_test, wc = lpw, nl = nl, batch_size=100, epochs=300)

results = ptah.classify_pmi_based(words_pmis, word2text_count, tokenized_test_texts, N)

results_net = neuro_ptic.get_dili_net_results(dili_net, words_pmis, word2id, word2text_count, tokenized_test_texts, N, lpw, nl)

print('Results of the conventional ptic classifier:')
print(metrics.classification_report(results, targets_test, digits=3, target_names=labels_names))

print('Results of the neural network classifier:')
print(metrics.classification_report(results_net, targets_test, digits=3, target_names=labels_names))

