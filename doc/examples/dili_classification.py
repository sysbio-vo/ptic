from ptic import pmi_tfidf_classifier as ptah
import numpy as np
import pandas as pd
from sklearn import metrics
import torch
from ptic import neuro_ptic

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.random.seed(42)
path = "../datasets/"

data_raw = pd.read_csv(path + 'dili_raw.csv')
indices = np.random.permutation(data_raw.index)
data = data_raw.loc[indices]
data = data_raw.sample(frac=1)
data['Documents'] = data['Title'].map(str) + '. ' + data['Abstract'].map(str)
labels = data_raw.Label.unique()
nl = len(labels)
idx = int(data.shape[0] * 0.1)
test_data = data.iloc[:idx]
train_data = data.iloc[idx:]
targets_train = train_data["Label"].values
targets_test = test_data["Label"].values

tokenized_texts = ptah.tokenization(train_data, 'Documents')
tokenized_test_texts = ptah.tokenization(test_data, 'Documents')
N = len(tokenized_texts)

word2text_count = ptah.get_word_stat(tokenized_texts)
words_pmis = ptah.create_pmi_dict(tokenized_texts, targets_train, min_count=5)

X = neuro_ptic.get_pmi_vectors(words_pmis, word2text_count, tokenized_texts, N)
X_test = neuro_ptic.get_pmi_vectors(words_pmis, word2text_count, tokenized_test_texts, N)
Y = torch.from_numpy(targets_train).to(device)
Y_test = torch.from_numpy(targets_test).to(device)
dili_net = neuro_ptic.train(X=X.to(device), Y=Y, X_test=X_test.to(device), Y_test=Y_test, wc = len(words_pmis[0]), nl = nl, batch_size=100, epochs=1000)

results = ptah.classify_pmi_based(words_pmis, word2text_count, tokenized_test_texts, N)

results_net = neuro_ptic.get_dili_net_results(dili_net, words_pmis, word2text_count, tokenized_test_texts, N, min_diff=0.0)

print(metrics.classification_report(results, targets_test, digits=3, target_names=labels))

print(metrics.classification_report(results_net, targets_test, digits=3, target_names=labels))
