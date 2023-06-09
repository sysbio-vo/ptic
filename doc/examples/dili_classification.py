from ptic import pmi_tfidf_classifier as ptah
import numpy as np
import pandas as pd
from sklearn import metrics
import torch
from ptic import logit_ptic

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.random.seed(42)
path = "doc/datasets/"

data_raw = pd.read_csv(path + 'DILI_data.csv')
indices = np.random.permutation(data_raw.index)
data = data_raw.loc[indices]
data = data_raw.sample(frac=1)
data['Documents'] = data['Title'].map(str) + '. ' + data['Abstract'].map(str)
labels = data_raw.Label.unique()
labels_names = {str(labels[i]): i for i in range(len(labels))}
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

words = set()
for wpid in words_pmis:
    words.update(words_pmis[wpid].keys())
word2id = {w:id for id, w in enumerate(words)}
lpw = len(words)

X = logit_ptic.get_pmi_vectors(words_pmis, word2text_count, tokenized_texts, N)
X_test = logit_ptic.get_pmi_vectors(words_pmis, word2text_count, tokenized_test_texts, N)
Y = torch.from_numpy(targets_train)
Y_test = torch.from_numpy(targets_test)
ptic_logit = logit_ptic.train(X=X, Y=Y, X_test=X_test, Y_test=Y_test, wc = lpw, nl = nl, batch_size=100, epochs=300)

results = ptah.classify_pmi_based(words_pmis, word2text_count, tokenized_test_texts, N) 

results_net = logit_ptic.get_ptic_logit_results(ptic_logit, words_pmis, word2text_count, tokenized_test_texts, N)


print('Results of the conventional ptic classifier:')
print(metrics.classification_report(results, targets_test, digits=3, target_names=labels_names))

print('Results of the neural network classifier:')
print(metrics.classification_report(results_net, targets_test, digits=3, target_names=labels_names))
