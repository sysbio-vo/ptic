from ptic import pmi_tfidf_classifier as ptah
import numpy as np
import pandas as pd

path = "../datasets/"

# Data is taken from here
# https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
data_raw = pd.read_csv(path + 'IMDB_Dataset.csv')
indices = np.random.permutation(data_raw.index)
data = data_raw.loc[indices]
data = data_raw.sample(frac=1)
data = data.replace(to_replace=['negative', 'positive'], value=[0, 1])

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
results = ptah.classify_pmi_based(words_pmis, word2text_count, tokenized_test_texts, N)
precision = np.sum( np.logical_and(results, targets_test) ) / np.sum(results)
recall = np.sum( np.logical_and(results, targets_test) ) / np.sum(targets_test)
F1 = 2 * (recall * precision)/(recall + precision)
accuracy = (results == targets_test).mean()

print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1: ", F1)