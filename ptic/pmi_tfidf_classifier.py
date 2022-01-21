import pandas as pd
import numpy as np
import spacy
from tqdm import tqdm
from collections import defaultdict

nlp = spacy.load("en_core_sci_lg", disable=['ner', 'parser'])

def tokenize(string):
    doc = nlp.make_doc(string)
    words = [token.text.lower() for token in doc if token.is_alpha and not token.is_stop and len(token.text) > 1 ]
    return words

def tokenization(train_data, var_name):
    tokenized_texts = []
    #print("Tokenization....")
    for _, row in train_data.iterrows():
        #text = str(row['Abstract'])
        #text = str(row['Title']) + ' ' + str(row['Abstract'])
        text = str(row[var_name])
        words = tokenize(text)
        tokenized_texts.append(words)
    return tokenized_texts

# TFIDF (Term frequency and inverse document frequency)
def get_word_stat(tokenized_texts):
    '''Words counts in documents
    finds in how many documents this word
    is present
    '''
    texts_number = len(tokenized_texts)
    #print("Word Stat....")
    word2text_count = defaultdict(int)
    for text in tokenized_texts:
        uniquewords = set(text)
        for word in uniquewords:
            word2text_count[word] +=1
    return word2text_count

def get_doc_tfidf(words, word2text_count, N):
    num_words = len(words)
    word2tfidf = defaultdict(int)
    for word in words:
        if word2text_count[word] > 0:
            idf = np.log(N/(word2text_count[word]))
            word2tfidf[word] += (1/num_words) * idf
        else:
            word2tfidf[word] = 1
    return word2tfidf

def create_pmi_dict(tokenized_texts, targets, min_count=5):
    #print("PMI dictionary ....")
    np.seterr(divide = 'ignore')
    # words count
    d = {0:defaultdict(int), 1:defaultdict(int), 'tot':defaultdict(int)}
    for idx, words in enumerate(tokenized_texts):
        target = targets[idx]
        for w in words:
            d[ target ][w] += 1
    Dictionary = set(list(d[0].keys()) + list(d[1].keys()))
    d['tot'] = {w:d[0][w] + d[1][w] for w in Dictionary}
    # pmi calculation
    N_0 = sum(d[0].values())
    N_1 = sum(d[1].values())
    d[0] = {w: -np.log((v/N_0 + 10**(-15)) / (0.5 * d['tot'][w]/(N_0 + N_1))) / np.log(v/N_0 + 10**(-15))
            for w, v in d[0].items() if d['tot'][w] > min_count}
    d[1] = {w: -np.log((v/N_1+ 10**(-15)) / (0.5 * d['tot'][w]/(N_0 + N_1))) / np.log(v/N_1 + 10**(-15))
            for w, v in d[1].items() if d['tot'][w] > min_count}
    del d['tot']
    return d


def calc_collinearity(word, words_dict, n=10):
    new_word_emb = nlp(word).vector
    pmi_new = 0
    max_pmis_words = sorted(list(words_dict.items()), key=lambda x: x[1], reverse=True)[:n]
    for w, pmi in max_pmis_words:
        w_emb = nlp(w).vector
        cos_similarity = \
        np.dot(w_emb, new_word_emb)/(np.linalg.norm(w_emb) * np.linalg.norm(new_word_emb) + 1e-12)
        pmi_new += cos_similarity * pmi
    return pmi_new / n


def create_tot_pmitfidf(words, words_pmis, word2tfidf):
    tot_pmitfidf0 = []
    tot_pmitfidf1 = []
    for word in words:
        if word in words_pmis[0]:
            tot_pmitfidf0.append( words_pmis[0][word] * word2tfidf[word] )
        else:
            pmi0idf = pmiidf_net.forward( nlp(word).vector )
            #pmi0 = calc_collinearity(word, words_pmis[0])
            tot_pmitfidf0.append( pmi0 )
        if word in words_pmis[1]:
            tot_pmitfidf1.append( words_pmis[1][word] * word2tfidf[word] )
        else:
            pmi1 = calc_collinearity(word, words_pmis[1])
            tot_pmitfidf1.append( pmi1 )

    return tot_pmitfidf0, tot_pmitfidf1


def classify_pmi_based(words_pmis, word2text_count, tokenized_test_texts, N):
    results = np.zeros(len(tokenized_test_texts))
    for idx, words in enumerate(tokenized_test_texts):
        word2tfidf = get_doc_tfidf(words, word2text_count, N)
        # PMI - determines significance of the word for the class
        # TFIDF - determines significance of the word for the document
        #tot_pmi0, tot_pmi1 = create_tot_pmitfidf(words, words_pmis, word2tfidf)
        tot_pmi0 = [ words_pmis[0][w] * word2tfidf[w] for w in set(words) if w in words_pmis[0] ]
        tot_pmi1 = [ words_pmis[1][w] * word2tfidf[w] for w in set(words) if w in words_pmis[1] ]
        pmi0 = np.sum(tot_pmi0)
        pmi1 = np.sum(tot_pmi1)
        diff = pmi1 - pmi0
        if diff > 0.001:
            results[idx] = 1
    return results
