import numpy as np
import nltk
import string
from nltk.tokenize import word_tokenize
from collections import defaultdict
nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(nltk.corpus.stopwords.words('english') + list(string.punctuation))

def tokenize(s):
    return [i for i in word_tokenize(s.lower()) if i not in stop_words]

def tokenization(train_data, var_name):
    tokenized_texts = []
    for _, row in train_data.iterrows():
        text = str(row[var_name])
        words = tokenize(text)
        tokenized_texts.append(words)
    return tokenized_texts

def get_word_stat(tokenized_texts):
    '''
    Input: List[list[str]] - tokenized texts
    Return: Dict[word] - number of documents with `word`
    '''
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

def get_class_stat(targets):
    target2count = defaultdict(int)
    for target in targets:
        target2count[target]+=1
    return target2count;

def create_pmi_dict(tokenized_texts, targets, min_count=5):
    np.seterr(divide = 'ignore')
    ts = set(targets)
    target2count = get_class_stat(targets)
    ttc = sum(target2count.values())
    target2percent = {t:target2count[t]/ttc for t in ts}
    # words count
    d = {'tot':defaultdict(int)}
    d.update({t:defaultdict(int) for t in ts})
    Dictionary = set()
    for idx, words in enumerate(tokenized_texts):
        target = targets[idx]
        for w in words:
            d['tot'][w] += 1
            Dictionary.add(w)
            d[ target ][w] += 1
    # pmi calculation
    for t in ts:
      N_0 = sum(d[t].values())
      N = sum(d['tot'].values())
      d[t] = {w: -np.log((v/N + 10**(-15)) / (target2percent[t] * d['tot'][w]/(N))) / np.log(v/N + 10**(-15))
            for w, v in d[t].items() if d['tot'][w] > min_count}
      d[t]=dict(sorted(d[t].items(),key= lambda x:x[1], reverse=True))
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
        #PMI - determines significance of the word for the class
        #TFIDF - determines significance of the word for the document
        #tot_pmi = create_tot_pmitfidf(words, words_pmis, word2tfidf)
        tot_pmi = {}
        pmi = {}
        for k in words_pmis:
            tot_pmi[k] = [ words_pmis[k][w] * word2tfidf[w] for w in set(words) if w in words_pmis[k] ]
            pmi[k] = np.sum(tot_pmi[k])
        results[idx] = np.argmax(list(pmi.values()))
    return results
