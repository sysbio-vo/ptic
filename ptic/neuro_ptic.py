from gettext import npgettext
import torch
from tqdm import tqdm 
from pmi_tfidf_classifier import get_doc_tfidf
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_pmi_vectors(words_pmis, word2text_count, tokenized_text, N):
    lpw = len(words_pmis[0])
    word2id = {w:id for id, w in enumerate(words_pmis[0].keys())}
    vectors = torch.zeros(len(tokenized_text), lpw*2)
    for idx, words in tqdm(enumerate(tokenized_text)):
      word2tfidf = get_doc_tfidf(words, word2text_count, N)
      counter = 0
      for word in words:
        if word in words_pmis[0]:
         for wpid in range(len(words_pmis)):
            pmi = words_pmis[wpid][word]*word2tfidf[word]
            vectors[idx][lpw*wpid + word2id[word]]=pmi
         counter+=1
      # vectors[idx][lpw]=counter/len(words)
    return vectors

class net(torch.nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1):
        super(net, self).__init__()
        
        self.layer1 = torch.nn.Linear(in_features, out_features)
        self.drop = torch.nn.Dropout(p=dropout)
        # self.act = torch.nn.ReLU()
        # self.layer2 = torch.nn.Linear(n_hidden_neurons, out_features)
        self.act_out = torch.nn.Sigmoid()
    
    def forward(self, x):
        x = self.drop(x)
        x = self.layer1(x)
        # x = self.act(x)
        # x = self.layer2(x)
        x = self.act_out(x)
        return x

#train neural net
def train(X, Y, X_test, Y_test, wc, nl, batch_size=13, epochs=87):
    loss = torch.nn.CrossEntropyLoss()
    dili_net = net(wc*nl, 100, 1)
    dili_net.to(device)
    optimizer = torch.optim.AdamW(dili_net.parameters(), 
                             lr=1.0e-3)
    for epoch in tqdm(range(epochs)):
        loss_vals=0
        order = np.random.permutation(len(X))
        for start_index in range(0, len(X), batch_size):
            optimizer.zero_grad()
            batch_indices = order[start_index:start_index+batch_size]
            x_batch = X[batch_indices]
            y_batch = Y[batch_indices]
            preds = dili_net.forward(x_batch)
            
            loss_value = loss(preds, y_batch.float())
            loss_vals+=loss_value.item()
            loss_value.backward()
            optimizer.step()
        print('\n Loss value', loss_vals)

        if epoch % 100 == 0:
            test_preds = dili_net.forward(X_test)
            #torch argmax returns the index of the max value in the tensor
            test_preds = torch.argmax(test_preds, dim=1)
            # test_preds = torch.argmax(test_preds, axis=1)
  
            print((test_preds.squeeze() == Y_test).float().mean().cpu().numpy())

    return dili_net

def get_dili_net_results(dili_net, words_pmis, word2text_count, tokenized_text, N, min_diff=0.5):
  X = get_pmi_vectors(words_pmis, word2text_count, tokenized_text, N)
  preds = dili_net.forward(X.to(device))
  preds = torch.where(preds > min_diff, 1, 0).cpu().numpy()
  return preds
  