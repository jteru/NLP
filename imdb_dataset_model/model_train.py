from collections import Counter
from matplotlib import pyplot as plt
import math
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

import nltk
nltk.download('wordnet')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# data processing
data = pd.read_csv('~/nlp/imdb_dataset_model/IMDB Dataset.csv').to_numpy()

lemmatizer = WordNetLemmatizer()

def pre_process(input_sent):
  words = word_tokenize(input_sent.lower())
  filtered_words = [word.lower() for word in words if word.isalnum()]
  # filtered_words = [lemmatizer.lemmatize(word.lower()) for word in filtered_words]
  return ' '.join(filtered_words)

def build_vocabulary(sents):
  all_words = []

  for sent in sents:
    all_words += sent.split()
    
  vocab = Counter(all_words)
  vocab_dict = {}
  for v, c in vocab.items():
    if c > 10:
      vocab_dict[v] = len(vocab_dict)
  
  return vocab_dict

def get_text_features(processed_sents, vocab):
  text_vectors, max_seq_len = [], 0
  for sent in processed_sents:
    word_ids_in_sent = [vocab[word] for word in sent.split() if word in vocab]
    text_vectors.append(word_ids_in_sent)
    max_seq_len = max(max_seq_len, len(word_ids_in_sent))

  # padding
  for text_vector in text_vectors:
    padding = [len(vocab)] * (max_seq_len - len(text_vector))
    text_vector += padding # after this, len(text_vector) = max_seq_len

  return np.array(text_vectors)


np.random.shuffle(data)

X, y = data[:, 0], np.array([1 if d == 'positive' else 0 for d in data[:, 1]])

x_train = X[:int(0.5*len(X))]
y_train = y[:int(0.5*len(y))]


x_valid = X[int(0.5*len(X)) : int(0.5*len(X)) +  int(0.25*len(X))]
y_valid = y[int(0.5*len(X)) : int(0.5*len(X)) +  int(0.25*len(X))]


x_test = X[int(0.5*len(X)) +  int(0.25*len(X)) :]
y_test = y[int(0.5*len(X)) +  int(0.25*len(X)) :]


print('preprocessing data...')
processed_sents_train = [pre_process(sent) for sent in tqdm(x_train)]
processed_sents_test = [pre_process(sent) for sent in tqdm(x_test)]
processed_sents_valid = [pre_process(sent) for sent in tqdm(x_valid)]

print('building vocabulary...')
vocab = build_vocabulary(processed_sents_train)

print('computing features...')
x_train = get_text_features(processed_sents_train, vocab)
x_test = get_text_features(processed_sents_test, vocab)
x_valid = get_text_features(processed_sents_valid, vocab)

x_train, y_train, x_test, y_test, x_valid, y_valid = map(
    torch.LongTensor, (x_train, y_train[:, np.newaxis], x_test, y_test[:, np.newaxis], x_valid, y_valid[:, np.newaxis])
)

train_dataset = TensorDataset(x_train, y_train)

dev_name = 'cpu'
device = torch.device(dev_name)

# model definition
class mlp_classifier(nn.Module):
  def __init__(self, vocab_len):
    super().__init__()

    self.emb = nn.Embedding(vocab_len + 1, 300, padding_idx=vocab_len)

    self.layer2 = nn.Linear(300, 500)
    self.layer3 = nn.Linear(500 ,1)
    

  def forward(self, xb):
    h =  self.emb(xb) # 3d
    h = torch.mean(h, dim=1) # 2d
    h = F.relu(h)
    h =  self.layer2(h) # pre-activation outputs are called 'logits'
    h = F.relu(h)
    h = self.layer3(h)
    out = torch.sigmoid(h) # only addition
    return out
  
  def predict(self, xb):
    with torch.no_grad():
      out = self.forward(xb) # shape of out: batch_size, 1; each element between [0, 1]
    return (out >= 0.5).type(torch.uint8)
  
model = mlp_classifier(len(vocab)).to(device)

# params
lr = 0.01  # learning rate
epochs = 25  # how many epochs to train for
bs = 30  # batch size
patience = 5

# loss function
loss_func = F.binary_cross_entropy

# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

# model evaluation
test_data_set = TensorDataset(x_test,y_test)
test_data_loader = DataLoader(test_data_set, batch_size = bs)
accuracy_test = 0 
for xb, yb in test_data_loader:
    y_pred = model.predict(xb.to(device))
    accuracy_test += accuracy_score(yb, y_pred.cpu()).item()
print('Test accuracy of model before training: ', (accuracy_test / len(test_data_loader)))


valid_data_set = TensorDataset(x_valid, y_valid)
valid_data_loader = DataLoader(valid_data_set, batch_size = bs)


# training loop
train_dataset = TensorDataset(x_train, y_train)
train_data_loader = DataLoader(train_dataset, batch_size=bs, shuffle = True)


count = 0
max_accuracy = 0

for epoch in range(epochs):
    total_loss = 0
    for xb, yb in tqdm(train_data_loader):
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = loss_func(pred, yb.float())
        loss.backward()
        
        total_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
    print('Loss at epoch %d : %f' % (epoch, total_loss / len(train_data_loader)))
    
    accuracy_valid = 0 
    for xv, yv in valid_data_loader:
        y_pred_v = model.predict(xv.to(device))
        accuracy_valid += accuracy_score(yv, y_pred_v.cpu()).item()
    print("Valid set accuracy after epoch %d : %f" % (epoch,accuracy_valid / len(valid_data_loader)))
    
    if accuracy_valid <= max_accuracy:
        count += 1
        if count == patience:
            break
    else:
        file = 'model_train.pth'
        torch.save(model, file)

        max_accuracy = max(max_accuracy, accuracy_valid)
        count = 0

# model evaluation
test_data_set = TensorDataset(x_test,y_test)
test_data_loader = DataLoader(test_data_set, batch_size = bs)
accuracy_test = 0 
for xb, yb in test_data_loader:
    y_pred = model.predict(xb.to(device))
    accuracy_test += accuracy_score(yb, y_pred.cpu())
print('Test accuracy of model before training: ', (accuracy_test / len(test_data_loader)))
