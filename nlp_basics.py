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
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

def pre_process(input_sent):
  words = word_tokenize(input_sent.lower())
  filtered_words = [word.lower() for word in words if word.isalnum()]
  # filtered_words = [lemmatizer.lemmatize(word) for word in filtered_words]
  # filtered_words = [ps.stem(word) for word in filtered_words]
  return ' '.join(filtered_words)

def build_vocabulary(sents):
  all_words = []

  for sent in sents:
    all_words += sent.split()

  vocab = Counter(all_words)
  vocab_dict = {}
  for k, c in vocab.items():
    if c > 10:
      vocab_dict[k] = len(vocab_dict)
  
  return vocab_dict

def get_text_features(processed_sents, vocab):
  text_vectors = []
  for sent in processed_sents:
    text_vectors.append([vocab[word] for word in sent.split() if word in vocab])

  text_features = np.zeros((len(text_vectors), len(vocab)))
  for i, text_vector in enumerate(text_vectors):
    text_features[i][text_vector] = 1
  
  return text_features

# data processing
data = pd.read_csv('~/nlp/IMDB Dataset.csv').to_numpy()

np.random.shuffle(data)
X, y = data[:, 0], np.array([1 if d == 'positive' else 0 for d in data[:, 1]])

x_train, y_train = X[:int(0.5*len(X))], y[:int(0.5*len(y))]
x_test, y_test = X[int(0.5*len(X)):], y[int(0.5*len(y)):]

processed_sents_train = [pre_process(sent) for sent in x_train]
processed_sents_test = [pre_process(sent) for sent in x_test]

vocab = build_vocabulary(processed_sents_train)

x_train = get_text_features(processed_sents_train, vocab)
x_test = get_text_features(processed_sents_test, vocab)

x_train, y_train, x_test, y_test = map(
    torch.FloatTensor, (x_train, y_train[:, np.newaxis], x_test, y_test[:, np.newaxis])
)

n, inp_dim = x_train.shape

train_dataset = TensorDataset(x_train, y_train)

dev_name = 'cpu'
device = torch.device(dev_name)

# model definition
class mlp_classifier(nn.Module):
  def __init__(self, inp_dim):
    super().__init__()
    self.layer1 = nn.Linear(inp_dim, 800)
    self.layer2 = nn.Linear(800, 1)

  def forward(self, xb):
    h =  self.layer1(xb)
    h = F.tanh(h)
    h =  self.layer2(h) # pre-activation outputs are called 'logits'
    out = F.sigmoid(h) # only addition
    return out
  
  def predict(self, xb):
    with torch.no_grad():
      out = self.forward(xb) # shape of out: batch_size, 1; each element between [0, 1]
    return (out >= 0.5).type(torch.uint8)
  
model = mlp_classifier(inp_dim).to(device)

# params
lr = 0.01  # learning rate
epochs = 25  # how many epochs to train for
bs = 10  # batch size

# loss function
loss_func = F.binary_cross_entropy

# optimizer
optimizer = optim.SGD(model.parameters(), lr=lr)

# model evaluation
y_pred = model.predict(x_test.to(device))
print('Test accuracy of model before training: ', accuracy_score(y_test, y_pred.cpu()))

# training loop
train_data_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)

for epoch in range(epochs):
  for xb, yb in tqdm(train_data_loader):
    xb, yb = xb.to(device), yb.to(device)
    pred = model(xb)
    loss = loss_func(pred, yb)
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
  print('Loss at epoch %d : %f' % (epoch, loss))

# model evaluation
y_pred = model.predict(x_test.to(device))
print('Test accuracy of model after training: ', accuracy_score(y_test, y_pred.cpu()))