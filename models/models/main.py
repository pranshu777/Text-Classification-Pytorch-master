
import os
import time
import load_patents
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from models.selfAttention import SelfAttention
from models.LSTM import LSTMClassifier
from models.LSTM_Attn import AttentionModel
from models.RCNN import RCNN
from models.RNN import RNN
from bayes_opt import BayesianOptimization

def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)

def train_model(model, train_iter, epoch, batch_size, learning_rate):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.cuda()
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=learning_rate)
    steps = 0
    model.train()
    for idx, batch in enumerate(train_iter):
        text = batch.text[0]
        target = batch.label
        target = torch.autograd.Variable(target).long()
        if torch.cuda.is_available():
            text = text.cuda()
            target = target.cuda()
        if (text.size()[0] is not batch_size):# One of the batch returned by BucketIterator has length different than batch_size.
            continue
        optim.zero_grad()
        prediction = model(text)
        loss = F.cross_entropy(prediction, target)
        num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
        acc = 100.0 * num_corrects/len(batch)
        loss.backward()
        clip_gradient(model, 1e-1)
        optim.step()
        steps += 1

        #if steps % 100 == 0:
        #    print (f'Epoch: {epoch+1}, Idx: {idx+1}, Training Loss: {loss.item():.4f}, Training Accuracy: {acc.item(): .2f}%')

        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()

    return total_epoch_loss/len(train_iter), total_epoch_acc/len(train_iter)

def eval_model(model, val_iter, batch_size):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            text = batch.text[0]
            if (text.size()[0] is not batch_size):
                continue
            target = batch.label
            target = torch.autograd.Variable(target).long()
            if torch.cuda.is_available():
                text = text.cuda()
                target = target.cuda()
            prediction = model(text)
            loss = F.cross_entropy(prediction, target)
            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
            acc = 100.0 * num_corrects/len(batch)
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

    return total_epoch_loss/len(val_iter), total_epoch_acc/len(val_iter)

def objective(batch_size, hidden_size, learning_rate):
    batch_size = int(batch_size)
    hidden_size = int(hidden_size)
    TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter = load_patents.load_dataset(batch_size)
    output_size = 2
    embedding_length = 300
    weights = word_embeddings
    #model = LSTMClassifier(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
    #model = AttentionModel(batch_size, output_size, hidden_size, vocab_size, embedding_length, weights)
    #model = RNN(batch_size, output_size, hidden_size, vocab_size, embedding_length, weights)
    #model = RCNN(batch_size, output_size, hidden_size, vocab_size, embedding_length, weights)
    model = SelfAttention(batch_size, output_size, hidden_size, vocab_size, embedding_length, weights)
    loss_fn = F.cross_entropy

    for epoch in range(10):
        #(model, train_iter, epoch, batch_size, learning_rate)
        train_loss, train_acc = train_model(model, train_iter, epoch, batch_size, learning_rate)
        val_loss, val_acc = eval_model(model, valid_iter, batch_size)

    test_loss, test_acc = eval_model(model, test_iter, batch_size)
    print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')

    return test_acc

pbounds = {'batch_size': (32, 64), 'hidden_size': (100, 300), 'learning_rate':(0.001, 0.01)}


optimizer = BayesianOptimization(
    f=objective,
    pbounds=pbounds,
    random_state=1,
)

optimizer.maximize(
    init_points=15,
    n_iter=15,
)

print(optimizer.max)
'''
Let us now predict the sentiment on a single sentence just for the testing purpose.

test_sen1 = "This is one of the best creation of Nolan. I can say, it's his magnum opus. Loved the soundtrack and especially those creative dialogues."
test_sen2 = "Ohh, such a ridiculous movie. Not gonna recommend it to anyone. Complete waste of time and money."

test_sen1 = TEXT.preprocess(test_sen1)
test_sen1 = [[TEXT.vocab.stoi[x] for x in test_sen1]]

test_sen2 = TEXT.preprocess(test_sen2)
test_sen2 = [[TEXT.vocab.stoi[x] for x in test_sen2]]

test_sen = np.asarray(test_sen1)
test_sen = torch.LongTensor(test_sen)
test_tensor = Variable(test_sen, volatile=True)
test_tensor = test_tensor.cuda()
model.eval()
output = model(test_tensor, 1)
out = F.softmax(output, 1)
if (torch.argmax(out[0]) == 1):
    print ("Sentiment: Positive")
else:
    print ("Sentiment: Negative")
'''
