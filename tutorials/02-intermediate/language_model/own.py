import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from data_utils import Dictionary, Corpus

# Hyper Parameter
embed_size = 128
hidden_size = 1024
num_layers = 1
num_epochs = 10
num_samples = 1000
batch_size = 100
seq_length = 30
lerning_rate = 2e-3

train_path = './data/train.txt'
sample_path = './data/sample.txt'
corpus = Corpus()
ids = corpus.get_data(train_path, batch_size)
vocab_size = len(corpus.dictionary)
num_batches = ids.size(1) // seq_length

class RNNLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNLM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()
    
    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-0.1, 0.1)
    
    def forward(self, x, h):
        # Embed word ids to vectors
        x = self.embed(x)
        # Forward propagate RNN
        out, h = self.lstm(x, h)
        # Reshape output to (batchsize * sequence_length, hidden_size)
        out = out.contiguous().view(out.size(0) * out.size(1), out.size(2))
        # Decode hidden states pf all time step
        out = self.linear(out)
        return out, h
rnnlm = RNNLM(vocab_size, embed_size, hidden_size, num_layers)
rnnlm.cuda()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnnlm.parameters(), lr=lerning_rate)

# Truncated Baclpropagation
def detach(states):
    return [state.detach() for state in states]

# Train
for epoch in range(num_epochs):
    states = (Variable(torch.zeros(num_layers, batch_size, hidden_size)).cuda(),
                Variable(torch.zeros(num_layers, batch_size, hidden_size)).cuda())
    for i in range(0, ids.size(1) - seq_length, seq_length):
        inputs = Variable(ids[:, i:i+seq_length]).cuda()
        targets = Variable(ids[:, (i+1):(i+1)+seq_length].contiguous()).cuda()

        rnnlm.zero_grad()
        states = detach(states)
        outputs, states = rnnlm(inputs, states)
        loss = criterion(outputs, targets.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm(rnnlm.parameters(), 0.5)
        optimizer.step()

        step = (i + 1) // seq_length
        if step % 100 == 0:
            print("Epoch {}/{}, Step: {}/{}, Loss: {}, Perplexity: {}"
                .format(epoch+1, num_epochs, step, num_batches, loss.data[0], np.exp(loss.data[0])))

# Sampling
with open(sample_path, 'w') as f:
    # Set initial hidden one memory states
    state = (Variable(torch.zeros(num_layers, 1, hidden_size)).cuda(),
            Variable(torch.zeros(num_layers, 1, hidden_size)).cuda())
    # Select one word id randomly
    prob = torch.one(vocab_size)
    input = Variable(torch.multinomial(prob, num_samples=1).unsqueeze(1),
        volatile=True).cuda()

    for i in range(num_samples):
        # Forward propagate rnn
        output, state = rnnlm(input, state)
        # Sample a word id
        prob = output.squeeze().data.exp().cpu()
        word_id = torch.multinomial(prob, 1)[0]
        # Feed sampled word id to next time step
        input.data.fill_(word_id)
        # File write
        word = corpus.dictionary.idx2word[word_id]
        word = '\n' if word == '<eos>' else word + ' '
        f.write(word)

        if (i+1) % 100 ==0:
            print("Sampled {}/{} words and save to {}".format(i+1, num_samples, sample_path))
    
