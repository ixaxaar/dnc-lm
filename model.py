import torch.nn as nn
from torch.autograd import Variable

from dnc import DNC


class RNNModel(nn.Module):
  """Container module with an encoder, a recurrent module, and a decoder."""

  def __init__(
      self,
      rnn_type,
      ntoken,
      ninp,
      nhid,
      nlayers,
      dropout=0.5,
      tie_weights=False,
      nr_cells=5,
      read_heads=2,
      cell_size=10,
      gpu_id=-1,
      independent_linears=True
  ):
    super(RNNModel, self).__init__()
    self.drop = nn.Dropout(dropout)
    self.encoder = nn.Embedding(ntoken, ninp)
    if rnn_type in ['LSTM', 'GRU']:
      self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
    elif rnn_type.lower() == 'dnc':
      if nhid != ninp:
        raise ValueError('When using DNC units, nhid must be equal to emsize')
      self.rnn = DNC(
          'lstm',
          hidden_size=nhid,
          num_layers=nlayers,
          nr_cells=nr_cells,
          read_heads=read_heads,
          cell_size=cell_size,
          gpu_id=gpu_id,
          independent_linears=independent_linears
      )
    else:
      try:
        nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
      except KeyError:
        raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
      self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
    self.decoder = nn.Linear(nhid, ntoken)

    # Optionally tie weights as in:
    # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
    # https://arxiv.org/abs/1608.05859
    # and
    # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
    # https://arxiv.org/abs/1611.01462
    if tie_weights:
      if nhid != ninp:
        raise ValueError('When using the tied flag, nhid must be equal to emsize')
      self.decoder.weight = self.encoder.weight

    self.init_weights()

    self.rnn_type = rnn_type
    self.nhid = nhid
    self.nlayers = nlayers

  def init_weights(self):
    initrange = 0.1
    self.encoder.weight.data.uniform_(-initrange, initrange)
    self.decoder.bias.data.fill_(0)
    self.decoder.weight.data.uniform_(-initrange, initrange)

  def forward(self, input, hidden, reset_experience=False):
    emb = self.drop(self.encoder(input))
    if self.rnn_type.lower() == 'dnc':
      emb = emb.transpose(0, 1)
      output, hidden = self.rnn(emb, hidden, reset_experience=reset_experience)
      output = output.transpose(0, 1)
    else:
      output, hidden = self.rnn(emb, hidden)
    output = self.drop(output)
    decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
    return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

  def init_hidden(self, bsz):
    weight = next(self.parameters()).data
    if self.rnn_type == 'LSTM':
      return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
              Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
    elif self.rnn_type.lower() == 'dnc':
      return None
    else:
      return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
