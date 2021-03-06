import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import time

import data
import model

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/penn', help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, DNC)')
parser.add_argument('--emsize', type=int, default=200, help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200, help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2, help='number of layers')
parser.add_argument('--lr', type=float, default=20, help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N', help='batch size')
parser.add_argument('--bptt', type=int, default=35, help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true', help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--cuda', type=int, default=-1, help='Cuda GPU ID, -1 for CPU')
parser.add_argument('--log-interval', type=int, default=200, metavar='N', help='report interval')
parser.add_argument('--save', type=str,  default='model.pt', help='path to save the final model')

parser.add_argument('--nr_cells', type=int, default=5, help='Number of memory cells of the DNC')
parser.add_argument('--read_heads', type=int, default=2, help='Number of read heads of the DNC')
parser.add_argument('--cell_size', type=int, default=10, help='Cell sizes of DNC')
parser.add_argument('--reset_experience', type=str, default='0',
                    help='Whether to reset the DNCs memory after each forward pass')

parser.add_argument('--optim', type=str, default='sgd', help='Optimizer to use, supports sgd, adam, rmsprop')

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
  if args.cuda == -1:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
  else:
    torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)


def batchify(data, bsz):
  # Work out how cleanly we can divide the dataset into bsz parts.
  nbatch = data.size(0) // bsz
  # Trim off any extra elements that wouldn't cleanly fit (remainders).
  data = data.narrow(0, 0, nbatch * bsz)
  # Evenly divide the data across the bsz batches.
  data = data.view(bsz, -1).t().contiguous()
  if args.cuda != -1:
    data = data.cuda(args.cuda)
  return data

eval_batch_size = args.batch_size
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

best_filename = None

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model = model.RNNModel(
    args.model,
    ntokens,
    args.emsize,
    args.nhid,
    args.nlayers,
    args.dropout,
    args.tied,
    nr_cells=args.nr_cells,
    read_heads=args.read_heads,
    cell_size=args.cell_size,
    gpu_id=args.cuda
)
if args.cuda != -1:
  model.cuda(args.cuda)

criterion = nn.CrossEntropyLoss()
if args.optim == 'sgd':
  optim = torch.optim.SGD(model.parameters(), lr=args.lr)
elif args.optim == 'adam':
  optim = torch.optim.Adam(model.parameters(), lr=args.lr)
elif args.optim == 'rmsprop':
  optim = torch.optim.RMSprop(model.parameters(), lr=args.lr, eps=1e-10)

###############################################################################
# Training code
###############################################################################


def repackage_hidden(h):
  """Wraps hidden states in new Variables, to detach them from their history."""
  if type(h) == Variable:
    return Variable(h.data)
  else:
    return tuple(repackage_hidden(v) for v in h)


def repackage_hidden_dnc(h):
  if h is None:
    return None

  (chx, mhxs, _) = h
  chx = repackage_hidden(chx)
  if type(mhxs) is list:
    mhxs = [dict([(k, repackage_hidden(v)) for k, v in mhx.items()]) for mhx in mhxs]
  else:
    mhxs = dict([(k, repackage_hidden(v)) for k, v in mhxs.items()])
  return (chx, mhxs, None)


def get_batch(source, i, evaluation=False):
  seq_len = min(args.bptt, len(source) - 1 - i)
  data = Variable(source[i:i + seq_len], volatile=evaluation)
  target = Variable(source[i + 1:i + 1 + seq_len].view(-1))
  return data, target


def evaluate(data_source, h=None):
  # Turn on evaluation mode which disables dropout.
  model.eval()
  total_loss = 0
  ntokens = len(corpus.dictionary)
  hidden = h if h is not None else model.init_hidden(eval_batch_size)
  for i in range(0, data_source.size(0) - 1, args.bptt):
    data, targets = get_batch(data_source, i, evaluation=True)
    if args.model.lower() == 'dnc':
      output, hidden = model(data, hidden, reset_experience=False)
    else:
      output, hidden = model(data, hidden)
    output_flat = output.view(-1, ntokens)
    total_loss += len(data) * criterion(output_flat, targets).data
    if args.model.lower() != 'dnc':
      hidden = repackage_hidden(hidden)
    else:
      hidden = repackage_hidden_dnc(hidden)
  return total_loss[0] / len(data_source)


def train(h=None):
  # Turn on training mode which enables dropout.
  model.train()
  total_loss = 0
  start_time = time.time()
  ntokens = len(corpus.dictionary)
  hidden = h if h is not None else model.init_hidden(args.batch_size)
  for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
    data, targets = get_batch(train_data, i)
    # Starting each batch, we detach the hidden state from how it was previously produced.
    # If we didn't, the model would try backpropagating all the way to start of the dataset.
    if args.model.lower() != 'dnc':
      hidden = repackage_hidden(hidden)
    else:
      hidden = repackage_hidden_dnc(hidden)
    model.zero_grad()
    optim.zero_grad()
    if args.model.lower() == 'dnc':
      output, hidden = model(data, hidden, reset_experience=False)
    else:
      output, hidden = model(data, hidden)
    loss = criterion(output.view(-1, ntokens), targets)
    loss.backward()

    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
    torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
    # for p in model.parameters():
    #   p.data.add_(-lr, p.grad.data)
    optim.step()

    total_loss += loss.data

    if batch % args.log_interval == 0 and batch > 0:
      cur_loss = total_loss[0] / args.log_interval
      elapsed = time.time() - start_time
      print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.10f} | ms/batch {:5.2f} | '
            'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
      total_loss = 0
      start_time = time.time()

  return hidden

# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
ghx = None

try:
  for epoch in range(1, args.epochs + 1):
    epoch_start_time = time.time()
    ghx = train(ghx)
    val_loss = evaluate(val_data, ghx)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f} '.format(
              epoch, (time.time() - epoch_start_time),
              val_loss, math.exp(val_loss)))
    print('-' * 89)
    # Save the model if the validation loss is the best we've seen so far.
    if not best_val_loss or val_loss < best_val_loss:
      best_filename = args.save + '-' + str(time.ctime()) + '-' + str(val_loss) + '.t7'
      with open(best_filename, 'wb') as f:
        torch.save(model, f)
      best_val_loss = val_loss
    else:
      # Anneal the learning rate if no improvement has been seen in the validation dataset.
      lr /= 2.5
      if args.optim == 'sgd':
        optim = torch.optim.SGD(model.parameters(), lr=lr)
      elif args.optim == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=lr)
      elif args.optim == 'rmsprop':
        optim = torch.optim.RMSprop(model.parameters(), lr=lr, eps=1e-10)

except KeyboardInterrupt:
  print('-' * 89)
  print('Exiting from training early')

# Load the best saved model.
with open(best_filename, 'rb') as f:
  model = torch.load(f)

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
