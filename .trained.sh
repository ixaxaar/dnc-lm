python train.py\
  --batch_size 64\
  --n_layers 2\
  --hidden_size 200\
  --n_epochs 1000\
  --cuda 0\
  --nr_cells 32\
  --read_heads 4\
  --cell_size 32\
  --rnn_type dnc\
  --learning_rate 0.0001\
  --print_every 5\
  ./input.txt


python train.py\
  --batch_size 256\
  --n_layers 2\
  --hidden_size 200\
  --n_epochs 1000\
  --chunk_len 200\
  --cuda 1\
  --nr_cells 32\
  --read_heads 4\
  --cell_size 32\
  --rnn_type dnc\
  --learning_rate 0.001\
  --print_every 10\
  --reset_experience 0\
  --train_file ./train.txt\
  --eval_file ./valid.txt

python main.py --cuda 0 --emsize 650 --nhid 650 --dropout 0.5 --epochs 40


python main.py --cuda 1 --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --model DNC --nr_cells 32 --read_heads 4 --cell_size 32


python main.py --cuda 1 --emsize 650 --nhid 650 --dropout 0 --epochs 40 --model DNC --nr_cells 32 --read_heads 4 --cell_size 32 --batch_size 500 --log-interval 10 --lr 5.0



python main.py --cuda 0 --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --model DNC --nr_cells 32 --read_heads 4 --cell_size 64 --batch_size 500 --log-interval 15 --lr 0.0001

# smaller batch size faster convergence
python main.py --cuda 1 --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --model DNC --nr_cells 32 --read_heads 4 --cell_size 64 --batch_size 100 --log-interval 20 --lr 0.0001

# larger learning rate faster convergence
python main.py --cuda 0 --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --model DNC --nr_cells 32 --read_heads 4 --cell_size 64 --batch_size 100 --log-interval 15 --lr 0.001

# explore smaller hidden size - failed
python main.py --cuda 0 --emsize 300 --nhid 300 --dropout 0.5 --epochs 40 --model DNC --nr_cells 32 --read_heads 4 --cell_size 64 --batch_size 100 --log-interval 15 --lr 0.001

# explore less dropout and smaller hidden size with larger lr - failed
python main.py --cuda 0 --emsize 300 --nhid 300 --dropout 0.3 --epochs 40 --model DNC --nr_cells 32 --read_heads 4 --cell_size 64 --batch_size 100 --log-interval 15 --lr 0.001

# explore more hidden size - failed
python main.py --cuda 1 --emsize 1500 --nhid 1500 --dropout 0.5 --epochs 40 --model DNC --nr_cells 32 --read_heads 4 --cell_size 64 --batch_size 100 --log-interval 15 --lr 0.001


# shared memory
python main.py --cuda 0 --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --model DNC --nr_cells 32 --read_heads 4 --cell_size 64 --batch_size 100 --log-interval 15 --lr 0.001

# shared memory - failed
python main.py --cuda 0 --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --model DNC --nr_cells 32 --read_heads 4 --cell_size 64 --batch_size 100 --log-interval 15 --lr 0.0001

# unshared hidden state between layers - killed when started overfitting - revisit
python main.py --cuda 1 --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --model DNC --nr_cells 32 --read_heads 4 --cell_size 64 --batch_size 100 --log-interval 15 --lr 0.001

# unshared hidden etc with larger cell sizes
python main.py --cuda 1 --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --model DNC --nr_cells 32 --read_heads 4 --cell_size 128 --batch_size 100 --log-interval 15 --lr 0.001

# with only one hidden inter epochs and valid time
python main.py --cuda 1 --emsize 300 --nhid 300 --dropout 0.5 --epochs 40 --model DNC --nr_cells 32 --read_heads 4 --cell_size 300 --batch_size 100 --log-interval 15 --lr 0.001


# with reset memory in eval - 48.12, 105.21, 256.58
python main.py --cuda 1 --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --model DNC --nr_cells 32 --read_heads 4 --cell_size 64 --batch_size 100 --log-interval 15 --lr 0.001

# with reset memory in eval - 43.76, 107.68
python main.py --cuda 1 --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --model DNC --nr_cells 32 --read_heads 4 --cell_size 64 --batch_size 100 --log-interval 15 --lr 0.001

# try with SGD - 168.00, 164.31, 158.86
python main.py --cuda 1 --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --model DNC --nr_cells 32 --read_heads 4 --cell_size 64 --batch_size 100 --log-interval 15 --lr 1.0

# try ith RMSprop - 89.05, 117.95, 110.93
python main.py --cuda 1 --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --model DNC --nr_cells 32 --read_heads 4 --cell_size 64 --batch_size 100 --log-interval 15 --lr 0.0001 --optim rmsprop

# no reset experience, adam
python main.py --cuda 0 --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --model DNC --nr_cells 32 --read_heads 4 --cell_size 64 --batch_size 100 --log-interval 15 --lr 0.001 --optim adam

# with shared memory throughout layers
python main.py --cuda 0 --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --model DNC --nr_cells 32 --read_heads 4 --cell_size 64 --batch_size 100 --log-interval 15 --lr 0.001 --optim adam
python main.py --cuda 0 --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --model DNC --nr_cells 32 --read_heads 4 --cell_size 64 --batch_size 100 --log-interval 15 --lr 0.0001 --optim rmsprop

