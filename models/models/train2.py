import argparse
from main2 import *


parser = argparse.ArgumentParser(description='Trains model on the given dataset')
parser.add_argument('-s','--start-epoch', type=int, default=0, help='index of the first epoch')
parser.add_argument('-e','--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('-a','--arch', type=str, default='LSTM', help='model architecture (CNN/LSTM_Attn/LSTM/RCNN/RNN/selfAttn)')
parser.add_argument('-v','--version', type=str, default='v0', help='version of the model')
parser.add_argument('-w','--weights-version-load', type=str, default='v0', help='version of the model weight checkpoint to load')
parser.add_argument('-x','--weights-version-save', type=str, default='v0', help='version of the model weight checkpoint to load')
parser.add_argument('-o','--set-optimizer', type=str, default='adam', help="optimizer ('adam' or 'sgd')")
parser.add_argument('-l','--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('-b','--batch_size', type=int, default=32, help='batch size')


# parser.add_argument('-d','--weight-decay', type=float, default=0.0, help='weight decay')
# parser.add_argument('-F','--focal-loss', type=int, default=0, help='give 1 for focal loss')
# parser.add_argument('-p','--print-freq', type=int, default=2000, help='multiplier for the variance loss')
# parser.add_argument('-W','--num-of-workers', type=int, default=4, help='number of workers for the dataprep')
# parser.add_argument('-U','--user', type=str, default='sk7685', help='hpc username to be used when determining paths')
# parser.add_argument('-D','--drop_prob', type=float, default=0.0, help='dropout probability at the last layer')
# parser.add_argument('-R','--drop-2d', type=int, default=0, help='insert dropout inbetween resnet modules')


args = parser.parse_args()
print('\nversion name: ' + args.version +'\n')

train_and_val(args)
