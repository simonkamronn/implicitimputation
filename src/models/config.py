import argparse
import torch


def get_config():
    parser = argparse.ArgumentParser(description='Denoising autoencoders')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N', help='number of epochs to train (default: 2)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--lr', type=int, default=0.003, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=.5, metavar='D', help='input dropout')
    parser.add_argument('--model', type=str, default='dae', metavar='M', help='Model selection')
    parser.add_argument('--layers', type=int, default=[400, 50], nargs='+', help='Units in each layer')
    parser.add_argument('--blocks', type=int, default=1, help='Number of stacked blocks')
    parser.add_argument('--name', type=str, default='', help='Experiment name')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args