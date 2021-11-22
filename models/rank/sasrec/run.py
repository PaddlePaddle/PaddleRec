#!/usr/bin/env python3
import argparse
import os

import paddle

from sasrec.model import SASRec
from sasrec.data import WarpSampler
from sasrec.utils import set_seed, data_partition
from sasrec.train import train
from sasrec.eval import evaluate

set_seed(42)

parser = argparse.ArgumentParser(description='SASRec training')
# data
parser.add_argument('--dataset_path', metavar='DIR',
                    default='data/preprocessed/ml-1m.txt')
# learning
learn = parser.add_argument_group('Learning options')
learn.add_argument('--lr', type=float, default=0.001, help='initial learning rate [default: 0.01]')
learn.add_argument('--epochs', type=int, default=1000, help='number of epochs for train')
learn.add_argument('--batch_size', type=int, default=128, help='batch size for training')
learn.add_argument('--optimizer', default='AdamW',
                   help='Type of optimizer. Adagrad|Adam|AdamW are supported [default: Adagrad]')
# model
model_cfg = parser.add_argument_group('Model options')
model_cfg.add_argument('--hidden_units', type=int, default=50,
                       help='hidden size of LSTM [default: 300]')
model_cfg.add_argument('--maxlen', type=int, default=200,
                       help='hidden size of LSTM [default: 300]')
model_cfg.add_argument('--dropout', type=float, default=0.2, help='the probability for dropout')
model_cfg.add_argument('--l2_emb', type=float, default=0.0, help='penalty term coefficient')
model_cfg.add_argument('--num_blocks', type=int, default=2,
                       help='d_a size [default: 150]')
model_cfg.add_argument('--num_heads', type=int, default=1,
                       help='row size of sentence embedding [default: 30]')
# device
device = parser.add_argument_group('Device options')
device.add_argument('--num_workers', default=8, type=int, help='Number of workers used in data-loading')
device.add_argument('--cuda', action='store_true', default=True, help='enable the gpu')
device.add_argument('--device', type=int, default=None)

# experiment options
experiment = parser.add_argument_group('Experiment options')
experiment.add_argument('--continue_from', default='', help='Continue from checkpoint model')
experiment.add_argument('--checkpoint', dest='checkpoint', default=True, action='store_true',
                        help='Enables checkpoint saving of model')
experiment.add_argument('--checkpoint_per_batch', default=10000, type=int,
                        help='Save checkpoint per batch. 0 means never save [default: 10000]')
experiment.add_argument('--save_folder', default='output/',
                        help='Location to save epoch models, training configurations and results.')
experiment.add_argument('--log_config', default=True, action='store_true', help='Store experiment configuration')
experiment.add_argument('--log_result', default=True, action='store_true', help='Store experiment result')
experiment.add_argument('--log_interval', type=int, default=30,
                        help='how many steps to wait before logging training status')
experiment.add_argument('--val_interval', type=int, default=800,
                        help='how many steps to wait before vaidation')
experiment.add_argument('--val_start_batch', type=int, default=8000,
                        help='how many steps to wait before vaidation')
experiment.add_argument('--save_interval', type=int, default=20,
                        help='how many epochs to wait before saving')
experiment.add_argument('--test', type=bool, default=False, help='test only')
experiment.add_argument('--model_path', type=str, default=False, help='test only')


def main():
    print(paddle.__version__)
    args = parser.parse_args()

    # gpu
    if args.cuda and args.device:
        paddle.set_device(f"gpu:{args.device}")
    print(paddle.get_device())

    dataset = data_partition(args.dataset_path)

    [user_train, _, _, usernum, itemnum] = dataset
    num_batch = len(user_train) // args.batch_size  # tail? + ((len(user_train) % args.batch_size) != 0)
    print("batches / epoch:", num_batch)

    seq_len = 0.0
    for u in user_train:
        seq_len += len(user_train[u])
    print('\nAverage sequence length: %.2f' % (seq_len / len(user_train)))

    # make save folder
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    # configuration
    print("\nConfiguration:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}:".format(attr.capitalize().replace('_', ' ')).ljust(25) + "{}".format(value))

    # log result
    if args.log_result:
        with open(os.path.join(args.save_folder, 'result.csv'), 'w') as r:
            r.write('{:s},{:s},{:s},{:s},{:s}'.format('epoch', 'batch', 'loss', 'acc', 'lr'))

    # model
    model = SASRec(itemnum, args)
    print(model)

    if not args.test:  # train
        # dataloader
        sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen,
                              n_workers=args.num_workers)
        train(sampler, model, args, num_batch, dataset)
        sampler.close()
    else:  # test
        print("=> loading weights from '{}'".format(args.model_path))
        assert os.path.isfile(args.model_path), "=> no checkpoint found at '{}'".format(args.model_path)
        checkpoint = paddle.load(args.model_path)
        model.set_state_dict(checkpoint['state_dict'])
        evaluate(dataset, model, checkpoint['epoch'], 0, args, is_val=False)


if __name__ == '__main__':
    main()
