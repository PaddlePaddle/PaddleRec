#!/usr/bin/env python3
import os

import paddle
from paddle import optimizer
import paddle.nn.functional as F

from sasrec.eval import evaluate


class MyBCEWithLogitLoss(paddle.nn.Layer):
    def __init__(self):
        super(MyBCEWithLogitLoss, self).__init__()

    def forward(self, pos_logits, neg_logits, labels):
        return paddle.sum(
            - paddle.log(F.sigmoid(pos_logits) + 1e-24) * labels -
            paddle.log(1 - F.sigmoid(neg_logits) + 1e-24) * labels,
            axis=(0, 1)
        ) / paddle.sum(labels, axis=(0, 1))


def train(sampler, model, args, num_batch, dataset):
    clip = None
    # optimization scheme
    if args.optimizer == 'Adam':
        optim = optimizer.Adam(parameters=model.parameters(), learning_rate=args.lr, grad_clip=clip)
    elif args.optimizer == 'Adagrad':
        optim = optimizer.Adagrad(parameters=model.parameters(), learning_rate=args.lr, grad_clip=clip)
    elif args.optimizer == 'AdamW':
        optim = optimizer.AdamW(parameters=model.parameters(), learning_rate=args.lr, grad_clip=clip)

    # loss
    # criterion = nn.BCEWithLogitsLoss()
    criterion = MyBCEWithLogitLoss()

    # continue training from checkpoint model
    if args.continue_from:
        print("=> loading checkpoint from '{}'".format(args.continue_from))
        assert os.path.isfile(args.continue_from), "=> no checkpoint found at '{}'".format(args.continue_from)
        checkpoint = paddle.load(args.continue_from)
        start_epoch = checkpoint['epoch']
        best_pair = checkpoint.get('best_pair', None)
        model.set_state_dict(checkpoint['state_dict'])
    else:
        start_epoch = 1
        best_pair = None

    model.train()

    tot_batch = 0
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_loss = 0
        for i_batch in range(num_batch):
            tot_batch += 1
            u, seq, pos, neg = sampler.next_batch()  # tuples to ndarray
            u, seq, pos, neg = paddle.to_tensor(u, dtype='int64'), paddle.to_tensor(seq,
                                                                                    dtype='int64'), paddle.to_tensor(
                pos), paddle.to_tensor(neg)
            pos_logits, neg_logits = model(seq, pos, neg)  # ()

            targets = (pos != 0).astype(dtype='int32')
            # targets = targets.reshape((args.batch_size*args.maxlen, -1))
            loss = criterion(pos_logits, neg_logits, targets)
            for param in model.item_emb.parameters():
                loss += args.l2_emb * paddle.norm(param)
            loss.backward()
            epoch_loss += loss.numpy()[0]
            optim.step()
            optim.clear_grad()

            # validation
            if tot_batch >= args.val_start_batch and tot_batch % args.val_interval == 0 and i_batch != 0:
                valid_pair = evaluate(dataset, model, epoch, i_batch, args, is_val=True)
                if best_pair is None or valid_pair > best_pair:
                    best_pair = valid_pair
                    file_path = '%s/SASRec_best.pth.tar' % (args.save_folder)
                    print("=> found better validated model, saving to %s" % file_path)
                    save_checkpoint(model,
                                    {'epoch': epoch,
                                     'optimizer': optim.state_dict(),
                                     'best_pair': best_pair},
                                    file_path)

        print('Epoch {:3} - loss: {:.4f}  lr: {:.5f}'.format(epoch,
                                                             epoch_loss / num_batch,
                                                             optim._learning_rate,
                                                             ))

        if args.checkpoint and epoch % args.save_interval == 0:
            file_path = '%s/SASRec_epoch_%d.pth.tar' % (args.save_folder, epoch)
            print("\r=> saving checkpoint model to %s" % file_path)
            save_checkpoint(model, {'epoch': epoch,
                                    'optimizer': optim.state_dict(),
                                    'best_pair': best_pair},
                            file_path)


def save_checkpoint(model, state, filename):
    state['state_dict'] = model.state_dict()
    paddle.save(state, filename)
