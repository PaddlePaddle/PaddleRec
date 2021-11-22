import numpy as np
import paddle
import paddle.nn as nn


class SASRec(paddle.nn.Layer):
    def __init__(self, item_num, args):
        super(SASRec, self).__init__()
        self.item_emb = nn.Embedding(item_num + 1, args.hidden_units)  # [pad] is 0
        self.pos_emb = nn.Embedding(args.maxlen, args.hidden_units)
        self.emb_dropout = paddle.nn.Dropout(p=args.dropout)

        self.subsequent_mask = (paddle.triu(paddle.ones((args.maxlen, args.maxlen))) == 0)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=args.hidden_units,
                                                        nhead=args.num_heads,
                                                        dim_feedforward=args.hidden_units,
                                                        dropout=args.dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=args.num_blocks)

    def position_encoding(self, seqs):
        seqs_embed = self.item_emb(seqs)  # (batch_size, max_len, embed_size)
        positions = np.tile(np.array(range(seqs.shape[1])), [seqs.shape[0], 1])
        position_embed = self.pos_emb(paddle.to_tensor(positions, dtype='int64'))
        return self.emb_dropout(seqs_embed + position_embed)

    def forward(self, log_seqs, pos_seqs, neg_seqs):
        # all input seqs: (batch_size, seq_len)
        seqs_embed = self.position_encoding(log_seqs)  # (batch_size, seq_len, embed_size)
        log_feats = self.encoder(seqs_embed, self.subsequent_mask)  # (batch_size, seq_len, embed_size)

        pos_embed = self.item_emb(pos_seqs)  # (batch_size, seq_len, embed_size)
        neg_embed = self.item_emb(neg_seqs)

        pos_logits = (log_feats * pos_embed).sum(axis=-1)
        neg_logits = (log_feats * neg_embed).sum(axis=-1)

        return pos_logits, neg_logits

    def predict(self, log_seqs, item_indices):  # for inference
        seqs = self.position_encoding(log_seqs)
        log_feats = self.encoder(seqs, self.subsequent_mask)  # (batch_size, seq_len, embed_size)

        final_feat = log_feats[:, -1, :]
        item_embs = self.item_emb(paddle.to_tensor(item_indices, dtype='int64'))

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits
