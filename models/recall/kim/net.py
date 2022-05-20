import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

MAX_SENTENCE = 30
MAX_ALL = 50
MAX_SENT_LENGTH = MAX_SENTENCE
MAX_SENTS = MAX_ALL
max_entity_num = 10
num = 100
num1 = 200
num2 = 100
npratio = 4


def time_distributed(x, layer):
    shape = x.shape
    x = x.reshape([-1, *shape[2:]])
    x = layer(x)
    return x.reshape([*shape[:2], *x.shape[-(x.ndim - 1):]])


class Attention(nn.Layer):

    def __init__(self, input_dim, nb_head, size_per_head):
        super(Attention, self).__init__()
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head * size_per_head

        self.WQ = nn.Linear(input_dim, self.output_dim, bias_attr=False)
        self.WK = nn.Linear(input_dim, self.output_dim, bias_attr=False)
        self.WV = nn.Linear(input_dim, self.output_dim, bias_attr=False)

    def mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = F.one_hot(seq_len[:, 0], paddle.shape(inputs)[1])
            mask = 1 - paddle.cumsum(mask, 1)
            for _ in range(len(inputs.shape) - 2):
                mask = paddle.unsqueeze(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12

    def forward(self, q, k, v, q_len=None, v_len=None):
        # 对Q、K、V做线性变换
        Q_seq = self.WQ(q)
        Q_seq = paddle.reshape(Q_seq, (-1, paddle.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = paddle.transpose(Q_seq, (0, 2, 1, 3))
        K_seq = self.WK(k)
        K_seq = paddle.reshape(K_seq, (-1, paddle.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = paddle.transpose(K_seq, (0, 2, 1, 3))
        V_seq = self.WV(v)
        V_seq = paddle.reshape(V_seq, (-1, paddle.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = paddle.transpose(V_seq, (0, 2, 1, 3))
        # 计算内积，然后mask，然后softmax
        A = paddle.matmul(Q_seq, K_seq, transpose_y=True) / self.size_per_head ** 0.5
        A = paddle.transpose(A, (0, 3, 2, 1))
        A = self.mask(A, v_len, 'add')
        A = paddle.transpose(A, (0, 3, 2, 1))
        A = F.softmax(A, -1)
        # 输出并mask
        O_seq = paddle.matmul(A, V_seq)
        O_seq = paddle.transpose(O_seq, (0, 2, 1, 3))
        O_seq = paddle.reshape(O_seq, (-1, paddle.shape(O_seq)[1], self.output_dim))
        O_seq = self.mask(O_seq, q_len, 'mul')
        return O_seq


class AttentivePooling(nn.Layer):
    def __init__(self, dim1=10, dim2=100, dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.dense1 = nn.Sequential(nn.Linear(dim2, 200), nn.Tanh())
        self.dense2 = nn.Sequential(nn.Linear(200, 1))

    def forward(self, vecs_input):
        user_vecs = self.dropout(vecs_input)
        user_attn = self.dense1(user_vecs.astype('float32'))
        user_attn = self.dense2(user_attn)
        user_att = F.softmax(user_attn, axis=-2)
        user_vec = paddle.sum(user_vecs * user_att, axis=-2)
        return user_vec


class GraphCoAttNet(nn.Layer):
    def __init__(self, num, input_dim=100):
        super(GraphCoAttNet, self).__init__()
        self.num = num
        self.attn = Attention(input_dim, 5, 20)
        self.dense1 = nn.Linear(input_dim, 100)
        self.dense2 = nn.Linear(input_dim, 100)
        self.dense3 = nn.Linear(input_dim, 100)
        # self.dense4 = nn.Linear(input_dim, 100)
        self.dense5 = nn.Linear(input_dim, 1)

    def forward(self, entity_input):
        entity_emb, candidate_emb = entity_input.split(2, -2)
        entity_vecs = self.attn(entity_emb, entity_emb, entity_emb)  # bz,num,100
        entity_co_att = self.dense1(entity_vecs)
        candidate_co_att = self.dense2(candidate_emb)
        S = paddle.matmul(entity_co_att, candidate_co_att, transpose_y=True)  # bz,num,num

        entity_self_att = self.dense3(entity_vecs)

        # candidate_co_att = self.dense4(candidate_emb)
        entity_co_att = paddle.matmul(S, candidate_emb)
        entity_att = entity_self_att + entity_co_att
        entity_att = F.tanh(entity_att)  # bz,num,100
        entity_att = self.dense5(entity_att)
        entity_vec = paddle.sum(entity_vecs * entity_att, axis=-2)  # bz,100
        return entity_vec


class ContextEncoder(nn.Layer):
    def __init__(self,
                 vocab_size,
                 embedding_size,
                 nb_head=20,
                 size_per_head=20,
                 hidden_size=400,
                 dropout=0.2,
                 title_word_embedding_matrix=None):
        super().__init__()
        self.title_word_embedding = nn.Embedding(vocab_size, embedding_size)
        if title_word_embedding_matrix is not None:
            self.title_word_embedding.weight.set_value(title_word_embedding_matrix)
        self.dropout = nn.Dropout(dropout)
        self.word_rep1 = nn.Sequential(nn.Conv1D(embedding_size, hidden_size, 3, padding=1, data_format='NLC'),
                                       nn.ReLU())
        self.word_rep2 = Attention(embedding_size, nb_head, size_per_head)

    def forward(self, sentence_input):
        word_vecs = self.dropout(self.title_word_embedding(sentence_input))
        word_rep1 = self.word_rep1(word_vecs)
        word_rep2 = F.relu(self.word_rep2(word_vecs, word_vecs, word_vecs))
        return self.dropout(word_rep1 + word_rep2)


class EntityEncoder(nn.Layer):
    def __init__(self, dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pooling = AttentivePooling()

    def forward(self, entity_input):
        droped_rep = self.dropout(entity_input)
        entity_vec = self.pooling(droped_rep)
        return entity_vec


class PairPair(nn.Layer):
    def __init__(self, max_entity_num):
        super().__init__()
        self.max_entity_num = max_entity_num
        self.entity_encoder = EntityEncoder()
        self.gat_fuse = nn.Linear(200, 100)
        self.gcat = GraphCoAttNet(max_entity_num)
        self.merge = nn.Linear(200, 100)
        self.gcat0 = GraphCoAttNet(max_entity_num)

    def forward(self, entity_input):
        entity_input = entity_input.astype('float32')
        user_entity_input, news_entity_input = entity_input.split(2, -2)
        user_entity_zerohop = user_entity_input[..., max_entity_num * max_entity_num:, :]  # bz,1,100
        user_entity_onehop = user_entity_input[..., :max_entity_num * max_entity_num, :]
        user_entity_onehop = user_entity_onehop.reshape((-1, max_entity_num, max_entity_num, 100))
        # user_entity_onehop = user_entity_onehop.reshape((-1, max_entity_num, 100))

        user_can = self.entity_encoder(user_entity_onehop).reshape(
            [-1, max_entity_num, 100])  # (max_entity_num,100)
        user_can = paddle.concat([user_can, user_entity_zerohop], -1)
        user_can = self.gat_fuse(user_can.astype('float32'))  # (max_entity_num,100)
        user_can = user_can.reshape((-1, max_entity_num * 100,))
        user_can = user_can.unsqueeze(-2).tile([1, max_entity_num, 1])
        user_can = user_can.reshape((-1, max_entity_num, max_entity_num, 100))

        news_entity_zerohop = news_entity_input[..., max_entity_num * max_entity_num:, :]  # bz,1,100
        news_entity_onehop = news_entity_input[..., :max_entity_num * max_entity_num, :]
        news_entity_onehop = news_entity_onehop.reshape((-1, max_entity_num, max_entity_num, 100))

        news_can = self.entity_encoder(news_entity_onehop).reshape(
            [-1, max_entity_num, 100])  # (max_entity_num,100)
        news_can = paddle.concat([news_can, user_entity_zerohop], -1)
        news_can = self.gat_fuse(news_can.astype('float32'))  # (max_entity_num,100)
        news_can = news_can.reshape((-1, max_entity_num * 100,))
        news_can = news_can.unsqueeze(-2).tile([1, max_entity_num, 1])
        news_can = news_can.reshape((-1, max_entity_num, max_entity_num, 100))

        user_entity_onehop = paddle.concat([user_entity_onehop, news_can], -2)  # bz,max_entity_num,max_entity_num*2,100
        news_entity_onehop = paddle.concat([news_entity_onehop, user_can], -2)  # bz,max_entity_num,max_entity_num*2,100

        user_entity_onehop = time_distributed(user_entity_onehop, self.gcat)
        news_entity_onehop = time_distributed(news_entity_onehop, self.gcat)

        user_entity_vecs = paddle.concat([user_entity_zerohop, user_entity_onehop], -1)
        news_entity_vecs = paddle.concat([news_entity_zerohop, news_entity_onehop], -1)

        user_entity_vecs = self.merge(user_entity_vecs)
        news_entity_vecs = self.merge(news_entity_vecs)

        user_entity_vecs = paddle.concat([user_entity_vecs, news_entity_zerohop], -2)
        news_entity_vecs = paddle.concat([news_entity_vecs, user_entity_zerohop], -2)

        user_entity_vec = self.gcat0(user_entity_vecs)
        news_entity_vec = self.gcat0(news_entity_vecs)

        vec = paddle.concat([user_entity_vec, news_entity_vec], -1)
        return vec


class PairModel(nn.Layer):
    def __init__(self,
                 vocab_size,
                 embedding_size,
                 hidden_size=400,
                 MAX_SENTS=50,
                 max_entity_num=10,
                 title_word_embedding_matrix=None):
        super(PairModel, self).__init__()
        self.MAX_SENTS = MAX_SENTS
        self.max_entity_num = max_entity_num
        self.hidden_size = hidden_size
        self.merge_layer = nn.Linear(500, hidden_size)
        self.context_encoder = ContextEncoder(vocab_size, embedding_size, 20, 20,
                                              title_word_embedding_matrix=title_word_embedding_matrix)
        self.attn_layer1 = nn.Sequential(nn.Linear(hidden_size, 200), nn.Tanh())
        self.attn_layer2 = nn.Linear(200, 1)

        self.pair_graph = PairPair(max_entity_num)

        self.match_att = nn.Sequential(nn.Linear(hidden_size, 100), nn.Tanh(), nn.Linear(100, 1))
        self.match_reduce = nn.Linear(hidden_size, 100)

    def forward(self, title_inputs, entity_inputs, one_hop_inputs, clicked_title_input, clicked_entity_input,
                clicked_one_hop_input):
        clicked_title_word_vecs = time_distributed(clicked_title_input, self.context_encoder)
        candi_title_word_vecs = self.context_encoder(title_inputs)
        clicked_title_att_vecs = self.attn_layer1(clicked_title_word_vecs)
        clicked_title_att = self.attn_layer2(clicked_title_att_vecs).reshape([-1, MAX_SENTS, MAX_SENTENCE])

        candi_title_att_vecs = self.attn_layer1(candi_title_word_vecs)
        candi_title_att0 = self.attn_layer2(candi_title_att_vecs)
        candi_title_att = candi_title_att0.squeeze(-1).unsqueeze(1).tile([1, self.MAX_SENTS, 1])

        clicked_title_att_vecs = clicked_title_att_vecs.reshape([-1, MAX_SENTS * MAX_SENTENCE, 200])  # (bz,50*30,200)
        candi_title_att_vecs = candi_title_att_vecs.transpose([0, 2, 1])  # (bz,200,30)
        cross_att = paddle.matmul(clicked_title_att_vecs, candi_title_att_vecs)  # (bz,50*30,30)
        cross_att_candi = F.softmax(cross_att, -1)  # (bz,50*30,30)
        cross_att_candi = paddle.matmul(cross_att_candi, candi_title_att0)
        cross_att_candi = cross_att_candi.reshape([-1, MAX_SENTS, MAX_SENTENCE]) * 0.001

        clicked_title_att += cross_att_candi
        clicked_title_att = F.softmax(clicked_title_att, -1)

        cross_att_click = cross_att.reshape((-1, MAX_SENTS, MAX_SENTENCE, MAX_SENTENCE))
        cross_att_click = cross_att_click.transpose((0, 1, 3, 2))  # (bz,50,30,30)
        clicked_title_att_re = clicked_title_att.reshape((-1, MAX_SENTS, 1, MAX_SENTENCE))  # (bz,50,1,30,)

        cross_att_click = (cross_att_click * clicked_title_att_re).sum(-2) * 0.001

        candi_title_att = candi_title_att + cross_att_click
        candi_title_att = F.softmax(candi_title_att, -1)  # (bz,50, 30)

        candi_title_vecs = paddle.matmul(candi_title_att, candi_title_word_vecs)  # (bz,50,400)

        clicked_title_att = (clicked_title_att).reshape((-1, MAX_SENTS, MAX_SENTENCE, 1))  # (bz, 50, 30, 1)

        clicked_title_vecs = (clicked_title_word_vecs * clicked_title_att).sum(-2)  # (bz, 50, 30)

        clicked_onehop = clicked_one_hop_input.reshape([-1, MAX_SENTS, max_entity_num * max_entity_num, 100])
        clicked_entity = paddle.concat([clicked_onehop, clicked_entity_input], -2)

        news_onehop = one_hop_inputs.reshape((-1, max_entity_num * max_entity_num, 100))  # (bz,100,100)
        news_entity = paddle.concat([news_onehop, entity_inputs, ], -2)  # (bz,110,100)
        news_entity = news_entity.reshape((-1, max_entity_num * (max_entity_num + 1) * 100,))

        news_entity = news_entity.unsqueeze(1).tile([1, MAX_SENTS, 1])
        news_entity = news_entity.reshape((-1, MAX_SENTS, max_entity_num * (max_entity_num + 1), 100))

        entity_emb = paddle.concat([clicked_entity, news_entity], -2)  # [16, 50, 220, 100]
        entity_vecs = time_distributed(entity_emb, self.pair_graph)

        user_entity_vecs, news_entity_vecs = entity_vecs.split(2, -1)

        user_vecs = paddle.concat([clicked_title_vecs, user_entity_vecs], -1)
        user_vecs = self.merge_layer(user_vecs)

        news_vecs = paddle.concat([candi_title_vecs, news_entity_vecs], -1)
        news_vecs = self.merge_layer(news_vecs)

        user_att1 = self.match_att(user_vecs)
        user_att = user_att1.reshape((-1, MAX_SENTS))

        news_att1 = self.match_att(news_vecs)
        news_att = user_att1.reshape((-1, MAX_SENTS))

        cross_user_vecs = self.match_reduce(user_vecs)  # (bz,50,100)
        cross_news_vecs = self.match_reduce(news_vecs)  # (bz,50,100)
        cross_news_vecs = cross_news_vecs.transpose((0, 2, 1))  # (bz,100,50)
        cross_att = paddle.matmul(cross_user_vecs, cross_news_vecs)  # (bz,50,50)

        cross_user_att = F.softmax(cross_att, -1)  # (bz,50,50)
        cross_user_att = paddle.matmul(cross_user_att, news_att1)
        cross_user_att = cross_user_att.reshape((-1, MAX_SENTS,))
        cross_user_att = cross_user_att * 0.01
        user_att = user_att + cross_user_att
        user_att = F.softmax(user_att, -1)

        cross_news_att = cross_att.transpose((0, 2, 1))  # (bz,50,50)
        cross_news_att = F.softmax(cross_news_att)  # (bz,50,50)
        cross_news_att = paddle.matmul(cross_news_att, user_att1)
        cross_news_att = cross_news_att.reshape([-1, MAX_SENTS])
        cross_news_att = cross_news_att * 0.01
        news_att = news_att + cross_news_att
        news_att = F.softmax(news_att)

        user_vec = paddle.matmul(user_att.unsqueeze(1), user_vecs).squeeze(1)
        news_vec = paddle.matmul(news_att.unsqueeze(1), news_vecs).squeeze(1)

        score = (user_vec * news_vec).sum(-1, keepdim=True)  # (bz,50)

        return score


class KIMLayer(nn.Layer):
    def __init__(self,
                 vocab_size,
                 embedding_size,
                 hidden_size=400,
                 MAX_SENTS=50,
                 max_entity_num=10,
                 title_word_embedding_matrix=None):
        super().__init__()
        self.model = PairModel(vocab_size,
                               embedding_size,
                               hidden_size,
                               MAX_SENTS,
                               max_entity_num,
                               title_word_embedding_matrix)

    def forward(self, title_inputs, entity_inputs, one_hop_inputs, clicked_title_input, clicked_entity_input,
                clicked_one_hop_input):
        if self.training:
            doc_score = []
            for i in range(5):
                inp = (
                    title_inputs[:, i, :, ],
                    entity_inputs[:, i, :, :],
                    one_hop_inputs[:, i, :, :, :],
                    clicked_title_input,
                    clicked_entity_input,
                    clicked_one_hop_input
                )
                score = self.model(*inp)
                doc_score.append(score)
            doc_score = paddle.concat(doc_score, -1)
        else:
            doc_score = self.model(title_inputs, entity_inputs, one_hop_inputs, clicked_title_input, clicked_entity_input,
                clicked_one_hop_input)
        return doc_score
