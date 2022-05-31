import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np

class Meta_Linear(nn.Linear): #used in MAML to forward input with fast weight
    def __init__(self, in_features, out_features):
        super(Meta_Linear, self).__init__(in_features, out_features)
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast, self.bias.fast) #weight.fast (fast weight) is the temporaily adapted weight
        else:
            out = super(Meta_Linear, self).forward(x)
        return out

class Meta_Embedding(nn.Embedding): #used in MAML to forward input with fast weight
    def __init__(self, num_embedding, embedding_dim):
        super(Meta_Embedding, self).__init__(num_embedding, embedding_dim)
        self.weight.fast = None

    def forward(self, x):
        if self.weight.fast is not None:
            out = F.embedding(x.astype('int64'), self.weight.fast,self._padding_idx,self._sparse)
        else:
            out = F.embedding(x.astype('int64'), self.weight,self._padding_idx,self._sparse)
        return out

class Emb(nn.Layer):
    def __init__(self, max_idxs, embedding_size=4):
        """
        fnames: feature names
        max_idxs: array of max_idx of each feature
        embedding_size: size of embedding
        dropout: prob for dropout, set None if no dropout
        use_cuda: bool, True for gpu or False for cpu
        """
        super(Emb, self).__init__()
        self.static_emb = StEmb(max_idxs[0], embedding_size)
        self.ad_emb = StEmb(max_idxs[2], embedding_size)
        self.dynamic_emb = DyEmb(max_idxs[1], embedding_size)

    def forward(self, x):
        static_emb = self.static_emb(x[0])
        dynamic_emb = self.dynamic_emb(x[1], x[2])
        concat_embeddings = paddle.concat([static_emb, dynamic_emb], 1)
        ad_emb = self.ad_emb(x[3])

        return concat_embeddings, ad_emb

class DyEmb(nn.Layer):
    def __init__(self, max_idxs, embedding_size=4):
        """
        fnames: feature names
        max_idxs: array of max_idx of each feature
        embedding_size: size of embedding
        dropout: prob for dropout, set None if no dropout
        method: 'avg' or 'sum'
        use_cuda: bool, True for gpu or False for cpu
        """
        super(DyEmb, self).__init__()
        self.max_idxs = max_idxs
        self.embedding_size = embedding_size

        self.embeddings = nn.LayerList(
            [Meta_Embedding(max_idxs + 1, self.embedding_size) for max_idxs in self.max_idxs])

    def masked_fill(self, x, mask, value):
        y = paddle.full(x.shape, value, x.dtype)
        return paddle.where(mask, y, x)

    def forward(self, dynamic_ids, dynamic_lengths):
        """
        input: relative id
        dynamic_ids: Batch_size * Field_size * Max_feature_size
        dynamic_lengths: Batch_size * Field_size
        return: Batch_size * Field_size * Embedding_size
        """
        concat_embeddings = []
        batch_size = dynamic_lengths.shape[0]

        dynamic_list = list()
        dynamic_list.append(dynamic_ids[:,0:10])
        dynamic_list.append(dynamic_ids[:,10:20])
        dynamic_list.append(dynamic_ids[:,20:30])
        dynamic_list.append(dynamic_ids[:,30:35])
        dynamic_list.append(dynamic_ids[:,35:40])
        dynamic_list.append(dynamic_ids[:,40:45])
        dynamic_list.append(dynamic_ids[:,45:50])

        for i in range(len(self.max_idxs)):
            # B*M
            dynamic_ids_tensor = dynamic_list[i]
            dynamic_lengths_tensor = dynamic_lengths[:,i].astype(float)

            # embedding layer B*M*E
            dynamic_embeddings_tensor = self.embeddings[i](dynamic_ids_tensor)
            # average B*M*E --AVG--> B*E
            dynamic_lengths_tensor = dynamic_lengths_tensor.unsqueeze(1)
            mask = (paddle.arange(paddle.shape(dynamic_embeddings_tensor).item(1)).unsqueeze(0).astype(float)< dynamic_lengths_tensor.unsqueeze(1))
            mask = mask.squeeze(1).unsqueeze(2)

            dynamic_embedding = self.masked_fill(dynamic_embeddings_tensor, mask == 0, 0)
       
            dynamic_lengths_tensor[dynamic_lengths_tensor == 0] = 1

            dynamic_embedding = (dynamic_embedding.sum(axis=1) / dynamic_lengths_tensor.astype('float32')).unsqueeze(1)
            concat_embeddings.append(paddle.reshape(dynamic_embedding,[batch_size,1,self.embedding_size]))
        # B*F*E
        concat_embeddings = paddle.concat(concat_embeddings, 1)

        return concat_embeddings


class StEmb(nn.Layer):
    def __init__(self, max_idxs, embedding_size=4):
        """
        fnames: feature names
        max_idxs: array of max_idx of each feature
        embedding_size: size of embedding
        dropout: prob for dropout, set None if no dropout
        use_cuda: bool, True for gpu or False for cpu
        """
        super(StEmb, self).__init__()
        self.max_idxs = max_idxs ## feature max    list
        self.embedding_size = embedding_size
        self.embeddings = nn.LayerList(
            [Meta_Embedding(max_idx + 1, self.embedding_size) for max_idx in self.max_idxs])

    def forward(self, static_ids):
        """
        input: relative id
        static_ids: Batch_size * Field_size
        return: Batch_size * Field_size * Embedding_size
        """
        concat_embeddings = []
        batch_size = static_ids.shape[0]
        feature_size = static_ids.shape[1]   ## batch * feature_size

        for i in range(feature_size):
            # B*1
            static_ids_tensor = static_ids[:,i]
            static_embeddings_tensor = self.embeddings[i](static_ids_tensor.astype('int64'))

            concat_embeddings.append(paddle.reshape(static_embeddings_tensor,[batch_size, 1, self.embedding_size]))
        # B*F*E
        concat_embeddings = paddle.concat(concat_embeddings, 1)
        return concat_embeddings


class MultiLayerPerceptron(nn.Layer):
    def __init__(self, input_dim, embed_dims):
        super().__init__()
        layers = []
        for embed_dim in embed_dims:
            layers.append(Meta_Linear(input_dim, embed_dim))
            layers.append(nn.ReLU())
            input_dim = embed_dim
        self.mlp = nn.LayerList(layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        out1 = self.mlp[0](x)
        out2 = self.mlp[1](out1)
        return out2


class WideAndDeepModel(nn.Layer):
    """
    A pytorch implementation of wide and deep learning.
    Reference:
        HT Cheng, et al. Wide & Deep Learning for Recommender Systems, 2016.
    """

    def __init__(self, max_ids, embed_dim, mlp_dims, num_expert, num_output):
        super().__init__()
        self.embedding = Emb(max_ids, embed_dim)
        self.embed_output_dim = (len(max_ids[0]) + len(max_ids[1])) * embed_dim
        self.ad_embed_dim = (len(max_ids[2]) + 1) * embed_dim 
        expert = []
        for i in range(num_expert):
            expert.append(MultiLayerPerceptron(self.embed_output_dim, mlp_dims))
        self.mlp = nn.LayerList(expert)
        output_layer = []
        for i in range(num_output):
            output_layer.append(Meta_Linear(mlp_dims[-1], 1))
        self.output_layer = nn.LayerList(output_layer)

        self.attention_layer = nn.Sequential(Meta_Linear(self.ad_embed_dim, mlp_dims[-1]),
                                                   nn.ReLU(),
                                                   Meta_Linear(mlp_dims[-1], num_expert),
                                                   nn.Softmax(axis=1))
        self.output_attention_layer = nn.Sequential(Meta_Linear(self.ad_embed_dim, mlp_dims[-1]),
                                                   nn.ReLU(),
                                                   Meta_Linear(mlp_dims[-1], num_output),
                                                   nn.Softmax(axis=1))

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        emb, ad_emb = self.embedding(x)
        ad_emb = paddle.concat([paddle.mean(emb, axis=1, keepdim=True), ad_emb], 1)

        fea = 0
        att = self.attention_layer(paddle.reshape(ad_emb,[-1,self.ad_embed_dim]))
        for i in range(len(self.mlp)):
            fea += (att[:, i].unsqueeze(1) * self.mlp[i](paddle.reshape(emb,[-1,self.embed_output_dim])))

        result = 0
        att2 = self.output_attention_layer(paddle.reshape(ad_emb,[-1,self.ad_embed_dim]))
        for i in range(len(self.output_layer)):
            result += (att2[:, i].unsqueeze(1) * F.sigmoid(self.output_layer[i](fea)))

        return result.squeeze(1)