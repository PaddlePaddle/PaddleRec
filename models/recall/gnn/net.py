#encoding=utf-8
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math

class MLP(nn.Layer):
    def __init__(self,insize,outsize,bias=True,act=True):
        super(MLP, self).__init__()
        self.stdv = 1 / math.sqrt(outsize)
        self.act = act
        if bias:
            self.linear=nn.Linear(insize,
                                  outsize,
                                  weight_attr=paddle.ParamAttr(
                                      initializer=nn.initializer.Uniform(low=-self.stdv,high=self.stdv)),
                                  bias_attr=paddle.ParamAttr(
                                      initializer=nn.initializer.Uniform(low=-self.stdv,high=self.stdv)))
        else:
            self.linear = nn.Linear(insize,
                                    outsize,
                                    weight_attr=paddle.ParamAttr(
                                        initializer=nn.initializer.Uniform(low=-self.stdv, high=self.stdv)),
                                    bias_attr=False)
        self.relu=nn.ReLU()

    def forward(self,x):
        if self.act:
            y=self.relu(self.linear.forward(x))
        else:
            y = self.linear.forward(x)
        return y

class SRGNN(nn.Layer):
    def __init__(self,dict_size,hidden_size,step,batch_size=None):
        super(SRGNN, self).__init__()
        self.dict_size = dict_size
        self.hidden_size = hidden_size
        self.step = step
        self.stdv=1/math.sqrt(self.hidden_size)
        self.batch_size = batch_size

        self.emb=nn.Embedding(
            self.dict_size,
            self.hidden_size,
            weight_attr=paddle.ParamAttr(
                name="SparseFeatFactors",
                initializer=nn.initializer.Uniform(low=-self.stdv,high=self.stdv)))

        self.state_in_update=[]
        self.state_out_update = []
        self.grus=[]
        for i in range(self.step):
            self.state_in_update.append(MLP(self.hidden_size,self.hidden_size))
            self.state_out_update.append(MLP(self.hidden_size,self.hidden_size))
            self.grus.append(nn.GRUCell(self.hidden_size*2,self.hidden_size))

        self.seq_lin=MLP(self.hidden_size,self.hidden_size,False)
        self.last_lin=MLP(self.hidden_size,self.hidden_size)
        self.att_weight=MLP(self.hidden_size,1,False)
        self.att_bias=paddle.create_parameter(shape=[self.hidden_size],
                                              dtype="float32",
                                              default_initializer=nn.initializer.Constant(value=0.0))
        self.final_att_fc=MLP(self.hidden_size*2,self.hidden_size,False)

    def forward(self, inputs):
        items_emb=self.emb(inputs[0])
        if self.batch_size is None:
            bs=inputs[0].shape[0]
        else:
            bs=self.batch_size
        uniq_len=inputs[0].shape[1]
        sql_len=inputs[1].shape[1]
        pre_state = items_emb
        for i in range(self.step):
            pre_state=paddle.reshape(pre_state,(bs,-1,self.hidden_size))
            state_in=self.state_in_update[i](pre_state)
            state_out=self.state_out_update[i](pre_state)
            state_adj_in=paddle.matmul(inputs[3],state_in)
            state_adj_out=paddle.matmul(inputs[4],state_out)
            gru_input=paddle.concat([state_adj_in,state_adj_out],axis=2)
            gru_input =paddle.reshape(gru_input,(-1,self.hidden_size*2))
            # print(gru_input.shape,pre_state.shape)
            _,pre_state=self.grus[i](gru_input,paddle.reshape(pre_state,(-1,self.hidden_size)))
        final_state=paddle.reshape(pre_state,(bs,-1,self.hidden_size))

        seq_len=inputs[1].shape[1]
        indices=paddle.arange(0,bs,dtype="int32")
        indices=paddle.reshape(indices,(-1,1))
        indices1=paddle.expand(indices,(bs,seq_len))
        indices1=paddle.unsqueeze(indices1,axis=-1)
        seq_indices=paddle.concat([indices1,inputs[1]],axis=2)
        last_indices=paddle.concat([indices,inputs[2]],axis=1)
        seq=paddle.gather_nd(final_state,seq_indices)
        last=paddle.gather_nd(final_state,last_indices)

        att=self.att_weight(F.sigmoid(paddle.add(paddle.add(seq.transpose([1,0,2]),last),self.att_bias)).transpose([1,0,2]))
        att *= inputs[5]
        # print(seq.shape,att.shape)
        att_mask=paddle.multiply(seq,att)
        global_att=paddle.sum(att_mask,axis=1)
        global_emd=self.final_att_fc(paddle.concat([global_att,last],axis=1))

        all_vocab=paddle.arange(1,self.dict_size,dtype="int64")
        all_emb=self.emb(all_vocab)

        probs=paddle.matmul(global_emd,all_emb,transpose_y=True)

        # loss = F.cross_entropy(input=probs, label=inputs[6])
        # acc=recallk(probs,inputs[6])

        return probs











