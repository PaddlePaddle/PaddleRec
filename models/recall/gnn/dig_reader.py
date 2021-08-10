#encoding=utf-8

import numpy as np
from paddle.io import Dataset

class RecDataset(Dataset):
    def __init__(self,file_list,config=None):
        super(RecDataset, self).__init__()
        self.file_list = file_list
        self.config = config
        self.init()

    def init(self,):
        res = []
        global_max_seq_len=0
        global_max_uniq_len=0
        for f in self.file_list:
            with open(f, "r") as fin:
                for line in fin:
                    line = line.strip().split('\t')
                    seq=[int(l) for l in line[0].split(',')]
                    global_max_seq_len = max(global_max_seq_len, len(seq))
                    global_max_uniq_len = max(global_max_uniq_len, len(np.unique(seq)))
                    res.append(tuple([seq, int(line[1])]))
        self.input = res
        self.global_max_seq_len=global_max_seq_len
        #[0]这个pad数据下标不存在
        self.global_max_uniq_len=global_max_uniq_len+1
        print(global_max_seq_len,global_max_uniq_len)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        ses=self.input[idx][0]
        label=self.input[idx][1]
        last_id = len(ses) - 1
        ses += [0] * (self.global_max_seq_len - len(ses))
        nodes=np.unique(ses)
        items=nodes.tolist()+(self.global_max_uniq_len-len(nodes))*[0]

        adj=np.zeros((self.global_max_uniq_len,self.global_max_uniq_len))
        for i in range(len(ses)-1):
            if ses[i+1]==0:
                break
            u=np.where(nodes==ses[i])[0][0]
            v=np.where(nodes==ses[i+1])[0][0]
            adj[u][v]=1

        u_deg_in = np.sum(adj, 0)
        u_deg_in[np.where(u_deg_in == 0)] = 1
        adj_in=np.divide(adj, u_deg_in).transpose()
        u_deg_out = np.sum(adj, 1)
        u_deg_out[np.where(u_deg_out == 0)] = 1
        adj_out=np.divide(adj.transpose(), u_deg_out).transpose()

        seq_index=[np.where(nodes == i)[0][0] for i in ses]
        last_index=np.where(nodes == ses[last_id])[0][0]
        mask=[[1] * last_id + [0] *(self.global_max_seq_len - last_id)]

        items=np.array(items).astype("int64")
        seq_index=np.array(seq_index).astype("int32").reshape(-1,1)
        last_index = np.array([last_index]).astype("int32")
        adj_in = np.array(adj_in).astype("float32")
        adj_out = np.array(adj_out).astype("float32")
        mask = np.array(mask).astype("float32").reshape(-1, 1)
        label = np.array([label]).astype("int64")

        return items, seq_index, last_index, adj_in, adj_out, mask, label

if __name__=='__main__':
    import paddle
    from paddle.io import DataLoader
    import os
    import glob
    file_list = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),'data/train/*.txt'))
    dataset=RecDataset(file_list)

    use_gpu=True
    place = paddle.set_device('gpu' if use_gpu else 'cpu')
    loader = DataLoader(dataset, batch_size=12, places=place, drop_last=False)
    for batch_data in loader:
        print('batch_data',len(batch_data))
        for i in batch_data:
            print(i.shape)
        break
