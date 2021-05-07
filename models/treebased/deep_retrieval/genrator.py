import random
n = 1000
emb_size = 10
item_count = 1000
path = "./train_data/fake_data.txt"

with open(path,"w") as f:
    for i in range(n):
        for j in range(emb_size):
            f.write(str(random.uniform(-1,1)) + " ")
        f.write(str(random.randint(0,item_count-1)) + "\n")
