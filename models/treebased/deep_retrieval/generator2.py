import random
n = 1000
file_count = 5
emb_size = 10
item_count = 100
path = "./train_data/fake_data"
path_set = [ path + str(i) for i in range(file_count)]
list = [[] for i in range(file_count)]
arr = [i for i in range(item_count)]
for i in range(item_count):
    a = random.randint(0, item_count - i - 1) + i
    t = arr[i]
    arr[i] = arr[a]
    arr[a] = t

item_each = item_count//file_count
ind = 0
for i in range(file_count):
    if i < file_count - 1:
        count = item_each
    else:
        count = item_count - item_each * (file_count - 1)
    for j in range(count):
        list[i].append(arr[ind])
        ind = ind + 1

belong = {}
files = [open(path_set[i],"w") for i in range(file_count)]
for i in range(file_count):
    print(i, list[i])
    for j in list[i]:
        belong[j] = i

for i in range(n):
    t = random.randint(0,item_count-1)
    f = files[belong[t]]
    for j in range(emb_size):
        f.write(str(random.uniform(-1, 1)) + " ")
    f.write(str(t) + "\n")

for f in files:
    f.close()



# with open(path,"w") as f:
#     for i in range(n):
#         for j in range(emb_size):
#             f.write(str(random.uniform(-1,1)) + " ")
#         f.write(str(random.randint(0,item_count-1)) + "\n")




