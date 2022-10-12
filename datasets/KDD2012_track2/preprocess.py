# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from tqdm import tqdm
import random


def preprocess():
    ## preprocess origin dataset, filter out the required parameters
    ## The parameters needed is as below:
    ## click - 0
    ## adID  - 3
    ## advertiseID => cate? - 4
    ## position - 6
    ## userID - -1
    ad_id_dict = {}
    advertise_id_dict = {}
    cnt_ad = 1
    cnt_cate = 1
    fi = open("training.txt", "r")
    fo = open("train_phrase1.txt", "w")
    # All data: 149639106
    print("Step 1. Preprocess origin dataset...")
    for i in tqdm(range(149639106)):
        line = fi.readline()
        if not line:
            break
        else:
            line = line.strip('\n')
            line = line.split("\t")
            ## if click not in [0, 1], then change it to 1
            ## because 'click' maybe greater than 1
            if line[0] != '0' and line[0] != '1':
                line[0] = '1'
            ## The parameters adID and advertiseID are needed to be remarked
            ad_id = line[3]
            advertise_id = line[4]
            if ad_id_dict.get(ad_id, -1) == -1:
                ad_id_dict[ad_id] = cnt_ad
                ad_id = cnt_ad
                cnt_ad += 1
            else:
                ad_id = ad_id_dict.get(ad_id)
            if advertise_id_dict.get(advertise_id, -1) == -1:
                advertise_id_dict[advertise_id] = cnt_cate
                advertise_id = cnt_cate
                cnt_cate += 1
            else:
                advertise_id = advertise_id_dict.get(advertise_id)
            fo.writelines("{};{};{};{};{}\n".format(line[
                -1], ad_id, advertise_id, line[6], line[0]))
    fi.close()
    fo.close()


def gen_DIN(num=5000000):
    #$ Step 2. Generate dataset for model DIN(optional)
    fi = open("train_phrase1.txt", "r")
    fo = open("train_din_tmp.txt", "w")
    user_id = ""
    max_ad = 0
    max_context = 0
    his_ad = []
    his_cat = []
    cnt_total = 0
    print("Step 2.1 Generate dataset for model DIN(optional)...")
    for i in tqdm(range(num)):
        line = fi.readline()
        if not line:
            break
        else:
            line = line.strip('\n')
            line = line.split(";")
            max_ad = max(max_ad, int(line[1]))
            max_context = max(max_context, int(line[2]))
            if user_id != line[0]:
                # reset
                his_ad.clear()
                his_cat.clear()
                user_id = line[0]
            else:
                ## If click = 1, then append the data to his.
                ## If his is not null, then add positive cases.
                if line[4] == '1':
                    if len(his_ad) != 0:
                        cnt_total += 1
                        fo.writelines("{};{};{};{};{};{}\n".format(' '.join(
                            his_ad), ' '.join(his_cat), line[1], line[2], line[
                                3], line[4]))
                    his_ad.append(line[1])
                    his_cat.append(line[2])
                ## If click = 0 and his is not null, then add negative cases.
                if line[4] == '0':
                    if len(his_ad) != 0:
                        cnt_total += 1
                        fo.writelines("{};{};{};{};{};{}\n".format(' '.join(
                            his_ad), ' '.join(his_cat), line[1], line[2], line[
                                3], line[4]))
    fi.close()
    fo.close()
    print("Total dataset lines : ", cnt_total)
    print("Step 2.2 Partitioning the dataset for model DIN...")
    fi = open("train_din_tmp.txt", "r")
    fo_train = open("train_din.txt", "w")
    fo_test = open("test_din.txt", "w")
    cnt_train = 0
    cnt_test = 0
    random.seed(2022)
    train_set = []
    test_set = []
    for i in tqdm(range(cnt_total)):
        line = fi.readline()
        if random.random() <= 0.2:
            cnt_test += 1
            test_set.append(line)
        else:
            cnt_train += 1
            train_set.append(line)
    ## shuffle
    print("Step 2.3 Shuffling the dataset...")
    random.shuffle(train_set)
    random.shuffle(test_set)
    ## save to file
    print("Step 2.4 Saving to file...")
    for _, line in enumerate(tqdm(train_set)):
        fo_train.writelines(line)
    for _, line in enumerate(tqdm(test_set)):
        fo_test.writelines(line)
    print("Train dataset lines : ", cnt_train)
    print("Test dataset lines : ", cnt_test)
    print("---" * 20)
    print(
        "Please remember the result as is shown below, \nyou have to copy them to file 'config.yaml' or 'config_bigdata.yaml'"
    )
    print("max_item", max_ad)
    print("max_context", max_context)
    print("---" * 20)


def gen_DPIN(num=5000000):
    ## Step 3. Generate dataset for model DPIN
    fi = open("train_phrase1.txt", "r")
    fo = open("train_dpin_tmp.txt", "w")
    user_id = ""
    max_ad = 0
    max_context = 0
    his_ad = []
    his_cat = []
    his_pos = []
    cnt_total = 0
    print("Step 3.1 Generating dataset for model DPIN...")
    for i in tqdm(range(num)):
        line = fi.readline()
        if not line:
            break
        else:
            line = line.strip('\n')
            line = line.split(";")
            max_ad = max(max_ad, int(line[1]))
            max_context = max(max_context, int(line[2]))
            if user_id != line[0]:
                # reset
                his_ad.clear()
                his_cat.clear()
                his_pos.clear()
                user_id = line[0]
            else:
                ## If click = 1, then append the data to his.
                ## If his is not null, then add positive cases.
                if line[4] == '1':
                    if len(his_ad) != 0:
                        cnt_total += 1
                        fo.writelines("{};{};{};{};{};{};{}\n".format(' '.join(
                            his_ad), ' '.join(his_cat), ' '.join(
                                his_pos), line[1], line[2], line[3], line[4]))
                    his_ad.append(line[1])
                    his_cat.append(line[2])
                    his_pos.append(line[3])
                ## If click = 0 and his is not null, then add negative cases.
                if line[4] == '0':
                    if len(his_ad) != 0:
                        cnt_total += 1
                        fo.writelines("{};{};{};{};{};{};{}\n".format(' '.join(
                            his_ad), ' '.join(his_cat), ' '.join(
                                his_pos), line[1], line[2], line[3], line[4]))
    fi.close()
    fo.close()
    print("Total dataset lines : ", cnt_total)
    print("Step 3.2 Partitioning the dataset for model DPIN...")
    fi = open("train_dpin_tmp.txt", "r")
    fo_train = open("train_dpin.txt", "w")
    fo_test = open("test_dpin.txt", "w")
    cnt_train = 0
    cnt_test = 0
    random.seed(2022)
    train_set = []
    test_set = []
    for i in tqdm(range(cnt_total)):
        line = fi.readline()
        if random.random() <= 0.2:
            cnt_test += 1
            test_set.append(line)
        else:
            cnt_train += 1
            train_set.append(line)
    ## shuffle
    print("Step 3.3 Shuffling the dataset...")
    random.shuffle(train_set)
    random.shuffle(test_set)
    ## save to file
    print("Step 3.4 Saving to file...")
    for _, line in enumerate(tqdm(train_set)):
        fo_train.writelines(line)
    for _, line in enumerate(tqdm(test_set)):
        fo_test.writelines(line)
    print("Train dataset lines : ", cnt_train)
    print("Test dataset lines : ", cnt_test)
    print("---" * 20)
    print(
        "Please remember the result as is shown below, \nyou have to copy them to file 'config.yaml' or 'config_bigdata.yaml'"
    )
    print("max_item", max_ad)
    print("max_context", max_context)
    print("---" * 20)


if __name__ == '__main__':
    # Step 1. Preprocess the data download from KDD Cup 2012, Track 2
    # !!!! Please make sure that you have downloaded the data from Kaggle !!!
    # The data input is stored at 'training.txt'
    # The processed data will be stored at 'train_phrase1.txt'
    preprocess()

    # Step 2. Generate dataset for model DIN(optional)
    # The processed data will be stored at 'train_din.txt'
    # gen_DIN(50000000)

    # Step 3. Generate dataset for model DPIN
    # The processed data will be stored at 'train_dpin.txt'
    gen_DPIN(50000000)
