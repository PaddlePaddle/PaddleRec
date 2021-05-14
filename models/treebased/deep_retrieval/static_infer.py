from __future__ import print_function
import os
import warnings
import logging
import paddle
import sys
import paddle.distributed.fleet as fleet
import paddle.distributed.fleet.base.role_maker as role_maker

__dir__ = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../../tools')))

from utils.static_ps.reader_helper import get_reader
from utils.utils_single import load_yaml, load_static_model_class, get_abs_model, reset_auc
from utils.save_load import save_static_model, load_static_model
from operator import itemgetter, attrgetter
from paddle.io import DistributedBatchSampler, DataLoader
import numpy as np
import time
import paddle.static as static
import argparse
import paddle.fluid as fluid

logging.basicConfig(
    format='%(asctime)s - %(levelname)s- %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("PaddleRec train static script")
    parser.add_argument("-m", "--config_yaml", type=str)
    args = parser.parse_args()
    args.abs_dir = os.path.dirname(os.path.abspath(args.config_yaml))
    args.config_yaml = get_abs_model(args.config_yaml)
    return args


def main(args):
    paddle.seed(12345)

    config = load_yaml(args.config_yaml)
    config["yaml_path"] = args.config_yaml
    config["config_abs_dir"] = args.abs_dir
    model_save_path = config.get("runner.model_save_path", "model_output")
    # load static model class
    static_model_class = load_static_model_class(config)
    user_embedding_size = config.get(
        "hyper_parameters.user_embedding_size")
    input_data = paddle.static.data(
        name="user_embedding_input",
        shape=[None, user_embedding_size],
        dtype='float32')
    print("input_data.shape", input_data.shape)
    print(input_data.shape[0])
    print(input_data.shape[1])
    fetch_vars = static_model_class.net(input_data, is_infer=True, is_static=True)
    program1 = static_model_class.model.em_startup_program
    main_program = static_model_class.model.em_main_program
    with paddle.static.program_guard(main_program, program1):
        print("in sub graph")
        user_label = static.data(name="user_label", shape=[None], dtype="int64")
        score = static_model_class.model.rerank(input_data, user_label)
    logger.info("cpu_num: {}".format(os.getenv("CPU_NUM")))

    use_gpu = config.get("runner.use_gpu", True)
    data_dir = config.get("runner.test_data_dir", None)
    item_path_volume = config.get("hyper_parameters.item_path_volume")
    epoch = str(config.get("runner.infer_start_epoch", "0"))
    batch_size = config.get("runner.train_batch_size", None)
    width = config.get("hyper_parameters.width")
    recall_num = config.get("hyper_parameters.recall_num")
    static_user_embedding_size = config.get(
        "hyper_parameters.user_embedding_size")

    os.environ["CPU_NUM"] = str(config.get("runner.thread_num", 1))

    file_list = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]
    place = paddle.set_device('gpu' if use_gpu else 'cpu')
    exe = paddle.static.Executor(place)

    beam_size = static_model_class.beam_search_num * 4
    last_epoch_id = config.get("last_epoch", -1)
    exe.run(paddle.static.default_startup_program())
    exe.run(program1)
    print("model_save_path", model_save_path)
    print("epoch", epoch)
    load_static_model(
        paddle.static.default_main_program(),
        os.path.join(model_save_path, epoch),
        prefix='rec_static')
    static_model_class.load_item_path(model_save_path, epoch)
    batch_emb = []
    batch_items = []
    graph_index = static_model_class.graph_index._graph
    static_model_class.model.init_metrics()
    net_emb_input = []
    net_item_input = []
    net_lens = []
    label_list = []
    total_len = 0
    cur_len = 0

    def run_batch():
        nonlocal batch_emb, batch_items, net_emb_input, net_item_input, net_lens, label_list, total_len, cur_len,recall_num
        o = exe.run(
            program=paddle.static.default_main_program(),
            feed={"user_embedding_input": np.array(batch_emb).astype("float32")},
            fetch_list=[fetch_vars])
        path_list = np.array(o[0])
        for emb, kd_path, label in zip(batch_emb, path_list, batch_items):
            print("emb----",emb,kd_path,label)
            path = []
            for i in kd_path:
                path.append(static_model_class.graph_index.kd_represent_to_path_id(i))
            recall_items = graph_index.gather_unique_items_of_paths(path)
            if len(recall_items) < recall_num:
                print("recall ",recall_items, label)
                static_model_class.model.calculate_metric(recall_items, label)
                continue
            net_emb_input = net_emb_input + ([emb] * len(recall_items))
            total_len = total_len + len(recall_items)
            net_lens.append(len(recall_items))
            net_item_input = net_item_input + recall_items
            label_list.append(label)
            if total_len >= 1:
                outs = exe.run(program=main_program,
                               feed={'user_embedding_input': np.array(net_emb_input).astype("float32"),
                                     "user_label": np.array(net_item_input).astype("int64")},
                               fetch_list=[score])
                item_score = np.array(outs[0])
                cur_len = 0
                for i in range(len(net_lens)):
                    if net_lens[i] == 0:
                        static_model_class.model.calculate_metric([0], label_list[i])
                    else:
                        sub_list = [(x, y) for x, y in zip(item_score[cur_len:cur_len + net_lens[i]],
                                                           net_item_input[cur_len:cur_len + net_lens[i]])]
                        sub_list = sorted(sub_list, key=itemgetter(0), reverse=True)[0:recall_num]
                        sub_item_list = [int(x[1]) for x in sub_list ]
                        static_model_class.model.calculate_metric(sub_item_list, label_list[i])
                    cur_len = cur_len + net_lens[i]
                total_len = 0
                cur_len = 0
                net_item_input = []
                net_lens = []
                net_emb_input = []
                label_list = []
        batch_items = []
        batch_emb = []
        acc, recall = static_model_class.model.final_metrics()
        logger.info("accuracy {}, recall {}".format(acc, recall))

    for file in file_list:
        with open(file, "r") as f:
            for l in f:
                arr = l.split(" ")
                batch_emb.append(arr[0:len(arr) - 1])
                batch_items.append([int(x) for x in arr[-1].split(",")])
                if len(batch_items) >= batch_size:
                    run_batch()

    if (len(batch_items) > 0):
        run_batch()


if __name__ == "__main__":
    paddle.enable_static()
    args = parse_args()
    main(args)

