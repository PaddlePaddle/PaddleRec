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
from utils.save_load import save_static_model
from operator import itemgetter, attrgetter
from paddle.io import DistributedBatchSampler, DataLoader
import numpy as np
import time
import argparse

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def create_data_loader(config, place, graph_index, mode="train"):
    if mode == "train":
        data_dir = config.get("runner.train_data_dir", None)
        batch_size = config.get('runner.train_batch_size', None)
        reader_path = config.get('runner.train_reader_path', 'reader')
    else:
        data_dir = config.get("runner.test_data_dir", None)
        batch_size = config.get('runner.infer_batch_size', None)
        reader_path = config.get('runner.infer_reader_path', 'reader')
    distributed_training = config.get("runner.distributed_training", True)
    data_partition = config.get("runner.data_partition", False)
    config_abs_dir = config.get("config_abs_dir", None)
    data_dir = os.path.join(config_abs_dir, data_dir)
    file_list = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]
    sub_file_list = fleet.util.get_file_shard(file_list) if distributed_training == True and data_partition == True else file_list
    print("sub_file_list = ",sub_file_list)
    user_define_reader = config.get('runner.user_define_reader', False)
    logger.info("reader path:{}".format(reader_path))
    from importlib import import_module
    reader_class = import_module(reader_path)
    dataset = reader_class.RecDataset(sub_file_list, config=config, graph_index=graph_index)
    loader = DataLoader(
        dataset, batch_size=batch_size, places=place, drop_last=True)
    return loader


def parse_args():
    parser = argparse.ArgumentParser("PaddleRec train static script")
    parser.add_argument("-m", "--config_yaml", type=str)
    args = parser.parse_args()
    args.abs_dir = os.path.dirname(os.path.abspath(args.config_yaml))
    args.config_yaml = get_abs_model(args.config_yaml)
    return args


def main(args):
    paddle.seed(12345)
    item_to_user_emb = {};

    def clear_item_to_user_emb():
        item_to_user_emb = {}

    def add_item_to_user_emb(item_id, user_emb):
        if not item_id in item_to_user_emb:
            item_to_user_emb[item_id] = []
        item_to_user_emb[item_id].append(user_emb)

    # load config

    config = load_yaml(args.config_yaml)
    config["yaml_path"] = args.config_yaml
    config["config_abs_dir"] = args.abs_dir
    # load static model class
    static_model_class = load_static_model_class(config)
    distributed_training = config.get("runner.distributed_training", True)
    if distributed_training:
        os.environ["PADDLE_WITH_GLOO"] = "1"
        role = role_maker.PaddleCloudRoleMaker(init_gloo = True)
        fleet.init(role)
    input_data = static_model_class.create_feeds()
    input_data_names = [data.name for data in input_data]

    fetch_vars = static_model_class.net(input_data,is_static = True)
    logger.info("cpu_num: {}".format(os.getenv("CPU_NUM")))
    strategy = None
    if distributed_training == True:
        strategy = fleet.DistributedStrategy()
        strategy.a_sync = True
        static_model_class.create_optimizer(strategy, fleet)
    else:
        static_model_class.create_optimizer()

    use_gpu = config.get("runner.use_gpu", True)
    use_auc = config.get("runner.use_auc", False)
    auc_num = config.get("runner.auc_num", 1)
    train_data_dir = config.get("runner.train_data_dir", None)
    item_path_volume = config.get("hyper_parameters.item_path_volume")
    epochs = config.get("runner.epochs", None)
    print_interval = config.get("runner.print_interval", None)
    model_save_path = config.get("runner.model_save_path", "model_output")
    model_init_path = config.get("runner.model_init_path", None)
    batch_size = config.get("runner.train_batch_size", None)
    width = config.get("hyper_parameters.width")
    static_user_embedding_size = config.get(
        "hyper_parameters.user_embedding_size")
    em_execution_interval = config.get("hyper_parameters.em_execution_interval", 4)
    pernalize_path_to_item_count = config.get("hyper_parameters.pernalize_path_to_item_count")

    os.environ["CPU_NUM"] = str(config.get("runner.thread_num", 1))
    logger.info("**************common.configs**********")
    logger.info(
        "use_gpu: {}, train_data_dir: {}, epochs: {}, print_interval: {}, model_save_path: {}".
            format(use_gpu, train_data_dir, epochs, print_interval,
                   model_save_path))
    logger.info("**************common.configs**********")

    place = paddle.set_device('gpu' if use_gpu else 'cpu')
    exe = paddle.static.Executor(place)

    beam_size = static_model_class.item_path_volume * 4
    last_epoch_id = config.get("last_epoch", -1)
    em_main_program = static_model_class.model.em_main_program
    em_startup_program = static_model_class.model.em_startup_program
    with paddle.static.program_guard(em_main_program, em_startup_program):
        static_user_emb = paddle.static.data(
            name="static_user_emb",
            shape=[None, static_user_embedding_size],
            dtype='float32')
        static_saved_path, static_final_prob = static_model_class.model.generate_candidate_path_for_item(
            static_user_emb, beam_size)
    if distributed_training:
        if fleet.is_server():
            print("start server")
            fleet.init_server()
            fleet.run_server()
            return

    train_dataloader = create_data_loader(config=config, place=place, graph_index=static_model_class.graph_index)
    exe.run(em_startup_program)
    exe.run(paddle.static.default_startup_program())
    if distributed_training:
        fleet.init_worker()
    for epoch_id in range(last_epoch_id + 1, epochs):
        epoch_begin = time.time()
        interval_begin = time.time()
        train_reader_cost = 0.0
        train_run_cost = 0.0
        total_samples = 0
        reader_start = time.time()
        record_item_to_user_emb = False
        if (epoch_id + 1) % em_execution_interval == 0:
            record_item_to_user_emb = True
            clear_item_to_user_emb()
        for batch_id, batch in enumerate(train_dataloader()):

            if record_item_to_user_emb:
                input_emb = np.array(batch[1])
                item_set = np.array(batch[0])
                for user_emb, items in zip(input_emb, item_set):
                    add_item_to_user_emb(items[0], user_emb)

            print("start to train----------------------------")
            # fetch_batch_var = dataloader_train(epoch_id, train_dataloader,
            #                                    input_data_names, fetch_vars,
            #                                    exe, config)
            train_start = time.time()
            fetch_batch_var = exe.run(
                program=paddle.static.default_main_program(),
                feed=dict(zip(input_data_names, batch)),
                fetch_list=[var for _, var in fetch_vars.items()])
            train_run_cost += time.time() - train_start
            total_samples += batch_size
            if batch_id % print_interval == 0:
                metric_str = ""
                for var_idx, var_name in enumerate(fetch_vars):
                    metric_str += "{}: {}, ".format(var_name,
                                                    fetch_batch_var[var_idx])
                logger.info(
                    "epoch: {}, batch_id: {}, ".format(epoch_id,
                                                       batch_id) + metric_str +
                    "avg_reader_cost: {:.5f} sec, avg_batch_cost: {:.5f} sec, avg_samples: {:.5f}, ips: {:.5f} images/sec".
                    format(train_reader_cost / print_interval, (
                            train_reader_cost + train_run_cost) / print_interval,
                           total_samples / print_interval, total_samples / (
                                   train_reader_cost + train_run_cost)))
                train_reader_cost = 0.0
                train_run_cost = 0.0
                total_samples = 0
            reader_start = time.time()
            print("train ends--------------------------------")
            # metric_str = ""
            # for var_idx, var_name in enumerate(fetch_vars):
            #     metric_str += "{}: {}, ".format(var_name,
            #                                     fetch_batch_var[var_idx])
            # logger.info("epoch: {} done, ".format(epoch_id) + metric_str +
            #             "epoch time: {:.2f} s".format(time.time() -
            #                                           epoch_begin))

        if record_item_to_user_emb:
            static_model_class.graph_index.reset_graph_mapping()
            item_to_path_map = {}
            item_to_path_score_map = {}            
            for key in item_to_user_emb:
                user_emb = item_to_user_emb[key]
                fetch_batch_var = exe.run(
                    program=em_main_program,
                    feed={"static_user_emb": np.array(user_emb)},
                    fetch_list=[static_saved_path.name, static_final_prob.name])
                path, pro = fetch_batch_var[0:2]
                list = []
                score_list = []                
                path = np.array(path).astype("int64")
                pro = np.array(pro).astype("float")
                if not pernalize_path_to_item_count:
                    for single_path, single_pro in zip(path, pro):
                        list.append((single_pro, single_path))
                    sorted(list, key=itemgetter(0), reverse=True)
                    topk = []
                    for i in range(static_model_class.item_path_volume):
                        topk.append(static_model_class.graph_index.kd_represent_to_path_id(list[i][1]))

                    topk = np.array(topk).astype("int64")
                    static_model_class.graph_index._graph.add_item(key, topk)
                else:
                    for single_path,single_pro in zip(path,pro):
                        list.append(static_model_class.graph_index.kd_represent_to_path_id(single_path))
                        score_list.append(single_pro)
                    item_to_path_map[key] = list
                    item_to_path_score_map[key] = score_list
            if pernalize_path_to_item_count:
                #print("in update j path",item_to_path_score_map,item_to_path_score_map)
                static_model_class.graph_index.update_Jpath_of_item(item_to_path_map,item_to_path_score_map, 10,0.1)
            path_dict = static_model_class.graph_index._graph.get_item_path_dict()
            print("display new path mapping")
            for key in path_dict:
                print(key, "------>", path_dict[key])
        if epoch_id == epochs - 1 and  distributed_training:
            print("enter")
            path_dict = static_model_class.graph_index._graph.get_item_path_dict()
            list = []
            for key in path_dict:
                list = list + [key]
                list = list + path_dict[key]
            print("before gather list",list)
            final_list = fleet.util.all_gather_list(list,"worker")
            static_model_class.graph_index._graph.reset_mapping()
            item_length_for_each = 1 + item_path_volume
            item_count = len(final_list)//item_length_for_each
            for i in range(item_count):
                start = i * item_length_for_each
                key = final_list[start]
                path_list = final_list[start + 1: start + item_length_for_each]
                static_model_class.graph_index._graph.add_item(key,path_list)

            path_dict = static_model_class.graph_index._graph.get_item_path_dict()
            print("display final path mapping")
            for key in path_dict:
                print(key, "------>", path_dict[key])
        if distributed_training:
            print("index - ",fleet.worker_index())
        if not distributed_training or fleet.worker_index() == 0:
            print("saved epoch_id",epoch_id)
            static_model_class.save_item_path(model_save_path, epoch_id)
            save_static_model(
                paddle.static.default_main_program(),
                model_save_path,
                epoch_id,
                prefix='rec_static')
        else:
            print("not the worker to save model")

    if distributed_training:
        fleet.stop_worker()



if __name__ == "__main__":
    paddle.enable_static()
    args = parse_args()
    main(args)

