from __future__ import print_function, division
import numpy as np
import torch
import torch.utils.model_zoo as model_zoo
import random
import json
import os
import time
import copy
import traceback
from multiprocessing import Process
from client.bcosclient import BcosClient
from client.datatype_parser import DatatypeParser
from client.common.compiler import Compiler
from client_config import client_config
from fllib.update import LocalUpdate
from fllib.utils import agg_func, exp_details, get_dataset, proto_aggregation
from fllib.options import args_parser
from fllib.models.models import CNNMnist, CNNFemnist
from fllib.models.resnet import resnet18

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


# 序列化和反序列化
def serialize(data):
    json_data = json.dumps(data)
    return json_data


def deserialize(json_data):
    data = json.loads(json_data)
    return data


# 节点角色常量
ROLE_TRAINER = "trainer"  # 训练节点
ROLE_AGG = "agg"  # 聚合节点

# 轮询的时间间隔，单位秒
QUERY_INTERVAL = 5

# 从文件加载abi定义
if os.path.isfile(client_config.solc_path) or os.path.isfile(client_config.solcjs_path):
    Compiler.compile_file("contracts/CBHFLPrecompiled.sol")
abi_file = "contracts/CBHFLPrecompiled.abi"
data_parser = DatatypeParser()
data_parser.load_abi_file(abi_file)
contract_abi = data_parser.contract_abi

# 定义合约地址
to_address = "0x0000000000000000000000000000000000005006"


# 写一个节点的工作流程
def run_one_node(node_id, args, role, train_dataset, local_model, user_data):
    """指定一个node id，并启动一个进程"""

    trained_epoch = -1
    node_index = int(node_id.split('_')[-1])
    role = role

    local_model = local_model
    local_protos = None

    # 初始化bcos客户端
    try:
        client = BcosClient()
        # 为了更好模拟真实多个客户端场景，给不同的客户端分配不同的地址
        client.set_from_account_signer(node_id)
        print(f"{node_id} initializing....")
    except Exception as e:
        client.finish()
        traceback.print_exc()

    def local_training(global_epoch):
        print(f"{node_id} begin training, global epoch: {global_epoch}")
        try:
            global_protos, epoch = client.call(to_address, contract_abi, "QueryGlobalProtos")
            if global_protos == "":
                global_protos = []
            else:
                global_protos = deserialize(global_protos)

            nonlocal local_model
            model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_data)
            w, loss, acc, protos = model.update_weights_het(
                args, node_index, global_protos, model=copy.deepcopy(local_model), global_round=global_epoch)

            # 得到本地protos，然后序列化
            nonlocal local_protos
            local_protos = agg_func(protos)
            for k in local_protos:
                local_protos[k] = local_protos[k].tolist()
            proto_str = serialize(local_protos)
            client.sendRawTransactionGetReceipt(
                to_address, contract_abi, "UploadLocalProtos", [proto_str, epoch])

            local_model.load_state_dict(w, strict=True)

            nonlocal trained_epoch
            trained_epoch = epoch

        except Exception as _:
            client.finish()
            traceback.print_exc()

        return

    def local_aggregating():
        print(f"{node_id} begin aggregating..")
        try:
            updates, = client.call(to_address, contract_abi, "QueryProtosUpdates")
            updates = deserialize(updates)

            # 如果更新的节点数不够，就不进行聚合
            if len(updates) != args.num_users:
                return

            # 获取所有节点更新的protos，然后聚合
            update_protos = [torch.tensor(protos) for _, protos in updates.items()]
            global_protos = proto_aggregation(update_protos)

            _, epoch = client.call(to_address, contract_abi, "QueryCurrentEpoch")
            client.sendRawTransactionGetReceipt(
                to_address, contract_abi, "UpdateGlobalProtos", [global_protos, epoch])

            nonlocal trained_epoch
            trained_epoch = epoch

        except Exception as e:
            client.finish()
            traceback.print_exc()

        return

    def wait():
        time.sleep(random.uniform(QUERY_INTERVAL, QUERY_INTERVAL*3))
        return

    def main_loop():
        # 注册节点
        try:
            client.sendRawTransactionGetReceipt(
                to_address, contract_abi, "RegisterNode", [role])
            print("{} registered successfully".format(node_id))

            while True:
                global_epoch, = client.call(to_address, contract_abi, "QueryCurrentEpoch")

                if global_epoch > args.rounds:
                    break

                if global_epoch <= trained_epoch:
                    wait()
                    continue

                if role == ROLE_TRAINER:
                    local_training(global_epoch)

                if role == ROLE_AGG:
                    local_aggregating()

                wait()

        except Exception as e:
            client.finish()
            traceback.print_exc()

        return

    main_loop()

    # 关闭客户端
    client.finish()


if __name__ == "__main__":
    args = args_parser()
    exp_details(args)

    # set random seeds
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda':
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # load dataset and user groups
    n_list = np.random.randint(max(2, args.ways - args.stdev), min(
        args.num_classes, args.ways + args.stdev + 1), args.num_users)
    if args.dataset == 'mnist':
        k_list = np.random.randint(
            args.shots - args.stdev + 1, args.shots + args.stdev - 1, args.num_users)
    elif args.dataset == 'cifar10':
        k_list = np.random.randint(
            args.shots - args.stdev + 1, args.shots + args.stdev + 1, args.num_users)
    elif args.dataset == 'cifar100':
        k_list = np.random.randint(args.shots, args.shots + 1, args.num_users)
    elif args.dataset == 'femnist':
        k_list = np.random.randint(
            args.shots - args.stdev + 1, args.shots + args.stdev + 1, args.num_users)
    train_dataset, test_dataset, user_groups, user_groups_lt, classes_list, classes_list_gt = get_dataset(
        args, n_list, k_list)

    # Build models
    local_model_list = []
    for i in range(args.num_users):
        if args.dataset == 'mnist':
            if args.mode == 'model_heter':  # 根据不同的用户索引i，设置不同的out_channels，out_channels是干啥的？
                if i < 7:
                    args.out_channels = 18
                elif i >= 7 and i < 14:
                    args.out_channels = 20
                else:
                    args.out_channels = 22
            else:
                args.out_channels = 20

            local_model = CNNMnist(args=args)

        elif args.dataset == 'femnist':
            if args.mode == 'model_heter':
                if i < 7:
                    args.out_channels = 18
                elif i >= 7 and i < 14:
                    args.out_channels = 20
                else:
                    args.out_channels = 22
            else:
                args.out_channels = 20
            local_model = CNNFemnist(args=args)

        elif args.dataset == 'cifar100' or args.dataset == 'cifar10':
            if args.mode == 'model_heter':
                if i < 10:
                    args.stride = [1, 4]
                else:
                    args.stride = [2, 2]
            else:
                args.stride = [2, 2]
            resnet = resnet18(args, pretrained=False,
                              num_classes=args.num_classes)
            initial_weight = model_zoo.load_url(model_urls['resnet18'])
            local_model = resnet
            initial_weight_1 = local_model.state_dict()
            for key in initial_weight.keys():
                if key[0:3] == 'fc.' or key[0:5] == 'conv1' or key[0:3] == 'bn1':
                    initial_weight[key] = initial_weight_1[key]

            local_model.load_state_dict(initial_weight)

        local_model.to(args.device)
        local_model.train()
        local_model_list.append(local_model)

    process_pool = []
    for i in range(args.num_users):
        node_id = f'node_{i}'
        p = Process(target=run_one_node, args=(node_id, args, ROLE_TRAINER,
                    train_dataset, copy.deepcopy(local_model_list[i]), user_groups[i]))
        p.start()
        process_pool.append(p)
        time.sleep(1)

    p = Process(target=run_one_node, args=(
        f'node_{args.num_users}', args, ROLE_AGG, None, None, None))
    p.start()
    process_pool.append(p)
    # for i in range(args.num_users):
    #     run_one_node(f'node_{i}', args, ROLE_TRAINER,
    #                  train_dataset, local_model_list[i], user_groups[i])

    for p in process_pool:
        p.join()
