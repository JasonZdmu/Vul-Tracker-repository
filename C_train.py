seed = 2023
import os

os.environ["PYTHONHASHSEED"] = str(seed)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import random

random.seed(seed)
import numpy as np

np.random.seed(seed)
import torch

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False


from transformers import AutoTokenizer, AutoModel
from torch_geometric.data import HeteroData
from typing import Dict, List, Union


import torch.nn.functional as F
from torch import nn

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv

from torch_geometric.data import HeteroData
import json
import sys

from itertools import chain

from genPyG import *
from genPairs import *
from genBatch import *
from model import *
from eval import *
from genMiniGraphs import genAllMiniGraphs

# %%
def get_all_data():
    with open("miniGraphs.json") as f:
        miniGraphs = json.load(f) #打开名为 “miniGraphs.json” 的文件，并将其内容加载到 miniGraphs 变量中。

    dataset1 = json.load(open("dataset1.json")) #打开dataset1.json,导入到变量dataset1
    dataset2 = json.load(open("dataset2.json"))
    dataset3 = json.load(open("dataset3.json"))

    #返回包含以上所有数据的元组
    return (miniGraphs, dataset1, dataset2, dataset3)



# %%
def init_model(device,pyg):
    criterion = torch.nn.BCELoss() #二元交叉熵损失函数，一种常用于二分类问题的损失函数。
    rgcnModel = RGCN(device, 768,1, 768*2,pyg) #初始化了一个 HAN 模型。
    rgcnModel = rgcnModel.to(device)#将模型移动到指定的设备上
    rankNetModel = rankNet(768 * 2)#初始化了一个 rankNet 模型，输入维度是 768 * 2。
    rankNetModel = rankNetModel.to(device)#将 rankNetModel 移动到指定的设备上
    

    optimizer = torch.optim.Adam(
        chain(rgcnModel.parameters(), rankNetModel.parameters()), lr=5e-6
    ) #定义了一个优化器，使用的是 Adam 算法。优化器会更新 hanModel 和 rankNetModel 的参数以最小化损失函数。学习率设置为 5e-6。

    return rgcnModel, rankNetModel, optimizer, criterion


# %%
def divide_lst(lst, n, k):#这个函数可以用于将一个列表分割成多个子列表，其中前 k - 1 个子列表的长度为 n，最后一个子列表包含剩余的元素。
    cnt = 0
    all_list = []
    for i in range(0, len(lst), n):#遍历 lst，步长为 n。每次循环处理一段长度为 n 的子列表。
        if cnt < k - 1:
            all_list.append(lst[i : i + n])
        else:
            all_list.append(lst[i:])
            break
        cnt = cnt + 1
    return all_list


# %%
def get_sub_minigraphs(fdirs, all_minigraphs):#这个函数可以用于从一个大的图集合中提取出一部分子图。
    sub_minigraphs = {}
    for fdir in fdirs:
        sub_minigraphs[fdir] = all_minigraphs[fdir]
    return sub_minigraphs


# %%
# used for k cross fold validation
def divide_minigraphs(all_minigraphs, k):#这个函数的主要目的是将 all_minigraphs 中的所有子图分割成 k 个部分。
    all_fdirs = []
    for fdir in all_minigraphs.keys():
        all_fdirs.append(fdir)  #遍历 all_minigraphs 中的每一个键 fdir，并将其添加到 all_fdirs 中。
    #random.shuffle(all_fdirs)

    all_sub_minigraphs = []
    all_sub_fdirs = []
    for sub_fdirs in divide_lst(all_fdirs, int(len(all_fdirs) / k), k): #将 all_fdirs 分割成 k 个部分，每个部分包含 int(len(all_fdirs) / k) 个元素。然后，遍历每一个部分 sub_fdirs。
        if len(sub_fdirs) == 0:
            continue
        all_sub_fdirs.append(sub_fdirs)
        all_sub_minigraphs.append(get_sub_minigraphs(sub_fdirs, all_minigraphs))

    return all_sub_minigraphs, all_sub_fdirs


# %%
def get_all_batchlist(mini_graphs, k, max_pair):#这个函数的主要目的是将 mini_graphs 中的所有子图分割成 k 个部分，然后为每个部分生成一组批次列表。
    all_batch_list = []
    pair_cnt = 0
    all_sub_minigraphs, all_sub_fdirs = divide_minigraphs(mini_graphs, k)

    for sub_minigraph in all_sub_minigraphs:
        all_pairs = get_all_pairs(sub_minigraph, max_pair)#使用 get_all_pairs 函数为 sub_minigraph 生成一组对，最多 max_pair 对。
        pair_cnt = pair_cnt + len(all_pairs)#更新对的总数 pair_cnt。
        batch_list = combinePair(all_pairs, 128)#使用 combinePair 函数将 all_pairs 组合成一组批次，每个批次包含 128 对。
        all_batch_list.append(batch_list)#将 batch_list 添加到 all_batch_list 中。

    return all_batch_list, all_sub_fdirs, pair_cnt



# %%
# %%
def train_batchlist(batches, rgcnModel, rankNetModel, optimizer, criterion, device):
    #这个函数的主要目的是训练模型并计算损失。
    all_loss = []#初始化一个空列表 all_loss，用于存储每个批次的损失。
    #将 hanModel 和 rankNetModel 设置为训练模式。
    rgcnModel.train()
    rankNetModel.train()

    for batch in batches:
        pyg1 = batch.pyg1.clone().to(device)#将 batch 中的 pyg1 和 pyg2 克隆并移动到指定的设备上（CPU 或 GPU）。
        pyg2 = batch.pyg2.clone().to(device)

        del_index1 = batch.del_index1.to(device)
        del_index2 = batch.del_index2.to(device)

        probs = batch.probs.to(device)
        x = rgcnModel(pyg1, del_index1)#通过 hanModel 计算 pyg1 和 pyg2 的输出。
        y = rgcnModel(pyg2, del_index2)

        optimizer.zero_grad()#在进行新的优化之前，先将优化器中的梯度清零。
        preds = rankNetModel(x, y)#通过 rankNetModel 计算预测值 preds。
        loss = criterion(preds, probs)#使用损失函数 criterion 计算 preds 和 probs 之间的损失。


        loss.backward()#计算损失的反向传播。
        optimizer.step()#更新模型的参数。

        all_loss.append(loss.cpu().detach().item())#将损失转移到 CPU，从计算图中分离出来，然后转换为 Python 数字，并添加到 all_loss 中。

    return sum(all_loss)#函数返回所有批次的总损失。


        commit_index = batch
        loss_list = list()
        commit_index_list = list()
        confidence_mul_loss_list = list()

        lm_logits = loss.logits
        loss_fct = torch.nn.BCELoss(ignore_index=0, reduction='none')
        # logger.info("lm_logits.size:{}".format(lm_logits.size))
        loss_batch = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), target_ids.view(-1)).view(-1,
                                                                                                    lm_logits.size(1))
        loss_batch = torch.nanmean(loss_batch.masked_fill(target_ids == preds, torch.nan), dim=1)
        loss_list += loss_batch.tolist()

        commit_index_list += commit_index.tolist()

        loss_batch = [loss * confidence_list[commit_index[idx]] for idx, loss in enumerate(loss_batch)]
        confidence_mul_loss_list += loss_batch
        loss_batch = torch.stack(loss_batch, dim=0)
        loss = torch.mean(loss_batch)


            # Get distribution by EM algorithm
            ratio, avg, std = EM(torch.tensor(score_list, device=args.device, dtype=torch.float64))
            if avg[0] < avg[1]:
                clean_idx = 0
            else:
                clean_idx = 1
            # Get Confidence of loss of each commit
            # NOT IN ORDER
            logger.info("Calculating the clean_weight ...")
            with multiprocessing.Pool(min(multiprocessing.cpu_count(), args.max_cpu_count)) as pool:
                commit_data = [(idx, score, ratio[clean_idx], avg[clean_idx], std[clean_idx] * 9) for idx, score in
                               enumerate(score_list)]
                idx_clean_weight_list = pool.map(cal_Prob, commit_data)
            logger.info("Calculating the noisy_weight ...")
            with multiprocessing.Pool(min(multiprocessing.cpu_count(), args.max_cpu_count)) as pool:
                commit_data = [(idx, score, ratio[1 - clean_idx], avg[1 - clean_idx], std[1 - clean_idx]) for idx, score
                               in enumerate(score_list)]
                idx_noisy_weight_list = pool.map(cal_Prob, commit_data)
            # IN ORDER
            sorted_idx_clean_weight_list = sorted(idx_clean_weight_list, key=lambda x: x[0])
            sorted_idx_noisy_weight_list = sorted(idx_noisy_weight_list, key=lambda x: x[0])
            clean_weight_list = [idx_weight[1] for idx_weight in sorted_idx_clean_weight_list]
            noisy_weight_list = [idx_weight[1] for idx_weight in sorted_idx_noisy_weight_list]

            # clean_weight_list = [cal_Prob(each_data, ratio[clean_idx], avg[clean_idx], std[clean_idx]*9) for each_data in score_list]
            # noisy_weight_list = [cal_Prob(each_data, ratio[1-clean_idx], avg[1-clean_idx], std[1-clean_idx]) for each_data in score_list]
            logger.info("Calculating the confidence ...")
            confidence_list = []
            for idx in range(len(score_list)):
                confidence_list.append((clean_weight_list[idx] + (np.power(args.base_number, -score_list[idx])) *
                                        noisy_weight_list[idx]) / (clean_weight_list[idx] + noisy_weight_list[idx]))
            os.makedirs(args.res_dir, exist_ok=True)
            with open(os.path.join(args.res_dir, "confidence_list_cur_epoch_{}.pickle".format(cur_epoch)), "wb") as f:
                pickle.dump(confidence_list, f)
            logger.info("confidence_list_cur_epoch_{}.pickle saved.".format(cur_epoch))





# %%
# #这个函数的主要目的是在给定的设备上验证模型的性能，并计算损失。
def validate_batchlist(batches, rgcnModel, rankNetModel, criterion, device):
    all_loss = []
    rgcnModel.eval()
    rankNetModel.eval()#将 hanModel 和 rankNetModel 设置为评估模式。

    for batch in batches:
        with torch.no_grad(): #在这个上下文管理器中，PyTorch 不会计算和存储梯度，这可以节省内存并加速计算，特别是在评估模式下
            pyg1 = batch.pyg1.clone().to(device)
            pyg2 = batch.pyg2.clone().to(device)

            del_index1 = batch.del_index1.to(device)
            del_index2 = batch.del_index2.to(device)

            probs = batch.probs.to(device)
            x = rgcnModel(pyg1, del_index1)#通过 hanModel 计算 pyg1 和 pyg2 的输出。
            y = rgcnModel(pyg2, del_index2)

            preds = rankNetModel(x, y)#通过 rankNetModel 计算预测值 preds。
            loss = criterion(preds, probs)#使用损失函数 criterion 计算 preds 和 probs 之间的损失。
            all_loss.append(loss.cpu().detach().item())#将损失转移到 CPU，从计算图中分离出来，然后转换为 Python 数字，并添加到 all_loss 中。

    return sum(all_loss) #函数返回所有批次的总损失。


def do_cross_fold_valid(device, K):
    all_mini_graphs, dataset1, dataset2, dataset3 = get_all_data()#使用 get_all_data 函数获取所有的数据和小图
    all_data = []

    high_ranking_folders = {}

    all_data.extend(dataset1)
    all_data.extend(dataset2)
    all_data.extend(dataset3)
    # print(all_data)
    # print(np.array(all_data).shape)

    random.shuffle(all_data)

    all_data_list = divide_lst(all_data, int(len(all_data) * 0.1), K)#使用 divide_lst 函数将 all_data 分割成 K 个部分，每个部分包含 len(all_data) * 0.1 个元素。
    for i in range(0, len(all_data_list)):
        testset = all_data_list[i]#对于 all_data_list 中的每个部分，将其作为测试集，其余部分作为训练集。
        trainset = []

        for j in range(len(all_data_list)):
            if j != i:
                trainset.extend(all_data_list[j])

        random.shuffle(trainset)

        max_pair = 100
        #从 all_mini_graphs 中提取出 trainset 指定的子图。
        mini_graphs = get_sub_minigraphs(trainset, all_mini_graphs)

        all_batch_list, all_sub_fdirs, pair_cnt = get_all_batchlist(
            mini_graphs, 1, max_pair=max_pair
        )
        #获取一些可能用于后续处理的映射和数据。
        all_true_cid_map = get_true_cid_map(all_data)
        dir_to_minigraphs = get_dir_to_minigraphs(
            get_sub_minigraphs(all_data, all_mini_graphs)
        )
        # print(trainset)
        # print(dir_to_minigraphs)



        rgcnModel, rankNetModel, optimizer, criterion = init_model(
            device, all_batch_list[0][0].pyg1
        )#初始化模型和训练设置。



        epochs = 20

        all_info = []
        test_info = []
        for epoch in range(epochs):

            total_train_loss = 0
            total_tp1 = 0
            total_fp1 = 0
            total_tp2 = 0
            total_fp2 = 0
            total_tp3 = 0
            total_fp3 = 0
            total_t = 0

            total_train_loss = total_train_loss + train_batchlist(
                all_batch_list[0], rgcnModel, rankNetModel, optimizer, criterion, device
            )

            eval(trainset, dir_to_minigraphs, rgcnModel, rankNetModel, device)

            tp1, fp1, t = eval_top(
                trainset,
                dir_to_minigraphs,
                rgcnModel,
                rankNetModel,
                device,
                all_true_cid_map,
                1,
            )
            tp2, fp2, t = eval_top(
                trainset,
                dir_to_minigraphs,
                rgcnModel,
                rankNetModel,
                device,
                all_true_cid_map,
                2,
            )
            tp3, fp3, t = eval_top(
                trainset,
                dir_to_minigraphs,
                rgcnModel,
                rankNetModel,
                device,
                all_true_cid_map,
                3,
            )
            total_t = total_t + t
            total_tp1 = total_tp1 + tp1
            total_fp1 = total_fp1 + fp1
            total_tp2 = total_tp2 + tp2
            total_fp2 = total_fp2 + fp2
            total_tp3 = total_tp3 + tp3
            total_fp3 = total_fp3 + fp3
            cur_f1_score = (
                2 * (total_tp1 / (total_tp1 + total_fp1)) * (total_tp1 / total_t)
            ) / (total_tp1 / (total_tp1 + total_fp1) + total_tp1 / total_t)
            info = {}
            info["epoch"] = epoch
            info["pair_cnt"] = pair_cnt
            info["train_loss"] = total_train_loss
            info["train_f1_score"] = (
                2 * (total_tp1 / (total_tp1 + total_fp1)) * (total_tp1 / total_t)
            ) / (total_tp1 / (total_tp1 + total_fp1) + total_tp1 / total_t)
            info["train_top1_f1_precision"] = total_tp1 / (total_tp1 + total_fp1)
            info["train_top1_f1_recall"] = total_tp1 / total_t
            info["train_top2_f1_precision"] = total_tp2 / (total_tp2 + total_fp2)
            info["train_top2_f1_recall"] = total_tp2 / total_t
            info["train_top3_f1_precision"] = total_tp3 / (total_tp3 + total_fp3)
            info["train_top3_f1_recall"] = total_tp3 / total_t
            
            total_tp1 = 0
            total_fp1 = 0
            total_tp2 = 0
            total_fp2 = 0
            total_tp3 = 0
            total_fp3 = 0
            total_t = 0
            eval(testset, dir_to_minigraphs, rgcnModel, rankNetModel, device)
            tp1, fp1, t = eval_top(
                testset,
                dir_to_minigraphs,
                rgcnModel,
                rankNetModel,
                device,
                all_true_cid_map,
                1,
            )
            tp2, fp2, t = eval_top(
                testset,
                dir_to_minigraphs,
                rgcnModel,
                rankNetModel,
                device,
                all_true_cid_map,
                2,
            )
            tp3, fp3, t = eval_top(
                testset,
                dir_to_minigraphs,
                rgcnModel,
                rankNetModel,
                device,
                all_true_cid_map,
                3,
            )
            total_t = total_t + t
            total_tp1 = total_tp1 + tp1
            total_fp1 = total_fp1 + fp1
            total_tp2 = total_tp2 + tp2
            total_fp2 = total_fp2 + fp2
            total_tp3 = total_tp3 + tp3
            total_fp3 = total_fp3 + fp3
            cur_f1_score = (
                2 * (total_tp1 / (total_tp1 + total_fp1)) * (total_tp1 / total_t)
            ) / (total_tp1 / (total_tp1 + total_fp1) + total_tp1 / total_t)
            info["test_f1_score"] = (
                2 * (total_tp1 / (total_tp1 + total_fp1)) * (total_tp1 / total_t)
            ) / (total_tp1 / (total_tp1 + total_fp1) + total_tp1 / total_t)
            info["test_top1_f1_precision"] = total_tp1 / (total_tp1 + total_fp1)
            info["test_top1_f1_recall"] = total_tp1 / total_t
            info["test_top2_f1_precision"] = total_tp2 / (total_tp2 + total_fp2)
            info["test_top2_f1_recall"] = total_tp2 / total_t
            info["test_top3_f1_precision"] = total_tp3 / (total_tp3 + total_fp3)
            info["test_top3_f1_recall"] = total_tp3 / total_t
            info["test recall@top1"] = eval_recall_topk(testset, dir_to_minigraphs, 1)
            info["test recall@top2"] = eval_recall_topk(testset, dir_to_minigraphs, 2)
            info["test recall@top3"] = eval_recall_topk(testset, dir_to_minigraphs, 3)
            info["mean_first_rank"] = eval_mean_first_rank(testset, dir_to_minigraphs, high_ranking_folders, epoch)
            all_info.append(info)



            with open(f"./crossfold_result/{i}_{epoch}.json", "w") as f:
                json.dump(all_info, f)



    with open("output.json", 'w') as json_file:
                json.dump(high_ranking_folders, json_file)


def do_cross_project_predict(device):
    all_mini_graphs, dataset1, dataset2, dataset3 = get_all_data()
    
    high_ranking_folders = {}
    
    testset = dataset1
    trainset = []
    trainset.extend(dataset2)
    trainset.extend(dataset3)
    all_data = []
    all_data.extend(dataset1)
    all_data.extend(dataset2)
    all_data.extend(dataset3)

    # print(trainset)
    # print(np.array(trainset).shape)

    max_pair = 100
    mini_graphs = get_sub_minigraphs(trainset, all_mini_graphs)
    all_batch_list, all_sub_fdirs, pair_cnt = get_all_batchlist(
        mini_graphs, 1, max_pair=max_pair
    )
    all_true_cid_map = get_true_cid_map(all_data)
    dir_to_minigraphs = get_dir_to_minigraphs(
        get_sub_minigraphs(all_data, all_mini_graphs)
    )
    # print(trainset)
    # print(dir_to_minigraphs)
    gcnModel, rankNetModel, optimizer, criterion = init_model(
        device, all_batch_list[0][0].pyg1
    )
    epochs = 20
    all_info = []
    for epoch in range(epochs):
        total_train_loss = 0
        total_tp1 = 0
        total_fp1 = 0
        total_tp2 = 0
        total_fp2 = 0
        total_tp3 = 0
        total_fp3 = 0
        total_t = 0
        total_train_loss = total_train_loss + train_batchlist(
            all_batch_list[0], gcnModel, rankNetModel, optimizer, criterion, device
        )
        eval(trainset, dir_to_minigraphs, gcnModel, rankNetModel, device)
        tp1, fp1, t = eval_top(
            trainset,
            dir_to_minigraphs,
            gcnModel,
            rankNetModel,
            device,
            all_true_cid_map,
            1,
        )
        tp2, fp2, t = eval_top(
            trainset,
            dir_to_minigraphs,
            gcnModel,
            rankNetModel,
            device,
            all_true_cid_map,
            2,
        )
        tp3, fp3, t = eval_top(
            trainset,
            dir_to_minigraphs,
            gcnModel,
            rankNetModel,
            device,
            all_true_cid_map,
            3,
        )
        total_t = total_t + t
        total_tp1 = total_tp1 + tp1
        total_fp1 = total_fp1 + fp1
        total_tp2 = total_tp2 + tp2
        total_fp2 = total_fp2 + fp2
        total_tp3 = total_tp3 + tp3
        total_fp3 = total_fp3 + fp3
        cur_f1_score = (
            2 * (total_tp1 / (total_tp1 + total_fp1)) * (total_tp1 / total_t)
        ) / (total_tp1 / (total_tp1 + total_fp1) + total_tp1 / total_t)
        info = {}
        info["epoch"] = epoch
        info["pair_cnt"] = pair_cnt
        info["train_loss"] = total_train_loss
        info["train_f1_score"] = (
            2 * (total_tp1 / (total_tp1 + total_fp1)) * (total_tp1 / total_t)
        ) / (total_tp1 / (total_tp1 + total_fp1) + total_tp1 / total_t)
        info["train_top1_f1_precision"] = total_tp1 / (total_tp1 + total_fp1)
        info["train_top1_f1_recall"] = total_tp1 / total_t
        info["train_top2_f1_precision"] = total_tp2 / (total_tp2 + total_fp2)
        info["train_top2_f1_recall"] = total_tp2 / total_t
        info["train_top3_f1_precision"] = total_tp3 / (total_tp3 + total_fp3)
        info["train_top3_f1_recall"] = total_tp3 / total_t
        
        total_tp1 = 0
        total_fp1 = 0
        total_tp2 = 0
        total_fp2 = 0
        total_tp3 = 0
        total_fp3 = 0
        total_t = 0
        eval(testset, dir_to_minigraphs, gcnModel, rankNetModel, device)
        tp1, fp1, t = eval_top(
            testset,
            dir_to_minigraphs,
            gcnModel,
            rankNetModel,
            device,
            all_true_cid_map,
            1,
        )
        tp2, fp2, t = eval_top(
            testset,
            dir_to_minigraphs,
            gcnModel,
            rankNetModel,
            device,
            all_true_cid_map,
            2,
        )
        tp3, fp3, t = eval_top(
            testset,
            dir_to_minigraphs,
            gcnModel,
            rankNetModel,
            device,
            all_true_cid_map,
            3,
        )
        total_t = total_t + t
        total_tp1 = total_tp1 + tp1
        total_fp1 = total_fp1 + fp1
        total_tp2 = total_tp2 + tp2
        total_fp2 = total_fp2 + fp2
        total_tp3 = total_tp3 + tp3
        total_fp3 = total_fp3 + fp3
        cur_f1_score = (
            2 * (total_tp1 / (total_tp1 + total_fp1)) * (total_tp1 / total_t)
        ) / (total_tp1 / (total_tp1 + total_fp1) + total_tp1 / total_t)
        info["test_f1_score"] = (
            2 * (total_tp1 / (total_tp1 + total_fp1)) * (total_tp1 / total_t)
        ) / (total_tp1 / (total_tp1 + total_fp1) + total_tp1 / total_t)
        info["test_top1_f1_precision"] = total_tp1 / (total_tp1 + total_fp1)
        info["test_top1_f1_recall"] = total_tp1 / total_t
        info["test_top2_f1_precision"] = total_tp2 / (total_tp2 + total_fp2)
        info["test_top2_f1_recall"] = total_tp2 / total_t
        info["test_top3_f1_precision"] = total_tp3 / (total_tp3 + total_fp3)
        info["test_top3_f1_recall"] = total_tp3 / total_t
        info["test recall@top1"] = eval_recall_topk(testset, dir_to_minigraphs, 1)
        info["test recall@top2"] = eval_recall_topk(testset, dir_to_minigraphs, 2)
        info["test recall@top3"] = eval_recall_topk(testset, dir_to_minigraphs, 3)
        info["mean_first_rank"] = eval_mean_first_rank(testset,dir_to_minigraphs,high_ranking_folders, epoch)
        all_info.append(info)
       
        with open(f"./crossproject_result/{epoch}.json", "w") as f:
            json.dump(all_info, f)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available()  else 'cpu')
    # device = ''
    genAllMiniGraphs()
    #do_cross_fold_valid(device, 10)
    do_cross_project_predict(device)
