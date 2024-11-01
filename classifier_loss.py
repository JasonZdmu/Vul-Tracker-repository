# giang temporarily switch EnsembleModel to EnsembleModelFileLevelCNN
# for lineLSTM and lineGRU, just use EnsembleModel
# for hunk-level FCN, use EnsembleModelHunkLevelFCN

import os
import json
import utils
import scipy
from torch.utils.data import DataLoader
from entities import EnsembleDataset, EnsemblePcaDataset
from model import EnsembleModel
import torch
from torch import cuda
from torch import nn as nn
from transformers import AdamW
from transformers import get_scheduler
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
from sklearn import metrics
import csv
from transformers import get_linear_schedule_with_warmup, RobertaTokenizer, T5ForConditionalGeneration, RobertaModel
import multiprocessing
import argparse
import pickle
import logging
from loss import FocalLoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
import math
from variant_ensemble import write_feature_to_file

directory = os.path.dirname(os.path.abspath(__file__))
dataset_name = 'ase_dataset_sept_19_2021.csv'
# dataset_name = 'huawei_sub_dataset.csv'

FINAL_MODEL_PATH = None
JAVA_RESULT_PATH = None
PYTHON_RESULT_PATH = None

TRAIN_BATCH_SIZE = 128
VALIDATION_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128

TRAIN_PARAMS = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 8}
VALIDATION_PARAMS = {'batch_size': VALIDATION_BATCH_SIZE, 'shuffle': True, 'num_workers': 8}
TEST_PARAMS = {'batch_size': TEST_BATCH_SIZE, 'shuffle': True, 'num_workers': 8}

LEARNING_RATE = 1e-7
NUMBER_OF_EPOCHS = 20
use_cuda = cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
random_seed = 109
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

logger = logging.getLogger(__name__)

def EM(data, k = 2, max_step = 200, threhold = 0.00001):

    phais = torch.tensor([[1.0/k] for i in range(k)], device=data.device)
    mean = torch.tensor([[i] for i in range(k)], device=data.device)
    std = torch.tensor([[1] for i in range(k)], device=data.device)
    pi_times_2 = torch.tensor(2 * math.pi)
    for i in range(max_step):
        # Qs = e_step(data,phais,mean,std)
        data_k = data.repeat(k).reshape(k, data.shape[0])
        exponent = torch.pow((data_k - mean),2)*(-1/(2*std))
        Qs = (torch.exp(exponent)/torch.sqrt(pi_times_2*std)*phais)
        Qs = Qs / torch.sum(Qs, dim=0, keepdim=True)
        # phais, mean, std= m_step(data,phais,mean,std,Qs)
        gama_j = torch.sum(Qs, dim=1)
        new_phais = (gama_j/data.shape[0]).reshape(k, 1)
        new_mean = (torch.sum(data*Qs, dim=1)/gama_j).reshape(k, 1)
        X_i_mu_j = torch.pow((data_k - mean),2)
        # new_std = (torch.sum((X_i_mu_j*Qs).transpose(0,1), axis=1) / gama_j).reshape(k, 1)
        new_std = (torch.sum(X_i_mu_j*Qs, axis=1) /gama_j ).reshape(k, 1)
        if i > 0 and False not in (torch.abs(new_mean - mean) < threhold):
            break
        phais, mean, std = new_phais, new_mean, new_std
    return phais[:,0].tolist(), mean[:,0].tolist(), std[:,0].tolist()

def cal_Prob(one_commit_data):
    idx, score, phais, mean, std = one_commit_data
    return idx, scipy.stats.norm(mean, std).pdf(score)*phais

def prevaluate(args, model, loc_model, eval_dataset, tokenizer,idx, eval_when_training=False, eval_with_mask=False):
    #build dataloader
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0)
    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)
    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()
    eval_loss, num = 0, 0
    loss_list = list()
    commit_index_list = list()
    bar = tqdm(eval_dataloader, total=len(eval_dataloader))
    for batch in bar:
        (input_ids, _, repair_input_ids, commit_index) = [x.to(args.device) for x in batch]
        if eval_with_mask:
            vul_query_mask = loc_model(input_ids)
            outputs = model(input_ids=input_ids, vul_query_mask=vul_query_mask, repair_input_ids=repair_input_ids)
        else:
            outputs = model(input_ids=input_ids, vul_query_mask=None, repair_input_ids=repair_input_ids)
        loss = outputs.loss
        lm_logits = outputs.logits
        loss_fct = CrossEntropyLoss(ignore_index=0, reduction='none')
        # logger.info("lm_logits.size:{}".format(lm_logits.size))
        loss_batch = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), repair_input_ids.view(-1)).view(-1, lm_logits.size(1))
        loss_batch = torch.nanmean(loss_batch.masked_fill(repair_input_ids==tokenizer.pad_token_id, torch.nan), dim=1)
        loss_list += loss_batch.tolist()
        commit_index_list += commit_index.tolist()

        eval_loss += loss.item()
        num += 1
    pathh = "bbb"
    os.makedirs(pathh, exist_ok=True)
    with open(os.path.join(pathh, "loss_list_cur_epoch_{}.pickle".format(idx)), "wb") as f:
        pickle.dump(loss_list, f)
    with open(os.path.join(pathh, "commit_index_list_cur_epoch_{}.pickle".format(idx)), "wb") as f:
        pickle.dump(commit_index_list, f)
    if idx >= 3:
        score_file_path = os.path.join(pathh, "loss_list_cur_epoch_{}.pickle".format(idx))
        commit_index_file = os.path.join(pathh, "commit_index_list_cur_epoch_{}.pickle".format(idx))
        if score_file_path.split(".")[-1] == "loss":
            with open(score_file_path) as score_f:
                score_list = score_f.read().split("\n")
        elif score_file_path.split(".")[-1] == "pickle":
            with open(score_file_path, "rb") as score_f:
                score_list = pickle.load(score_f)
        with open(commit_index_file, "rb") as f:
            commit_index_list = pickle.load(f)
        logger.info("There are {} commit messages are calculated in the probability".format(len(commit_index_list)))
        if set(commit_index_list) != set(range(len(commit_index_list))):
            logger.info("Some indexes are missing in commit_index_list")
        unsorted_score_list = [float(item) for item in score_list]
        score_list = []
        for i in np.argsort(commit_index_list):
            score_list.append(unsorted_score_list[i])

        # Get distribution by EM algorithm
        ratio, avg, std = EM(torch.tensor(score_list, device=args.device, dtype=torch.float64))
        if avg[0] < avg[1]:
            clean_idx = 0
        else:
            clean_idx = 1
        # Get Confidence of loss of each commit
        # NOT IN ORDER
        logger.info("Calculating the clean_weight ...")
        with multiprocessing.Pool(min(multiprocessing.cpu_count(), 24)) as pool:
            commit_data = [(idxb, score, ratio[clean_idx], avg[clean_idx], std[clean_idx]*9) for idxb, score in enumerate(score_list)]
            idx_clean_weight_list = pool.map(cal_Prob, commit_data)
        logger.info("Calculating the noisy_weight ...")
        with multiprocessing.Pool(min(multiprocessing.cpu_count(), 24)) as pool:
            commit_data = [(idxb, score, ratio[1-clean_idx], avg[1-clean_idx], std[1-clean_idx]) for idxb, score in enumerate(score_list)]
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
        fenzi = []
        fenmu = []
        cleann = []
        noisyy = []
        for idxc in range(len(score_list)):
            denom = clean_weight_list[idxc]+noisy_weight_list[idxc]
            if denom == 0:
                con = 1
            else:
                con = (clean_weight_list[idxc]+(np.power(1.6, -score_list[idxc]))*noisy_weight_list[idxc])/(clean_weight_list[idxc]+noisy_weight_list[idxc])
            confidence_list.append(con)
            cleann.append(clean_weight_list[idxc])
            noisyy.append(noisy_weight_list[idxc])
            fenzi.append(clean_weight_list[idxc]+(np.power(1.6, -score_list[idxc]))*noisy_weight_list[idxc])
            fenmu.append(clean_weight_list[idxc]+noisy_weight_list[idxc])
        logger.info("Calculate over ..")
        os.makedirs(pathh,exist_ok=True)
        with open(os.path.join(pathh, "cleann_cur_epoch_{}.pickle".format(idx)), "wb") as f:
            pickle.dump(cleann, f)
        with open(os.path.join(pathh, "noisyy_cur_epoch_{}.pickle".format(idx)), "wb") as f:
            pickle.dump(noisyy, f)
        with open(os.path.join(pathh, "confidence_list_cur_epoch_{}.pickle".format(idx)), "wb") as f:
            pickle.dump(confidence_list, f)
        with open(os.path.join(pathh, "fenzi_cur_epoch_{}.pickle".format(idx)), "wb") as f:
            pickle.dump(fenzi, f)
        with open(os.path.join(pathh, "fenmu_cur_epoch_{}.pickle".format(idx)), "wb") as f:
            pickle.dump(fenmu, f)
    eval_loss = round(eval_loss/num, 5)
    model.train()
    logger.info("***** Eval results *****")
    #logger.info(f"Evaluation Loss: {str(eval_loss)}")
    return confidence_list

def write_prob_to_file(file_path, urls, probs):
    with open(file_path, 'w') as file:
        writer = csv.writer(file)
        for i, url in enumerate(urls):
            writer.writerow([url, probs[i]])


def read_features_from_file(file_path):
    file_path = os.path.join(directory, file_path)
    with open(file_path, 'r') as reader:
        data = json.loads(reader.read())

    return data


def read_feature_list(file_path_list, reshape=False, need_list=False, need_extend=False):
    url_to_feature = {}
    for file_path in file_path_list:
        data = read_features_from_file(file_path)
        for url, feature in data.items():
            if url not in url_to_feature:
                url_to_feature[url] = []
            if not need_extend:
                url_to_feature[url].append(feature)
            else:
                url_to_feature[url].extend(feature)
    if not reshape:
        return url_to_feature
    else:
        url_to_combined = {}
        if reshape:
            for url in url_to_feature.keys():
                features = url_to_feature[url]
                combine = []
                for feature in features:
                    combine.extend(feature)
                if not need_list:
                    combine = torch.FloatTensor(combine)
                url_to_combined[url] = combine

        return url_to_combined


def predict_test_data(model, testing_generator, device, need_prob=False, need_features=False):
    y_pred = []
    y_test = []
    probs = []
    features = []
    urls = []
    with torch.no_grad():
        model.eval()
        for ids, url_batch, feature_1, feature_2, feature_3, feature_5, feature_6, feature_7, feature_8, label_batch in tqdm(
                testing_generator):
            feature_1 = feature_1.to(device)
            feature_2 = feature_2.to(device)
            feature_3 = feature_3.to(device)
            feature_5 = feature_5.to(device)
            feature_6 = feature_6.to(device)
            feature_7 = feature_7.to(device)
            feature_8 = feature_8.to(device)

            label_batch = label_batch.to(device)

            outs, pca_features = model(feature_1, feature_2, feature_3, feature_5, feature_6, feature_7, feature_8,
                                       need_features=True)

            outs = F.softmax(outs, dim=1)

            y_pred.extend(torch.argmax(outs, dim=1).tolist())
            y_test.extend(label_batch.tolist())
            probs.extend(outs[:, 1].tolist())
            urls.extend(list(url_batch))
            features.extend(list(pca_features.tolist()))

        precision = metrics.precision_score(y_pred=y_pred, y_true=y_test)
        recall = metrics.recall_score(y_pred=y_pred, y_true=y_test)
        f1 = metrics.f1_score(y_pred=y_pred, y_true=y_test)
        try:
            auc = metrics.roc_auc_score(y_true=y_test, y_score=probs)
        except Exception:
            auc = 0

    print("Finish testing")

    if not need_prob:
        return precision, recall, f1, auc
    else:
        return precision, recall, f1, auc, urls, probs, features


def train(model, learning_rate, number_of_epochs, training_generator, test_java_generator, test_python_generator):
    # loss_function = nn.NLLLoss()
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    criterion = FocalLoss(alpha=1, gamma=2, reduction='mean')
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = NUMBER_OF_EPOCHS * len(training_generator)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    train_losses = []

    for epoch in range(number_of_epochs):
        model.train()
        loss_list = list()
        commit_index_list = list()
        confidence_list2 = list()
        total_loss = 0
        current_batch = 0
        for ids, url_batch, feature_1, feature_2, feature_3, feature_5, feature_6, feature_7, feature_8, label_batch in tqdm(
                training_generator):
            feature_1 = feature_1.to(device)
            feature_2 = feature_2.to(device)
            feature_3 = feature_3.to(device)
            feature_5 = feature_5.to(device)
            feature_6 = feature_6.to(device)
            feature_7 = feature_7.to(device)
            feature_8 = feature_8.to(device)

            label_batch = label_batch.to(device)

            outs = model(feature_1, feature_2, feature_3, feature_5, feature_6, feature_7, feature_8)
            # outs = F.log_softmax(outs, dim=1)
            # loss = loss_function(outs, label_batch)
            """1"""
            lm_logits = outs.logits
            loss_fct = CrossEntropyLoss(ignore_index=0, reduction='none')
            # logger.info("lm_logits.size:{}".format(lm_logits.size))
            loss_batch = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), repair_input_ids.view(-1)).view(-1,
                                                                                                          lm_logits.size(
                                                                                                              1))
            loss_batch = torch.nanmean(loss_batch.masked_fill(repair_input_ids == tokenizer.pad_token_id, torch.nan),
                                       dim=1)
            loss_list += loss_batch.tolist()
            commit_index_list += commit_index.tolist()
            if epoch > 3:
                loss_batch = [loss * confidence_list[commit_index[idxa]] for idxa, loss in enumerate(loss_batch)]
                loss_batch = torch.stack(loss_batch, dim=0)
                loss = torch.mean(loss_batch)
            """1"""
            loss.backward()
            train_losses.append(loss.item())
            model.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            total_loss += loss.detach().item()

            current_batch += 1
            if current_batch % 50 == 0:
                print("Train commit iter {}, total loss {}, average loss {}".format(current_batch, np.sum(train_losses),
                                                                                    np.average(train_losses)))

        print("epoch {}, training commit loss {}".format(epoch, np.sum(train_losses)))
        train_losses = []
        model.eval()

    print("Result on Java testing dataset...")
    precision, recall, f1, auc, urls, probs, features = predict_test_data(model=model,
                                                                          testing_generator=test_java_generator,
                                                                          device=device, need_prob=True)
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1: {}".format(f1))
    print("AUC: {}".format(auc))
    print("-" * 32)

    write_prob_to_file(JAVA_RESULT_PATH, urls, probs)

    print("Result on Python testing dataset...")
    precision, recall, f1, auc, urls, probs, features = predict_test_data(model=model,
                                                                          testing_generator=test_python_generator,
                                                                          device=device, need_prob=True)

    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1: {}".format(f1))
    print("AUC: {}".format(auc))
    print("-" * 32)

    write_prob_to_file(PYTHON_RESULT_PATH, urls, probs)

    torch.save(model.state_dict(), FINAL_MODEL_PATH)

    return model


def do_train(args):
    global FINAL_MODEL_PATH
    global JAVA_RESULT_PATH
    global PYTHON_RESULT_PATH
    FINAL_MODEL_PATH = args.model_path
    if FINAL_MODEL_PATH is None or FINAL_MODEL_PATH == '':
        raise Exception("Model path must not be None or empty")

    JAVA_RESULT_PATH = args.java_result_path
    if JAVA_RESULT_PATH is None or JAVA_RESULT_PATH == '':
        raise Exception("Java result path must not be None or empty")

    PYTHON_RESULT_PATH = args.python_result_path
    if PYTHON_RESULT_PATH is None or PYTHON_RESULT_PATH == '':
        raise Exception("Java result path must not be None or empty")

    variant_to_drop = []
    if args.variant_to_drop is not None:
        for variant in args.variant_to_drop:
            variant_to_drop.append(int(variant))

    train_feature_path = [
        'features/feature_variant_1_train.txt',
        'features/feature_variant_2_train.txt',
        'features/feature_variant_3_train.txt',
        'features/feature_variant_5_train.txt',
        'features/feature_variant_6_train.txt',
        'features/feature_variant_7_train.txt',
        'features/feature_variant_8_train.txt'
    ]

    val_feature_path = [
        'features/feature_variant_1_val.txt',
        'features/feature_variant_2_val.txt',
        'features/feature_variant_3_val.txt',
        'features/feature_variant_5_val.txt',
        'features/feature_variant_6_val.txt',
        'features/feature_variant_7_val.txt',
        'features/feature_variant_8_val.txt'
    ]

    test_java_feature_path = [
        'features/feature_variant_1_test_java.txt',
        'features/feature_variant_2_test_java.txt',
        'features/feature_variant_3_test_java.txt',
        'features/feature_variant_5_test_java.txt',
        'features/feature_variant_6_test_java.txt',
        'features/feature_variant_7_test_java.txt',
        'features/feature_variant_8_test_java.txt'
    ]

    test_python_feature_path = [
        'features/feature_variant_1_test_python.txt',
        'features/feature_variant_2_test_python.txt',
        'features/feature_variant_3_test_python.txt',
        'features/feature_variant_5_test_python.txt',
        'features/feature_variant_6_test_python.txt',
        'features/feature_variant_7_test_python.txt',
        'features/feature_variant_8_test_python.txt'
    ]

    print("Reading data...")
    url_to_features = {}
    print("Reading train data")
    url_to_features.update(read_feature_list(train_feature_path))
    print("Reading test java data")
    url_to_features.update(read_feature_list(test_java_feature_path))
    print("Reading test python data")
    url_to_features.update(read_feature_list(test_python_feature_path))

    print("Finish reading")
    url_data, label_data = utils.get_data(dataset_name)

    feature_data = {}
    feature_data['train'] = []
    feature_data['test_java'] = []
    feature_data['test_python'] = []

    for url in url_data['train']:
        feature_data['train'].append(url_to_features[url])

    for url in url_data['test_java']:
        feature_data['test_java'].append(url_to_features[url])

    for url in url_data['test_python']:
        feature_data['test_python'].append(url_to_features[url])

    val_ids, test_java_ids, test_python_ids = [], [], []
    index = 0
    id_to_url = {}
    id_to_label = {}
    id_to_feature = {}

    for i, url in enumerate(url_data['train']):
        val_ids.append(index)
        id_to_url[index] = url
        id_to_label[index] = label_data['train'][i]
        id_to_feature[index] = feature_data['train'][i]
        index += 1

    for i, url in enumerate(url_data['test_java']):
        test_java_ids.append(index)
        id_to_url[index] = url
        id_to_label[index] = label_data['test_java'][i]
        id_to_feature[index] = feature_data['test_java'][i]
        index += 1

    for i, url in enumerate(url_data['test_python']):
        test_python_ids.append(index)
        id_to_url[index] = url
        id_to_label[index] = label_data['test_python'][i]
        id_to_feature[index] = feature_data['test_python'][i]
        index += 1

    training_set = EnsembleDataset(val_ids, id_to_label, id_to_url, id_to_feature)
    test_java_set = EnsembleDataset(test_java_ids, id_to_label, id_to_url, id_to_feature)
    test_python_set = EnsembleDataset(test_python_ids, id_to_label, id_to_url, id_to_feature)

    training_generator = DataLoader(training_set, **TRAIN_PARAMS)
    test_java_generator = DataLoader(test_java_set, **TEST_PARAMS)
    test_python_generator = DataLoader(test_python_set, **TEST_PARAMS)

    model = EnsembleModel(args.ablation_study, variant_to_drop)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    train(model=model,
          learning_rate=LEARNING_RATE,
          number_of_epochs=NUMBER_OF_EPOCHS,
          training_generator=training_generator,
          test_java_generator=test_java_generator,
          test_python_generator=test_python_generator)


def infer_dataset(model_path, partition, ablation_study, variant_to_drop, prob_path):
    # val_feature_path = [
    #     'features/feature_variant_1_val.txt',
    #     'features/feature_variant_2_val.txt',
    #     'features/feature_variant_3_val.txt',
    #     'features/feature_variant_5_val.txt',
    #     'features/feature_variant_6_val.txt',
    #     'features/feature_variant_7_val.txt',
    #     'features/feature_variant_8_val.txt'
    # ]

    test_java_feature_path = [
        'features/feature_variant_1_test_java.txt',
        'features/feature_variant_2_test_java.txt',
        'features/feature_variant_3_test_java.txt',
        'features/feature_variant_5_test_java.txt',
        'features/feature_variant_6_test_java.txt',
        'features/feature_variant_7_test_java.txt',
        'features/feature_variant_8_test_java.txt'
    ]

    test_python_feature_path = [
        'features/feature_variant_1_test_python.txt',
        'features/feature_variant_2_test_python.txt',
        'features/feature_variant_3_test_python.txt',
        'features/feature_variant_5_test_python.txt',
        'features/feature_variant_6_test_python.txt',
        'features/feature_variant_7_test_python.txt',
        'features/feature_variant_8_test_python.txt'
    ]

    # model = EnsembleModelHunkLevelFCN(False, [])

    model = EnsembleModel(ablation_study, variant_to_drop)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.load_state_dict(torch.load('model/patch_ensemble_model.sav'))
    model.to(device)

    print("Reading data")
    url_to_features = read_feature_list(test_python_feature_path)

    print("Finish reading")
    url_data, label_data = utils.get_data(dataset_name)

    feature_data = {}
    feature_data[partition] = []

    for url in url_data[partition]:
        feature_data[partition].append(url_to_features[url])

    val_ids = []
    index = 0
    id_to_url = {}
    id_to_label = {}
    id_to_feature = {}

    for i, url in enumerate(url_data[partition]):
        val_ids.append(index)
        id_to_url[index] = url
        id_to_label[index] = label_data[partition][i]
        id_to_feature[index] = feature_data[partition][i]
        index += 1

    val_set = EnsembleDataset(val_ids, id_to_label, id_to_url, id_to_feature)

    val_generator = DataLoader(val_set, **TEST_PARAMS)

    print("Result on dataset...")
    precision, recall, f1, auc, urls, probs, features = predict_test_data(model=model,
                                                                          testing_generator=val_generator,
                                                                          device=device, need_prob=True)

    write_prob_to_file(prob_path, urls, probs)
    # write_feature_to_file(feature_path, urls, features)
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1: {}".format(f1))
    print("AUC: {}".format(auc))
    print("-" * 32)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ensemble Classifier')
    parser.add_argument('--ablation_study',
                        type=bool,
                        default=False,
                        help='Do ablation study or not')
    parser.add_argument('-v',
                        '--variant_to_drop',
                        action='append',
                        required=False,
                        help='Select index of variant to drop, 1, 2, 3, 5, 6, 7, 8')
    parser.add_argument('--model_path',
                        type=str,
                        help='IMPORTANT select path to save model')
    parser.add_argument('--java_result_path',
                        type=str,
                        help='path to save prediction for Java projects')
    parser.add_argument('--python_result_path',
                        type=str,
                        help='path to save prediction for Python projects')
    args = parser.parse_args()
    do_train(args)

    # infer_dataset('test_python', 'features/feature_ensemble_test_python.txt')