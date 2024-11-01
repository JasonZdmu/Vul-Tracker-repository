import os
import json
import utils
from torch.utils.data import DataLoader, Dataset
from entities import EnsembleDataset, EnsemblePcaDataset
from model import EnsembleModel
import torch
from torch import cuda, nn
from transformers import AdamW, get_scheduler
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
from sklearn import metrics
import csv
import argparse
from loss import FocalLoss

directory = os.path.dirname(os.path.abspath(__file__))
dataset_name = 'ase_dataset_sept_19_2021.csv'

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
NUMBER_OF_EPOCHS = 80
EVAL_INTERVAL = 5  # 每5个epoch进行一次验证
use_cuda = cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
random_seed = 109
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


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


def select_high_confidence_samples(model, training_generator, confidence_threshold=0.9):
    model.eval()
    high_confidence_ids = []
    id_to_confidence = {}

    with torch.no_grad():
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
            probs = F.softmax(outs, dim=1)
            confidences, predictions = torch.max(probs, dim=1)

            for i, id in enumerate(ids):
                confidence = confidences[i].item()
                if confidence >= confidence_threshold:
                    high_confidence_ids.append(id.item())
                id_to_confidence[id.item()] = confidence

    return high_confidence_ids, id_to_confidence

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


def get_C_hat_transpose(model, gold_loader, num_classes):
    model.eval()
    probs = []
    labels = []
    with torch.no_grad():
        for ids, url_batch, feature_1, feature_2, feature_3, feature_5, feature_6, feature_7, feature_8, label_batch in tqdm(
                gold_loader):
            feature_1 = feature_1.to(device)
            feature_2 = feature_2.to(device)
            feature_3 = feature_3.to(device)
            feature_5 = feature_5.to(device)
            feature_6 = feature_6.to(device)
            feature_7 = feature_7.to(device)
            feature_8 = feature_8.to(device)
            label_batch = label_batch.to(device)

            outs = model(feature_1, feature_2, feature_3, feature_5, feature_6, feature_7, feature_8)
            pred = F.softmax(outs, dim=1)
            probs.extend(pred.cpu().numpy())
            labels.extend(label_batch.cpu().numpy())

    probs = np.array(probs)
    labels = np.array(labels)
    C_hat = np.zeros((num_classes, num_classes))
    for label in range(num_classes):
        indices = np.where(labels == label)
        if len(indices[0]) > 0:
            C_hat[label] = np.mean(probs[indices], axis=0)
    return torch.FloatTensor(C_hat.T).to(device)


def train_phase1(model, optimizer, lr_scheduler, criterion, training_generator):
    model.train()
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

        optimizer.zero_grad()
        outs = model(feature_1, feature_2, feature_3, feature_5, feature_6, feature_7, feature_8)
        loss = criterion(outs, label_batch)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()


def train_phase2(model, optimizer, lr_scheduler, criterion, training_loader, gold_loader, C_hat_transpose):
    model.train()
    for (silver_data, gold_data) in zip(training_loader, gold_loader):
        silver_ids, silver_urls, silver_feature_1, silver_feature_2, silver_feature_3, silver_feature_5, silver_feature_6, silver_feature_7, silver_feature_8, silver_label_batch = silver_data
        gold_ids, gold_urls, gold_feature_1, gold_feature_2, gold_feature_3, gold_feature_5, gold_feature_6, gold_feature_7, gold_feature_8, gold_label_batch = gold_data

        silver_feature_1 = silver_feature_1.to(device)
        silver_feature_2 = silver_feature_2.to(device)
        silver_feature_3 = silver_feature_3.to(device)
        silver_feature_5 = silver_feature_5.to(device)
        silver_feature_6 = silver_feature_6.to(device)
        silver_feature_7 = silver_feature_7.to(device)
        silver_feature_8 = silver_feature_8.to(device)
        silver_label_batch = silver_label_batch.to(device)

        gold_feature_1 = gold_feature_1.to(device)
        gold_feature_2 = gold_feature_2.to(device)
        gold_feature_3 = gold_feature_3.to(device)
        gold_feature_5 = gold_feature_5.to(device)
        gold_feature_6 = gold_feature_6.to(device)
        gold_feature_7 = gold_feature_7.to(device)
        gold_feature_8 = gold_feature_8.to(device)
        gold_label_batch = gold_label_batch.to(device)

        optimizer.zero_grad()

        # Silver loss
        silver_outs = model(silver_feature_1, silver_feature_2, silver_feature_3, silver_feature_5, silver_feature_6,
                            silver_feature_7, silver_feature_8)
        silver_soft_outs = F.softmax(silver_outs, dim=1)
        silver_loss = -torch.sum(torch.log(torch.sum(C_hat_transpose[silver_label_batch] * silver_soft_outs, dim=1)))

        # Gold loss
        gold_outs = model(gold_feature_1, gold_feature_2, gold_feature_3, gold_feature_5, gold_feature_6,
                          gold_feature_7, gold_feature_8)
        gold_loss = criterion(gold_outs, gold_label_batch)

        loss = (silver_loss + gold_loss) / (len(silver_label_batch) + len(gold_label_batch))

        loss.backward()
        optimizer.step()
        lr_scheduler.step()


def train(model, learning_rate, number_of_epochs, training_generator, test_java_generator, test_python_generator, eval_interval=EVAL_INTERVAL, confidence_threshold=0.9):
    criterion = FocalLoss(alpha=1, gamma=2, reduction='mean')
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = number_of_epochs * len(training_generator)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    for epoch in range(number_of_epochs):
        print(f"Epoch {epoch + 1}/{number_of_epochs}")

        # Phase 1: Train on silver data
        train_phase1(model, optimizer, lr_scheduler, criterion, training_generator)

        # Select high confidence samples for gold data
        high_confidence_ids, id_to_confidence = select_high_confidence_samples(model, training_generator, confidence_threshold)

        # Create gold_loader based on high confidence samples
        high_confidence_dataset = torch.utils.data.Subset(training_generator.dataset, high_confidence_ids)
        gold_loader = DataLoader(high_confidence_dataset, batch_size=VALIDATION_BATCH_SIZE, shuffle=False, num_workers=8)

        # Estimate C_hat
        num_classes = model.module.num_classes if isinstance(model, nn.DataParallel) else model.num_classes
        C_hat_transpose = get_C_hat_transpose(model, gold_loader, num_classes)

        # Phase 2: Train with corrected loss using gold data
        train_phase2(model, optimizer, lr_scheduler, criterion, training_generator, gold_loader, C_hat_transpose)

        if (epoch + 1) % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                print("Result on Java testing dataset...")
                precision, recall, f1, auc, urls, probs, features = predict_test_data(model=model, testing_generator=test_java_generator, device=device, need_prob=True)
                print(f"Precision: {precision}, Recall: {recall}, F1: {f1}, AUC: {auc}")
                write_prob_to_file(JAVA_RESULT_PATH, urls, probs)

                print("Result on Python testing dataset...")
                precision, recall, f1, auc, urls, probs, features = predict_test_data(model=model, testing_generator=test_python_generator, device=device, need_prob=True)
                print(f"Precision: {precision}, Recall: {recall}, F1: {f1}, AUC: {auc}")
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
        raise Exception("Python result path must not be None or empty")

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
    model.num_classes = 2  # Set the number of classes

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)

    train(model=model, learning_rate=LEARNING_RATE, number_of_epochs=NUMBER_OF_EPOCHS,
          training_generator=training_generator, test_java_generator=test_java_generator,
          test_python_generator=test_python_generator, eval_interval=5)




def infer_dataset(model_path, partition, ablation_study, variant_to_drop, prob_path):
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

    model = EnsembleModel(ablation_study, variant_to_drop)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
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
    precision, recall, f1, auc, urls, probs, features = predict_test_data(model=model, testing_generator=val_generator,
                                                                          device=device, need_prob=True)

    write_prob_to_file(prob_path, urls, probs)
    print(f"Precision: {precision}, Recall: {recall}, F1: {f1}, AUC: {auc}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ensemble Classifier')
    parser.add_argument('--ablation_study', type=bool, default=False, help='Do ablation study or not')
    parser.add_argument('-v', '--variant_to_drop', action='append', required=False,
                        help='Select index of variant to drop, 1, 2, 3, 5, 6, 7, 8')
    parser.add_argument('--model_path', type=str, help='IMPORTANT select path to save model')
    parser.add_argument('--java_result_path', type=str, help='path to save prediction for Java projects')
    parser.add_argument('--python_result_path', type=str, help='path to save prediction for Python projects')
    args = parser.parse_args()
    do_train(args)
