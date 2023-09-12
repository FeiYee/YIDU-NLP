# -*- coding:utf-8 -*-
import sys
sys.path.append("../")
import os
import argparse
import json
import torch.optim as optim
import numpy as np
import copy
import torch
import shutil
device = "cuda" if torch.cuda.is_available() else "cpu"

from pytorch_transformers import AdamW, WarmupLinearSchedule
from process_data.ner_process import ner_data_process_machine
from model.bert_bilstm_crf import bert_bilstm_crf
from model.bilstm_crf import bilstm_crf
from model.bert_crf import bert_crf
from torch.utils import data
from tensorboardX import SummaryWriter
from evaluation.ner_evaluation import f1_eva_ner, f1_eva_ner_label_level, sta_error_type

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser()
# 数据集参数
parser.add_argument("-tr","--train_file_path", help="训练文件路径，如果指定此参数则代表训练模型，并需要指定验证集", default=None)
parser.add_argument("-de","--dev_file_path", help="验证集路径", default=None)
parser.add_argument("-te","--test_file_path", help="测试文件路径，如果指定此参数，需要指定模型路径参数", default=None)
parser.add_argument("-mfp","--model_file_path", default="data/")
parser.add_argument("-mn","--model_name", default="v1")


# 模型结构参数
parser.add_argument("-ms","--model_structure", help="模型结构", default="bert_bilstm_crf")
parser.add_argument("-mp","--model_path", default="./chinese_L-12_H-768_A-12/chinese_L-12_H-768_A-12")
parser.add_argument("-ft","--fine_tuning", help="微调", default=False)
parser.add_argument("-pos","--pretrain_output_size", help="预训练模型的输出维度", default=768)


# lstm的参数
parser.add_argument("-lhs","--lstm_hidden_size", help="lstm的隐藏单元数", default=384)
parser.add_argument("-nl","--num_layers", help="lstm的层数", default=1)
parser.add_argument("-bid","--bidirectional", help="lstm是否双向", default=True)
parser.add_argument("-dr","--dropout_ratio", help="dropout_ratio", default=0.5)


# 数据构造参数
parser.add_argument("-bs","--batch_size", help="batch_size", default=512)
parser.add_argument("-lr","--learning_rate", help="learning_rate", default=1e-5)
parser.add_argument("-sml","--sentence_max_length", help="sentence_max_length", default=300)

# 训练参数
parser.add_argument("-e","--epochs", help="epochs", default=20)

def str_to_bool(string):
    return True if str(string).lower() == 'true' else False


args = parser.parse_args()
model_config = {
    "train_file_path":args.train_file_path,
    "dev_file_path" :args.dev_file_path,
    "test_file_path" : args.test_file_path,
    "model_file_path" : args.model_file_path,
    "model_name" : args.model_name,
    "model_structure" : args.model_structure,
    "model_path":args.model_path,
    "fine_tuning":str_to_bool(args.fine_tuning),
    "pretrain_output_size":int(args.pretrain_output_size),
    "lstm_hidden_size" : int(args.lstm_hidden_size),
    "num_layers" : int(args.num_layers),
    "bidirectional" : str_to_bool(args.bidirectional),
    "dropout_ratio" : float(args.dropout_ratio),
    "batch_size" : int(args.batch_size),
    "learning_rate" : float(args.learning_rate),
    "sentence_max_length" : int(args.sentence_max_length),

    "epochs" : int(args.epochs)}


if model_config["train_file_path"]:

    if os.path.exists(os.path.join(model_config["model_file_path"],'Result',model_config["model_name"])):
        # os.rmdir(os.path.join(model_config["model_file_path"],'Result',model_config["model_name"]))
        shutil.rmtree(os.path.join(model_config["model_file_path"],'Result',model_config["model_name"]))

    summary_writer = SummaryWriter(os.path.join(model_config["model_file_path"],'Result',model_config["model_name"]))

    train_dataset = ner_data_process_machine(file_path = model_config["train_file_path"],
                                             model_structure = model_config["model_structure"],
                                             sentence_max_len = model_config["sentence_max_length"],
                                             tokenizer_path = model_config["model_path"])
    train_dataloader = data.DataLoader(train_dataset,
                                       batch_size=model_config["batch_size"], shuffle=True, num_workers=0)

    model_config["label2id"] = train_dataset.get_label2id()

    with open(model_config["model_file_path"] + model_config["model_name"] + "_model_config.json", "w+", encoding="utf-8") as f:
        f.write(json.dumps(model_config, ensure_ascii=False))

    dev_dataset = ner_data_process_machine(
        file_path = model_config["dev_file_path"],
        sentence_max_len = model_config["sentence_max_length"],
        model_structure=model_config["model_structure"],
        label2id = model_config["label2id"],
        tokenizer = train_dataset.get_tokenizer())
    # dev_dataloader = data.dataloader(dev_dataset,
    #                                    batch_size=model_config["batch_size"], shuffle=True, num_workers=4)

    # 得到验证集的loss和F1，p，r
    dev_text_list, dev_x, dev_y, dev_start_index, dev_sen_len, dev_z = \
        dev_dataset.extract_data(copy.deepcopy(dev_dataset.get_data()))
    source_dev_data, transform_back_dev_data = dev_dataset.transform_data_back(
            dev_dataset.uni_data(copy.deepcopy(dev_text_list), copy.deepcopy(dev_x),
                                 copy.deepcopy(dev_y), copy.deepcopy(dev_start_index), copy.deepcopy(dev_sen_len), copy.deepcopy(dev_z)))

    print("验证集上限P R F1：", f1_eva_ner(transform_back_dev_data, source_dev_data))

    print(sta_error_type(source_dev_data, source_dev_data))

    # dev_x_input = torch.tensor(dev_x, dtype=torch.long).to(device)
    # dev_y = torch.tensor(dev_y, dtype=torch.long).to(device)

    if model_config["model_structure"] == "bert_bilstm_crf":
        model = bert_bilstm_crf(
            pretrain_model_path = model_config["model_path"],
            lstm_hidden_size = model_config["lstm_hidden_size"],
            num_layers = model_config["num_layers"],
            dropout_ratio = model_config["dropout_ratio"],
            bidirectional = model_config["bidirectional"],
            lable_num = len(model_config["label2id"]),
            device = device)
    elif model_config["model_structure"] == "bert_crf":
        model = bert_crf(pretrain_model_path = model_config["model_path"],
                         lable_num = len(model_config["label2id"]),
                         device = device)
    model.to(device)

    num_total_steps = model_config["epochs"] * len(train_dataset) / model_config["batch_size"]
    num_warmup_steps = int(0.1 * num_total_steps)

    if model_config["fine_tuning"]:
        warm_up_params = []
    non_warm_up_params = []

    for name, param in model.named_parameters():
        if "bert" in name:
            if model_config["fine_tuning"]:
                warm_up_params.append(param)
            else:
                param.requires_grad = False
        else:
            non_warm_up_params.append(param)

    model.model_structure()

    if model_config["fine_tuning"]:
        warm_up_optimizer = AdamW(warm_up_params, lr=model_config["learning_rate"],
                                  correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
        scheduler = WarmupLinearSchedule(warm_up_optimizer, warmup_steps=num_warmup_steps,
                                         t_total=num_total_steps)  # PyTorch scheduler
    non_warm_up_optimizer = optim.Adam(non_warm_up_params, lr=model_config["learning_rate"] * 10)

    best_f1 = 0

    for i in range(model_config["epochs"]):
        loss_list = []
        model.train()
        for batch_data in train_dataloader:
            batch_data_array = np.array(batch_data)
            # x_input = batch_data_array[:, 1].tolist()
            # y = batch_data[:, 2].tolist()

            # _, x, y, __, sen_len = train_dataset.extract_data(batch_data)
            _, x, y, __, sen_len, z = batch_data[0], batch_data[1], batch_data[2], batch_data[3], batch_data[4], batch_data[5]

            # x_input = torch.tensor(x, dtype=torch.long).to(device)
            # y = torch.tensor(y, 0.T, dtype=torch.long).to(device)

            output = model(x, z)
            loss = model.get_loss(output, model_config["sentence_max_length"], sen_len, y = y)

            loss.backward()

            if model_config["fine_tuning"]:
                warm_up_optimizer.step()
                scheduler.step()
                warm_up_optimizer.zero_grad()
            non_warm_up_optimizer.step()
            non_warm_up_optimizer.zero_grad()

            loss_list.append(loss.item())

        model.eval()
        dev_output, dev_loss = model.decode(dev_x, max_len = model_config["sentence_max_length"],
                                            sen_len = dev_sen_len, use_cuda = True, dev_y = dev_y, dev_z=dev_z)

        summary_writer.add_scalars("loss",{"train_loss" : np.mean(loss_list), "dev_loss" : dev_loss.item()}, i)

        source_dev_data, predict_dev_data = \
            dev_dataset.transform_data_back(
                dev_dataset.uni_data(dev_text_list, dev_x, dev_output, dev_start_index, dev_sen_len, dev_z))

        p, r, f1 = f1_eva_ner(predict_dev_data, source_dev_data)

        summary_writer.add_scalars("dev_evaluation_total", {"p": p, "r": r, "f1": f1}, i)
        ans_dict = f1_eva_ner_label_level(predict_dev_data, source_dev_data)
        for key, v in ans_dict.items():
            summary_writer.add_scalars(key, v, i)

        if f1 > best_f1:
            best_f1 = f1
            model.save_model(model_config["model_file_path"]+model_config["model_name"])
            print("错误类型和个数：", sta_error_type(source_dev_data, predict_dev_data))
            print("label lavel:", ans_dict)
        print("训练完成：{}， 训练集loss：{}，验证集loss：{}，验证集P：{}，验证集R：{}，验证集F1：{}，最高F1：{}".format(
            i, np.mean(loss_list), dev_loss, p, r, f1, best_f1))
        print("-"*100)

if model_config["test_file_path"]:

    with open(model_config["model_file_path"] + model_config["model_name"] + "_model_config.json", encoding="utf-8") as f:
        config = json.load(f)

    config["test_file_path"] =  model_config["test_file_path"]
    print(config)

    if config["model_structure"] == "bert_bilstm_crf":
        model = bert_bilstm_crf(
            pretrain_model_path=config["model_path"],
            lstm_hidden_size=config["lstm_hidden_size"],
            num_layers=config["num_layers"],
            dropout_ratio=config["dropout_ratio"],
            bidirectional=config["bidirectional"],
            lable_num=len(config["label2id"]),
            device=device)
    elif config["model_structure"] == "bert_crf":
        model = bert_crf(pretrain_model_path = config["model_path"],
                         lable_num = len(config["label2id"]),
                         device = device)
    elif config["model_structure"] == "w2v_bilstm_crf":
        model = bilstm_crf(
            pretrain_model_path = config["model_path"],
            pretrain_output_size = config["pretrain_output_size"],
            lstm_hidden_size = config["lstm_hidden_size"],
            num_layers = config["num_layers"],
            dropout_ratio = config["dropout_ratio"],
            bidirectional = config["bidirectional"],
            lable_num = len(config["label2id"]),
            device = device)

    model.load_state_dict(torch.load(config["model_file_path"]+config["model_name"]))
    model.to(device)
    model.eval()

    test_dataset = ner_data_process_machine(
        file_path=config["test_file_path"],
        sentence_max_len=config["sentence_max_length"],
        model_structure=config["model_structure"],
        label2id=config["label2id"],
        tokenizer_path=config["model_path"])
    # dev_dataloader = data.dataloader(dev_dataset,
    #                                    batch_size=model_config["batch_size"], shuffle=True, num_workers=4)
    # 得到验证集的loss和F1，p，r
    test_text_list, test_x, test_y, test_start_index, test_sen_len, test_z = \
        test_dataset.extract_data(copy.deepcopy(test_dataset.get_data()))
    source_test_data, transform_back_test_data = test_dataset.transform_data_back(
        test_dataset.uni_data(copy.deepcopy(test_text_list), copy.deepcopy(test_x),
                             copy.deepcopy(test_y), copy.deepcopy(test_start_index),
                              copy.deepcopy(test_sen_len), copy.deepcopy(test_z)))

    print("测试集上限P R F1：", f1_eva_ner(transform_back_test_data, source_test_data))

    # test_x_input = torch.tensor(test_x, dtype=torch.long).to(device)
    # test_y = torch.tensor(test_y, dtype=torch.long).to(device)

    test_output, test_loss = model.decode(test_x, max_len=config["sentence_max_length"],
                                        sen_len=test_sen_len, use_cuda=True, dev_y=test_y, dev_z=test_z)

    source_test_data, predict_test_data = \
        test_dataset.transform_data_back(
            test_dataset.uni_data(test_text_list, test_x, test_output, test_start_index, test_sen_len, test_z))

    p, r, f1 = f1_eva_ner(predict_test_data, source_test_data)
    print("测试total p, r, f1表现：{}，{}，{}".format(p, r, f1))
    ans_dict = f1_eva_ner_label_level(predict_test_data, source_test_data)
    print("eva_ner_label_level: ", ans_dict)
    print("-"*200)
    print("预测结果：",predict_test_data)