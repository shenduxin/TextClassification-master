# coding: UTF-8
import time
import torch
import numpy as np
from train_eval_fusion import train, init_network
from importlib import import_module
import argparse

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True,
                    help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN,'
                         ' Transformer, BERT, ERNIE, XLNet,XLNet_BERT_cat,XLNet_BERT_cat')
parser.add_argument('--dataset', default='AGNews', help='THUCNews or AGNews')
args = parser.parse_args()

if __name__ == '__main__':
    dataset = 'datasets/' + args.dataset  # 数据集

    model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
    if model_name == 'FastText':
        from utils_fasttext import build_dataset, build_iterator, get_time_dif
    elif model_name == 'BERT' or model_name == 'ERNIE' or model_name == 'XLNet' or model_name=='XLNet_BERT_cat' or model_name=='XLNet_BERT_cat':
        from utils_bert import build_dataset, build_iterator, get_time_dif
    else:
        from utils import build_dataset, build_iterator, get_time_dif

    x = import_module('models.' + model_name)
    config1 = x.Config1(dataset)
    config2 = x.Config2(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    if model_name == 'BERT' or model_name == 'ERNIE' or model_name == 'XLNet' or model_name=='XLNet_BERT_cat' or model_name=='XLNet_BERT_cat':
        train_data, dev_data, test_data = build_dataset(config1)
    else:
        if args.dataset == 'AGNews':
            use_word = True
        else:
            use_word = False
        vocab, train_data, dev_data, test_data = build_dataset(config1, use_word)
        config1.n_vocab = len(vocab)
    train_iter = build_iterator(train_data, config1)
    dev_iter = build_iterator(dev_data, config1)
    test_iter = build_iterator(test_data, config1)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.Model(config1,config2).to(config1.device)
    #model.load_state_dict(torch.load(''))
    if model_name != 'Transformer' and model_name != 'BERT' and model_name != 'ERNIE' and model_name != 'XLNet' and model_name!='XLNet_BERT_cat' and model_name!='XLNet_BERT_cat':
        init_network(model)
    print(model.parameters)
    train(config1,config2, model, train_iter, dev_iter, test_iter)
