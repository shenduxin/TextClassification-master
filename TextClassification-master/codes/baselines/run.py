# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True,
                    help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN,'
                         ' Transformer, BERT, ERNIE, XLNet')
parser.add_argument('--dataset', default='AGNews', help='THUCNews or AGNews')
parser.add_argument('--use_sep',action='store_true',help="Use [sep] to partion the title and content. Only available for pretrained LM model (XLNet,BERT,ERNIE) and for AGNews.")
parser.add_argument('--use_lookahead',action='store_true',help='Use LookAhead or not')
parser.add_argument('--require_improvement',type=int)
parser.add_argument('--num_epochs',type=int)
parser.add_argument('--batch_size',type=int)
parser.add_argument('--pad_size',type=int)
parser.add_argument('--learning_rate',type=float)

args = parser.parse_args()

if __name__ == '__main__':
    dataset = 'datasets/' + args.dataset  # 数据集
    if args.use_sep:
        dataset = dataset + '_SEP'

    model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
    if model_name == 'FastText':
        from utils_fasttext import build_dataset, build_iterator, get_time_dif
    elif model_name == 'BERT' or model_name == 'ERNIE' or model_name == 'XLNet':
        from utils_bert import build_dataset, build_iterator, get_time_dif
    else:
        from utils import build_dataset, build_iterator, get_time_dif

    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    config.__dict__.update({k:v for k,v in args.__dict__.items() if v is not None})
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    if model_name == 'BERT' or model_name == 'ERNIE' or model_name == 'XLNet':
        if args.use_sep:
            train_data, dev_data, test_data = build_dataset(config,use_sep=True)
        else:
            train_data, dev_data, test_data = build_dataset(config)
    else:
        if args.dataset == 'AGNews':
            use_word = True
        else:
            use_word = False
        vocab, train_data, dev_data, test_data = build_dataset(config, use_word)
        config.n_vocab = len(vocab)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.Model(config).to(config.device)
    if model_name != 'Transformer' and model_name != 'BERT' and model_name != 'ERNIE' and model_name != 'XLNet':
        init_network(model)
    print(model.parameters)
    train(config, model, train_iter, dev_iter, test_iter)
