# coding: UTF-8
import torch
from tqdm import tqdm
import time
from datetime import timedelta

XLNET_CLS, BERT_CLS = '<cls>', '[CLS]'  # padding符号, bert中综合信息符号


def build_dataset(config,use_sep=False):

    def load_dataset(path, pad_size=32,use_sep=False):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                if use_sep:
                    title, content, label = lin.split('\t')
                    encoded = config.tokenizer.encode_plus(title,content,max_length=pad_size,pad_to_max_length=True)
                    contents.append((encoded['input_ids'],int(label),encoded['token_type_ids'],encoded['attention_mask']))
                else:
                    content, label = lin.split('\t')
                    # token = config.tokenizer.tokenize(content)
                    # if config.model_name == 'XLNet':
                    #     token = token + [XLNET_CLS] # XLNet 风格的顺序
                    # else:
                    #     token = [BERT_CLS] + token
                    # seq_len = len(token)
                    # mask = []
                    # token_ids = config.tokenizer.convert_tokens_to_ids(token)
                    # if pad_size:
                    #     if len(token) < pad_size:
                    #         if config.model_name == 'XLNet':
                    #             mask = [0] * (pad_size - len(token)) + [1] * len(token)
                    #             token_ids = ([0] * (pad_size - len(token))) + token_ids
                    #         else:
                    #             mask = [1] * len(token) + [0] * (pad_size - len(token))
                    #             token_ids += ([0] * (pad_size - len(token)))
                    #     else:
                    #         mask = [1] * pad_size
                    #         token_ids = token_ids[:pad_size]
                    #         seq_len = pad_size
                    # contents.append((token_ids, int(label), seq_len, mask))
                    encoded = config.tokenizer.encode_plus(content,max_length=pad_size,pad_to_max_length=True)
                    contents.append((encoded['input_ids'],int(label),encoded['token_type_ids'],encoded['attention_mask']))
        return contents
    train = load_dataset(config.train_path, config.pad_size,use_sep=use_sep)
    dev = load_dataset(config.dev_path, config.pad_size,use_sep=use_sep)
    test = load_dataset(config.test_path, config.pad_size,use_sep=use_sep)
    return train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        # seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        token_types = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, token_types, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
