# -*- encoding: utf-8 -*-
import time
import random
import models
import torch
import logging
import fire
import json
import numpy as np
import torch.optim as optim

from tqdm import trange
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import utils
import config
from data import Data
from metrics import f1_score, f1_score_ent_rel, eval_file
from config import opt

def load_data(path):
    '''
    加载数据，返回json数组.
    '''
    data = []
    data_lines = open(path, encoding='utf-8').readlines()
    for line in data_lines:
        line_json = json.loads(line)
        if len(line_json['postag']) == 0:
            continue
        if 'spo_list' in line_json.keys() and len(line_json['spo_list']) == 0:
            continue
        data.append(line_json)
    return data

def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))


def collate_fn(batch):
    data, label, rel = zip(*batch)
    return data, label, rel

def set_up(opt):
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if opt.use_gpu:
        torch.cuda.manual_seed_all(opt.seed)


def train(**kwargs):

    # 1 config
    opt.parse(kwargs)
    set_up(opt)
    if opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)
    # 2 model
    model = getattr(models, opt.model)(opt)
    if opt.use_gpu:
        model.cuda()
    if opt.load_ckpt:
        logging.info("{} load ckpt from {}".format(now(), opt.ckpt))
        model.load(opt.ckpt_path)

    # 3 data
    train_data = Data(opt, case=0)
    train_data_loader = DataLoader(train_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers, collate_fn=collate_fn)
    dev_data = Data(opt, case=1)
    dev_data_loader = DataLoader(dev_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, collate_fn=collate_fn)

    utils.set_logger(opt.log_dir)
    logging.info("lamba: {}, na num: {}, data set:{}".format(opt.lam, opt.naNum, opt.dataset))
    logging.info("CNN k:{}, filter num:{}, seq length:{}, tuple max length:{}".format(opt.filters, opt.filter_num, opt.seq_length, opt.tuple_max_len))
    logging.info('{};train data: {}; dev data: {}'.format(now(), len(train_data), len(dev_data)))

    # 4 optimiezer
    train_steps = (len(train_data) + opt.batch_size - 1) // opt.batch_size
    dev_steps = (len(dev_data) + opt.batch_size - 1) // opt.batch_size

    if opt.full_finetuning:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]
    optimizer = optim.Adam(optimizer_grouped_parameters, lr=opt.lr)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(1 + 0.95*epoch))
    # training
    for epoch in range(opt.epochs):
        logging.info("{}; epoch:{}/{};training....".format(now(), epoch, opt.epochs))
        model.train()
        scheduler.step()
        loss_avg = utils.RunningAverage()
        loss_tag_avg = utils.RunningAverage()
        loss_rel_avg = utils.RunningAverage()

        data_interator = enumerate(train_data_loader)
        t = trange(train_steps)
        for i in t:
            idx, data = next(data_interator)
            sens , tags = list(map(lambda x: torch.LongTensor(x), data[:2]))
            entRels = data[-1]
            if opt.use_gpu:
                sens = sens.cuda()
                tags = tags.cuda()
            loss_tags, loss_rels = model(sens, tags, entRels)
            loss = opt.lam * loss_tags + (1-opt.lam)*loss_rels
            # 梯度裁剪
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=opt.clip_grad)
            optimizer.step()

            loss_avg.update(loss.item())
            loss_tag_avg.update(loss_tags.item())
            loss_rel_avg.update(loss_rels.item())
            t.set_postfix(loss='{:05.3f}/{:05.3f}'.format(loss_avg(), loss.item()), \
                          tag_loss='{:05.3f}/{:05.3f}'.format(loss_tag_avg(), loss_tags.item()), \
                          rel_loss='{:05.3f}/{:05.3f}'.format(loss_rel_avg(), loss_rels.item()))

        print("saving model ..")
        model.save(opt, epoch)
        #print("evaluate train set: ")
        #evaluate(opt, model, train_steps, train_data_loader, epoch, case='train')
        print("evaluate dev set: ")
        evaluate(opt, model, dev_steps, dev_data_loader, epoch, case='dev')


def evaluate(opt, model, steps, data_loader, epoch, case='dev'):
    model.eval()
    predicts = []
    goldens = []
    g_entRel_t = []
    p_entRel_t = []
    tag2id = json.loads(open(opt.tag2id_dir, 'r').readline())
    id2tag = {tag2id[k]: k for k in tag2id.keys()}
    with torch.no_grad():
        data_interator = enumerate(data_loader)
        t = trange(steps)
        for i in t:
            idx, data = next(data_interator)
            sens, g_tags = list(map(lambda x: torch.LongTensor(x), data[:2]))
            g_entRel = data[-1]
            if opt.use_gpu:
                sens = sens.cuda()
                g_tags = g_tags.cuda()
            p_tags, all_out = model(sens, None, None)
            if 'crf' not in opt.model.lower():
                p_tags = torch.max(p_tags, 2)[1]
            if opt.use_gpu:
                g_tags = g_tags.cpu()
                if 'crf' not in opt.model.lower():
                    p_tags = p_tags.cpu()
            g_tags = g_tags.tolist()
            goldens.extend([id2tag.get(idx) for indices in g_tags for idx in indices])
            if 'crf' not in opt.model.lower():
                p_tags = p_tags.tolist()
            predicts.extend([id2tag.get(idx) for indices in p_tags for idx in indices])
            g_entRel_t.extend(g_entRel)
            p_entRel_t.extend(all_out)
        # 测试单纯的位置对应准确率
    assert len(g_entRel_t) == len(p_entRel_t)
    p_t, r_t, f_t = f1_score_ent_rel(g_entRel_t, p_entRel_t)
    logging.info("epoch {}; POS: pre: {}; rel: {}; f1: {}".format(epoch, p_t, r_t, f_t))

    # 测试实际转换为文字的准确率
    if case =='dev':
        data_path = opt.dev_data_dir
    else:
        data_path = opt.train_data_dir
    json_data = load_data(data_path)
    # assert len(json_data) == len(p_entRel_t)
    predict_data =  utils.get_text_spolist(opt, p_entRel_t, json_data)
    p, r, f = eval_file(predict_data, data_path)
    logging.info("epoch {}; REL: pre:{};rel:{};f1:{}".format(epoch, p,r,f))

    assert len(g_tags) == len(p_tags)
    p, r, f = f1_score(goldens, predicts)
    logging.info("epoch {}; NER: pre: {}; rel: {}; f1: {}".format(epoch, p, r, f))

def tofile(**kwargs):
    opt.parse(kwargs)
    if opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)
    # 2 model
    model = getattr(models, opt.model)(opt)
    if opt.use_gpu:
        model.cuda()
    print("{} load ckpt from: {}".format(now(), opt.ckpt_path))
    model.load(opt.ckpt_path)
    model.eval()
    data = Data(opt, case=opt.case+1)
    data_loader = DataLoader(data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, collate_fn=collate_fn)
    print("predict case:{},data num:{}".format(opt.case, len(data)))



    tag2id = json.loads(open(opt.tag2id_dir, 'r').readline())
    id2tag = {tag2id[k]: k for k in tag2id.keys()}

    steps = (len(data) + opt.batch_size - 1) // opt.batch_size
    data_interator = enumerate(data_loader)
    t = trange(steps)
    p_entRel_t = []
    dev_entRel_t = []
    pred_tags = []
    true_tags = []
    with torch.no_grad():
        for i in t:
            idx, data = next(data_interator)
            sens, true_tag = list(map(lambda x: torch.LongTensor(x), data[:2]))
            dev_entRel = data[-1]
            if opt.use_gpu:
                sens = sens.cuda()
            p_tags, all_out = model(sens, None, None)
            if 'crf' not in opt.model.lower():
                p_tags = torch.max(p_tags, 2)[1]
            if opt.use_gpu:
                if 'crf' not in opt.model.lower():
                    p_tags = p_tags.cpu()
            p_entRel_t.extend(all_out)
            dev_entRel_t.extend(dev_entRel)
            true_tags.extend(true_tag.tolist())
            if 'crf' not in opt.model.lower():
                pred_tags.extend(p_tags.tolist())
            else:
                pred_tags.extend(p_tags)

    if opt.case == 0:
        data_path = opt.dev_data_dir
    elif opt.case == 1:
        data_path = opt.test1_data_dir
    else:
        data_path = opt.test2_data_dir
    json_data = load_data(data_path)[:len(true_tags)]
    # assert len(json_data) == len(p_entRel_t)
    predict_data =  utils.get_text_spolist(opt, p_entRel_t, json_data)
    if opt.case == 0:
        p, r, f = eval_file(predict_data, opt.dev_data_dir)
        print("predict res: pre:{};rel:{};f1:{}".format(p,r,f))
    with open('out/pred_out', 'w') as f:
        for p_data in predict_data:
            f.write(json.dumps(p_data, ensure_ascii=False)+'\n')

    if opt.case == 0:
        dev_data = utils.get_text_spolist(opt,dev_entRel_t, json_data)
        p, r, f = eval_file(dev_data, opt.dev_data_dir)
        print("origin dev res: pre:{};rel:{};f1:{}".format(p,r,f))
        with open('./out/true_out', 'w') as f:
            for p_data in dev_data:
                f.write(json.dumps(p_data, ensure_ascii=False)+'\n')
        utils.write_tags(opt, true_tags, pred_tags, json_data, './out/tag_out', id2tag)

if __name__ == "__main__":
    fire.Fire()
