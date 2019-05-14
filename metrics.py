"""Thanks to https://github.com/chakki-works/seqeval
Metrics to assess performance on sequence labeling task given prediction
Functions named as ``*_score`` return a scalar value to maximize: the higher
the better
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import defaultdict
import numpy as np
import json
import utils

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

def f1_score_ent_rel(g_entRel, p_entRel):
    """
    g_entRel [[[s1,e1, s2, e2, r],[]], [[], []], ....]
    """
    acc, pre, rel = 0.0, 0.0, 0.0
    # 针对dev， 没有对dev补NA关系
    #   train, need
    for idx, g_sen in enumerate(g_entRel):
        p_sen = p_entRel[idx]
        pre += len(p_sen)
        rel += len(g_sen)
        for p_s in p_sen:
            if p_s in g_sen:
                acc+=1
    p = 100 * acc / pre if pre > 0 else 0
    r = 100 * acc / rel if rel > 0 else 0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0
    return p, r, f1

def get_entities(seq, suffix=False):
    """Gets entities from sequence.

    Args:
        seq (list): sequence of labels.

    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).

    Example:
        >>> from seqeval.metrics.sequence_labeling import get_entities
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> get_entities(seq)
        [('PER', 0, 1), ('LOC', 3, 3)]
    """
    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]

    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        if suffix:
            tag = chunk[-1]
            type_ = chunk.split('-')[0]
        else:
            tag = chunk[0]
            type_ = chunk.split('-')[-1]

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i-1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks


def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'S': chunk_end = True

    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'S': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True

    if prev_tag == 'E' and tag == 'E': chunk_start = True
    if prev_tag == 'E' and tag == 'I': chunk_start = True
    if prev_tag == 'S' and tag == 'E': chunk_start = True
    if prev_tag == 'S' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start


def f1_score(y_true, y_pred, average='micro', digits=2, suffix=False):
    """Compute the F1 score.

    The F1 score can be interpreted as a weighted average of the precision and
    recall, where an F1 score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F1 score are
    equal. The formula for the F1 score is::

        F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.

    Returns:
        score : float.

    Example:
        >>> from seqeval.metrics import f1_score
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> f1_score(y_true, y_pred)
        0.50
    """
    true_entities = set(get_entities(y_true, suffix))
    pred_entities = set(get_entities(y_pred, suffix))

    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)

    p = 100 * nb_correct / nb_pred if nb_pred > 0 else 0
    r = 100 * nb_correct / nb_true if nb_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0

    return p, r, score


def accuracy_score(y_true, y_pred):
    """Accuracy classification score.

    In multilabel classification, this function computes subset accuracy:
    the set of labels predicted for a sample must *exactly* match the
    corresponding set of labels in y_true.

    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.

    Returns:
        score : float.

    Example:
        >>> from seqeval.metrics import accuracy_score
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> accuracy_score(y_true, y_pred)
        0.80
    """
    if any(isinstance(s, list) for s in y_true):
        y_true = [item for sublist in y_true for item in sublist]
        y_pred = [item for sublist in y_pred for item in sublist]

    nb_correct = sum(y_t==y_p for y_t, y_p in zip(y_true, y_pred))
    nb_true = len(y_true)

    score = nb_correct / nb_true

    return score


def classification_report(y_true, y_pred, digits=2, suffix=False):
    """Build a text report showing the main classification metrics.

    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a classifier.
        digits : int. Number of digits for formatting output floating point values.

    Returns:
        report : string. Text summary of the precision, recall, F1 score for each class.

    Examples:
        >>> from seqeval.metrics import classification_report
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> print(classification_report(y_true, y_pred))
                     precision    recall  f1-score   support
        <BLANKLINE>
               MISC       0.00      0.00      0.00         1
                PER       1.00      1.00      1.00         1
        <BLANKLINE>
        avg / total       0.50      0.50      0.50         2
        <BLANKLINE>
    """
    true_entities = set(get_entities(y_true, suffix))
    pred_entities = set(get_entities(y_pred, suffix))

    name_width = 0
    d1 = defaultdict(set)
    d2 = defaultdict(set)
    for e in true_entities:
        d1[e[0]].add((e[1], e[2]))
        name_width = max(name_width, len(e[0]))
    for e in pred_entities:
        d2[e[0]].add((e[1], e[2]))

    last_line_heading = 'avg / total'
    width = max(name_width, len(last_line_heading), digits)

    headers = ["precision", "recall", "f1-score", "support"]
    head_fmt = u'{:>{width}s} ' + u' {:>9}' * len(headers)
    report = head_fmt.format(u'', *headers, width=width)
    report += u'\n\n'

    row_fmt = u'{:>{width}s} ' + u' {:>9.{digits}f}' * 3 + u' {:>9}\n'

    ps, rs, f1s, s = [], [], [], []
    for type_name, true_entities in d1.items():
        pred_entities = d2[type_name]
        nb_correct = len(true_entities & pred_entities)
        nb_pred = len(pred_entities)
        nb_true = len(true_entities)

        p = 100 * nb_correct / nb_pred if nb_pred > 0 else 0
        r = 100 * nb_correct / nb_true if nb_true > 0 else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0

        report += row_fmt.format(*[type_name, p, r, f1, nb_true], width=width, digits=digits)

        ps.append(p)
        rs.append(r)
        f1s.append(f1)
        s.append(nb_true)

    report += u'\n'

    # compute averages
    report += row_fmt.format(last_line_heading,
                             np.average(ps, weights=s),
                             np.average(rs, weights=s),
                             np.average(f1s, weights=s),
                             np.sum(s),
                             width=width, digits=digits)

    return report

def get_sent2triple_set(json_datas):
    sen2set = {}
    for data in json_datas:
        sent = data['text']
        spo_set = set()
        for spo_item in data['spo_list']:
            s = spo_item['subject'].lower()
            o = spo_item['object'].lower()
            r = spo_item['predicate']
            if r == 'NA':
                continue
            spo_set.add((o, r, s))
        sen2set[sent] = spo_set
    return sen2set

def eval_file(predict_json, golden_file_path):
    '''
    传入两个数据格式如下文件
    {"text": "《离开》是由张宇谱曲，演唱",
    "spo_list": [{"subject": "离开", "predicate": "歌手", "object": "张宇", "subject_type": "歌曲", "object_type": "人物"},
    {"subject": "离开", "predicate": "作词", "object": "张宇", "subject_type": "歌曲", "object_type": "人物"}]}
    '''
    correct_sum, predict_sum, recall_sum = 0.0, 0.0, 0.0
    golden_data = load_data(golden_file_path)[:len(predict_json)]
    pred_sent2set = get_sent2triple_set(predict_json)
    golden_sent2set = get_sent2triple_set(golden_data)
    for sent in golden_sent2set.keys():
        golden_set = golden_sent2set[sent]
        pred_set = pred_sent2set.get(sent, set())
        recall_sum += len(golden_set)
        predict_sum += len(pred_set)
        for each in pred_set:
            if each in golden_set:
                correct_sum += 1
    pre = correct_sum / predict_sum
    rel = correct_sum / recall_sum
    f1_num = pre + rel
    if f1_num == 0:
        f1_num = 10000
    f1 = 2*pre*rel/f1_num
    return 100*pre, 100*rel, 100*f1

def judge_data_quality(opt):
    train_npy_data = np.load(opt.npy_data_root+'train/relations.npy')
    dev_npy_data = np.load(opt.npy_data_root+'dev/relations.npy')
    train_data = []
    dev_data= []
    for i in train_npy_data:
        train_data.append(i)
    for i in dev_npy_data:
        dev_data.append(i)
    json_data = load_data(opt.train_data_dir)
    train_data =  utils.get_text_spolist(opt, train_data, json_data)
    json_data = load_data(opt.dev_data_dir)
    dev_data =  utils.get_text_spolist(opt, dev_data, json_data)
    print("judge train data..")
    p, r, f = eval_file(train_data, opt.train_data_dir)
    print("train_data p:{};r:{};f1:{}".format(p, r, f))
    print("judge dev data..")
    p, r, f = eval_file(dev_data, opt.dev_data_dir)
    print("dev_data p:{};r:{};f1:{}".format(p, r, f))
