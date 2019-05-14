import logging
import json
from pytorch_pretrained_bert import BertTokenizer

class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

def get_text_spolist(opt, p_entRel_t, json_data):
    id2r = json.loads(open(opt.id2r_dir, 'r').readline())
    tokenizer = BertTokenizer.from_pretrained(opt.bert_vocab_unk, do_lower_case=True)
    predictg_data = []
    for idx, p_tuples in enumerate(p_entRel_t):
        data_unit = {}
        data = json_data[idx]
        text = data['text']
        # 得到时按同样方式加工
        text = text.strip().replace(' ', '$')
        word_list = tokenizer.tokenize(text)
        word_list = [word.replace('#', '')for word in word_list]
        spo_list = []
        for p_sample in p_tuples:
            o_s, o_e, s_s, s_e, r = p_sample
            if r == 49:
                continue
            obj, sbj = '',''
            if max(o_s, o_e, s_s, s_e) >= len(word_list):
                continue
            for i in range(o_s, o_e+1):
                obj = obj + word_list[i]
            for i in range(s_s, s_e+1):
                sbj = sbj + word_list[i]
            #将@替换回来
            obj = obj.replace('$', ' ')
            sbj = sbj.replace('$', ' ')
            spo_unit = {}
            spo_unit['object'] = obj
            spo_unit['subject'] = sbj
            spo_unit['predicate'] = id2r[str(r)]
            spo_list.append(spo_unit)
        # 替换回来:
        text = text.replace('$', ' ')
        data_unit['text'] = text
        data_unit['spo_list'] = spo_list
        predictg_data.append(data_unit)
    return predictg_data

def norm_length(origin_list):
    """
    规范化标签格式长度
    input: ['球', '星'， '姚', '明']
    """

    return origin_list
    norm_list = []
    for i in  origin_list:
        if len(i) < 5:
            i = i + (5 - len(i)) * ' '
        else:
            i = i[:5]
        norm_list.append(i)
    return norm_list

def write_tags(opt, true_tags, pred_tags, json_data, out_dir, id2tag):
    f = open(out_dir, 'w')
    tokenizer = BertTokenizer.from_pretrained(opt.bert_vocab_unk, do_lower_case=True)
    for idx, data in enumerate(json_data):
        text = data['text']
        # 得到时按同样方式加工
        text = text.strip().replace(' ', '$')
        word_list = tokenizer.tokenize(text)
        word_list = norm_length([word.replace('#', '')for word in word_list])
        true_tag = true_tags[idx][:len(word_list)]
        pred_tag = pred_tags[idx][:len(word_list)]
        true_tag = norm_length([id2tag[i] for i in true_tag])
        pred_tag = norm_length([id2tag[i] for i in pred_tag])
        sens =  "".join(word_list)
        t_tag = " ".join(true_tag)
        p_tag = " ".join(pred_tag)
        f.write(sens+'\n')
        f.write(t_tag+'\n')
        f.write(p_tag+'\n')
        f.write("------------------------------------------------------------------\n")
    f.close()
