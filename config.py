# -*- coding: utf-8 -*-

class Config(object):
    # -----------数据集选择--------------------#
    dataset = 'small'       # large(没有合并type关系)/small(合并type关系)
    naNum = 2               # 每个例子中补充的最大NA关系数目
    tag_nums = 19*2+1       # tag类型数量
    rel_nums = 50           # 关系数量

    # -------------dir ----------------#
    bert_model_dir = './bert-base-chinese/bert-base-chinese.tar.gz'
    bert_vocab_dir = './bert-base-chinese/vocab.txt'
    bert_vocab_unk = './bert-base-chinese/vocab.txt'

    npy_data_root = './data/'+dataset+'/npy_data/'
    origin_data_root = './data/'+dataset+'/origin_data/'
    json_data_root = './data/'+dataset+'/json_data/'

    id2type_dir = json_data_root + 'id2type.json'
    type2id_dir = json_data_root + 'type2id.json'
    tag2id_dir = json_data_root + 'tag2id.json'
    r2id_dir = json_data_root + 'r2id.json'
    id2r_dir = json_data_root + 'id2r.json'
    id2tag_dir = json_data_root + 'id2tag.json'
    type2types_dir = json_data_root + 'type2types.json'

    schema_dir_old = origin_data_root + 'all_50_schemas_old'
    schema_dir_new = origin_data_root + 'all_50_schemas_new'
    train_data_dir = origin_data_root + 'train_data.json'
    dev_data_dir = origin_data_root + 'dev_data.json'
    test1_data_dir = origin_data_root + 'test1_data_postag.json'
    test2_data_dir = origin_data_root + 'test2_data_postag.json'

    log_dir = './log'
    #  -------------- 模型超参数 -----------#
    k = 9
    filters = [5, 9, 13]               # CNN卷积核宽度
    filter_num = 230 # CNN卷积核个数
    seq_length = 180
    tuple_max_len = 13
    bert_hidden_size = 768   # bert隐层维度，固定
    lam = 0.85           # 越大tag越重要`

    # --------------main.py ----------------#
    load_ckpt = False
    ckpt_path = './checkpoints/BERT_CNN_CRF_sl:180_k:[5, 9, 13]_fn:230_lam:0.85_lr:3e-05_epoch:9'
    num_workers = 1
    seed = 9979
    epochs = 10
    batch_size = 20
    use_gpu = 1
    gpu_id = 0
    # ------------optimizer ------------------#
    lr = 3e-5
    full_finetuning = True
    optimizer = 'Adam'
    model = 'BERT_MUL_CNN'  # 'BERT_CNN_CRF'
    clip_grad = 2  # 梯度的最大值

    # ----------预测数据集----------#
    case = 1 # 0:dev(并测试数据质量) 1:test1, 2:test2

    def parse(self, kwargs):
        '''
        user can update the default hyperparamter
        '''
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception('opt has No key: {}'.format(k))
            setattr(self, k, v)

        print('*************************************************')
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print("{} => {}".format(k, getattr(self, k)))

        print('*************************************************')
opt = Config()
