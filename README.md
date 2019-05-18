# 项目目录
```
|—— models      # 存放模型目录 存放网页相关前端配置
|—— modules     # 存放自己封装的encoder
|—— out         
|—— bert-base-chinese 
|—— |—— bert-base-chinese.tar.gz # bert预训练参数
|—— |—— vocab.txt # bert词典库
|—— data        # 存放数据
|—— |—— Data.py
|—— |—— small #自己合并了实体类型的数据
|—— |——|—— json_data  
|—— |——|—— npy_data  
|—— |——|——|——train
|—— |——|——|——dev
|—— |——|——|——test1
|—— |——|——|——test2
|—— |——|—— origin_data  # 存放原始数据
|—— analysis_result.ipynb  # 用来分析错误结果
|—— checkpoints # 存放训练模型参数
|—— config.py     
|—— helpData.py # 数据预处理函数
|—— mian.py     # 主函数
|—— metrics.py  # 测评函数
|—— README.md
```
# 项目环境(主要环境)
- Ubuntu 16.04
- Pytorch 1.x
- Python 3.x
# 运行方式
- 克隆项目

```
git clone https://github.com/Wangpeiyi9979/IE-Bert-CNN.git
```
- 准备数据
    - 在这里[下载](https://pan.baidu.com/s/1DG1aVcDzbKG3ubkj8Q8nHQ)数据,提取码`59fg`。
    - 解压数据，放在`data/small/origin_data/`文件夹下
- 准备Bert预训练模型
    - 在这里[下载](https://pan.baidu.com/s/1EGkPB628ewXJhqqgrHBfDw), 提取码`uolz`。将下载后的压缩文件放在`bert-base-chinese`文件夹下
- 在data/small/目录下按项目结构中所示创建所需目录
- 回到主目录，执行

```
python helpData.py
```
- 开始训练

```
python main train
```
- 预测
    - 将config.py中的`ckpt_path`更改为训练后的模型地址.
    执行：
    ```
    python main tofile --case=1
    ```
    预测结果存放在`out`文件夹下.

- 结果： f1:0.81
