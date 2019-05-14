# CCF信息抽取竞赛

## 比赛信息
- 网址:http://lic2019.ccf.org.cn/kg
- 时间节点:
    - 2/25:启动报名，发放训练数据和开发数据
    - 3/31:报名截止，发放第一批测试数据
    - 5/13:发放最终测试数据.
    - 5/20:结果提交截止.
    - 5/31:公布结果，接受系统报告和论文.
    - 8/24:技术交流及颁奖

------

## 数据样例
- 训练集数据示例:
```
{"postag": [{"word": "如何", "pos": "r"}, {"word": "演", "pos": "v"}, {"word": "好", "pos": "a"}, {"word": "自己", "pos": "r"}, {"word": "的", "pos": "u"}, {"word": "角色", "pos": "n"}, {"word": "，", "pos": "w"}, {"word": "请", "pos": "v"}, {"word": "读", "pos": "v"}, {"word": "《", "pos": "w"}, {"word": "演员自我修养", "pos": "nw"}, {"word": "》", "pos": "w"}, {"word": "《", "pos": "w"}, {"word": "喜剧之王", "pos": "nw"}, {"word": "》", "pos": "w"}, {"word": "周星驰", "pos": "nr"}, {"word": "崛起", "pos": "v"}, {"word": "于", "pos": "p"}, {"word": "穷困潦倒", "pos": "a"}, {"word": "之中", "pos": "f"}, {"word": "的", "pos": "u"}, {"word": "独门", "pos": "n"}, {"word": "秘笈", "pos": "n"}], "text": "如何演好自己的角色，请读《演员自我修养》《喜剧之王》周星驰崛起于穷困潦倒之中的独门秘笈", "spo_list": [{"predicate": "主演", "object_type": "人物", "subject_type": "影视作品", "object": "周星驰", "subject": "喜剧之王"}]}
```
- 验证集数据示例:
```
{"postag": [{"word": "查尔斯", "pos": "nr"}, {"word": "·", "pos": "w"}, {"word": "阿兰基斯", "pos": "nr"}, {"word": "（", "pos": "w"}, {"word": "Charles Aránguiz", "pos": "nz"}, {"word": "）", "pos": "w"}, {"word": "，", "pos": "w"}, {"word": "1989年4月17日", "pos": "t"}, {"word": "出生", "pos": "v"}, {"word": "于", "pos": "p"}, {"word": "智利圣地亚哥", "pos": "ns"}, {"word": "，", "pos": "w"}, {"word": "智利", "pos": "ns"}, {"word": "职业", "pos": "n"}, {"word": "足球", "pos": "n"}, {"word": "运动员", "pos": "n"}, {"word": "，", "pos": "w"}, {"word": "司职", "pos": "v"}, {"word": "中场", "pos": "n"}, {"word": "，", "pos": "w"}, {"word": "效力", "pos": "v"}, {"word": "于", "pos": "p"}, {"word": "德国", "pos": "ns"}, {"word": "足球", "pos": "n"}, {"word": "甲级", "pos": "a"}, {"word": "联赛", "pos": "n"}, {"word": "勒沃库森足球俱乐部", "pos": "nt"}], "text": "查尔斯·阿兰基斯（Charles Aránguiz），1989年4月17日出生于智利圣地亚哥，智利职业足球运动员，司职中场，效力于德国足球甲级联赛勒沃库森足球俱乐部", "spo_list": [{"predicate": "出生地", "object_type": "地点", "subject_type": "人物", "object": "圣地亚哥", "subject": "查尔斯·阿兰基斯"}, {"predicate": "出生日期", "object_type": "Date", "subject_type": "人物", "object": "1989年4月17日", "subject": "查尔斯·阿兰基斯"}]}
```
- schema示例:
```
{"object_type": "地点", "predicate": "祖籍", "subject_type": "人物"}
```
-----
## 一些常识
- pos tag:
    - 名词n、时间词t、处所词s、方位词f、数词m、
    - 量词q、区别词b、代词r、动词v、形容词a、状态词z、副词d、
    - 介词p、连词c、助词u、语气词y、叹词e、拟声词o、成语i、
    - 习惯用语l、简称j、前接成分h、后接成分k、语素g、非语素字x、标点符号w

## 相关资料
- https://github.com/crownpku/Information-Extraction-Chinese# IE-Bert-CNN
