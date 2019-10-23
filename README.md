# 聊天机器人

聊天机器人的 PyTorch 实现。


## 依赖

- Python 3.6.8
- PyTorch 1.3.0

## 数据集

100w豆瓣语料

## 用法

### 数据预处理
提取训练和验证样本：
```bash
$ python extract.py
$ python pre_process.py
```

### 训练
```bash
$ python train.py
```

要想可视化训练过程，在终端中运行：
```bash
$ tensorboard --logdir path_to_current_dir/logs
```

### Demo
下载 [预训练模型](https://github.com/foamliu/Scene-Classification/releases/download/v1.0/model.85-0.7657.hdf5) 然后执行:

```bash
$ python demo.py
```

下面第一行是英文例句（数据集），第二行是人翻中文例句（数据集），之后一行是机翻（本模型）中文句子（实时生成）。

<pre>



</pre>