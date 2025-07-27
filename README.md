# Times-Series-Library 工程结构通俗解析 #

## 1. 主程序 run.py ##
终端 or shell脚本输入 ----  `argparse` ---> 记录下所有参数的args

args ---- `args match` ---> 提取args信息

args信息 ---- `Exp` ---> 根据信息实例化每个任务对应的Exp对象并执行

## 2. 任务执行程序Exp ##
每个Exp类下都对应一个任务，实现了同一任务下的不同模型的train, validate, test<br><br>
具体调用哪个Exp以及Exp下使用哪个model,取决于args解析出的任务信息和模型信息

## 3. models和layers ##
每个models的各个block定义在 `./models/模型名称.py` 的文件下，基本就是按照原文的网络设计搭建起来网络架构

各个block可能用到的一些可复用的功能函数放在 `./layers/函数功能.py` 的文件下

## 4. 数据迭代器data_provider ##
将已经预处理好的数据或者源数据放在 `./data/数据集名称/` 这个目录下

相应的数据集合的dataset实现放在 `./data_provider/data_factory.py` 下，每个数据集对应一个，如果有特殊的collate_fn也在该文件下实现

相应的数据迭代器的loader实现放在 `./data_provider/data_loader.py` 下，主要是根据相应的args的任务信息加载数据迭代器

## 5. 启动脚本scripts ##
基本上每个模型所需的相关参数都很长，所以最后是写成一个shell脚本启动，`./scripts` 下有原作者本来的shell脚本示例

# 代码重构相关文件 #
1. run的args部分加入自己所需的参数，并更新`./utils/print_args.py`
2. `./models`和`./layers`下实现自己的模型
3. 如果实现的模型已经有合适的Exp那最好，否则要么是新写一个模型要么是给比较接近的Exp再加一个分支

# 目前以实现的方法和对应的Exp #
exp_lstm(使用了lstm的交通预测任务) ------- MDTP

exp_prediction(没有使用lstm的较为通用的交通预测任务) ------- Trajnet

# 数据链接 #
[百度网盘数据链接](https://pan.baidu.com/s/1s3VafVC22W18ktWrjqEaRg?pwd=ss52)



