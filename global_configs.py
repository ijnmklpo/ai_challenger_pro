# -*- coding:utf-8 -*-
# 
# @Author: ijnmklpo
# @Time: 2018/9/6 上午11:25
# @Desc: 全局变量的定义

import numpy as np
import os

## 服务器端根目录
# root_path = '/data/ryh/datas/ai_challenger_data/ai_challenger_sentiment_analysis_trainingset_20180816'  # 训练集目录
root_path = '/data/ryh/datas/ai_challenger_data/ai_challenger_sentiment_analysis_validationset_20180816'    # 验证集目录
# root_path = '/data/ryh/datas/ai_challenger_data/ai_challenger_sentiment_analysis_testa_20180816'    # 测试集目录

## 本地根目录
# root_path = '/Users/renyihao/Datas/ai_challenger_data/ai_challenger_sentiment_analysis_trainingset_20180816'


# 原始的语料文件
raw_corpus_path = os.path.join(root_path, 'sentiment_analysis.csv')

# 预处理后的语料（分词前）
preprocessed_corpus_path = os.path.join(root_path, 'preprocessed_corpus.csv')

# 分词后的语料文件
cutted_corpus_path = os.path.join(root_path, 'cutted_corpus.csv')

# 词典文件
vocab_path = os.path.join(root_path, 'vocab.csv')

# 索引化后的语料文件
indexed_corpus_path = os.path.join(root_path, 'indexed_corpus.csv')

# 索引化后的语料文件
tfrecord_corpus_path = os.path.join(root_path, 'corpus.tfrecord')


# 模型中用的目录结构
train_root_path = '/data/ryh/datas/ai_challenger_data/ai_challenger_sentiment_analysis_trainingset_20180816'  # 训练集根目录
valid_root_path = '/data/ryh/datas/ai_challenger_data/ai_challenger_sentiment_analysis_validationset_20180816'  # 验证集根目录

train_tfrecord_corpus_path = os.path.join(train_root_path, 'corpus.tfrecord')
valid_tfrecord_corpus_path = os.path.join(valid_root_path, 'corpus.tfrecord')


if __name__ == '__main__':
    pass