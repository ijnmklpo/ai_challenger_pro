# -*- coding:utf-8 -*-
# 
# @Author: ijnmklpo
# @Time: 2018/9/7 上午9:23
# @Desc: 语料类


import pandas as pd
import jieba
import re
try:
    import data_preprocess.Emotion_Filter as ef
    import data_preprocess.lanconv as lanconv
except:
    import Emotion_Filter as ef
    import lanconv as lanconv
import tensorflow as tf
import os
import logging
import multiprocessing as multP
import utils
import tensorflow.contrib.data as cb_data

import global_configs


try:
    jieba.load_userdict('userdict.txt')
except:
    jieba.load_userdict('data_preprocess/userdict.txt')




class CorpusBleach(object):
    '''
    对文本做预处理，构建词典，以及切割
    '''
    docs = None
    text_col = None
    cut_docs = None
    ind_docs = None
    vocab = None    # 不大好存成dataframe。如果要存成csv，只能用encoding='utf-8'来存，这时候直接用pd.read_csv()读取，会取不到空格对应的行，如果用普通文件打开然后读，因为utf-8会转义一些特殊字符，会导致一些特殊字符没法正常读取，很僵。

    def __init__(self, file_path, text_col_name, limit_len=0):
        """
        读入整个训练集数据，为了降低oov对线上结果的影响，只用训练集数据进行构造词典
        """
        self.docs = pd.read_csv(file_path, encoding='utf-8')
        self.text_col = text_col_name
        if limit_len > 0:
            self.docs = self.docs[0: limit_len]
        print('get corpus!')



    def remove_emotions(self, emotion_path):
        """
        从文档中去除颜文字，颜文字字典还需要扩充的。预处理的第1步。
        :param emotion_path:
        :return:
        """
        print('remove emotions...')
        emotion_filter = ef.Emotion_Filter(emotion_path)

        for i, row in self.docs.iterrows():
            if i % 1000 == 0:
                print('remove emotions: %d' % (i))
            self.docs.loc[[i], [self.text_col]] = utils.text_remove_emotions(row[self.text_col], emotion_filter)


    def shrink_punctuations(self):
        """
        将重复的标点符号只保留一个。
        不知道为什么，这个正则必须要写在正则串前面加r才能用，按理说不用停止转义的呀。
        这个函数要在remove_emotions执行完再用比较好，不然会把一些颜文字弄乱掉。
        预处理第2步。
        :param text:
        :return:
        """
        print('shrink punctuations...')
        reg_exp = r' |(\*|\!|\@|\#|\$|\%|\^|\&|\*|\=|\+|\-|\<|\{|\}|\[|\]|\:|\'|\"|\(|\)|\#|\/|\\\\|\||\?|\.|\,|\<|\>|！|、|。|，|《|》|【|】|~|￥|（|）|——|_|\||\||…|·|\s)\1+'
        pattern = re.compile(reg_exp)

        for i, row in self.docs.iterrows():
            if i % 1000 == 0:
                print('shrink punctuations: %d' % (i))
            self.docs.loc[[i], [self.text_col]] = utils.text_shrink_punctuations(row[self.text_col], pattern)  # \1表示表达式中第一个括号里匹配到的内容


    def remove_useless_chars(self, chars=' \'\"\f\n\r\t\v\\/~({<[【+=-_、@#￥%……&*^'):
        """
        去掉文本中前后的空格和换行等符号，并且用正则去掉中间的无用符号。预处理第3步。
        :param chars: 要去掉的字符系列。
        :return:
        """
        print('remove useless words...')

        for i, row in self.docs.iterrows():
            if i % 1000 == 0:
                print('remove useless chars: %d' % (i))
            self.docs.loc[[i], [self.text_col]] = utils.text_remove_useless_chars(row[self.text_col], chars=chars)


    def cht_to_chs(self):
        """
        繁体转简体。预处理第4步。
        :return:
        """
        print('cht to chs...')
        for i, row in self.docs.iterrows():
            if i % 1000 == 0:
                print('transform to simplified: %d' % (i))
            self.docs.loc[[i], [self.text_col]] = utils.text_cht_to_chs(row[self.text_col])


    def remove_short_text(self, min_len=0):
        """
        去掉比较短的文本用来训练，这个在训练集里可以用，测试集就不用了。
        :param min_len:
        :return:
        """
        print('drop short sentences...')
        remove_idx_list = []
        for i, row in self.docs.iterrows():
            if i % 1000 == 0:
                print('remove short text: %d' % (i))
            if len(row[self.text_col]) <= min_len:
                remove_idx_list.append(i)
        print('drop idxs:', remove_idx_list)
        self.docs.drop(self.docs.index[remove_idx_list], inplace=True)


    def label_assemble(self):
        '''
        将语料中多个label合并到一个list里
        :return:
        '''
        print('label assemble...')
        label_list = []
        for i, row in self.docs.iterrows():
            if i % 1000 == 0:
                print('assemble label in doc: %d' % (i))
            label_list.append([row[]])
        print('drop idxs:', remove_idx_list)
        self.docs.drop(self.docs.index[remove_idx_list], inplace=True)



    def flow_process(self, emotion_path):
        """
        整体进行语料的预处理，并不包含保存
        :param emotion_path:
        :return:
        """
        self.remove_emotions(emotion_path)
        self.remove_useless_chars()
        self.shrink_punctuations()
        self.cht_to_chs()
        self.remove_short_text()



    def word_cut(self):
        """
        用jieba分词切语料
        :return:
        """
        self.doc_ids = []
        self.cut_docs = []
        self.doc_labels = []
        for i, row in self.docs.iterrows():
            if i % 100 == 0:
                print(i)
            # print(row[self.text_col])
            seg_list = jieba.cut(row[self.text_col])
            seg_list = [item for item in seg_list]
            if len(seg_list) == 0:
                continue
            else:
                self.doc_ids.append(row[self.id_col])
                self.cut_docs.append(seg_list)
                self.doc_labels.append(row[self.label_col])


    def init_vocab(self):
        self.vocab = set()
        vocab_list = ['__blank__']
        if self.cut_docs is None:
            print('no cut docs, start cutting...')
            self.word_cut()
        for i, seg_list in enumerate(self.cut_docs):
            if i % 100 == 0:
                print(i)
            vocab_list.extend(seg_list)
        vocab_list = list(set(vocab_list))
        self.vocab = vocab_list



if __name__ == '__main__':
    # ## 对语料进行预处理，未分词
    # cps = CorpusBleach(global_configs.raw_corpus_path,text_col_name='content')
    # print(cps.docs.describe())
    # print(cps.docs.head())
    # cps.flow_process('./data_preprocess/emotion_dict.txt')
    # cps.docs.to_csv(global_configs.preprocessed_corpus_path, index=False, sep=',')


    # 对预处理后的语料进行分词
    cps = CorpusBleach(global_configs.preprocessed_corpus_path, text_col_name='content')
    print(cps.docs['content'][35])
