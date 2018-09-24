# -*- coding:utf-8 -*-
# 
# @Author: ijnmklpo
# @Time: 2018/9/7 上午8:51
# @Desc: 放一些通用的方法，比如对单个句子进行预处理的方法，这种会在多个场景下被调用

try:
    import data_preprocess.Emotion_Filter as ef
except:
    from data_preprocess import Emotion_Filter as ef
try:
    import data_preprocess.lanconv as lanconv
except:
    import lanconv as lanconv
import tensorflow as tf
import re
import logging
import pandas as pd
import jieba
# 自定义的分词词典
jieba.load_userdict('data_preprocess/userdict.txt')

# 表情词典的目录
emotion_path = 'data_preprocess/emotion_dict.txt'


def text_remove_emotions(text, emotion_filter=None):
    '''
    去掉emoji表情和颜文字，通过词典来实现的
    :param text: str。要处理的文本
    :param emotion_filter: Emotion_Filter。过滤器类。
    :return:
    '''
    if emotion_filter is None:
        emotion_filter = ef.Emotion_Filter(emotion_path)
    return emotion_filter.reg_filter(text)


def text_shrink_punctuations(text, pattern=None):
    '''
    因为有些人为了表达情绪，会连续用多个标点，但是这种标点并不包含太多语义信息，所以要缩减标点符号，比如把'~~~~'转化成'~'，把'。。。'转化成'。'
    :param text: 要处理的文本
    :param pattern: 可以传入要处理的标点正则匹配模式，不传的话会默认一个值（通常用默认的就好了）
    :return:
    '''
    if pattern is None:
        reg_exp = r'( |\*|\!|\@|\#|\$|\%|\^|\&|\*|\=|\+|\-|\<|\{|\}|\[|\]|\:|\'|\"|\(|\)|\#|\/|\\\\|\||\?|\.|\,|\<|\>|！|、|。|，|《|》|【|】|~|￥|（|）|——|_|\||\||…|·|\s)\1+'
        pattern = re.compile(reg_exp)
    return pattern.sub(r'\1', text)  # \1表示表达式中第一个括号里匹配到的内容


def text_remove_useless_chars(text, chars=None):
    '''
    除去句子前后无用的特殊符号

    :param text: str。要处理的文本。
    :param chars: str。要匹配的strip模式。
    :return:
    '''
    if chars is None:
        chars = ' \'\"\f\n\r\t\v\\/~({<[【+=-_、@#￥%……&*^'

    text_out = text.strip(chars)  # 去掉前后无用的字符

    pattern2 = re.compile(r' |\'|\"|\s|\r|\f|\n|\t|\v')
    text_out = pattern2.sub(r'。', text_out)
    pattern3 = re.compile(r',')
    text_out = pattern3.sub(r'，', text_out)

    return text_out


def text_cht_to_chs(text):
    '''
    繁体转简体。
    :param text: str。要处理的文本。
    :return:
    '''
    text_out = lanconv.Converter('zh-hans').convert(text)
    text_out.encode('utf-8')
    return text_out


def init_vocab(cutted_corpus_list):
    '''
    初始化词典。
    :param cutted_corpus_list: list[list[]]。切割过的语料文件。
    :return:
    '''
    vocab_list = []
    out_list = ['__blank__']
    for cut_corpus in cutted_corpus_list:
        for i, seg_list in enumerate(cut_corpus):
            if i % 1000 == 0:
                print(i)
            if len(seg_list) == 0:
                print(1)
            vocab_list.extend(seg_list)
    vocab_list = list(set(vocab_list))
    out_list.extend(vocab_list)     # 为了保证'__blank__'下标为0
    return out_list

def save_vocab(vocab, vocab_path, encoding='utf-8'):
    '''
    保存词典到某个文件
    :param vocab:
    :param vocab_path:
    :param encoding:
    :return:
    '''
    writer = open(vocab_path, 'w', encoding=encoding)
    for item in vocab:
        writer.write(item + '\n')


def one_sentence_transform(comment, vocab):
    comment = text_remove_emotions(comment)
    comment = text_remove_useless_chars(comment)
    comment = text_shrink_punctuations(comment)
    comment = text_cht_to_chs(comment)
    comment = jieba.cut(comment)
    cut_comment = [item for item in comment]
    ind_comment = []
    for token in cut_comment:
        try:
            token_index = vocab.index(token)
            ind_comment.append(token_index)
        except:
            # 如果token不在词典里，填充一个空的token，index=0
            token_index = 0
            ind_comment.append(token_index)
    return cut_comment, ind_comment



def get_optimizer(loss, learning_rate=0.001):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(loss)
    return optimizer


def accuracy(logits, labels, begin, size):
    p = tf.slice(logits, begin, size)

    max_idx_p = tf.argmax(p, 1)
    max_idx_p = tf.cast(max_idx_p, dtype=tf.int32)
    correct_pred = tf.equal(max_idx_p, labels)
    acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return acc




if __name__ == '__main__':
    pass