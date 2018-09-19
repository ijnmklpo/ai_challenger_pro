# -*- coding:utf-8 -*-
# 
# @Author: ijnmklpo
# @Time: 2018/9/7 上午8:52
# @Desc: 通过词典去除表情的类

import numpy as np



import re


reg_embedding = '(\*|\!|\@|\#|\$|\%|\^|\&|\*|\=|\+|\-|\<|\{|\}|\[|\]|\:|\'|\"|\(|\)|\#|\/|\\\\|\||\?|\.|\,|\<|\>)'

class Emotion_Filter(object):
    emotion_dict = None
    emotion_reg = None
    emotion_reg_pattern = None

    def __init__(self, dict_file):
        self.emotion_list = []
        self.emotion_reg = ''
        with open(dict_file, 'r') as emotions:
            for i, emotion in enumerate(emotions):
                self.emotion_list.append(emotion.strip())
                if self.emotion_reg == '':
                    self.emotion_reg += self.__reg_fit(emotion)
                else:
                    self.emotion_reg += '|' + self.__reg_fit(emotion)
        self.emotion_reg_pattern = re.compile(self.emotion_reg)

    def __reg_fit(self, emotion):
        return re.sub(reg_embedding, '\\\\\\1', emotion.strip())


    def reg_filter(self, text):
        return re.sub(self.emotion_reg_pattern, '', text)


if __name__ == '__main__':
    a = Emotion_Filter('emotion_dict.txt')
    print(a.reg_filter('@_@@_@@_@哩个就系晕表情啊~~~~~^_^'))