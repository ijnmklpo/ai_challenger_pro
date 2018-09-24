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



def ind_task(args):
    if len(args) != 4:
        print('Wrong args!')
        return
    proc_id, doc_num, docs, vocab = args

    doc_ids = (docs['id']).tolist()
    cut_docs = (docs['content'].map(lambda x: x.split('||'))).tolist()
    doc_labels = None
    try:
        doc_labels = (docs['label_list']).tolist()
    except:
        pass
    print()
    out_doc_list = []
    if doc_labels is not None:
        for i, (id, comment, label) in enumerate(zip(doc_ids, cut_docs, doc_labels)):
            if i % 100 == 0:
                print('train corpus. p: %d; current doc:%d; percentage:%f' % (proc_id, i, round(i / doc_num * 100, 3)))
            catch_words = []
            # print(id)
            # print(comment)
            # print(label)
            for token in comment:
                try:
                    token_index = vocab.index(token)
                    catch_words.append(token_index)
                except:
                    # 如果token不在词典里，填充一个空的token，index=0
                    token_index = 0
                    catch_words.append(token_index)
            if len(catch_words) > 0:
                out_doc_list.append([id, catch_words, label])
        print('end multiprocess.')
        return out_doc_list
    else:
        for i, (id, comment) in enumerate(zip(doc_ids, cut_docs)):
            if i % 100 == 0:
                print('test corpus. p: %d; current doc:%d; percentage:%f' % (proc_id, i, round(i / doc_num * 100, 3)))

            catch_words = []
            for token in comment:
                try:
                    token_index = vocab.index(token)
                    catch_words.append(token_index)
                except:
                    pass
                if len(catch_words) > 0:
                    out_doc_list.append([id, catch_words, 'pseudo_label'])
        print('end multiprocess %d.' % (proc_id))
        return out_doc_list



class CorpusBleach(object):
    '''
    对文本做预处理，构建词典，以及切割
    '''
    docs = None
    id_col = 'id'
    text_col = 'content'
    label_col = 'label_list'
    cut_docs = None
    ind_docs = None
    vocab = None    # 不大好存成dataframe。如果要存成csv，只能用encoding='utf-8'来存，这时候直接用pd.read_csv()读取，会取不到空格对应的行，如果用普通文件打开然后读，因为utf-8会转义一些特殊字符，会导致一些特殊字符没法正常读取，很僵。

    def __init__(self, file_path, limit_len=0, sep=',', mode=0, vocab_path=None):
        """
        读入整个训练集数据，为了降低oov对线上结果的影响，只用训练集数据进行构造词典
        """
        self.docs = pd.read_csv(file_path, encoding='utf-8', sep=sep)
        if limit_len > 0:
            self.docs = self.docs[0: limit_len]
        print('get corpus!')

        if vocab_path is not None:
            open_f = open(vocab_path, 'r')
            vocab_list = []
            for i, line in enumerate(open_f):
                vocab_list.append(line.strip('\n'))
            self.vocab = vocab_list

        if mode == 1:
            print('catch cutted docs...')
            cutted_docs = self.docs[self.text_col].map(lambda x: x.split('||'))
            self.cut_docs = cutted_docs.tolist()
            print('catch cutted docs end.')
        elif mode == 2:
            print('catch ind docs...')
            ind_docs = self.docs[self.text_col].map(lambda x: [int(elem) for elem in x.strip('[]').split(', ')])
            self.ind_docs = ind_docs.tolist()
            print('catch ind docs end.')


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
            labels = [str(row['location_traffic_convenience']),
                      str(row['location_distance_from_business_district']),
                      str(row['location_easy_to_find']),
                      str(row['service_wait_time']),
                      str(row['service_waiters_attitude']),
                      str(row['service_parking_convenience']),
                      str(row['service_serving_speed']),
                      str(row['price_level']),
                      str(row['price_cost_effective']),
                      str(row['price_discount']),
                      str(row['environment_decoration']),
                      str(row['environment_noise']),
                      str(row['environment_space']),
                      str(row['environment_cleaness']),
                      str(row['dish_portion']),
                      str(row['dish_taste']),
                      str(row['dish_look']),
                      str(row['dish_recommendation']),
                      str(row['others_overall_experience']),
                      str(row['others_willing_to_consume_again'])
                       ]
            label_list.append('||'.join(labels))
        self.docs.insert(0, 'label_list', label_list)



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
        self.label_assemble()


    def word_cut(self):
        """
        用jieba分词切语料
        :return:
        """
        cutted_docs = []
        for i, row in self.docs.iterrows():
            if i % 100 == 0:
                print('cut word: %d' % (i))
            # print(row[self.text_col])
            seg_list = jieba.cut(row[self.text_col])
            seg_list = [item for item in seg_list]
            cutted_docs.append(seg_list)
            self.docs.loc[[i], [self.text_col]] = '||'.join(seg_list)
        self.cut_docs = cutted_docs



    def init_vocab(self):
        self.vocab = set()
        vocab_list = ['__blank__']  # 第1位作为oov位
        vocab_words = []
        if self.cut_docs is None:
            print('no cut docs, start cutting...')
            self.word_cut()
        for i, seg_list in enumerate(self.cut_docs):
            if i % 100 == 0:
                print('init vocab: %d' % (i))
            vocab_words.extend(seg_list)
        vocab_list.extend(list(set(vocab_words)))
        self.vocab = vocab_list


    def save_vocab(self, vocab_path, encoding='utf-8'):
        writer = open(vocab_path, 'w', encoding=encoding)
        for item in self.vocab:
            writer.write(item + '\n')


    def index_corpus_and_save(self, out_path, proc_num=0):
        """
        文本索引化，多线程版本
        :return:
        """
        self.ind_docs = []
        if self.cut_docs is None:
            print('No cut docs.')
            return 0

        self.ind_ids = []
        self.ind_docs = []
        self.ind_labels = []

        if proc_num <= 0:
            proc_num = max(1, multP.cpu_count() - 1)  # 确定进程数量
        proc_pool = multP.Pool(processes=proc_num)
        print('core num: %d' % (proc_num))

        doc_num = len(self.cut_docs)
        seg_len = int(doc_num / proc_num)
        doc_parts = [self.docs[ind * seg_len:(ind+1)*seg_len] for ind in range(proc_num)]

        chunk_list = []
        for i, doc_part in enumerate(doc_parts):
            chunk_list.append([i, seg_len, doc_part, self.vocab])

        # 下面开始执行多进程方法
        ind_ids_docs_labels = []
        for out_docs in proc_pool.imap(ind_task, chunk_list):
            ind_ids_docs_labels.extend(out_docs)
        proc_pool.close()
        proc_pool.join()

        # print(ind_ids_docs_labels)

        ind_doc_df = pd.DataFrame(ind_ids_docs_labels, columns=[self.id_col, self.text_col, self.label_col])
        ind_doc_df.to_csv(out_path, index=False)


    def make_tfrecord(self, output_file_path, max_len=0):
        '''
        制作tfrecord
        :param output_file_path:
        :return:
        '''

        def func(x):
            return [int(item)+2 for item in x.strip().split('||')]

        doc_ids = self.docs[self.id_col].map(lambda x: str(x)).tolist()
        ind_docs = self.ind_docs
        try:
            doc_labels = self.docs[self.label_col].map(lambda x: func(x)).tolist()
        except:
            doc_labels = [-1 for _ in range(20)]
        if len(doc_ids) != len(ind_docs) or len(doc_labels) != len(ind_docs):
            print('length not match！')
            return 1
        tot_len = len(doc_ids)
        with tf.python_io.TFRecordWriter(output_file_path) as writer:
            for i, (id, comment, label) in enumerate(zip(doc_ids, ind_docs, doc_labels)):
                if i % 100 == 0:
                    print('%d, percentage: %f' % (i, i / tot_len))
                if max_len > 0:
                    if len(comment) > max_len:
                        comment = comment[0: max_len]
                    else:
                        comment.extend([0 for _ in range(max_len - len(comment))])
                effect_len = len(comment)
                id_feat = tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(id, 'utf-8')]))
                effect_len_feat = tf.train.Feature(int64_list=tf.train.Int64List(value=[effect_len]))
                label_feat = tf.train.Feature(int64_list=tf.train.Int64List(value=label))
                comment_feat = tf.train.Feature(int64_list=tf.train.Int64List(value=comment))
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'id_feat': id_feat,
                            'effect_len_feat': effect_len_feat,
                            'label_feat': label_feat,
                            'comment_feat': comment_feat
                        }
                    )
                )

                writer.write(example.SerializeToString())  # 序列化为字符串



class CorpusLoader(object):
    """
    提供从tfrecord到dataset类型的转换（而并不到iterator）
    """
    tfrecord_path_list = None
    batch_size = None
    repeat_eposh_num = None
    shffle_buffer_size = None

    def __init__(self, tfrecord_path_list, batch_size, repeat_epoch_num, shuffle_buffer_size):
        self.tfrecord_path_list = tfrecord_path_list
        self.batch_size = batch_size
        self.repeat_epoch_num = repeat_epoch_num
        self.shuffle_buffer_size = shuffle_buffer_size

    def launch_tfrecorddataset(self):
        def __parse(serialized_example):
            """
            定义一个内部用的解析函数
            :param serialized_example:
            :return:
            """
            features = {
                'effect_len_feat': tf.FixedLenFeature([1], dtype=tf.int64),
                'id_feat': tf.FixedLenFeature([1], dtype=tf.string),
                'label_feat': tf.VarLenFeature(dtype=tf.int64),
                'comment_feat': tf.VarLenFeature(dtype=tf.int64)
            }
            parsed_example = tf.parse_single_example(serialized_example, features)

            id = tf.cast(parsed_example['id_feat'], dtype=tf.string)
            effect_len = tf.cast(parsed_example['effect_len_feat'], dtype=tf.int32)
            label = tf.sparse_tensor_to_dense(parsed_example['label_feat'])
            label = tf.cast(label, dtype=tf.int32)
            comment = tf.sparse_tensor_to_dense(parsed_example['comment_feat'])  # 变长的数组用VarLen存的时候是稀疏存储的，需要先转成密集型
            comment = tf.cast(comment, dtype=tf.int32)
            return id, effect_len, label, comment

        data_set = cb_data.TFRecordDataset(self.tfrecord_path_list)
        # 这个比较奇怪不知道为什么必须要加个括号。。必须要是个tuple？
        parsed_dataset = (data_set.map(__parse))
        if self.repeat_epoch_num is not None:
            # 指定重复的次数，队列会以这个epoch次数为长度
            parsed_dataset = parsed_dataset.repeat(self.repeat_epoch_num)
        if self.shuffle_buffer_size is not None:
            # 指定shuffle的范围，一般要选比整个数据集的长度大，才能整体打乱
            parsed_dataset = parsed_dataset.shuffle(buffer_size=self.shuffle_buffer_size)

        # 由于有变长数组，所以必须要padd，否则用parsed_dataset.batch(batch_size)就可以了
        # 像这个例子里，如果parse函数的返回为多个变量，则padded_shapes需要是一个tuple，每个元素对应该变量在该批次pad到的上限，-1为pad到最长
        parsed_dataset = parsed_dataset.padded_batch(self.batch_size, padded_shapes=([-1], [-1], [-1], [-1]))
        return parsed_dataset   # 包含四个字段：id, effect_len, label, comment

        # # 为解析出来的数据集弄一个迭代器
        # iterator = parsed_dataset.make_one_shot_iterator()
        # id_batch, effect_len_batch, label_batch, comment_batch = iterator.get_next()
        #
        # id_batch = tf.reshape(id_batch, [batch_size])
        # effect_len_batch = tf.reshape(effect_len_batch, [batch_size])
        # label_batch = tf.reshape(label_batch, [batch_size])
        #
        # return id_batch, effect_len_batch, label_batch, comment_batch




if __name__ == '__main__':
    # ## 对语料进行预处理，未分词
    # cps = CorpusBleach(global_configs.raw_corpus_path)
    # # print(cps.docs.describe())
    # # print(cps.docs.head())
    # cps.flow_process('./data_preprocess/emotion_dict.txt')
    # print(cps.docs.iloc[982])
    # cps.docs.to_csv(global_configs.preprocessed_corpus_path, index=False, encoding='utf-8', sep=',')


    # # 对预处理后的语料进行分词
    # cps = CorpusBleach(global_configs.preprocessed_corpus_path,mode=0, vocab_path=global_configs.vocab_path)
    # cps.init_vocab()
    # cps.docs.to_csv(global_configs.cutted_corpus_path, index=False, encoding='utf-8', sep=',')
    # cps.save_vocab(global_configs.vocab_path)
    #
    # # 分词后的语料索引化
    # cps = CorpusBleach(global_configs.cutted_corpus_path, mode=1, vocab_path=global_configs.vocab_path)
    # cps.index_corpus_and_save(global_configs.indexed_corpus_path)


    # # 每个评论词量的统计
    # cps = CorpusBleach(global_configs.indexed_corpus_path, mode=2, vocab_path=global_configs.vocab_path)
    # bottles = {'10': 0.0, '20': 0.0, '50': 0.0, '100': 0.0, '200': 0.0, '300': 0.0, '400': 0.0, '500': 0.0, '500+': 0.0}
    # tot_num = len(cps.ind_docs)
    # tot_word_num = 0.0
    # for line in cps.ind_docs:
    #     doc_len = len(line)
    #     tot_word_num += doc_len
    #     if doc_len <= 10:
    #         bottles['10'] += 1
    #     elif doc_len > 10 and doc_len <= 20:
    #         bottles['20'] += 1
    #     elif doc_len > 20 and doc_len <= 50:
    #         bottles['50'] += 1
    #     elif doc_len > 50 and doc_len <= 100:
    #         bottles['100'] += 1
    #     elif doc_len > 100 and doc_len <= 200:
    #         bottles['200'] += 1
    #     elif doc_len > 200 and doc_len <= 300:
    #         bottles['300'] += 1
    #     elif doc_len > 300 and doc_len <= 400:
    #         bottles['400'] += 1
    #     elif doc_len > 400 and doc_len <= 500:
    #         bottles['500'] += 1
    #     elif doc_len > 500:
    #         bottles['500+'] += 1
    # print(tot_word_num / tot_num)
    # for k, v in bottles.items():
    #     print('k: %s, v: %f, per: %f' % (k, v, v / tot_num))




    # 索引语料生成tfrecord
    cps = CorpusBleach(global_configs.indexed_corpus_path, mode=2, vocab_path=global_configs.vocab_path)
    cps.make_tfrecord(output_file_path=global_configs.tfrecord_corpus_path, max_len=400)


    # # 检查生成的tfrecord。dataset 取数据
    # data_batch_size = 100
    # train_tfrecord_path_list = [global_configs.tfrecord_corpus_path]
    # dataset_loader = CorpusLoader(train_tfrecord_path_list, batch_size=data_batch_size, repeat_epoch_num=5, shuffle_buffer_size=None)
    # train_dataset = dataset_loader.launch_tfrecorddataset()
    # train_iterator = train_dataset.make_one_shot_iterator()
    #
    # id_batch, effect_len_batch, label_batch, comment_batch = train_iterator.get_next()
    # id_batch = tf.reshape(id_batch, [data_batch_size])
    # effect_len_batch = tf.reshape(effect_len_batch, [data_batch_size])
    # # label_batch = tf.reshape(label_batch, [20, data_batch_size])  # 出来反正是个2维的，不用reshape成1维
    #
    #
    # with tf.Session() as sess:
    #     id_batch_out, effect_len_batch_out, label_batch_out, comment_batch_out = sess.run([id_batch, effect_len_batch, label_batch, comment_batch])
    #     print(id_batch_out)
    #     print(effect_len_batch_out)
    #     print(label_batch_out)