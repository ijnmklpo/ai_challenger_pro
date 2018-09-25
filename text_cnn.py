# -*- coding:utf-8 -*-
# 
# @Author: ijnmklpo
# @Time: 2018/9/23 下午3:26
# @Desc: text cnn模型

import numpy as np
import tensorflow as tf
from tensorflow.contrib.data import Iterator
import tensorflow.contrib.slim as slim
import os
import shutil   # 跟os类似，管理文件系统的，用来删除目录

import global_configs
import Corpus
import utils as ryh_utils


# ===================== 全局配置 =====================
## 超参配置
data_batch_size = 50
node_num = 100
the_keep_prob = 0.5
l2_reg = 3.0
learning_rate = 0.00007
'''
配置日志
版本1：
data_batch_size = 50
node_num = 100
keep_prob = 0.5
l2_reg = 3.0
learning_rate = 0.01
结果：迭代20w次，似乎还能继续往下降；lr->0.002训练至30w看看，20.5w的时候valid loss降到0.36+了，30w的时候vl降到了0.36；从30w开始lr->0.001训练到40w看看,40w基本也还是维持在0.36;再把lr->0.00007，跑60w次迭代，最终执行了100w次迭代，vl收敛在0.36，似乎真的没法再下降了，这版数据就这样吧。

'''

model_root_dir = './models/text_cnn/version1'
train_dir = os.path.join(model_root_dir, 'model_files')
summary_dir = os.path.join(model_root_dir, 'summarys')
summary_train_dir = os.path.join(summary_dir, 'train')
summary_valid_dir = os.path.join(summary_dir, 'valid')

# ====================/ 全局配置 /====================


class CNN_Model(object):
    class_num = None  # model
    vocab_len = None  # model
    word_vector_len = None  # model
    node_num = None
    batch_size = None

    def __init__(self, class_num, wv_shape, node_num, batch_size):
        '''
        wv_shape为[vocab_len, wv_len]
        :param class_num:
        :param wv_shape:
        :param node_num:
        '''
        self.class_num = class_num
        self.vocab_len = wv_shape[0]
        self.word_vector_len = wv_shape[1]
        self.node_num = node_num
        self.batch_size = batch_size

    def word_embedding(self, comment_batch, batch_size):
        wv_mat = tf.Variable(tf.truncated_normal([self.vocab_len, self.word_vector_len]), dtype=tf.float32)
        with tf.name_scope('word_embedding1'):
            comment_batch = tf.nn.embedding_lookup(wv_mat, comment_batch)
            comment_batch = tf.reshape(comment_batch, [batch_size, -1, self.word_vector_len, 1])
        return comment_batch


    def label_onehot(self, label_batch, class_num=4):
        '''
        这次的label比较特殊，是20个维度，每个维度4个取值。因此在最后做loss的时候，需要做20个softmax，然后再全加起来。
        :param label_batch:
        :param class_num:
        :return:
        '''
        label_onehot_list = []
        for i in range(20):
            label_slise = tf.slice(label_batch, begin=[0, i], size=[self.batch_size, 1])
            label_onehot_list.append(tf.one_hot(label_slise, class_num))
        label_onehot_list = tf.concat(values=label_onehot_list, axis=2)
        label_onehot_list = tf.reshape(label_onehot_list, shape=[self.batch_size, 20 * self.class_num])
        return label_onehot_list


    def model_stream(self, comment_batch, effect_doc_len, keep_prob, l2_reg):
        comment1_batch = self.word_embedding(comment_batch, batch_size=self.batch_size)
        comment1_batch = tf.reshape(comment1_batch, [self.batch_size, -1, self.word_vector_len, 1])

        # return comment1_batch
        regularizer = slim.l2_regularizer(l2_reg)
        with tf.name_scope('model_flow'):
            with tf.name_scope('width3_cnn'):
                net1 = slim.conv2d(comment1_batch, self.node_num, [3, self.word_vector_len], weights_regularizer=regularizer, padding='VALID')
                net1 = slim.max_pool2d(net1, [effect_doc_len-2, 1])  # max pooling
                net1 = tf.reshape(net1, [self.batch_size, self.node_num])
            with tf.name_scope('width4_cnn'):
                net2 = slim.conv2d(comment1_batch, self.node_num, [4, self.word_vector_len], weights_regularizer=regularizer, padding='VALID')
                net2 = slim.max_pool2d(net2, [effect_doc_len-3, 1])  # max pooling
                net2 = tf.reshape(net2, [self.batch_size, self.node_num])
            with tf.name_scope('width5_cnn'):
                net3 = slim.conv2d(comment1_batch, self.node_num, [5, self.word_vector_len], weights_regularizer=regularizer, padding='VALID')
                net3 = slim.max_pool2d(net3, [effect_doc_len-4, 1])  # max pooling
                net3 = tf.reshape(net3, [self.batch_size, self.node_num])
            with tf.name_scope('fully_connect'):
                net_all = tf.concat([net1,net2, net3],axis=1)
                net_all = slim.dropout(net_all, keep_prob=keep_prob)
                logits = slim.fully_connected(net_all, self.class_num * 20, activation_fn=None)
        return logits


    def loss_define(self, logits, labels):
        with tf.name_scope('train_loss') as train_loss_scope:
            # tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels_onehot, label_smoothing=0.0)
            loss = tf.losses.sigmoid_cross_entropy(logits=logits, multi_class_labels=labels, label_smoothing=0.0)
            loss = tf.losses.get_total_loss()   # 这个不能用，因为要算两个loss，用这个会累加在一块
        return loss


    def get_optimizer(self, loss, learning_rate=0.001):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(loss)
        return optimizer


    def accuracy(self, logits, labels, begin, size):
        p = tf.slice(logits, begin, size)

        max_idx_p = tf.argmax(p, 1)
        max_idx_p = tf.cast(max_idx_p, dtype=tf.int32)
        correct_pred = tf.equal(max_idx_p, labels)
        acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return acc


    def prediction(self, logits):
        pred_all = []
        for i in range(20):
            p = tf.slice(logits, [0, i*self.class_num], [self.batch_size, self.class_num])
            max_idx_p = tf.argmax(p, axis=1)
            max_idx_p = tf.cast(max_idx_p, dtype=tf.int32)
            pred_all.append(max_idx_p)
        pred_out = tf.stack(pred_all, axis=1)
        return pred_out



if __name__ == '__main__':
    # 载入训练集，并构造一个迭代器
    train_tfrecord_path_list = [global_configs.train_tfrecord_corpus_path]
    train_dataloader = Corpus.CorpusLoader(train_tfrecord_path_list, batch_size=data_batch_size, repeat_epoch_num=300, shuffle_buffer_size=150000)
    train_dataset = train_dataloader.launch_tfrecorddataset()
    train_iterator = train_dataset.make_one_shot_iterator()

    # 载入验证集，并构造一个迭代器
    valid_tfrecord_path_list = [global_configs.valid_tfrecord_corpus_path]
    valid_dataloader = Corpus.CorpusLoader(valid_tfrecord_path_list, batch_size=data_batch_size, repeat_epoch_num=200, shuffle_buffer_size=20000)
    valid_dataset = valid_dataloader.launch_tfrecorddataset()
    valid_iterator = valid_dataset.make_one_shot_iterator()

    # =================== 用handle导入，feedble ===================
    # 构造一个可导入(feedble)的句柄占位符，可以通过这个将训练集的句柄或者验证集的句柄传入
    handle = tf.placeholder(tf.string, shape=[])
    iterator = Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
    id_batch, effect_len_batch, label_batch, comment_batch = iterator.get_next()
    # 从迭代器中出来的是一个二维数组，而用到的id、effect_len和label是要一个一维数组，需要reshape以下
    id_batch = tf.reshape(id_batch, [data_batch_size])
    effect_len_batch = tf.reshape(effect_len_batch, [data_batch_size])
    # ==================/ 用handle导入，feedble /==================


    # ======================模型数据流============================
    # 全量词典长度：217528
    keep_prob = tf.placeholder(dtype=tf.float32)
    model = CNN_Model(4, [217528, 100], node_num, batch_size=data_batch_size)
    label_onehot_batch = model.label_onehot(label_batch)
    logits = model.model_stream(comment_batch, effect_doc_len=400, keep_prob=keep_prob, l2_reg=l2_reg)
    # =====================/模型数据流/===========================

    # ====================== 训练分支 ============================
    loss = model.loss_define(logits, label_onehot_batch)
    tf.summary.scalar('loss', loss)
    model_preds = model.prediction(logits)
    real_labels = model.prediction(label_onehot_batch)
    optimizer = model.get_optimizer(loss, learning_rate=learning_rate)
    # =====================/ 训练分支 /===========================

    # ===================== 要监控的变量 =====================
    summary_loss_flag = tf.placeholder(dtype=tf.int32)
    # tensorboard summary将指定的监控的值收集起来
    merged = tf.summary.merge_all()
    # ===================== 要监控的变量 =====================

    # ================指定已有的训练文件================
    if tf.gfile.Exists(train_dir) == False:
        tf.gfile.MakeDirs(train_dir)
    saver = tf.train.Saver()
    # ===============/指定已有的训练文件/===============


    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        sess.run(tf.global_variables_initializer())  # 所有变量初始化

        # ================== 配置tensorboard文件 ==================
        # if os.path.exists(summary_dir) == True:
        #     shutil.rmtree(summary_dir)
        summary_train_writer = tf.summary.FileWriter(summary_train_dir, sess.graph)
        summary_valid_writer = tf.summary.FileWriter(summary_valid_dir, sess.graph)
        # =================/ 配置tensorboard文件 /=================

        # 获得训练集和验证集的引用句柄，后面导入数据到模型用
        [train_iterator_handle, valid_iterator_handle] = sess.run([train_iterator.string_handle(), valid_iterator.string_handle()])

        init_step = 0
        # ================确认已有模型，load====================
        ckpt = tf.train.get_checkpoint_state(train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            init_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            saver.restore(sess=sess, save_path=ckpt.model_checkpoint_path)
        # ===============/确认已有模型，load/===================


        for step in range(init_step, init_step + 10000001):
            [loss_out, _] = sess.run([loss, optimizer], feed_dict={handle: train_iterator_handle, keep_prob: the_keep_prob})
            # =============== 保存summary ===============
            if step % 99 == 0:
                [rs1, train_loss_out] = sess.run([merged, loss], feed_dict={handle: train_iterator_handle, keep_prob: the_keep_prob})
                print(step, train_loss_out)
                summary_train_writer.add_summary(rs1, step)
            elif step % 100 == 0:
                [rs2, valid_loss_out] = sess.run([merged, loss], feed_dict={handle: valid_iterator_handle, keep_prob: the_keep_prob})
                print(step, valid_loss_out)
                summary_valid_writer.add_summary(rs2, step)
            # ==============/ 保存summary /==============

            # ================保存模型====================
            if step % 100000 == 0:
                saver.save(sess, os.path.join(train_dir, 'model.ckpt'), global_step=step)
                if step == init_step + 10000 * 50:
                    break
            # ===============/保存模型/===================

        summary_train_writer.close()
        summary_valid_writer.close()
        coord.request_stop()
        coord.join(threads)





        # for step in range(init_step+1, init_step + 10000001):
        #     [logits_out, loss_out, model_preds_out, real_labels_out, _] = sess.run([logits, loss, model_preds, real_labels, optimizer], feed_dict={handle: train_iterator_handle, keep_prob: the_keep_prob})
        #     if step % 50 == 0:
        #         print('step: %d, train loss: %f' % (step, loss_out))
        #     if step % 300 == 0:
        #         print(model_preds_out[0])
        #         print(real_labels_out[0])
        #     if step % 500 == 0:
        #         [logits_out, loss_out, model_preds_out, real_labels_out] = sess.run([logits, loss, model_preds, real_labels],feed_dict={handle: valid_iterator_handle, keep_prob: 1.0})
        #         print('===============================')
        #         print('step: %d, valid loss: %f' % (step, loss_out))
        #         print(model_preds_out[0])
        #         print(real_labels_out[0])
        #         print('===============================')
        #
        #     # =============== 保存summary ===============
        #     if step % 100 == 0:
        #         rs1 = sess.run(train_merged, feed_dict={handle: train_iterator_handle, keep_prob: the_keep_prob})
        #         rs2 = sess.run(valid_merged, feed_dict={handle: valid_iterator_handle, keep_prob: 1.0})
        #         summary_writer.add_summary(rs1, step)
        #         summary_writer.add_summary(rs2, step)
        #     # ==============/ 保存summary /==============
        #
        #     # ================保存模型====================
        #     if step % 20000 == 0:
        #         saver.save(sess, os.path.join(train_dir, 'model.ckpt'), global_step=step)
        #         if step == init_step + 20000 * 5:
        #             break
        #     # ===============/保存模型/===================
        #
        #
        # summary_writer.close()
        # coord.request_stop()
        # coord.join(threads)