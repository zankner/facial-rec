import tensorflow as tf
import time
import os
import numpy as np
from tqdm import trange
from ops import *

import sys
sys.path.insert(0, '../process')
from pipeline import * 


class Model(object):
    def __init__(self,sess,args):
        self.model_name = 'model_name'
        self.sess = sess

        self.checkpoint_dir = args.checkpoint_dir
        self.log_dir = args.log_dir

        self.restore = args.restore

        self.img_size = args.img_size
        self.z_dim = args.z_dim
        self.label_dim = args.label_dim

        self.channels = args.channels
        self.epochs = args.epochs
        self.batch_size = args.batch_size

        self.train_data, self.test_data = final_split(self.img_dim, self.batch_size)

        self.iterations = self.train_data['num_samples'] // self.batch_size
        self.use_bias = args.use_bias 

        self.learning_rate = args.lr


##############################################################################
#Generator: Generate model in this section
##############################################################################
    def network(self,x,reuse=False, is_training=True):
        with tf.variable_scope('network', reuse=reuse):
            x = conv(x, channels=self.channels, kernel=3,
                    stride=2, padding='SAME', use_bias=self.use_bias, scope='conv_0')
            x = max_pool(x)
            x = batch_norm(x,is_training,'batch_norm_0')
            x = relu(x)

            x = conv(x, channels=self.channels*2, kernel=3,
                    stride=1, padding='SAME', use_bias=self.use_bias, scope='conv_1')
            x = max_pool(x)
            x = batch_norm(x,is_training,'batch_norm_1')
            x = relu(x)

            x = dense(x,units=self.embedding_dim, use_bias=self.use_bias, scope='logit')

            return x

##############################################################################
#Model: Assemble model
##############################################################################
    def build_model(self):
        #Graph Input
        self.train_inputs = tf.placeholder(tf.float32,
                [self.batch_size, self.img_size,self.img_size,self.z_dim], name = 'train_inputs')
        self.train_labels = tf.placeholder(tf.float32,
                [self.batch_size, self.label_dim], name='train_labels')

        self.test_inputs = tf.placeholder(tf.float32, 
                [self.test_x.shape[0], self.img_size, self.img_size, self.z_dim], name = 'test_inputs')
        self.test_labels = tf.placeholder(tf.float32,
                [self.test_y.shape[0], self.label_dim], name='test_labels')

        self.lr = tf.placeholder(tf.float32, name='learning_rate')


        #Model
        self.train_logits = self.network(self.train_data['inputs'])
        self.test_logits = self.network(self.test_data['inputs'],reuse=True,is_training=False)

        self.train_loss = class_loss(logits=self.train_logits, labels=self.train_data['labels'])
        self.test_loss = class_loss(logits=self.test_logits, labels=self.test_data['labels'])


        #Training
        self.optim = optimizer(self.lr).minimize(self.train_loss)


        #Summary
        self.summary_train_loss = tf.summary.scalar('train_loss', self.train_loss)

        self.summary_test_loss = tf.summary.scalar('test_loss', self.test_loss)

        self.train_summary = tf.summary.merge([self.summary_train_loss])
        self.test_summary = tf.summary.merge([self.summary_test_loss])



##############################################################################
#Train: Train the model
##############################################################################
    def train(self):
        #initialize variables
        tf.global_variables_initializer().run()

        #saver for model
        self.saver = tf.train.Saver()

        #writer to write summary
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)

        start_time = time.time()

        for epoch in range(self.epochs):

            self.sess.run(init_op)

            t = trange(self.iterations)
            
            for batch in range(t):
                
                _, summary_str, train_loss = self.run([self.optim,self.train_summary,self.train_loss])

                test_summary_str, test_loss = self.run([self.test_summary, self.test_loss])

                t.set_postfix('Epoch: [%2d] [%5d/%5d] time: %4.4f, train_loss: %.2f, test_test %.2f'%(epoch, batch, self.iterations, time.time() - start_time, train_loss, test_loss))

    @property
    def model_dir(self):
        return '{}_{}_{}'.format(self.model_name, self.batch_size, self.lr)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self,checkpoint_dir):
        print('[*] Reading checkpoints...')
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            checkpoint_name = os.path.basename(checkpoint.model_checkpoint_path)
            self.saver.restore(self.sess,os.path.join(checkpoint_dir, checkpoint_name))
            counter = int(checkpoint_name.split('-')[-1])
            print('[*] Success to read {}'.format(checkpoint_name))
            return True, counter

        else:
            print('[*] Failed to find a checkpoint')
            return False, 0

    def test(self):
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()
        can_load, checkpoint_counter = self.load(self.checkpoint_dir)

        if can_load:
            print('[*} Load success ')
        else:
            print('[!] Load failed')

        test_feed_dict = {
            self.test_inputs : self.test_x,
            self.test_labels : self.test_y
        }

        test_accuracy = self.sess.run(self.test_accuracy, feed_dict = test_feed_dict)
        print('test_acc: {}'.format(test_accuracy))
train()
