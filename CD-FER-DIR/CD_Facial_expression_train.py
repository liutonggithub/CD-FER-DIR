# This code is implemented by tensorflow r0.12
# Date: March. 3th, 2023

from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from scipy.io import savemat
from ops import *
import pdb
import pandas as pd
from nonlocal_resnet import *
from get_fourier_image_myphaseconstant import *


class PFER_expression(object):
    def __init__(self,
                 session,  # TensorFlow session
                 size_image=128,  # size the input images 最好为2的倍数
                 size_kernel=5,  # size of the kernels in convolution and deconvolution
                 size_batch=36,  # mini-batch size for training and testing, must be square of an integer
                 num_input_channels=3,  # number of channels of input images
                 num_encoder_channels=64,  # number of channels of the first conv layer of encoder
                 num_fx=50,  # number of channels of the layer f(x)
                 num_categories=6,  # number of expressions in the training dataset
                 num_domains=4,  # number of domains in the training dataset
                 num_gen_channels=1024,  # number of channels of the first deconv layer of generator
                 enable_tile_label=True,  # enable to tile the label
                 tile_ratio=1.0,  # ratio of the length between tiled label and fx
                 is_training=True,  # flag for training or testing mode
                 save_dir='./PFER',  # path to save checkpoints, samples, and summary
                 ):

        self.session = session
        self.image_value_range = (-1, 1)
        self.size_image = size_image
        self.size_kernel = size_kernel
        self.size_batch = size_batch
        self.num_input_channels = num_input_channels
        self.num_encoder_channels = num_encoder_channels
        self.num_fx = num_fx
        self.num_categories = num_categories
        self.num_gen_channels = num_gen_channels
        self.enable_tile_label = enable_tile_label
        self.tile_ratio = tile_ratio
        self.is_training = is_training
        self.save_dir = save_dir
        self.num_domains = num_domains

        self.layer_num = int(np.log2(self.size_image)) - 3
        self.up_sample = True
        self.sn = True
        self.c_dim = 3  # 图片通道数
        self.gan_type = 'hinge'
        self.ld = 1.0

        # path of the file of trainset. the content style of trainJAFFE.txt: name expression-label domain-label
        self.pathtrain = './expression_data/'
        self.file_names = np.loadtxt(self.pathtrain + 'trainJAFFE.txt', dtype=bytes, delimiter=' ').astype(str)
        np.random.shuffle(self.file_names)
        self.len_trainset = len(self.file_names)
        self.num_batches = self.len_trainset // self.size_batch  # (3980//49=80)

        self.gen_names = np.loadtxt(self.pathtrain + 'genJAFFE.txt', dtype=bytes, delimiter=' ').astype(str)
        np.random.shuffle(self.gen_names)
        self.gen_trainset = len(self.gen_names)
        self.num_batches1 = self.gen_trainset // self.size_batch

        self.test_names = np.loadtxt(self.pathtrain + 'testJAFFE.txt', dtype=bytes, delimiter=' ').astype(str)
        np.random.shuffle(self.test_names)
        gen = open('testname.txt', 'w')  # name of the testset
        self.len_testset = len(self.test_names)
        self.num_batches2 = self.len_testset // self.size_batch
        for ii in range(self.test_names.shape[0]):
            gen.write(self.test_names[ii, 0] + ' ' + self.test_names[ii, 1] + '\n')
        gen.close()
        self.len_testset = len(self.test_names)
        self.num_batches2 = self.len_testset // self.size_batch

        gen = open('testlabel_name_ex_domain.txt', 'w')  # name of the testset
        self.len_testset = len(self.test_names)
        self.num_batches2 = self.len_testset // self.size_batch
        for ii in range(self.test_names.shape[0]):
            gen.write(self.test_names[ii, 0] + ' ' + self.test_names[ii, 1] + ' ' + self.test_names[ii, 2] + '\n')
        gen.close()
        self.len_testset = len(self.test_names)
        self.num_batches2 = self.len_testset // self.size_batch

        # ************************************* input to graph ********************************************************
        # 输入：
        # 图片self.input_image[49, 128, 128, 3]；
        # 表情标签onehot self.expression[49,7];域标签Onehot self.domain[49,4];身份均匀分布self.f_prior[49,50]
        self.input_image = tf.placeholder(  # input_image size [49, 128, 128, 3]
            tf.float32,
            shape=(self.size_batch, self.size_image, self.size_image, self.num_input_channels),
            name='input_images'
        )

        self.phase = spectrum_constant_phase(self.input_image)

        self.expression = tf.placeholder(  # expression label for G, D_att, and C_exp. onehot
            tf.float32,
            shape=(self.size_batch, self.num_categories),  #
            name='expression_labels'
        )

        self.domain = tf.placeholder(  # domain label for G and D_att, and C_exp. onehot
            tf.float32,
            shape=(self.size_batch, self.num_domains),  # [36,4]
            name='domain_labels'
        )

        self.f_prior = tf.placeholder(  # prior distribution of D_i
            tf.float32,
            shape=(self.size_batch, self.num_fx),  # [36,50]
            name='f_prior'
        )

        # ************************************* build the graph(建立抽象模型) *******************************************************
        print('\n\tBuilding graph ...')

        # 模型有5部分组成：
        # self.Gencoder、self.discriminator_i、self.Gdecoder、self.discriminator_att、self.discriminator_acc
        # G_encoder: input image --> f(x)
        self.f = self.Gencoder(
            image=self.input_image
        )

        self.e = self.Gencoder(
            image=self.input_image
        )

        self.d=self.Gencoder(
            image=self.input_image
        )

        # G_decoder: f(x) + expression + domain --> generated image
        self.G = self.Gdecoder(
            f=self.f,
            e=self.e,
            enable_tile_label=self.enable_tile_label,  # true
            tile_ratio=self.tile_ratio  # 1
        )

        # discriminator on identity
        self.D_f, self.D_f_logits = self.discriminator_i(
            f=self.f,
            is_training=self.is_training
        )

        # discriminator on G
        self.D_G, self.D_G_logits = self.discriminator_att(
            image=self.G,
            is_training=self.is_training
        )

        # discriminator on f_prior(更改了reuse_variables=True)
        self.D_f_prior, self.D_f_prior_logits = self.discriminator_i(
            f=self.f_prior,
            is_training=self.is_training,
            reuse_variables=True

        )

        # discriminator on input image
        self.D_input, self.D_input_logits = self.discriminator_att(
            image=self.phase,
            is_training=self.is_training,
            reuse_variables=True
        )

        # classifier on original facial images and generated facial images
        self.D_input_ex_logits = self.ex_classifier(
            image=self.e,
            resnet_size=101,
            is_training=self.is_training
        )

        self.D_input_domain_logits = self.domain_classifier(
            image=self.d,
            resnet_size=101,
            is_training=self.is_training
        )

        # ************************************* loss functions （定义损失函数）*******************************************************


        # loss function of generator G
        self.EG_loss = tf.reduce_mean(tf.abs(self.phase - self.G))

        """
        判别器的目的是：
        1.对于真实图片，D要为其打上标签1
        2.对于生成图片，D要为其打上标签0
        生成器的目的是：
        对于生成的图片，G希望D打上标签1
        """
        # loss function of discriminator on identity
        # 判别器判别真实身份信息，希望识别出来的标签为1
        self.D_f_loss_prior = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.D_f_prior_logits),
                                                    logits=self.D_f_prior_logits)
        )
        # 判别器识别生成器产出的身份信息，希望识别出来的标签为0
        # (Dz-z（编码器Gz生成的）) 使用tf.nn.sigmoid_cross_entropy_with_logits，self.D_f_logits不需要经过sigmoid
        self.D_f_loss_f = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.D_f_logits), logits=self.D_f_logits)
        )

        # 生成器希望判别器判别生成器产出的身份信息的标签为1；tf.ones_like()创建一个将所有元素都设置为1的张量
        # loss function of G on identity
        self.E_f_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.D_f_logits), logits=self.D_f_logits)
        )

        # 判别器判别真实图片，希望识别出来的标签为1
        # loss function of discriminator on image（Dimg-输入真实图片）
        self.D_att_loss_input, _ =discriminator_loss(self.gan_type, real=self.D_input_logits, fake=self.D_G_logits)
        # self.D_att_loss_input = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.D_input_logits),
        #                                             logits=self.D_input_logits)
        # )
        # 判别器识别生成器产出的图片，希望识别出来的标签为0
        # （Dimg-生成的图片）
        _, self.D_att_loss_G = discriminator_loss(self.gan_type, real=self.D_input_logits, fake=self.D_G_logits)
        # self.D_att_loss_G = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.D_G_logits), logits=self.D_G_logits)
        # )

        # 生成器希望判别器判别生成器产出的图片的标签为1；tf.ones_like()创建一个将所有元素都设置为1的张量
        # loss function of G on image
        self.G_att_loss = generator_loss(self.gan_type, fake=self.D_G_logits)
        # self.G_att_loss = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.D_G_logits), logits=self.D_G_logits)
        # )


        #  loss function of classifier on image（表情损失，域损失）
        # 1.两个函数作用是一样的，sparse_softmax_cross_entropy_with_logits处理的是真是标签，softmax_cross_entropy_with_logits_v2处理的是one-hot的标签；
        self.D_ex_loss_input = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.expression, logits=self.D_input_ex_logits))
        self.D_domain_loss_input = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.domain, logits=self.D_input_domain_logits))


        # pdb.set_trace()表示有效去除重影伪影的总变化量tv()
        tv_y_size = self.size_image  # 224
        tv_x_size = self.size_image
        self.tv_loss = (
                               (tf.nn.l2_loss(
                                   self.G[:, 1:, :, :] - self.G[:, :self.size_image - 1, :, :]) / tv_y_size) +
                               (tf.nn.l2_loss(self.G[:, :, 1:, :] - self.G[:, :, :self.size_image - 1,
                                                                    :]) / tv_x_size)) / self.size_batch

        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ACCURACY OPS$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # Accuracy of expression
        # tf.argmax表示找到最大值的位置(也就是预测的分类和实际的分类)，然后看看他们是否一致，是就返回true,不是就返回false,这样得到一个boolean数组。
        # tf.cast将boolean数组转成int数组，求和。或者最后求平均值，得到分类的准确率(怎么样，是不是很巧妙).数据最大值所在的索引值
        self.d_ex_count = tf.reduce_sum(
            tf.cast(tf.equal(tf.argmax(self.D_input_ex_logits, 1), tf.argmax(self.expression, 1)), 'int32'))
        # Accuracy of domain
        self.d_domain_count = tf.reduce_sum(
            tf.cast(tf.equal(tf.argmax(self.D_input_domain_logits, 1), tf.argmax(self.domain, 1)), 'int32'))


        # *********************************** trainable variables ****************************************************
        trainable_variables = tf.trainable_variables()

        # print (trainable_variables)
        # variables of G_encoder
        self.E_variables = [var for var in trainable_variables if 'E_' in var.name]
        # variables of G_decoder
        self.G_variables = [var for var in trainable_variables if 'G_' in var.name]
        # variables of discriminator on identity
        self.D_f_variables = [var for var in trainable_variables if 'D_f_' in var.name]
        # variables of discriminator on attributes
        self.D_att_variables = [var for var in trainable_variables if 'D_att_' in var.name]
        # variables of discriminator on expression
        self.D_acc_variables = [var for var in trainable_variables if 'D_acc_' in var.name]

        # ************************************* collect the summary ***************************************
        self.f_summary = tf.summary.histogram('f', self.f)
        self.f_prior_summary = tf.summary.histogram('f_prior', self.f_prior)
        self.EG_loss_summary = tf.summary.scalar('EG_loss', self.EG_loss)
        self.D_f_loss_f_summary = tf.summary.scalar('D_f_loss_f', self.D_f_loss_f)
        self.D_f_loss_prior_summary = tf.summary.scalar('D_f_loss_prior', self.D_f_loss_prior)
        self.E_f_loss_summary = tf.summary.scalar('E_f_loss', self.E_f_loss)
        self.D_f_logits_summary = tf.summary.histogram('D_f_logits', self.D_f_logits)
        self.D_f_prior_logits_summary = tf.summary.histogram('D_f_prior_logits', self.D_f_prior_logits)
        self.D_att_loss_input_summary = tf.summary.scalar('D_att_loss_input', self.D_att_loss_input)
        self.D_att_loss_G_summary = tf.summary.scalar('D_att_loss_G', self.D_att_loss_G)
        self.G_att_loss_summary = tf.summary.scalar('G_att_loss', self.G_att_loss)
        self.D_G_logits_summary = tf.summary.histogram('D_G_logits', self.D_G_logits)
        self.D_input_logits_summary = tf.summary.histogram('D_input_logits', self.D_input_logits)
        self.D_input_ex_logits_summary = tf.summary.histogram('D_input_ex_logits', self.D_input_ex_logits)
        self.D_ex_loss_input_summary = tf.summary.scalar('D_ex_loss_input_summary', self.D_ex_loss_input)
        self.d_ex_count_summary = tf.summary.scalar('d_ex_count', self.d_ex_count)
        self.d_domain_count_summary = tf.summary.scalar('d_domain_count', self.d_domain_count)

        # for saving the graph and variables max_to_keep用来设置保存模型的个数,保存最近的10个模型。保存模型
        self.saver = tf.train.Saver(max_to_keep=10)

    # get the train data and test data
    def get_batch_train_test(self, enable_shuffle=True, idx=0):
        # # *************************** load file names of images ******************************************************
        if self.is_training:
            if enable_shuffle:
                np.random.shuffle(self.file_names)  # 'trainJAFFE.txt'
            tt_files = self.file_names[idx * self.size_batch: idx * self.size_batch + self.size_batch]
            # path of the traindata
            # self.path = self.pathtrain + 'data/JAFFE_train/'
            self.path = self.pathtrain + 'JAFFE_train/'
        else:
            tt_files = self.test_names[idx * self.size_batch: idx * self.size_batch + self.size_batch]
            # path of the testdata
            # self.path = self.pathtrain + 'data/JAFFE_test/'
            self.path = self.pathtrain + 'JAFFE_test/'

        batch_images = np.zeros((self.size_batch, self.size_image, self.size_image, 3))
        for i in range(tt_files.shape[0]):
            sample = [load_image(
                image_path=self.path + tt_files[i, 0],  # i行1列（图片名称如.jpg）
                image_size=self.size_image,
                image_value_range=self.image_value_range,
                is_gray=(self.num_input_channels == 1),  # 图片通道数
            )]
            if self.num_input_channels == 1:
                batch_images[i, :, :, :] = np.array(sample).astype(np.float32)[:, :, :, None]
            else:
                batch_images[i, :, :, :] = np.array(sample).astype(np.float32)

        tt_label_expression = np.ones(
            shape=(len(tt_files), self.num_categories),  # 36行6列，数字都为1（假如是一个批次的图片）
            dtype=np.float32
        ) * self.image_value_range[0]  # self.image_value_range[0]为-1，相乘后为-1

        tt_label_domain = np.ones(
            shape=(len(tt_files), self.num_domains),  # 36行5列（假如是一个批次的图片）
            dtype=np.float32
        ) * self.image_value_range[0]


        for i, label in enumerate(tt_files[:, 1]):
            label = tt_files[i, 1].astype('int')
            # changing the expression label as onehot with the target as 1, others as -1;
            tt_label_expression[i, label] = self.image_value_range[-1]  # self.image_value_range[-1]=1

            # if sparse_softmax_cross_entropy_with_logits is used
            # expression label
            # tt_label_expression1[i]=label

            domain = tt_files[i, 2].astype('int')

            tt_label_domain[i, domain] = self.image_value_range[-1]

            # if sparse_softmax_cross_entropy_with_logits is used
            # domain label
            # tt_label_domain1[i]=domain
        # return batch_images, tt_label_expression, tt_label_domain, tt_label_expression1, tt_label_domain1, tt_files
        return batch_images, tt_label_expression, tt_label_domain, tt_files

    # get the gen data
    def get_batch_gen(self, DIS=True, idx=0):

        if DIS:
            print('dis')
            np.random.shuffle(self.gen_names)
        tt_files = self.gen_names[idx * self.size_batch: idx * self.size_batch + self.size_batch]
        # path of the traindata
        self.path = self.pathtrain + 'JAFFE_train/'
        batch_images = np.zeros((self.size_batch, self.size_image, self.size_image, 3))
        for i in range(tt_files.shape[0]):
            sample = [load_image(
                image_path=self.path + tt_files[i, 0],
                image_size=self.size_image,
                image_value_range=self.image_value_range,
                is_gray=(self.num_input_channels == 1),
            )]

            if self.num_input_channels == 1:
                batch_images[i, :, :, :] = np.array(sample).astype(np.float32)[:, :, :, None]
            else:
                batch_images[i, :, :, :] = np.array(sample).astype(np.float32)

        tt_label_expression = np.ones(
            shape=(len(tt_files), self.num_categories),
            dtype=np.float32
        ) * self.image_value_range[0]

        tt_label_domain = np.ones(
            shape=(len(tt_files), self.num_domains),
            dtype=np.float32
        ) * self.image_value_range[0]

        # tt_label_expression1 = np.ones(
        #     shape=(len(tt_files)),
        #     dtype=np.float32
        # ) * self.image_value_range[0]
        #
        # tt_label_domain1 = np.ones(
        #     shape=(len(tt_files)),
        #     dtype=np.float32
        # ) * self.image_value_range[0]

        for i, label in enumerate(tt_files[:, 1]):
            label = tt_files[i, 1].astype('int')
            tt_label_expression[i, label] = self.image_value_range[-1]
            # tt_label_expression1[i]=label

            domain = tt_files[i, 2].astype('int')
            # if domain == 41:
            #     domain = 0
            # elif domain == 130:
            #     domain = 1
            # elif domain == 50:
            #     domain = 2
            # elif domain == 51:
            #     domain = 3
            # elif domain == 140:
            #     domain = 4
            tt_label_domain[i, domain] = self.image_value_range[-1]
            # tt_label_domain1[i]=domain
        # return batch_images, tt_label_expression, tt_label_domain, tt_label_expression1, tt_label_domain1, tt_files
        return batch_images, tt_label_expression, tt_label_domain, tt_files

    # get the validation data to validate the generated images
    def get_batch_sample(self, idx=0):

        tt_files = self.test_names[idx * self.size_batch: idx * self.size_batch + self.size_batch]
        batch_images = np.zeros((self.size_batch, self.size_image, self.size_image, 3))
        for i in range(tt_files.shape[0]):
            sample = [load_image(
                # path of the testdata
                image_path=self.pathtrain + 'JAFFE_test/' + tt_files[i, 0],
                image_size=self.size_image,
                image_value_range=self.image_value_range,
                is_gray=(self.num_input_channels == 1),
            )]
            if self.num_input_channels == 1:
                batch_images[i, :, :, :] = np.array(sample).astype(np.float32)[:, :, :, None]
            else:
                batch_images[i, :, :, :] = np.array(sample).astype(np.float32)

        tt_label_expression = np.ones(
            shape=(len(tt_files), self.num_categories),
            dtype=np.float32
        ) * self.image_value_range[0]

        tt_label_domain = np.ones(
            shape=(len(tt_files), self.num_domains),
            dtype=np.float32
        ) * self.image_value_range[0]

        # tt_label_expression1 = np.ones(
        #     shape=(len(tt_files)),
        #     dtype=np.float32
        # ) * self.image_value_range[0]
        #
        # tt_label_domain1 = np.ones(
        #     shape=(len(tt_files)),
        #     dtype=np.float32
        # ) * self.image_value_range[0]

        for i, label in enumerate(tt_files[:, 1]):
            label = tt_files[i, 1].astype('int')
            tt_label_expression[i, label] = self.image_value_range[-1]

            # tt_label_expression1[i]=label

            domain = tt_files[i, 2].astype('int')
            # if domain == 41:
            #     domain = 0
            # elif domain == 130:
            #     domain = 1
            # elif domain == 50:
            #     domain = 2
            # elif domain == 51:
            #     domain = 3
            # elif domain == 140:
            #     domain = 4
            tt_label_domain[i, domain] = self.image_value_range[-1]

            # tt_label_domain1[i]=domain

        # return batch_images, tt_label_expression, tt_label_domain, tt_label_expression1, tt_label_domain1, tt_files
        return batch_images, tt_label_expression, tt_label_domain, tt_files

    # 梯度惩罚损失
    def gradient_penalty(self, real, fake):
        if self.gan_type == 'dragan' :
            shape = tf.shape(real)
            eps = tf.random_uniform(shape=shape, minval=0., maxval=1.)
            x_mean, x_var = tf.nn.moments(real, axes=[0, 1, 2, 3])
            x_std = tf.sqrt(x_var)  # magnitude of noise decides the size of local region
            noise = 0.5 * x_std * eps  # delta in paper

            # Author suggested U[0,1] in original paper, but he admitted it is bug in github
            # (https://github.com/kodalinaveen3/DRAGAN). It should be two-sided.

            alpha = tf.random_uniform(shape=[shape[0], 1, 1, 1], minval=-1., maxval=1.)
            interpolated = tf.clip_by_value(real + alpha * noise, -1., 1.)  # x_hat should be in the space of X

        else :
            alpha = tf.random_uniform(shape=[self.size_batch, 1, 1, 1], minval=0., maxval=1.)
            interpolated = alpha*real + (1. - alpha)*fake

        _, logit = self.discriminator_att(image=interpolated, y=self.expression, domain=self.domain,
                                          is_training=self.is_training, reuse_variables=True)

        grad = tf.gradients(logit, interpolated)[0]  # gradient of D(interpolated)
        grad_norm = tf.norm(flatten(grad), axis=1)  # l2 norm

        GP = 0
        if self.gan_type == 'wgan-gp' or self.gan_type == 'dragan':
            GP = self.ld * tf.reduce_mean(tf.square(grad_norm - 1.))
        # WGAN - LP
        elif self.gan_type == 'wgan-lp':
            GP = self.ld * tf.reduce_mean(tf.square(tf.maximum(0.0, grad_norm - 1.)))
        return GP



    # 定义训练方法
    def train(self,
              num_epochs=5100,  # number of epochs
              learning_rate=0.0002,  # learning rate of optimizer
              # learning_rate=0.00008,  # learning rate of optimizer
              beta1=0.5,  # parameter for Adam optimizer
              beta2=0.9,  # parameter for Adam optimizer
              decay_rate=0.99,  # learning rate decay (0, 1], 1 means no decay
              # decay_rate=1,  # learning rate decay (0, 1], 1 means no decay
              enable_shuffle=True,  # enable shuffle of the dataset
              use_trained_model=True,  # used the saved checkpoint to initialize the model
              ):

        # *********************************** optimizer （定义优化器，下降方法）**************************************************************
        # over all, there are four loss functions, weights may differ from the paper because of different datasets
        # self.loss_EG = self.EG_loss + 0.0005 * self.G_att_loss + 0.0005 * self.E_f_loss + 0.0001 * self.tv_loss+ 0.0001*self.loss_Ex  # slightly increase the params
        # 身份信息的判别器总loss=self.loss_Df
        self.loss_Df = self.D_f_loss_prior + self.D_f_loss_f
        # 表情分类损失+域分类损失
        self.loss_Ex = self.D_ex_loss_input + self.D_domain_loss_input

        if self.gan_type.__contains__('wgan') or self.gan_type == 'dragan':
            GP = self.gradient_penalty(real=self.input_image, fake=self.G)
        else:
            GP = 0

        # 图片的判别器总loss=self.loss_Datt
        self.loss_Datt = self.D_att_loss_input + self.D_att_loss_G + GP
        # 两个生成器的总loss
        self.loss_EG = self.EG_loss + 0.0001 * self.G_att_loss + 0.0001 * self.E_f_loss + 0.0001 * self.tv_loss

        # set learning rate decay学习率相关的参数的设定
        self.EG_global_step = tf.Variable(0, trainable=False, name='global_step')
        EG_learning_rate = tf.train.exponential_decay(
            learning_rate=learning_rate,
            global_step=self.EG_global_step,
            decay_steps=self.len_trainset / self.size_batch * 2,
            decay_rate=decay_rate,
            staircase=True
        )

        # optimizer for G_encoder + G_decoder（训练目标：最小化损失函数）
        # 优化器（训练目标：最小化损失函数）
        # 因为GAN中一共训练了两个网络，所以分别对G和D进行优化 两个生成器的损失self.loss_EG
        self.EG_optimizer = tf.train.AdamOptimizer(
            learning_rate=EG_learning_rate,
            beta1=beta1,
            beta2=beta2
        ).minimize(
            loss=self.loss_EG,
            global_step=self.EG_global_step,
            var_list=self.E_variables + self.G_variables
        )

        # optimizer for discriminator on f(x)
        self.D_f_optimizer = tf.train.AdamOptimizer(
            learning_rate=EG_learning_rate,
            beta1=beta1
        ).minimize(
            loss=self.loss_Df,
            var_list=self.D_f_variables
        )

        # optimizer for discriminator on attributes
        self.D_att_optimizer = tf.train.AdamOptimizer(
            learning_rate=EG_learning_rate,
            beta1=0,
            beta2=beta2
        ).minimize(
            loss=self.loss_Datt,
            var_list=self.D_att_variables
        )

        # optimizer for discriminator on expression
        self.D_ex_optimizer = tf.train.AdamOptimizer(
            learning_rate=EG_learning_rate,
            beta1=0.5
        ).minimize(
            loss=self.loss_Ex,
            var_list=self.D_acc_variables
        )

        # *********************************** tensorboard tf.summary记录想要可视化的值*************************************************************
        # for visualization (TensorBoard): $ tensorboard --logdir path/to/log-directory
        self.EG_learning_rate_summary = tf.summary.scalar('EG_learning_rate', EG_learning_rate)
        self.summary = tf.summary.merge([
            self.f_summary, self.f_prior_summary,
            self.D_f_loss_f_summary, self.D_f_loss_prior_summary,
            self.D_f_logits_summary, self.D_f_prior_logits_summary,
            self.EG_loss_summary, self.E_f_loss_summary,
            self.D_att_loss_input_summary, self.D_att_loss_G_summary,
            self.G_att_loss_summary, self.EG_learning_rate_summary,
            self.D_G_logits_summary, self.D_input_logits_summary,
            self.D_input_ex_logits_summary,
            self.D_ex_loss_input_summary,
            self.d_ex_count_summary, self.d_domain_count_summary
        ])
        self.writer = tf.summary.FileWriter(os.path.join(self.save_dir, '/summary/'),
                                            self.session.graph)  # 指定一个文件用来保存图。

        # ******************************************* training *******************************************************
        print('\n\tPreparing for training ...')

        # initialize the graph
        tf.global_variables_initializer().run()

        # load check point
        if use_trained_model:
            if self.load_checkpoint():
                print("\tSUCCESS ^_^")
            else:
                print("\tFAILED >_<!")

        # if sparse_softmax_cross_entropy_with_logits is used
        # sample_images, sample_label_expression, sample_label_domain, sample_label_expression1, sample_label_domain1, batch_files_name = self.get_batch_sample(0)

        sample_images, sample_label_expression, sample_label_domain, batch_files_name = self.get_batch_sample(0)
        sample_expression_label = list(map(lambda x: [[i, 0][i < 0] for i in x], sample_label_expression))
        sample_domain_label = list(map(lambda x: [[i, 0][i < 0] for i in x], sample_label_domain))

        for epoch in range(num_epochs):

            # to save the test results
            trainresult = open('./result/' + str(epoch) + 'a.txt', 'w')
            f1 = open('./result/' + str(epoch) + 'test.txt', 'w')
            f2 = open('./result/' + str(epoch) + 'index.txt', 'w')

            self.is_training = True
            DIS = True
            enable_shuffle = True

            for ind_batch in range(self.num_batches):
                self.is_training = True

                # if sparse_softmax_cross_entropy_with_logits is used
                # batch_images, batch_label_expression, batch_label_domain, batch_label_expression1, batch_label_domain1, batch_files_name = self.get_batch_train_test(enable_shuffle,ind_batch)
                batch_images, batch_label_expression, batch_label_domain, batch_files_name = self.get_batch_train_test(
                    enable_shuffle, ind_batch)

                # map batch_label_expression and batch_label_domain to onehot with the target as 1, others as 0
                expression_label = list(map(lambda x: [[i, 0][i < 0] for i in x], batch_label_expression))
                domain_label = list(map(lambda x: [[i, 0][i < 0] for i in x], batch_label_domain))

                enable_shuffle = False
                start_time = time.time()

                # prior distribution on the prior of f [-1,1]
                batch_f_prior = np.random.uniform(
                    self.image_value_range[0],
                    self.image_value_range[-1],
                    [self.size_batch, self.num_fx]
                ).astype(np.float32)

                # 使用fetches获取计算结果
                _, _, _ = self.session.run(
                    fetches=[
                        self.EG_optimizer,
                        self.D_f_optimizer,
                        self.D_att_optimizer
                    ],
                    feed_dict={
                        self.input_image: batch_images,
                        self.expression: batch_label_expression,
                        self.domain: batch_label_domain,
                        self.f_prior: batch_f_prior
                    }
                )

                # (1) if softmax_cross_entropy_with_logits is used
                _ = self.session.run(
                    fetches=[
                        self.D_ex_optimizer
                    ],
                    feed_dict={
                        self.input_image: batch_images,
                        self.expression: expression_label,
                        self.domain: domain_label
                    }
                )

                # (2) if sparse_softmax_cross_entropy_with_logits is used
                # _ = self.session.run(
                #     fetches=[
                #         self.D_ex_optimizer
                #     ],
                #     feed_dict={
                #         self.input_image: batch_images,
                #         self.expression1: batch_label_expression1,
                #         self.domain1: batch_label_domain1
                #     }
                # )

                _ = self.session.run(
                    fetches=[
                        self.EG_optimizer
                    ],
                    feed_dict={
                        self.input_image: batch_images,
                        self.expression: batch_label_expression,
                        self.domain: batch_label_domain,
                        self.f_prior: batch_f_prior
                    }
                )

                _ = self.session.run(
                    fetches=[
                        self.EG_optimizer
                    ],
                    feed_dict={
                        self.input_image: batch_images,
                        self.expression: batch_label_expression,
                        self.domain: batch_label_domain,
                        self.f_prior: batch_f_prior
                    }
                )

                _ = self.session.run(
                    fetches=[
                        self.EG_optimizer
                    ],
                    feed_dict={
                        self.input_image: batch_images,
                        self.expression: batch_label_expression,
                        self.domain: batch_label_domain,
                        self.f_prior: batch_f_prior
                    }
                )

                _ = self.session.run(
                    fetches=[
                        self.EG_optimizer
                    ],
                    feed_dict={
                        self.input_image: batch_images,
                        self.expression: batch_label_expression,
                        self.domain: batch_label_domain,
                        self.f_prior: batch_f_prior
                    }
                )

                # (1) if softmax_cross_entropy_with_logits is used
                Dex_err, Ddomain_err, D_ex, D_domain = self.session.run(
                    fetches=[
                        self.D_ex_loss_input,
                        self.D_domain_loss_input,
                        self.d_ex_count,
                        self.d_domain_count
                    ],
                    feed_dict={
                        self.input_image: batch_images,
                        self.expression: expression_label,
                        self.domain: domain_label,
                        self.f_prior: batch_f_prior
                    }
                )

                # (2) if sparse_softmax_cross_entropy_with_logits is used
                # Dex_err, Ddomain_err, D_ex, D_domain = self.session.run(
                #     fetches=[
                #         self.D_ex_loss_input,
                #         # self.D_domain_loss_G,
                #         self.D_domain_loss_input,
                #         self.d_ex_count,
                #         # self.g_ex_count,
                #         self.d_domain_count,
                #         # self.g_domain_count,
                #     ],
                #     feed_dict={
                #         self.input_image: batch_images,
                #         self.expression1: batch_label_expression1,
                #         self.domain1: batch_label_domain1,
                #         self.f_prior: batch_f_prior
                #     }
                # )

                EG_err, Ef_err, Df_err, Dfp_err, Gi_err, DiG_err, Di_err, TV = self.session.run(
                    fetches=[
                        self.EG_loss,
                        self.E_f_loss,
                        self.D_f_loss_f,
                        self.D_f_loss_prior,
                        self.G_att_loss,
                        self.D_att_loss_G,
                        self.D_att_loss_input,
                        self.tv_loss
                    ],
                    feed_dict={
                        self.input_image: batch_images,
                        self.expression: batch_label_expression,
                        self.domain: batch_label_domain,
                        self.f_prior: batch_f_prior
                    }
                )

                print("\nEpoch: [%3d/%3d] Batch: [%3d/%3d]\n\tEG_err=%.4f\tTV=%.4f" %
                      (epoch + 1, num_epochs, ind_batch + 1, self.num_batches, EG_err, TV))
                print("\t Accuracy Dex=%.4f /36 \t Ddomain =%.4f /36  " % (D_ex, D_domain))

                # using the generated images to train the classifier训练6个epoch后用再生成的图片训练分类器
                if epoch > 6:

                    # You can change parameter $add$ to add different times of generated images
                    if epoch > 6 and epoch < 10:
                        add = 1
                    elif epoch > 10 and epoch < 20:
                        add = 5
                    elif epoch > 20 and epoch < 30:
                        add = 8
                    elif epoch > 30:
                        add = 10

                    for addimg in range(add):
                        # if sparse_softmax_cross_entropy_with_logits is used
                        # gen_images, gen_label_expression, gen_label_domain, gen_label_expression1, gen_label_domain1, gen_files_name = self.get_batch_gen(DIS, ind_batch*add+addimg)

                        gen_images, gen_label_expression, gen_label_domain, gen_files_name = self.get_batch_gen(DIS,
                                                                                                              ind_batch * add + addimg)
                        DIS = False
                        gen_label_expressiononehot = list(
                            map(lambda x: [[i, 0][i < 0] for i in x], gen_label_expression))
                        gen_label_domainonehot = list(map(lambda x: [[i, 0][i < 0] for i in x], gen_label_domain))

                        f, G = self.session.run(
                            [self.f, self.G],
                            feed_dict={
                                self.input_image: gen_images,
                                self.expression: gen_label_expression,
                                self.domain: gen_label_domain
                            }
                        )

                        # (1)
                        _ = self.session.run(
                            fetches=[
                                self.D_ex_optimizer
                            ],
                            feed_dict={
                                self.input_image: G,
                                self.expression: gen_label_expressiononehot,
                                self.domain: gen_label_domainonehot
                            }
                        )
                        # (2)
                        # _ = self.session.run(
                        #     fetches=[
                        #         self.D_ex_optimizer
                        #     ],
                        #     feed_dict={
                        #         self.input_image: G,
                        #         self.expression1: gen_label_expression1,
                        #         self.domain1: gen_label_domain1
                        #     }
                        # )

                        # (1)
                        Dex_err, Ddomain_err, D_ex, D_domain = self.session.run(
                            fetches=[
                                self.D_ex_loss_input,
                                self.D_domain_loss_input,
                                self.d_ex_count,
                                self.d_domain_count
                            ],
                            feed_dict={
                                self.input_image: batch_images,
                                self.expression: expression_label,
                                self.domain: domain_label,
                                self.f_prior: batch_f_prior
                            }
                        )

                        # (2)
                        # Dex_err, Ddomain_err, D_ex, D_domain = self.session.run(
                        #     fetches=[
                        #         self.D_ex_loss_input,
                        #         # self.D_domain_loss_G,
                        #         self.D_domain_loss_input,
                        #         self.d_ex_count,
                        #         # self.g_ex_count,
                        #         self.d_domain_count,
                        #         # self.g_domain_count,
                        #     ],
                        #     feed_dict={
                        #         self.input_image: batch_images,
                        #         self.expression1: gen_label_expression1,
                        #         self.domain1: gen_label_domain1,
                        #         self.f_prior: batch_f_prior
                        #     }
                        # )

                print("\tEf=%.4f\tDf=%.4f\tDfp=%.4f" % (Ef_err, Df_err, Dfp_err))
                print("\tGi=%.4f\tDi=%.4f\tDiG=%.4f" % (Gi_err, Di_err, DiG_err))
                print("\tDex=%.4f\tDdomain=%.4f" % (Dex_err, Ddomain_err))
                print("\t Accuracy DGex=%.4f /36 \t DGdomain =%.4f /36  " % (D_ex, D_domain))
                result = 'epoch=' + str(epoch) + '\t' + 'num_batches=' + str(ind_batch) + '\t' + str(D_ex) + '\t' + '\n'
                trainresult.writelines(result)

                # estimate left run time
                elapse = time.time() - start_time
                time_left = ((num_epochs - epoch - 1) * self.num_batches + (self.num_batches - ind_batch - 1)) * elapse
                print("\tTime left: %02d:%02d:%02d" %
                      (int(time_left / 3600), int(time_left % 3600 / 60), time_left % 60))

                # add to summary
                # pdb.set_trace()
                # (1)使用Tensor.eval()时只能在同一步当中获取一个tensor值，并且每次使用 eval 和 run时，都会执行整个计算图
                summary = self.summary.eval(
                    feed_dict={
                        self.input_image: batch_images,
                        self.expression: batch_label_expression,
                        self.domain: batch_label_domain,
                        self.f_prior: batch_f_prior
                    }
                )

                # (2)
                # summary = self.summary.eval(
                #     feed_dict={
                #         self.input_image: batch_images,
                #         self.expression: batch_label_expression,
                #         self.domain: batch_label_domain,
                #         self.expression1: batch_label_expression1,
                #         self.domain1: batch_label_domain1,
                #         self.f_prior: batch_f_prior
                #     }
                # )

                self.writer.add_summary(summary, self.EG_global_step.eval())  # add_summary方法将训练过程以及训练步数保存

            trainresult.close()

            # save sample images for each epoch
            name = '{:02d}.png'.format(epoch + 1)
            # name = '{:02d}.JPG'.format(epoch + 1)
            self.sample(sample_images, sample_label_expression, sample_label_domain, name) #保存生成的图像
            self.test(sample_images, sample_label_domain, name, sample_label_expression)
            # self.test_acc(sample_images, sample_expression_label, sample_domain_label)
            # print (self.is_training)

            for ind_batch in range(self.num_batches2):
                self.is_training = False
                # batch_images, batch_label_expression, batch_label_domain, batch_label_expression1, batch_label_domain1, batch_files_name = self.get_batch_train_test(enable_shuffle, ind_batch)

                batch_images, batch_label_expression, batch_label_domain, batch_files_name = self.get_batch_train_test(
                    enable_shuffle, ind_batch)

                batch_label_expressiononehot = list(map(lambda x: [[i, 0][i < 0] for i in x], batch_label_expression))
                batch_label_domainonehot = list(map(lambda x: [[i, 0][i < 0] for i in x], batch_label_domain))

                accex, accdomain, accindex = self.test_acc(batch_images, batch_label_expressiononehot,
                                                         batch_label_domainonehot)

                re = str(accex) + '\t' + str(accdomain) + '\n'

                # Record the classified labels of each test image
                for jj in range(accindex.shape[0]):
                    resu = accindex[jj]
                    f2.writelines(str(resu) + '\n')
                # Record the number of right test image in each group (each batch_size)
                f1.writelines(re)

            f1.close()
            f2.close()
            # save checkpoint for each 10 epoch
            if np.mod(epoch, 10) == 9:
                self.save_checkpoint()

        # save the trained model
        self.save_checkpoint()
        # close the summary writer
        self.writer.close()

    def Gencoder(self, image, reuse_variables=False):
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            if reuse_variables:
                tf.get_variable_scope().reuse_variables()
            num_layers = int(np.log2(self.size_image)) - int(self.size_kernel / 2)  # size_kernel=5

            current = image
            # conv layers with stride 2
            for i in range(num_layers):
                name = 'E_conv' + str(i)
                current = conv2d(
                    input_map=current,
                    num_output_channels=self.num_encoder_channels * (2 ** i),
                    size_kernel=self.size_kernel,
                    name=name
                )
                current = tf.nn.relu(current)

            # fully connection layer
            name = 'E_fc'
            current = fc(
                input_vector=tf.reshape(current, [self.size_batch, -1]),
                num_output_length=self.num_fx,  # num_fx=50
                name=name
            )
            # output
            return tf.nn.tanh(current)

    def Gdecoder(self, f, e, reuse_variables=False, enable_tile_label=True, tile_ratio=1.0):
        with tf.variable_scope(tf.get_variable_scope()):
            if reuse_variables:
                tf.get_variable_scope().reuse_variables()  # 当tf.get_variable_scope().reuse == False，调用该函数会创建新的变量
            num_layers = int(np.log2(self.size_image)) - int(self.size_kernel / 2)  # int(7.8-5/2)=5.3 num_layers=5
            if enable_tile_label:
                duplicate = int(self.num_fx * tile_ratio / self.num_categories)  # num_categories=6 duplicate=7
            else:
                duplicate = 1

            f= tf.concat([f, e],0)
            # f = concat_label(f, e, duplicate=duplicate)#(49,99)
            if enable_tile_label:
                # duplicate = int(self.num_fx * tile_ratio / self.num_domains)
                duplicate = int(self.num_fx * tile_ratio / 2)
            else:
                duplicate = 1

            # size_mini_map = int(self.size_image / 2 ** num_layers)  # int(224/2的5次方)=7

            name = 'G_fc'
            current = fc(
                input_vector=f,
                # num_output_length=self.num_gen_channels * size_mini_map * size_mini_map,
                num_output_length=self.size_image,
                name=name
            )#[49,128]

            # reshape to cube for deconv[49,1,1,128]
            # current = tf.reshape(current, [-1, size_mini_map, size_mini_map, self.num_gen_channels])
            current = tf.reshape(current, [-1, 1, 1, self.size_image])
            current = tf.nn.relu(current)

            # Self Attention

            name = 'G_g_attention'
            fake_images = self.generator(current, name=name)#[49,128,128,3]
            return fake_images

    ######################################################################
    # Generator
    def generator(self, z, is_training=True, reuse=False, name='generator'):
        with tf.variable_scope(name, reuse=reuse):
            ch = 1024 #输入z:[49,1,1,128]
            x = deconv(z, channels=ch, kernel=4, stride=1, padding='VALID', use_bias=False, sn=self.sn, scope='deconv')
            x = batch_norm(x, is_training, scope='batch_norm')
            x = relu(x)  # (49,4,4,1024) self.layer_num=4

            for i in range(self.layer_num // 2):
                if self.up_sample:
                    x = up_sample(x, scale_factor=2)
                    x = conv(x, channels=ch // 2, kernel=3, stride=1, pad=1, sn=self.sn, scope='up_conv_' + str(i))
                    x = batch_norm(x, is_training, scope='batch_norm_' + str(i))
                    x = relu(x)

                else:
                    x = deconv(x, channels=ch // 2, kernel=4, stride=3, use_bias=False, sn=self.sn,
                               scope='deconv_' + str(i))
                    x = batch_norm(x, is_training, scope='batch_norm_' + str(i))
                    x = relu(x)

                ch = ch // 2

            # Self Attention x(49,16,16,256)
            x = self.attention(x, ch, sn=self.sn, scope="attention", reuse=reuse)

            for i in range(self.layer_num // 2, self.layer_num):  # x(49,28,28,256)->(49,112,112,64)
                if self.up_sample:
                    x = up_sample(x, scale_factor=2)
                    x = conv(x, channels=ch // 2, kernel=3, stride=1, pad=1, sn=self.sn, scope='up_conv_' + str(i))
                    x = batch_norm(x, is_training, scope='batch_norm_' + str(i))
                    x = relu(x)

                else:
                    x = deconv(x, channels=ch // 2, kernel=4, stride=2, use_bias=False, sn=self.sn,
                               scope='deconv_' + str(i))
                    x = batch_norm(x, is_training, scope='batch_norm_' + str(i))
                    x = relu(x)

                ch = ch // 2

            if self.up_sample:
                x = up_sample(x, scale_factor=2)
                x = conv(x, channels=self.c_dim, kernel=3, stride=1, pad=1, sn=self.sn, scope='G_conv_logit')
                x = tanh(x)

            else:
                x = deconv(x, channels=self.c_dim, kernel=4, stride=2, use_bias=False, sn=self.sn,
                           scope='G_deconv_logit')
                x = tanh(x)

            return x

    def attention(self, x, ch, sn=False, scope='attention', reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            f = conv(x, ch // 8, kernel=1, stride=1, sn=sn, scope='f_conv')  # [bs, h, w, c']
            g = conv(x, ch // 8, kernel=1, stride=1, sn=sn, scope='g_conv')  # [bs, h, w, c']
            h = conv(x, ch, kernel=1, stride=1, sn=sn, scope='h_conv')  # [bs, h, w, c]

            # N = h * w
            s = tf.matmul(hw_flatten(g), hw_flatten(f), transdomain_b=True)  # # [bs, N, N]

            beta = tf.nn.softmax(s)  # attention map

            o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
            gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

            o = tf.reshape(o, shape=x.shape)  # [bs, h, w, C]
            x = gamma * o + x

        return x

    ######################################################################
    def discriminator_i(self, f, is_training=True, reuse_variables=False, num_hidden_layer_channels=(64, 32, 16),
                        enable_bn=True):
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            if reuse_variables:
                tf.get_variable_scope().reuse_variables()
            current = f

            # fully connection layer
            for i in range(len(num_hidden_layer_channels)):
                name = 'D_f_fc' + str(i)
                current = fc(
                    input_vector=current,
                    num_output_length=num_hidden_layer_channels[i],
                    name=name
                )
                if enable_bn:
                    name = 'D_f_bn' + str(i)
                    current = tf.contrib.layers.batch_norm(
                        current,
                        scale=False,
                        is_training=is_training,
                        scope=name,
                        reuse=reuse_variables
                    )
                current = tf.nn.relu(current)
            # output layer
            name = 'D_f_fc' + str(i + 1)
            current = fc(
                input_vector=current,
                num_output_length=1,
                name=name
            )
            return tf.nn.sigmoid(current), current

    def discriminator_att(self, image, is_training=True, reuse_variables=False):
        with tf.variable_scope("D_att_", reuse=reuse_variables):
            if reuse_variables:
                tf.get_variable_scope().reuse_variables()
            current = image
            # current = concat_label(current, y)
            # current = concat_label(current, domain, int(self.num_categories / 2))  # current=(49,256,256,25)

            ch = 64#conv()后x为（49,64,64,64）
            x = conv(current, channels=ch, kernel=4, stride=2, pad=1, sn=self.sn, use_bias=False, scope='conv')
            x = leakyrelu(x, 0.2)

            for i in range(self.layer_num // 2):
                x = conv(x, channels=ch * 2, kernel=4, stride=2, pad=1, sn=self.sn, use_bias=False,
                         scope='conv_' + str(i))
                x = batch_norm(x, is_training, scope='batch_norm' + str(i))
                x = leakyrelu(x, 0.2)

                ch = ch * 2

            # Self Attention[49,16,16,256] 后面的尺寸：[49,8,8,512]->[49,4,4,1024]=>conv [49,1,1,4]
            x = self.attention(x, ch, sn=self.sn, scope="attention", reuse=reuse_variables)

            for i in range(self.layer_num // 2, self.layer_num):
                x = conv(x, channels=ch * 2, kernel=4, stride=2, pad=1, sn=self.sn, use_bias=False,
                         scope='conv_' + str(i))
                x = batch_norm(x, is_training, scope='batch_norm' + str(i))
                x = leakyrelu(x, 0.2)

                ch = ch * 2

            x = conv(x, channels=4, stride=1, sn=self.sn, use_bias=False, scope='D_logit')

            return tf.nn.sigmoid(x), x

    def ex_classifier(self, image, resnet_size, reuse_variables=False, is_training=True):
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            if reuse_variables:
                tf.get_variable_scope().reuse_variables()
            x = image
            output_channels = [256, 512, 1024, 2048]
            strides = [2, 2, 2, 2]

            if resnet_size == 50:
                stages = [3, 4, 6, 3]
            elif resnet_size == 101:
                stages = [3, 4, 23, 3]
            elif resnet_size == 152:
                stages = [3, 8, 36, 3]
            else:
                raise ValueError("resnet_size %d not implement" % resnet_size)

            # init net
            with tf.variable_scope("D_acc_init"):
                x = pad_conv("D_acc_init_conv", x, 7, 64, 2)

            # 4 stages
            for i in range(len(stages)):
                with tf.variable_scope("D_acc_stage_%d_block_%d" % (i, 0)):
                    if stages[i] == 4 or stages[i] == 6:
                        x = residual(x, output_channels[i], strides[i], "conv", is_training)
                        x = NonLocalBlock(x, output_channels[i], scope="Non-local_%d" % i)
                    else:
                        x = residual(x, output_channels[i], strides[i], "conv", is_training)

                for j in range(1, stages[i]):
                    with tf.variable_scope("D_acc_stage_%d_block_%d" % (i, j)):
                        if stages[i] == 4 or stages[i] == 6:
                            x = residual(x, output_channels[i], 1, "identity", is_training)
                            x = NonLocalBlock(x, output_channels[i], scope="Non_local_%d" % j)
                        else:
                            x = residual(x, output_channels[i], 1, "identity", is_training)

            with tf.variable_scope("D_acc_global_pool"):
                x = batch_norm_nonlocal("D_acc_bn", x, is_training)
                x = relu(x)
                x = global_avg_pool(x)

            # with tf.variable_scope("logit"):
            #     logit = fc(x, 10)

            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                # name = 'D_acc_fc1'
                # current = fc(
                #     input_vector=tf.reshape(x, [self.size_batch, -1]),
                #     num_output_length=4096,
                #     name=name
                # )
                # current = lrelu(current)
                # if self.is_training:
                #     current = tf.nn.dropout(current, 0.5)
                #
                # name = 'D_acc_fc2'
                # current = fc(
                #     input_vector=tf.reshape(current, [self.size_batch, -1]),
                #     num_output_length=4096,
                #     name=name
                # )
                # current = lrelu(current)
                # if self.is_training:
                #     current = tf.nn.dropout(current, 0.5)

                name = 'D_acc_fc3'
                current1 = nonlocalfc(x,self.num_categories)

                return current1

    def domain_classifier(self, image, resnet_size, reuse_variables=False, is_training=True):
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            if reuse_variables:
                tf.get_variable_scope().reuse_variables()
            x = image
            output_channels = [256, 512, 1024, 2048]
            strides = [2, 2, 2, 2]

            if resnet_size == 50:
                stages = [3, 4, 6, 3]
            elif resnet_size == 101:
                stages = [3, 4, 23, 3]
            elif resnet_size == 152:
                stages = [3, 8, 36, 3]
            else:
                raise ValueError("resnet_size %d not implement" % resnet_size)

            # init net
            with tf.variable_scope("D_acc_init"):
                x = pad_conv("D_acc_init_conv", x, 7, 64, 2)

            # 4 stages
            for i in range(len(stages)):
                with tf.variable_scope("D_acc_stage_%d_block_%d" % (i, 0)):
                    if stages[i] == 4 or stages[i] == 6:
                        x = residual(x, output_channels[i], strides[i], "conv", is_training)
                        x = NonLocalBlock(x, output_channels[i], scope="Non-local_%d" % i)
                    else:
                        x = residual(x, output_channels[i], strides[i], "conv", is_training)

                for j in range(1, stages[i]):
                    with tf.variable_scope("D_acc_stage_%d_block_%d" % (i, j)):
                        if stages[i] == 4 or stages[i] == 6:
                            x = residual(x, output_channels[i], 1, "identity", is_training)
                            x = NonLocalBlock(x, output_channels[i], scope="Non_local_%d" % j)
                        else:
                            x = residual(x, output_channels[i], 1, "identity", is_training)

            with tf.variable_scope("D_acc_global_pool"):
                x = batch_norm_nonlocal("D_acc_bn", x, is_training)
                x = relu(x)
                x = global_avg_pool(x)

            # with tf.variable_scope("logit"):
            #     logit = fc(x, 10)

            with tf.variable_scope(tf.get_variable_scope(), reuse=False):

                name = 'D_acc_fc4'
                current2 = nonlocalfc(x, self.num_domains)
                # current2 = fc(
                #     input_vector=tf.reshape(x, [self.size_batch, -1]),
                #     num_output_length=self.num_domains,
                #     name=name
                # )
                return current2



    def save_checkpoint(self):
        checkpoint_dir = os.path.join(self.save_dir, 'checkpoint')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(  # 保存模型
            sess=self.session,
            save_path=os.path.join(checkpoint_dir, 'model'),
            global_step=self.EG_global_step.eval()
        )

    def load_checkpoint(self):
        print("\n\tLoading pre-trained model ...")
        checkpoint_dir = os.path.join(self.save_dir, 'checkpoint')
        checkpoints = tf.train.get_checkpoint_state(checkpoint_dir)
        if checkpoints and checkpoints.model_checkpoint_path:
            checkpoints_name = os.path.basename(checkpoints.model_checkpoint_path)
            self.saver.restore(self.session, os.path.join(checkpoint_dir, checkpoints_name))  # 读取模型
            return True
        else:
            return False

    def sample(self, images, labels, domain, name):
        sample_dir = os.path.join(self.save_dir, 'samples')
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)

        f, G = self.session.run(
            [self.f, self.G],
            feed_dict={
                self.input_image: images,
                self.expression: labels,
                self.domain: domain
            }
        )
        size_frame = int(np.sqrt(self.size_batch))
        save_batch_images(
            batch_images=G,
            save_path=os.path.join(sample_dir, name),
            image_value_range=self.image_value_range,
            size_frame=[size_frame, size_frame]
        )

    def test(self, images, domain, name, expression):
        # pdb.set_trace()
        test_dir = os.path.join(self.save_dir, 'test')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        images = images[:int(np.sqrt(self.size_batch)), :, :, :]  # image[0:7,:,:,:]

        domain = domain[:int(np.sqrt(self.size_batch)), :]
        size_sample = images.shape[0]
        labels = np.arange(size_sample)
        labels = np.repeat(labels, size_sample)
        query_labels = np.ones(
            shape=(size_sample ** 2, size_sample),
            dtype=np.float32
        ) * self.image_value_range[0]
        for i in range(query_labels.shape[0]):
            query_labels[i, labels[i]] = self.image_value_range[-1]
        # pdb.set_trace()
        query_images = np.tile(images, [self.num_categories, 1, 1, 1])  # self.num_categories=7
        query_domain = np.tile(domain, [self.num_categories, 1])

        print('Generate the images reconstructed with Fourier phase information')
        f, G = self.session.run(
            [self.f, self.G],
            feed_dict={
                self.input_image: query_images,
                self.expression: query_labels,
                self.domain: query_domain
            }
        )
        save_batch_images(
            batch_images=query_images,
            save_path=os.path.join(test_dir, 'input.png'),
            image_value_range=self.image_value_range,
            size_frame=[size_sample, size_sample]
        )
        save_batch_images(
            batch_images=G,
            save_path=os.path.join(test_dir, name),
            image_value_range=self.image_value_range,
            size_frame=[size_sample, size_sample]
        )

    def test_acc(self, images, expression, domain):
        self.is_training = False
        test_images = images
        test_expression = expression
        test_domains = domain

        D_ex_acc, D_domain_acc = self.session.run(
            [self.d_ex_count, self.d_domain_count],
            feed_dict={
                self.input_image: test_images,
                self.expression: test_expression,
                self.domain: test_domains
            }
        )
        # if sparse_softmax_cross_entropy_with_logits is used
        # D_ex_acc, D_domain_acc= self.session.run(
        #             [self.d_ex_count, self.d_domain_count],
        #             feed_dict={
        #                 self.input_image: test_images,
        #                 self.expression1: test_expression,
        #                 self.domain1: test_domains
        #             }
        #         )

        lo = self.session.run(
            fetches=[self.D_input_ex_logits],
            feed_dict={
                self.input_image: test_images
            }
        )
        re = lo[0]
        index = self.session.run(tf.argmax(re, 1))  # axis=0按列计算；axis=1按行计算
        print(self.session.run(tf.argmax(re, 1)))
        print("test Accex =%.4f \t ACCdomain= %.4f " % (D_ex_acc, D_domain_acc))
        return D_ex_acc, D_domain_acc, index

    # generate
    def custom_test(self, testing_samples_dir):
        #pdb.set_trace()
        self.custom_test_names = np.loadtxt(testing_samples_dir + 'image_name_label.txt', dtype=bytes, delimiter=' ').astype(str)
        len_testset = len(self.custom_test_names)
        np.random.shuffle(self.custom_test_names)
        test_batches = len_testset // self.size_batch

        for i in range(test_batches):

            test_images, test_label_expression, test_label_domain, test_names = self.get_batch_custom_test(i)
            num_samples = int(np.sqrt(self.size_batch))

            domain_1 = np.ones(
                shape=(num_samples, 5),
                dtype=np.float32
            ) * self.image_value_range[0]
            domain_2 = np.ones(
                shape=(num_samples, 5),
                dtype=np.float32
            ) * self.image_value_range[0]
            domain_3 = np.ones(
                shape=(num_samples, 5),
                dtype=np.float32
            ) * self.image_value_range[0]
            domain_4 = np.ones(
                shape=(num_samples, 5),
                dtype=np.float32
            ) * self.image_value_range[0]
            domain_5 = np.ones(
                shape=(num_samples, 5),
                dtype=np.float32
            ) * self.image_value_range[0]

            for p in range(domain_1.shape[0]):
                domain_1[p, 0] = self.image_value_range[-1]
                domain_2[p, 1] = self.image_value_range[-1]
                domain_3[p, 2] = self.image_value_range[-1]
                domain_4[p, 3] = self.image_value_range[-1]
                domain_5[p, 4] = self.image_value_range[-1]

            if not self.load_checkpoint():
                print("\tFAILED >_<!")
                exit(0)
            else:
                print("\tSUCCESS ^_^")

            self.test(test_images, domain_1, str(i) + 'test_as_1.png', test_label_expression)
            self.test(test_images, domain_2, str(i) + 'test_as_2.png', test_label_expression)
            self.test(test_images, domain_3, str(i) + 'test_as_3.png', test_label_expression)
            self.test(test_images, domain_4, str(i) + 'test_as_4.png', test_label_expression)
            self.test(test_images, domain_5, str(i) + 'test_as_5.png', test_label_expression)

            print('\n\tDone! Results are saved as %s\n' % os.path.join(self.save_dir, 'test', 'test_as_xxx.png'))

    def get_batch_custom_test(self, idx=0):

        tt_files = self.custom_test_names[idx * self.size_batch: idx * self.size_batch + self.size_batch]
        batch_images = np.zeros((self.size_batch, self.size_image, self.size_image, 3))
        for i in range(tt_files.shape[0]):
            sample = [load_image(
                # image_path=self.pathtrain + 'FACE_AGING_train/' + tt_files[i, 0],
                # image_path=self.pathtrain + 'my_test/test_image_my/' + tt_files[i, 0],
                image_path=self.pathtrain + 'JAFFE_test/' + tt_files[i, 0],
                image_size=self.size_image,
                image_value_range=self.image_value_range,
                is_gray=(self.num_input_channels == 1),
            )]
            if self.num_input_channels == 1:
                batch_images[i, :, :, :] = np.array(sample).astype(np.float32)[:, :, :, None]
            else:
                batch_images[i, :, :, :] = np.array(sample).astype(np.float32)

        tt_label_expression = np.ones(
            shape=(len(tt_files), self.num_categories),
            dtype=np.float32
        ) * self.image_value_range[0]

        tt_label_domain = np.ones(
            shape=(len(tt_files), self.num_domains),
            dtype=np.float32
        ) * self.image_value_range[0]

        # tt_label_expression1 = np.ones(
        #     shape=(len(tt_files)),
        #     dtype=np.float32
        # ) * self.image_value_range[0]
        #
        # tt_label_domain1 = np.ones(
        #     shape=(len(tt_files)),
        #     dtype=np.float32
        # ) * self.image_value_range[0]

        for i, label in enumerate(tt_files[:, 1]):
            label = tt_files[i, 1].astype('int')
            tt_label_expression[i, label] = self.image_value_range[-1]

            # tt_label_expression1[i]=label

            domain = tt_files[i, 2].astype('int')

            tt_label_domain[i, domain] = self.image_value_range[-1]

            # tt_label_domain1[i]=domain

        # return batch_images, tt_label_expression, tt_label_domain, tt_label_expression1, tt_label_domain1, tt_files
        return batch_images, tt_label_expression, tt_label_domain, tt_files