import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import os
from PIL import Image
import numpy as np
import pickle

import utils


model_save_path = 'trained_model/alexnet.model'


def load_data(datafile, num_clss, save=True, save_path='dataset.pkl'):
    train_list = open(datafile,'r')
    labels = []
    images = []
    for line in train_list:
        tmp = line.strip().split(' ')
        fpath = tmp[0]
        img = load_image(fpath)
        img = utils.resize_image(img,224,224)
        np_img = utils.pil_to_nparray(img)
        images.append(np_img)

        index = int(tmp[1])
        label = np.zeros(num_clss)
        label[index] = 1
        labels.append(label)
    if save:
        pickle.dump((images, labels), open(save_path, 'wb'))
    return images, labels

def train(network, x, y):
    """alexnet 训练"""
    model = tflearn.DNN(network, checkpoint_path=model_save_path,
                        max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir='tensorboard_log')
    # 如果trained model文件已经存在，则读取
    if os.path.isfile(model_save_path):
        model.load(model_save_path)
    model.fit(x, y, n_epoch=20, validation_set=0.1, shuffle=True,
              show_metric=True, batch_size=64, snapshot_step=200,
              snapshot_epoch=False, run_id='alexnet_oxflowers17')
    model.save(model_save_path)


def load_image(img_path):
    # 读取一个image文件
    img = Image.open(img_path)
    return img


def create_alexnet(num_classes):
    """创建整个alexnet网络框架"""

    def create_alexnet(num_classes):
        # Building 'AlexNet'
        network = input_data(shape=[None, 224, 224, 3])
        network = conv_2d(network, 96, 11, strides=4, activation='relu')
        network = max_pool_2d(network, 3, strides=2)
        network = local_response_normalization(network)
        network = conv_2d(network, 256, 5, activation='relu')
        network = max_pool_2d(network, 3, strides=2)
        network = local_response_normalization(network)
        network = conv_2d(network, 384, 3, activation='relu')
        network = conv_2d(network, 384, 3, activation='relu')
        network = conv_2d(network, 256, 3, activation='relu')
        network = max_pool_2d(network, 3, strides=2)
        network = local_response_normalization(network)
        network = fully_connected(network, 4096, activation='tanh')
        network = dropout(network, 0.5)
        network = fully_connected(network, 4096, activation='tanh')
        network = dropout(network, 0.5)
        network = fully_connected(network, num_classes, activation='softmax')
        network = regression(network, optimizer='momentum',
                             loss='categorical_crossentropy',
                             learning_rate=0.001)
        return network

def predict(network, model_file):
    img_path = '17flowers/jpg/0/image_0034.jpg'
    imgs = []
    img = load_image(img_path)
    img = utils.resize_image(img, 224, 224)
    imgs.append(utils.pil_to_nparray(img))
    model = tflearn.DNN(network)
    model.load(model_file)
    prediction = model.predict(imgs)
    prediction = np.argmax(prediction, axis=1)
    return prediction


if __name__ == '__main__':
    is_training = False
    net = create_alexnet(17)
    if is_training:
        X, Y = load_data('train.txt', 17)
        # X, Y = load_from_pkl('dataset.pkl')
        train(net,X,Y)
    else:
        predicted = predict(net, model_save_path)
        print(predicted)