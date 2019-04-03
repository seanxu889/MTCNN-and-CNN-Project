#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 22:31:30 2019

@author: seanxu
"""

import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import re
import tensorflow as tf
import numpy as np
import time

def eachFile(path):
    pathDir = os.listdir(path)
    child_file_name = []
    full_child_file_list = []
    for allDir in pathDir:
        child = os.path.join('%s%s' % (path, allDir))
        full_child_file_list.append(child)
        child_file_name.append(allDir)

    return full_child_file_list, child_file_name

def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")

    isExists = os.path.exists(path)

    if isExists:
        print(path)
        print("Directory exists. ")
        return False
    else:
        os.makedirs(path)
        print(path)
        print("Directory established. ")
        return True

def importCroppedImg(file_path):
    _, file_name = eachFile(file_path)
    file_name = [n for n in file_name if n[0] != '.']

    face_img_train = []
    face_label_train = []
    plt.figure("image")
    for n in file_name:
        path = file_path + n + '/'
        name_list = os.listdir(path)
        for m in name_list:
            tmp_m = m.split('.')
            if tmp_m[0] == 'face':
                img = Image.open(path + m)
                img = img.resize((32, 32), Image.ANTIALIAS)          
                face_img_train.append(img)
                face_label_train.append(n)

    random.seed(100)
    random.shuffle(face_img_train)
    random.seed(100)
    random.shuffle(face_label_train)

    return face_img_train, face_label_train

if __name__ =='__main__':

    # Split dataset into training set and testing set with a desired scale
    filepath = './Project/'

    # Extract lable of images in training set and testing set
    train_dir = filepath + 'train/'
    test_dir = filepath + 'test/'
    train_path, train_name = eachFile(train_dir)
    test_path, test_name = eachFile(test_dir)

    # Build emotion dictionary
    emo_list = []
    for n in train_name:
        emo_list.append(re.match( r"[A-Z]+\.([A-Z]+)\d.\d+.tiff", n)[1])
    emo_list = set(emo_list)
    emo_list = sorted(emo_list)
    print(emo_list)

    emo_dict = {}
    count = 0
    for e in emo_list:
        emo_dict[e] = count
        count+=1

    # Build training/testing image and label vector
    file_path = './Project/FaceDetection/train/'
    face_cropped_train, face_lable_train = importCroppedImg(file_path)

    print(face_lable_train)

    train_img = []
    for img in face_cropped_train:
        img = img.resize((32, 32), Image.ANTIALIAS)
        train_img.append(np.array(img).reshape((1, -1))[0])

    train_img = np.array(train_img)/255.

    file_path = './Project/FaceDetection/test/'
    face_cropped_test, face_lable_test = importCroppedImg(file_path)

    test_img = []
    for img in face_cropped_test:
        img = img.resize((32, 32), Image.ANTIALIAS)
        test_img.append(np.array(img).reshape((1, -1))[0])

    test_img = np.array(test_img)/255.

    ## Test images from face localization.
    val_img = []
    _, filename = eachFile('mtcnn_output/resizedFace') # testing images, 32*32 RGB
    del filename[0]

    for f in filename:
        if f[0] is not '.':
            print(f)
            file_path = os.path.join('mtcnn_output/resizedFace', f)
            img = Image.open(file_path).convert('LA')
            img = np.array(img)[...,:1]

        val_img.append(np.array(img).reshape((1, -1))[0])

    val_img = np.array(val_img)/255.
    print(val_img.shape)
    
    train_label = []
    for i in face_lable_train:
        temp = np.zeros((1,7))
        temp[0][emo_dict[i]] = 1
        train_label.append(temp[0])

    train_label = np.array(train_label)

    test_label = []
    for i in face_lable_test:
        # print(emo_dict)
        # print(i)
        temp = np.zeros((1,7))
        temp[0][emo_dict[i]] = 1
        test_label.append(temp[0])

    test_label = np.array(test_label)

    # Define batch size and # of batches
    batch_size = 20
    num_batch = len(train_img)//batch_size

    def nextBatch(temp_img, temp_label):
        i  = random.randint(0,num_batch)
        batch_x = temp_img[i*batch_size:(i+1)*batch_size]
        batch_y = temp_label[i*batch_size:(i+1)*batch_size]
        return batch_x, batch_y

    batch_x, batch_y = nextBatch(train_img, train_label)

    # Build Convolutional Neural Network with training set
    img_size = 32
    category_size = 7

    def compute_accuracy(v_xs, v_ys):
        global prediction
        y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
        correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
        return result

    def prediction_label(v_xs):
        global prediction
        y_pre = sess.run(prediction, feed_dict={xs:v_xs, keep_prob:1})
        y_idx = []
        for row in y_pre:
            row = list(row)
            y_idx.append(row.index(max(row)))

        y_label = [list(emo_dict.keys())[i] for i in y_idx]
        return y_label

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    xs = tf.placeholder(tf.float32, [None, img_size*img_size])
    ys = tf.placeholder(tf.float32, [None, category_size])
    keep_prob = tf.placeholder(tf.float32)

    x_image = tf.reshape(xs, [-1, img_size, img_size, 1])

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # [64, 64, 32]
    h_pool1 = max_pool_2x2(h_conv1) # [32, 32, 32]

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # [32, 32, 64]
    h_pool2 = max_pool_2x2(h_conv2) # [16, 16, 64]
    
    W_fc1 = weight_variable([int(img_size/4)*int(img_size/4)*64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, int(img_size/4)*int(img_size/4)*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc3 = weight_variable([1024, category_size])
    b_fc3 = bias_variable([category_size])
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc3) + b_fc3)
    print(prediction)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1]))
    print(cross_entropy)

    train_step = tf.train.AdamOptimizer(0.0002).minimize(cross_entropy)
    merged_summary_op = tf.summary.merge_all()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    tic = time.clock()
    for n in range(2500):
        for i in range(num_batch):
            batch_x = train_img[i * batch_size: (i + 1) * batch_size]
            batch_y = train_label[i * batch_size: (i + 1) * batch_size]
            sess.run(train_step, feed_dict={xs: batch_x, ys: batch_y, keep_prob: 0.5})

        if n%50 == 0:
           saver.save(sess, "FaceDetection3/epoch_"+str(n)+"_model.ckpt")
           toc = time.clock()
           t_sec = toc - tic
           print('Time passed:'+str(t_sec))
           print(compute_accuracy(test_img, test_label))

    saver.restore(sess, "FaceDetection3/epoch_1975_model.ckpt")

    print(prediction_label(val_img))

    saver.save(sess, "FaceDetection3/model.ckpt") #save the final model