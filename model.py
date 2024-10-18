
import os
import sys

import cv2
import numpy as np
import tensorflow as tf
from anyio.streams import file
from utils import *

emotions=['angry','disgusted','fearful','happy','sad','surprised','neutral']

def depn(x):
    xImg=tf.reshape(x,[-1,48,48,1])
    wCnv1=varWeight([5,5,1,64])
    bCnv1=varBias([64])
    hCnv1=tf.nn.relu(conv2d(xImg,wCnv1)+bCnv1)

    hPol1=maxpool(hCnv1)

    norM1=tf.nn.lrn(hPol1,4,bias=1.0,alpha=0.001/9.0,beta=0.75)

    #conv2
    wCnv2=varWeight([3,3,64,64])
    bCnv2=varBias([64])
    hCnv2=tf.nn.relu(conv2d(hPol1,wCnv2)+bCnv2)
    norM2=tf.nn.lrn(hCnv2,4,bias=1.0,alpha=0.001/9.0,beta=0.75)
    hPol2=maxpool(norM2)

    #fully connected layer
    wFc1=varWeight([12*12*64,384])
    bFc1=varBias([384])
    hCnv3Flat=tf.reshape(hPol2,[-1,12*12*64])
    hFc1=tf.nn.relu(tf.matmul(hCnv3Flat,wFc1) + bFc1)

    wFc2=varWeight([384,192])
    bFc2=varBias([192])
    hFc2=tf.matmul(hFc1,wFc2) + bFc2

    # linear
    wFc3=varWeight([192,7])
    bFc3=varBias([7])
    yCnv=tf.add(tf.matmul(hFc2,wFc3),bFc3)

    return yCnv

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def maxpool(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME')

def varWeight(shape):
    init=tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init)

def varBias(shape):
    init=tf.constant(0.1, shape=shape)
    return tf.Variable(init)

def modelTrain(dataTrained):
    fer2024=dataInput(dataTrained)
    maxSteps=30001

    x=tf.placeholder(tf.float32,[None,2304])
    y=tf.placeholder(tf.float32,[None,7])

    yCnv=depn(x)

    cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=yCnv,labels=y))
    train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction=tf.equal(tf.argmax(yCnv,1),tf.argmax(y,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    with tf.Session() as sess:
        saver=tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        for step in range(maxSteps):
            batch=fer2024.train.next_batch(50)
            if step%100==0:
                train_accuracy=accuracy.eval(feed_dict={x:batch[0],y:batch[1]})
                print("step %d, training accuracy %g"%(step,train_accuracy))
                train_step.run(feed_dict={x:batch[0],y:batch[1]})

            if step+1==maxSteps:
                saver.save(sess,'./model.ckpt',global_step=step+1)
            if step%100==0:
                print('*Test accuracy %g'%accuracy.eval(feed_dict={x:fer2024.validation.images,y:fer2024.validation.labels}))


def predict(img=[[0.1]*2304]):
    x=tf.placeholder(tf.float32,[None,2304])
    yCnv=depn(x)

    saver=tf.train.Saver()
    probs=tf.nn.softmax(yCnv)
    y=tf.argmax(probs)

    with tf.Session() as sess:
        ckpt=tf.train.get_checkpoint_state('./models')
        print(ckpt.model_checkpoint_path)
        if ckpt.model_checkpoint_path and ckpt:
            saver.restore(sess,ckpt.model_checkpoint_path)
            print('Restore ssss')
            return sees.run(probs,feed_dict={x:img})


def imgToTensor(img):
    tensor=np.asarray(img).reshape(-1,2304)*1/255.0
    return tensor


def modelVal(modelPath,validFile):
    x=tf.placeholder(tf.float32,[None,2304])
    yCnv=depn(x)
    probs=tf.nn.softmax(yCnv)

    saver=tf.train.Saver()
    ckpt=tf.train.get_checkpoint_state(modelPath)

    with tf.Session() as sess:
        print(ckpt.model_checkpoint_path)
        if ckpt.model_checkpoint_path and ckpt:
            saver.restore(sess,ckpt.model_checkpoint_path)
            print('Restore Model Successfully')

            files=os.listdir(validFile)
            if file.endswith('.jpg'):
                imgFile=os.path.join(validFile,file)
                img=cv2.imread(imgFile,cv2.IMREAD_GRAYSCALE)
                tensort=imgToTensor(img)
                result=sess.run(probs,feed_dict={x:tensort})
                print(file,EMOTIONS[result.argmax()])