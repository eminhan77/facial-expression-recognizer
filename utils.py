import collections
import cv2
import numpy as np
import pandas as pd
from tensorflow.python.framework import dtypes,random_seed

def dataLoaded(dataFil):
    data=pd.read_csv(dataFil)
    pixels=data['pixels'].tolist()
    width,height=48,48
    faces=[]
    i=0
    for pixelSeq in pixels:
        face=[int(pixel) for pixel in pixelSeq.split(' ')]
        face=np.asarray(face).reshape(width,height)

        faces.append(face)
        faces=np.asarray(faces)
        faces=np.expand_dims(faces,-1)
        emotions=pd.get_dummies(data['emotion']).as_matrix()
        return faces,emotions



class DataSet(object):
    def __init__(self, images, labels,reshape=True,dtype=dtypes.float32,seed=None):
        seed1,seed2=random_seed.get_seed(seed)
        np.random.seed(seed1 if seed is None else seed2)
        if reshape:
            assert images.shape[3]==1
            images=images.reshape(images.shape[0],images.shape[1]*images.shape[2])

        if dtype==dtypes.float32:
            images=images.astype(np.float32)
            images=np.multiply(images,1.0/255.0)

        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

@property
def images(self):
    return self._images

@property
def labels(self):
    return self._labels

@property
def num_examples(self):
    return self._num_examples

@property
def epochs_completed(self):
    return self._epochs_completed

@property
def index_in_epoch(self):
    return self._index_in_epoch


def next_batch(self, batch_size, shuffle=True):
    start = self._index_in_epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
        perm0 = np.arange(self._num_examples)
        np.random.shuffle(perm0)
        self._images = self._images[perm0]
        self._labels = self._labels[perm0]

        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]

            # shuffling

            if shuffle:
                perm22 = np.arange(self._num_examples)
                np.random.shuffle(perm22)
                self._images = self._images[perm22]
                self._labels = self._labels[perm22]
                start = 0
                self._index_in_epoch = batch_size - rest_num_examples
                end=self._index_in_epoch
                imgs_new_part = self._images[start:end]
                labels_new_part = self._labels[start:end]
                return np.concatenate((images_rest_part,imgs_new_part),axis=0), np.concatenate((labels_rest_part, labels_new_part), axis=0)
            else:
                self._index_in_epoch+=batch_size
                end = self._index_in_epoch
                return self._images[start:end], self._labels[start:end]

def dataInput(train_dir,dtype=dtypes.float32,reshape=True,seed=None):
    trainSize=28709
    validSize=3589
    testSize=3589

    trainFcs,trainEmotions=dataLoaded(train_dir)
    print('dataset success')
    validationFcs=trainFcs[trainSize:trainSize+validSize]
    validationEmotions=trainEmotions[trainSize:trainSize+validSize]

    # testin data
    testFcs=trainFcs[trainSize+validSize]
    testEmotions=trainEmotions[trainSize+validSize:]

    #train data
    trainFcs=trainFcs[:trainSize]
    trainEmotions=trainEmotions[:trainSize]

    DataSets=collections.namedtuple('DataSets', ['train', 'validation', 'test'])
    train=DataSets(trainFcs,trainEmotions,reshape=reshape,seed=seed)
    validation=DataSets(validationFcs,validationEmotions,dtype=dtype,reshape=reshape,seed=seed)
    test=DataSets(testFcs,testEmotions,dtype=dtype,reshape=reshape,seed=seed)
    return DataSets(train=train,validation=validation,test=test)

def _test():
    import cv2
    i=dataInput('./data/fer2024/fer2024.csv')

    if __name__=='__main__':
        _test()