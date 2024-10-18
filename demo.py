
import cv2
import numpy as np
import sys
import tensorflow as tf
from model import predict,imgToTensor,depn

casc_path='./data/haarcascades/haarcascade_frontalface_default.xml'
cascade_classifier=cv2.CascadeClassifier(casc_path)
EMOTIONS=['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

def imgFormat(img):
    if len(img.shape)>2 and img.shape[2]==3:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        faces=cascade_classifier.detectMultiScale(img,scaleFactor=1.3,minNeighbors=5)

        if not len(faces)>0:
            return None,None
        maxFcs=faces[0]
        for face in faces:
            if face[2]*face[3]>maxFcs[2]*maxFcs[3]:
                maxFcs=face

                faceCor=maxFcs
                img=img[faceCor[1]:(faceCor[1]+faceCor[2]),faceCor[0]:(faceCor[0]+faceCor[3])]

                try:
                    img=cv2.resize(img,(48,48),interpolation=cv2.INTER_CUBIC)
                except Exception:
                    print("[+} Problem during resize")
                    return None,None
                return img,faceCor

def FaceDetect(img):
    if len(img.shape)>2 and img.shape[2]==3:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        faces=cascade_classifier.detectMultiScale(img,scaleFactor=1.3,minNeighbors=5)

    if not len(faces)>0:
        return None
    maxFace=faces[0]
    for face in faces:
        if face[2]*face[3]>maxFace[2]*maxFace[3]:
            maxFace=face
        faceImg=img[maxFace[1]:(maxFace[1]+maxFace[2]),maxFace[0]:(maxFace[0]+maxFace[3])]

        try:
            img=cv2.resize(faceImg,(48,48),interpolation=cv2.INTER_CUBIC)/255.
        except Exception:
            print("[+] Problem during resize")
            return None
        return faceImg


def imgResize(img,size):
        try:
            img=cv2.resize(img,size,interpolation=cv2.INTER_CUBIC)/255.
        except Exception:
            print("[+] Problem during resize")
            return None
        return img

def drEmotion():
    pass


def demo(modelPath,shwBox=False):
    facX=tf.placeholder(tf.float32,[None,2304])
    yCnv=depn(facX)
    probs=tf.nn.softmax(yCnv)

    saver=tf.train.Saver()
    ckpt=tf.train.get_checkpoint_state(modelPath)
    sees=tf.Session()
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess,ckpt.model_checkpoint_path)
        print('Restore model sucsses!!\nNOTE: Press SPACE on keyboard to capture face.')

        feelFace=[]
        for index,emotion in enumerate(EMOTIONS):
            feelFace.append(cv2.imread('./data/emojis/'+emotion+'.png',-1))
            videoCapture = cv2.VideoCapture(0)

            emojiFace=[]
            result=None

            while True:
                ret, frame = videoCapture.read()
                detectFace,faceCor=imgFormat(frame)
                if shwBox:
                    if faceCor is not None:
                        [x,y,w,h]=faceCor
                        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

                if cv2.waitKey(1) & 0xFF == ord(' '):
                    if detectFace is not None:
                        cv2.imread('a.jpg',detectFace)
                        tensor=imgToTensor(detectFace)
                        result=sees.run(probs,feed_dict={facX:tensor})

                        if result is not None:
                            for index,emotion in enumerate(EMOTIONS):
                                cv2.putText(frame,emotion,(10,index*20+20),cv2.FONT_HERSHEY_PLAIN,0.5,(0,255,0),1)
                                cv2.rectangle(frame, (130, index * 20 + 10), (130 + int(result[0][index] * 100), (index + 1) * 20 + 4),(255, 0, 0), -1)
                                emojiFace=feelFace[np.argmax(result[0])]

                            for c in range(0,3):
                                frame[200:320,10:130,c]=emoji_face[:, :, c] * (emoji_face[:, :, 3] / 255.0) + frame[200:320, 10:130, c] * (1.0 - emoji_face[:, :, 3] / 255.0)
                                cv2.imshow('face',frame)

                                if cv2.waitKey(1) & 0xFF == ord(' '):
                                    break
