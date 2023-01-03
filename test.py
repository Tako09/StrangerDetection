from facenet.src import facenet
import cv2
import os 
from mtcnn.mtcnn import MTCNN

# GPUが使用されてるかのテスト文
# from tensorflow.python.client import device_lib
# device_lib.list_local_devices()

model_path = 'facenet/20180402-114759/20180402-114759.pb'
facenet.load_model(model_path)
image = cv2.imread('check2.jpg')
prewhitened = facenet.prewhiten(image) # 正規化の処理
prewhitened = prewhitened.reshape(-1, prewhitened.shape[0], prewhitened.shape[1], prewhitened.shape[2])
print(prewhitened)

