import tensorflow as tf
import numpy as np
from facenet.src import facenet
from mtcnn.mtcnn import MTCNN
import cv2

pre_model_path = 'facenet/20180402-114759/20180402-114759.pb'

class FaceEmbedding(object):

  def __init__(self, model_path):
    # モデルを読み込んでグラフに展開
    facenet.load_model(model_path)
  
    self.embeddings = None
    self.myself_embeddings = None
    self.detector = MTCNN() # 顔領域の検出器
    self.faces = [] # 顔検出時の顔データ格納リスト
    self.sess = tf.Session()
    
    self.images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
    self.embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
    self.phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
    self.embedding_size = self.embeddings.get_shape()[1] # 512

  def delete(self):
    self.sess.close()
    
  def find_faces(self, img:list) -> bool:
    """MTCNNを使って顔部分のみを抽出する。複数人でも見つけることが可能

    Args:
        path (_type_): _description_

    Returns:
        _type_: _description_
    """
    self.detected_flg = False
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # RGB形式に変換
    faces = self.detector.detect_faces(img_rgb) # 顔領域を検出．画像中に複数の顔が検出されることも想定する
    for i, face in enumerate(faces):
        boundary_box = face['box']
        tmp_img = img[boundary_box[1]:boundary_box[1]+boundary_box[3], boundary_box[0]:boundary_box[0]+boundary_box[2]] # img[top : bottom, left : right]
        if face['confidence'] > 0.90:
            self.faces.append(tmp_img)
            self.detected_flg = True # 1回でも人物が発見されたらフラグをtrueにする
        else:
            continue
    return self.detected_flg

  def face_embeddings(self, image):
    """

    Args:
        image_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    prewhitened = facenet.prewhiten(image) # 正規化の処理
    prewhitened = prewhitened.reshape(-1, prewhitened.shape[0], prewhitened.shape[1], prewhitened.shape[2])
    feed_dict = { self.images_placeholder: prewhitened, self.phase_train_placeholder: False }
    embeddings = self.sess.run(self.embeddings, feed_dict=feed_dict)
    return embeddings
  
embedder = FaceEmbedding(pre_model_path)
image = cv2.imread('check1.jpg')
prewhitened = embedder.face_embeddings(image)
print(prewhitened)
print(prewhitened.shape)
embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
print(embeddings)
print(embeddings.get_shape()[1])
embedder.delete()