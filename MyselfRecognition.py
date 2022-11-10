"""
顔認識、本人識別モデルの作成、本人以外の場合殺意があるかどうかの判別を行うクラス・関数モジュール
"""
# dnn用

# その他パッケージ
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob
import json
import copy
from scipy import spatial
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# モデル作成用
from keras.models import model_from_json
from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN

'''
変数の定義
'''
current_dir = os.getcwd().replace(os.sep,'/') + '/' # pyファイル配下までのパスを取得

# ネットワークと学習済みモデルをロードする

class MyselfDetection():
  '''
  自分自身を識別するためのモデルを作成
  ユークリッド距離のしきい値にて自分か判断する
  '''
  
  def __init__(self, threshhold=0.5, K=1) -> None:
    self.embedder = FaceNet() # FaceNetモデル
    self.embeddings = None
    self.myself_embeddings = None
    self.threshhold = threshhold # しきい値
    self.detector = MTCNN() # 顔領域の検出器
    self.faces = [] # 顔検出時の顔データ格納リスト
    self.K = K # クラスタ数
    img = os.listdir(current_dir + 'data/myself')
    self.self_img = current_dir + 'data/myself/' + img[0]
    self.img = None
  
  def find_faces(self, path):
    # 顔検出用関数
    # これじゃ検出遅いか？(数秒かかってしまう)
    img = cv2.imread(path) # 画像読み込み
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # RGB形式に変換
    faces = self.detector.detect_faces(img_rgb) # 顔領域を検出．画像中に複数の顔が検出されることも想定する
    print(faces)
    for i, face in enumerate(faces):
        boundary_box = face['box']
        tmp_img = img[boundary_box[1]:boundary_box[1]+boundary_box[3], boundary_box[0]:boundary_box[0]+boundary_box[2]] # img[top : bottom, left : right]
        if face['confidence'] > 0.90:
            self.faces.append(tmp_img)
            return True # 発見フラグ
        else:
            return False
    return False
  
  def make_embedder(self):
    # 顔データの読み込み
    self.embeddings = self.embedder.embeddings(self.faces) # 潜在変数表現に変換
  
  def delete_embedder(self):
    del self.embeddings
    
  def register_face(self, label='Me', recursive=False):
    # 自分の顔を登録する
    if not recursive:
      self.img = cv2.imread(self.self_img)
      img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB) # RGB形式に変換
    else:
      img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB) # RGB形式に変換
    faces = self.detector.detect_faces(img_rgb)
    face = faces[0]
      
    if face['confidence'] < 0.90:
      return False
    bounding_box = face['box']
    keypoints = face['keypoints']
    
    # 顔部分の切り取り
    myself = []
    self.img = self.img[bounding_box[1]:bounding_box[1]+bounding_box[3], bounding_box[0]:bounding_box[0]+bounding_box[2]] # img[top : bottom, left : right]
    trimed_img = copy.copy(self.img) # コピーを作成
    myself.append(trimed_img)
    cv2.rectangle(self.img,
              (int(bounding_box[0]), int(keypoints['nose'][1])),
              (int(bounding_box[0]+bounding_box[2]), int(bounding_box[1]+bounding_box[3])),
              (0,0,0), -1)
    myself.append(self.img) # マスク加工されたイメージ
    self.myself_embeddings = self.embedder.embeddings(myself) # 潜在変数表現に変換
    cv2.imwrite('me1.jpg', myself[0]) # 確認用
    cv2.imwrite('me2.jpg', myself[1]) # 確認用
    
    if not recursive:
      self.register_face(label='Me', recursive=True)
    
    d = self.load_json()
    d = {
      label: self.myself_embeddings[0].tolist(),
      'Masked_'+label: self.myself_embeddings[1].tolist()
    } # 値の追加\
    self.update_json(d) # jsonのアップデート
    
    return True
    
  def load_json(self):
    try:
      with open('data/vector.json') as f: # ファイルの読み込み
        d = json.load(f)
      return d
    except Exception as e:
      print(e)
      return False
  
  def update_json(self, d):
    try:
      with open('data/vector.json', 'w') as f:
          json.dump(d, f, indent=4)
    except Exception as e:
      print(e)
      return False
    
  def Is_Me(self):
    d = self.load_json
    for i, embbeding in enumerate(self.embeddings):
      distance1 = spatial.distance.euclidean(np.array(d["Me"]), embbeding)
      distance2 = spatial.distance.euclidean(np.array(d["Masked_Me"]), embbeding)
      if distance1 < 0.5 or distance2 < 0.5:
        print("本人がいます。問題なし。")
        return True
    return False
    
  def show_graph(self):
    '''
    ベクトルが正しくとれているかプロットする
    '''
    kmeans = KMeans(n_clusters=self.K).fit(self.embeddings) # 圧縮前にクラスタリングしておく
    pred_label = kmeans.predict(self.embeddings)

    pca = PCA(n_components=2)
    pca.fit(self.embeddings) # 2次元にしてグラフにプロットできるようにする
    reduced_embeddings = pca.fit_transform(self.embeddings)
    
    x = reduced_embeddings[:, 0]
    y = reduced_embeddings[:, 1]
    plt.scatter(x, y)
    plt.legend()
    plt.show()
    print(pred_label)