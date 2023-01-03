"""
顔認識、本人識別モデルの作成クラス・関数モジュール
"""
# dnn用
# もしかしてkerasだから遅いのか？

# その他パッケージ
import numpy as np
import cv2
import os
import json
import copy
from scipy import spatial

# モデル作成用
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
  
  def __init__(self) -> None:
    self.embedder = FaceNet() # FaceNetモデル
    self.embeddings = None
    self.myself_embeddings = None
    self.detector = MTCNN() # 顔領域の検出器
    self.faces = [] # 顔検出時の顔データ格納リスト
    self.img = None
    self.json_dict = None
    self.detected_flg = False
  
  def find_faces(self, img):
    """動画に打ちっている顔認識関数。複数人でも見つけることが可能

    Args:
        path (_type_): _description_

    Returns:
        _type_: _description_
    """
    # 顔検出用関数
    # これじゃ検出遅いか？(数秒かかってしまう)
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
  
  def make_embedder(self):
    # 顔データの読み込み
    self.embeddings = self.embedder.embeddings(self.faces) # 潜在変数表現に変換
    
  def register_face(self, img, label='Me'):
    # 自分の顔を登録する 最終的には色々な人を登録できるようにする。
    img = cv2.imread(img)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # RGB形式に変換
    faces = self.detector.detect_faces(img_rgb) # セルフィーの写真しか入らないようにする
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
    cv2.imwrite('data\顔認証データ\me1.jpg', myself[0]) # 確認用 普通
    cv2.imwrite('data\顔認証データ\me2.jpg', myself[1]) # 確認用 マスク加工
    
    self.json_dict = self.load_json()
    self.json_dict = {
      label: self.myself_embeddings[0].tolist(),
      'Masked_'+label: self.myself_embeddings[1].tolist()
    } # 値の追加\
    self.update_json() # jsonのアップデート
    
    return True
    
  def load_json(self):
    """顔の登録情報の引き出し。

    Returns:
        bool: _description_
        str: _description_
    """
    try:
      with open('data/vector.json') as f: # ファイルの読み込み
        self.json_dict = json.load(f)
      if len(self.json_dict) == 0:
        return False
      return True
    except Exception as e:
      print(e)
      return e # strで返ってきたら読み込めないエラーということにする
  
  def update_json(self):
    """顔の登録情報のアップデート

    Returns:
        bool: _description_
        str: _description_
    """
    if len(self.json_dict) == 0:
      return False
    try:
      with open('data/vector.json', 'w') as f:
          json.dump(self.json_dict, f, indent=4)
      return True
    except Exception as e:
      print(e)
      return e # strで返ってきたら読み込めないエラーということにする
    
  def Is_Me(self):
    threshhold = 1.1 # google推奨
    flg = self.load_json()
    if flg and not self.json_dict.get('Me') is None and not self.json_dict.get('Masked_Me') is None:
      for i, embbeding in enumerate(self.embeddings):
        distance1 = spatial.distance.euclidean(np.array(self.json_dict["Me"]), embbeding)
        distance2 = spatial.distance.euclidean(np.array(self.json_dict["Masked_Me"]), embbeding)
        if distance1 < threshhold or distance2 < threshhold:
          print("本人がいます。問題なし。")
          return True
      print("知らない人がいます。")
      return False
    print("データが登録されていません。")
    return False
    