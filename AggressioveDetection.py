'''
認識された人物が攻撃的かどうかを判別するためのクラスを持つ
'''

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob

# モデル作成用
from keras.models import Sequential
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Activation, Conv2D, Flatten, Dense,Dropout
from keras.models import model_from_json

current_dir = os.getcwd().replace(os.sep,'/') + '/' # pyファイル配下までのパスを取得

class AggressioveDetection():
  '''
  Aggresive detection用の学習データとモデルを作成・向上させるためのクラス
  '''
  original_noraml_img = current_dir + 'data/origin/normal'
  original_angry_img = current_dir + 'data/origin/angry'
  train_noraml_img = current_dir + 'data/train_data/normal'
  train_angry_img = current_dir + 'data/train_data/angry'
  test_img = current_dir + 'data/test_data'
  anger_json_config = current_dir + 'model/anger_detection/detection_model.json'
  anger_saved_weights = current_dir + 'model/detection_weights.hdf5'
  
  def __init__(self, size=(50,50), dense_size=2, category=['normal', 'anger']) -> None:
    # aggresive判別モデルパス
    self.category = category # 分類項目
    self.dense_size = dense_size
    self.size = size
    self.epochs = 11
    self.model = None
    self.results = {}
    
  def make_model(self, loss='categorical_crossentropy', optimizers="Adadelta", metrics=['accuracy'], shape=(50, 50, 3), use_existed_model=False):
    #   モデルの形によって結果に大きな差が出ることもあるので、下記のコードで上手くいかない場合はCNN層を増やしたり減らしたり、活性化関数を変えてみたりしてください。
    #   変えると良いのはActivation関数やConv2D層などですかね。
    #   Dropoutは過学習を防ぐものなのであまり変えてもそんなに変わりません。
    #   Adadelta以外にもSGD, Adagrad, Adam, Adamax, RMSprop, Nadamなどがあるので試してみてください。
    print('モデルの作成')
    if use_existed_model:
      # 学習済モデルの読み込み
      print('作成済モデルを使用する')
      json_string = open(self.anger_json_config).read()
      self.model = model_from_json(json_string)
      self.model.compile(loss=loss, optimizer=optimizers, metrics=metrics)
      self.model.load_weights(self.anger_saved_weights)
    else:
      print('新しくモデルを作成する')
      self.model = Sequential()
      self.model.add(Conv2D(32, (3, 3), padding='same',input_shape=shape)) # インプットの形を教える。間違えてるかも
      self.model.add(Activation('relu'))
      self.model.add(Conv2D(32, (3, 3)))
      self.model.add(Activation('relu'))
      self.model.add(MaxPooling2D(pool_size=(2, 2)))
      self.model.add(Dropout(0.25))

      self.model.add(Conv2D(64, (3, 3), padding='same'))
      self.model.add(Activation('relu'))
      self.model.add(Conv2D(64, (3, 3)))
      self.model.add(Activation('relu'))
      self.model.add(MaxPooling2D(pool_size=(2, 2)))
      self.model.add(Dropout(0.25))

      self.model.add(Flatten())
      self.model.add(Dense(512))
      self.model.add(Activation('relu'))
      self.model.add(Dropout(0.5))
      self.model.add(Dense(self.dense_size))
      self.model.add(Activation('softmax')) # 影響はあるのかな？？
      self.model.compile(loss=loss, optimizer=optimizers, metrics=metrics)

    self.model.summary()

  def fit(self, train_X, train_y, epochs=11):
    # CNNの学習器を作成します。
    # また、学習した結果をを.jsonと.h5というファイルに格納することで、次回から毎度学習しなくても利用できるようにしておきます。
    # またoptimizersというのは最適化関数のことですが、これを変えると結構差が出たりするので、全部試してみるとよいです。
    # エポック数は200にしてありますが、適宜変更してください。
    self.epochs = epochs
    self.results = self.model.fit(train_X, train_y, validation_split=0.2, batch_size=128, epochs=epochs, verbose=1)

    model_json_str = self.model.to_json()
    open(self.anger_json_config, 'w').write(model_json_str)
    self.model.save_weights(self.anger_saved_weights);

  def show_graph(self):
    # 学習結果の表示
    x = range(self.epochs)

    plt.plot(x, self.results.history['accuracy'])
    plt.title("Accuracy")
    plt.show()

    plt.plot(x, self.results.history['val_accuracy'])
    plt.title("Val_Accuracy")
    plt.show()

  def make_predict(self):
      # モデルのテスト用関数
      # トレーニングとテストデータの作成
      X = []

      files = glob.glob(self.test_img + "/*.jpg")
      for i, file in enumerate(files):
        img = cv2.imread(file) # 画像データの読み込み
        img = cv2.resize(img, self.size)
        X.append(img)

      test_X = np.asarray(X).astype(np.float32)
      test_X = test_X / 255.0 # 正規化してる
      
      predict_y = self.model.predict(test_X)

      return predict_y

  def make_training_data(self):
    '''
    aggresive detectionの学習用データを作成
    オリジナル、ぼかし、色変化、回転、マスク加工をする。
    trainデータを返す
    '''
    pass # また考える
      # train_X = np.asarray(tmp_X).astype(np.float32)
      # train_y = np.asarray(tmp_y).astype(np.int32) # numpy arrayに変換
      # train_X / 255.0 # 正規化してる
      # train_y = np_utils.to_categorical(train_y, dense_size) # indexに応じてone-hot ベクトルに変換している。ラベル1の場合→[0,1,0,0,0]となる(カテゴリ数5の場合)

      # return train_X, train_y