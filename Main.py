import cv2
import os
import MyselfRecognition_keras as mr
import concurrent.futures
import pyautogui as pag
import random
import keyboard
import ctypes
from concurrent.futures import ProcessPoolExecutor


detector = mr.MyselfDetection()
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
capture = cv2.VideoCapture(0) # VideoCapture オブジェクトを取得します
# 動画ファイル保存用の設定
fps = int(capture.get(cv2.CAP_PROP_FPS))                    # カメラのFPSを取得
w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))              # カメラの横幅を取得
h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))             # カメラの縦幅を取得
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')        # 動画保存時のfourcc設定（mp4用）
# video = cv2.VideoWriter('video.mp4', fourcc, fps, (w, h))  # 動画の仕様（ファイル名、fourcc, FPS, サイズ）
pag.PAUSE = 1.0
width, height = pag.size() # 画面のサイズを取得
i = 0
pass_code = ['q', 't', 'a', 'k', 'u', 'y', 'a', 'p', 'c']
pass_code_length = len(pass_code)
unlock_flg = True
current_width = 0
current_height = 0

def run_video():
  while(True):
    ret, frame = capture.read() # フレームを取得
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): # press q to quit
      break
    
    detected_flg = detector.find_faces(frame) # TODO: 並行処理をさせて早くする
    
    if detected_flg:
      detector.make_embedder() # 動画に映った顔のベクトルを取得
      unlock_flg = detector.Is_Me()
    else:
      unlock_flg = False
      
  close_video(capture)

def close_video(capture):
  capture.release()
  cv2.destroyAllWindows()
  
def wall_paper():
  print("ロックします。")
  img = cv2.imread('data\wallpaper\wallpaperbetter.jpg')
  cv2.namedWindow('screen', cv2.WINDOW_NORMAL)
  cv2.setWindowProperty('screen', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
  cv2.imshow('screen', img)
  
def monitor_length(x):
  # 0 - モニターの長さまでの連続値を取得
  range_lst = []
  for i in range(0, x):
    range_lst.append(i+1)
  return range_lst

def detect_move():
  """マウスが動作するか検知する。動作した場合はスリープにする
  """
  while True:
      if not unlock_flg and current_width == 0 and current_height == 0:
        current_width, current_height = pag.position()
      elif not unlock_flg:
        continue
      else:
        tmp_width, tmp_height = pag.position()
        if current_width == tmp_width and current_height == tmp_height:
          continue
        else:
          # Windowsのスリープ
          ctypes.windll.PowrProf.SetSuspendState(0, 1, 0)
  # 値を初期化しておく
  current_width = 0
  current_height = 0

def read_passcode():
  """q を入力するとパスコードを取得するようにする。
  """
  while i < pass_code_length:
      if not unlock_flg and keyboard.read_key() == pass_code[i]:
        if i == 0:
          print("パスコードを入力してください")
        else:
          print(pass_code[i])
          i = i + 1
          if i == pass_code_length:
              print('パスコードの確認ができました')
              unlock_flg = True
          continue
      elif keyboard.read_key() != pass_code[i]:
          print("最初から入力してください")
          i = 0
          continue

if __name__ == "__main__":
  with ProcessPoolExecutor(max_workers=3) as executor:
    executor.submit(run_video)
    executor.submit(detect_move)
    executor.submit(read_passcode)
  # run_video()
  # TODO しきい値をあげて他の人の区別の精度が落ちないかチェックする
  # TODO 画面切り替え+マウス制御だけにする