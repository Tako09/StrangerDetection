import cv2
import os
import MyselfRecognition as mr
import time
import pyautogui


detector = mr.MyselfDetection()
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
capture = cv2.VideoCapture(0) # VideoCapture オブジェクトを取得します
# 動画ファイル保存用の設定
fps = int(capture.get(cv2.CAP_PROP_FPS))                    # カメラのFPSを取得
w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))              # カメラの横幅を取得
h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))             # カメラの縦幅を取得
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')        # 動画保存時のfourcc設定（mp4用）
# video = cv2.VideoWriter('video.mp4', fourcc, fps, (w, h))  # 動画の仕様（ファイル名、fourcc, FPS, サイズ） TODO: 知らない人が見てる時レコーディングする。

def run_video():
  while(True):
    # ret, frame = capture.read() # フレームを取得
    # cv2.imshow('Video', frame)
    frame = cv2.imread('check2.jpg')
    
    detected_flg = detector.find_faces(frame) # TODO: 並行処理をさせて早くする
    detected_flg = True
    if detected_flg:
      detector.make_embedder() # 動画に映った顔のベクトルを取得
      unlock_flg = detector.Is_Me()
      if not unlock_flg:
        print("ロックします。")
        wall_paper()
        # TODO pyatutoguiで自由に動けなくする
      else:
        print("ロックを解除します")
        cv2.destroyWindow('screen')
    else:
      wall_paper()
    
    # キー操作があればwhileループを抜ける
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(2)
  
  close_video(capture)

def close_video(capture):
  capture.release()
  cv2.destroyAllWindows()
  
def wall_paper():
  img = cv2.imread('data\wallpaper\wallpaperbetter.jpg')
  cv2.namedWindow('screen', cv2.WINDOW_NORMAL)
  cv2.setWindowProperty('screen', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
  cv2.imshow('screen', img)

if __name__ == "__main__":
  run_video()
  # run_video()
  # TODO しきい値をあげて他の人の区別の精度が落ちないかチェックする
  # TODO 画面切り替え+マウス制御だけにする