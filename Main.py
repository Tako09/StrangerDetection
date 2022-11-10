import cv2
import os
import MyselfRecognition as mr


obj = mr.MyselfDetection()

def main():
  # main関数
  run_video()

def run_video():
  while(True):
    capture = cv2.VideoCapture(0) # VideoCapture オブジェクトを取得します
    ret, frame = capture.read()
    wall_paper()

def close_video(capture):
  capture.release()
  cv2.destroyAllWindows()
  
def wall_paper():
  img = cv2.imread('data\wallpaper\wallpaperbetter.jpg')
  cv2.namedWindow('screen', cv2.WINDOW_NORMAL)
  cv2.setWindowProperty('screen', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
  cv2.imshow('screen', img)
  cv2.waitKey(1000)


if __name__ == "__main__":
  main()