from mira.detectors import MTCNN
import cv2
import time

file_path = r"D:\python\StrangerDetection\data\test"
detector = MTCNN() # 顔領域の検出器
img = cv2.imread(file_path) # 画像読み込み
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # RGB形式に変換
faces = detector.detect(img_rgb) # 顔領域を検出．画像中に複数の顔が検出されることも想定する