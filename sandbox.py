import keras
from frontend import YOLO
import cv2
from utils import draw_boxes, decode_netout
from time import time

yolo = YOLO(input_size=416,
            labels=['human'],
            max_box_per_image=10,
            anchors=[0.16, 0.40, 0.51, 1.24, 1.13, 2.82, 2.19, 4.54, 4.68, 4.86])
