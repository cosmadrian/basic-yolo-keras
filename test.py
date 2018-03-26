import keras
from frontend import YOLO
import cv2
from utils import draw_boxes, decode_netout
from time import time

yolo = YOLO(input_size=224,
            labels=['human'],
            max_box_per_image=10,
            anchors=[0.16, 0.40, 0.51, 1.24, 1.13, 2.82, 2.19, 4.54, 4.68, 4.86])

yolo.load_weights("yolo/yolo_weights.18-252.08.hdf5")

video_reader = cv2.VideoCapture(0)

while True:
    _, image = video_reader.read()

    start = time()
    boxes = yolo.predict(image)
    end = time()
    print(boxes, end-start)
    image = draw_boxes(image, boxes, ['human'])

    cv2.imshow('Person Detection', image)

    wait_key = cv2.waitKey(1)
    if wait_key == 27:
        break

cv2.destroyAllWindows()
video_reader.release()
