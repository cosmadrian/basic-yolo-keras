import numpy as np
import os
import xml.etree.ElementTree as ET
import tensorflow as tf
import keras
import copy
import cv2

class OutputObserver(keras.callbacks.Callback):
    def __init__(self, yolo, img, output_folder):
        self.yolo = yolo
        self.img = img
        self.output_folder = output_folder

    def on_epoch_end(self, epoch, logs={}):
        boxes = self.yolo.predict(self.img)
        rects = copy.deepcopy(self.img)
        rects = draw_boxes(rects, boxes, ['human'])
        cv2.imwrite(self.output_folder + '/output' + str(epoch) + '.png', rects)

class BoundBox:
    def __init__(self, x, y, w, h, c=None, classes=None):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

        self.c = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score


def bbox_iou(box1, box2):
    x1_min = box1.x - box1.w/2
    x1_max = box1.x + box1.w/2
    y1_min = box1.y - box1.h/2
    y1_max = box1.y + box1.h/2

    x2_min = box2.x - box2.w/2
    x2_max = box2.x + box2.w/2
    y2_min = box2.y - box2.h/2
    y2_max = box2.y + box2.h/2

    intersect_w = interval_overlap([x1_min, x1_max], [x2_min, x2_max])
    intersect_h = interval_overlap([y1_min, y1_max], [y2_min, y2_max])

    intersect = intersect_w * intersect_h

    union = box1.w * box1.h + box2.w * box2.h - intersect

    return float(intersect) / union


def interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3


def decode_netout(netout, obj_threshold=0.3, nms_threshold=0.3, anchors=[], nb_class=1):
    grid_h, grid_w, nb_box = netout.shape[:3]

    boxes = []

    # decode the output by the network
    netout[..., 4] = sigmoid(netout[..., 4])
    netout[..., 5:] = netout[..., 4][...,
                                     np.newaxis] * softmax(netout[..., 5:])
    netout[..., 5:] *= netout[..., 5:] > obj_threshold

    for row in range(grid_h):
        for col in range(grid_w):
            for b in range(nb_box):
                # from 4th element onwards are confidence and class classes
                classes = netout[row, col, b, 5:]

                if classes.any():
                    # first 4 elements are x, y, w, and h
                    x, y, w, h = netout[row, col, b, :4]

                    # center position, unit: image width
                    x = (col + sigmoid(x)) / grid_w
                    # center position, unit: image height
                    y = (row + sigmoid(y)) / grid_h
                    w = anchors[2 * b + 0] * \
                        np.exp(w) / grid_w  # unit: image width
                    h = anchors[2 * b + 1] * \
                        np.exp(h) / grid_h  # unit: image height
                    confidence = netout[row, col, b, 4]

                    box = BoundBox(x, y, w, h, confidence, classes)

                    boxes.append(box)

    # suppress non-maximal boxes
    for c in range(nb_class):
        sorted_indices = list(
            reversed(np.argsort([box.classes[c] for box in boxes])))

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] == 0:
                continue
            else:
                for j in range(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]

                    if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_threshold:
                        boxes[index_j].classes[c] = 0

    # remove the boxes which are less likely than a obj_threshold
    boxes = [box for box in boxes if box.get_score() > obj_threshold]

    return boxes


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def softmax(x, axis=-1, t=-100.):
    x = x - np.max(x)

    if np.min(x) < t:
        x = x/np.min(x)*t

    e_x = np.exp(x)

    return e_x / e_x.sum(axis, keepdims=True)


def draw_boxes(image, boxes, labels):

    for box in boxes:
        xmin = int((box.x - box.w/2) * image.shape[1])
        xmax = int((box.x + box.w/2) * image.shape[1])
        ymin = int((box.y - box.h/2) * image.shape[0])
        ymax = int((box.y + box.h/2) * image.shape[0])

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
        cv2.putText(image,
                    str(box.get_score()),
                    (xmin, ymin - 13),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1e-3 * image.shape[0],
                    (0, 255, 0), 2)

    return image


def step_lr_schedule(max_epochs, initial_lr):
    def _schedule(epoch):
        lr = 0
        if epoch / max_epochs < 0.33:
            lr = initial_lr

        if epoch / max_epochs >= 0.33 and epoch / max_epochs < 0.66:
            lr = initial_lr / 10

        if epoch / max_epochs >= 0.66:
            lr = initial_lr / 100

        print("Learning rate: ", lr)
        return lr

    return _schedule
