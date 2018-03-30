from keras.models import Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
import tensorflow as tf
import numpy as np
import cv2
from keras.optimizers import Adam
from preprocessing import BatchGenerator
from keras.callbacks import TerminateOnNaN, ModelCheckpoint, TensorBoard, LearningRateScheduler
from utils import BoundBox, bbox_iou, interval_overlap, decode_netout, step_lr_schedule, OutputObserver
from models.squeeze_net import squeeze_net_body
from models.darknet import darknet_body

def build_model(options, architecture):
    input_image = Input(shape=(options['IMAGE_H'], options['IMAGE_W'], 3))
    true_boxes  = Input(shape=(1, 1, 1, options['TRUE_BOX_BUFFER'] , 4))
    body = architecture(input_image)
    x = Conv2D(options['BOX'] * (5 + options['CLASS']), (1, 1), strides=(1,1), padding='same')(body)
    grid_h, grid_w = x.shape[1:3]
    grid_h, grid_w = int(grid_h), int(grid_w)
    print(grid_h, grid_w)

    output = Reshape((grid_h, grid_w, options['BOX'], 5 + options['CLASS']))(x)
    output = Lambda(lambda args: args[0])([output, true_boxes])

    return Model([input_image, true_boxes], output), true_boxes, grid_h, grid_w

model, true_boxes, grid_h, grid_w = build_model({
    'IMAGE_H': 416,
    'IMAGE_W': 416,
    'TRUE_BOX_BUFFER': 10,
    'CLASS': 1,
    'BOX': 5
    }, squeeze_net_body) # TODO use config to select architecture
model.summary()
