#! /usr/bin/env python
import argparse
import os
import numpy as np
from preprocessing import parse_annotation
from frontend import YOLO
import json


argparser = argparse.ArgumentParser(
    description='Train and validate YOLOv2 model on human dataset')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')


def _main_(args):

    config_path = args.conf

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    ###############################
    #   Parse the annotations
    ###############################
    print('parsing annotation')
    # parse annotations of the training set
    train_imgs, train_labels = parse_annotation(config['train']['train_annotation_file'],
                                                config['train']['train_image_folder'], config['model']['input_size'], size=config['train']['train_size'])

    # parse annotations of the validation set, if any, otherwise split the training set
    if os.path.exists(config['valid']['valid_annotation_file']):
        valid_imgs, valid_labels = parse_annotation(config['valid']['valid_annotation_file'],
                                                    config['valid']['valid_image_folder'],
                                                    config['model']['input_size'])
    else:
        train_valid_split = int(0.8*len(train_imgs))
        np.random.shuffle(train_imgs)

        valid_imgs = train_imgs[train_valid_split:]
        train_imgs = train_imgs[:train_valid_split]

    ###############################
    #   Construct the model
    ###############################

    print('constructing the model')
    yolo = YOLO(input_size=config['model']['input_size'],
                labels=config['model']['labels'],
                max_box_per_image=config['model']['max_box_per_image'],
                anchors=config['model']['anchors'],
                output_observer_img=config['train']['output_observer_img'],
                output_observer_out=config['train']['output_observer_out'])

    ###############################
    #   Load the pretrained weights (if any)
    ###############################

    if os.path.exists(config['train']['pretrained_weights']):
        print("Loading pre-trained weights in", config['train']['pretrained_weights'])
        yolo.load_weights(config['train']['pretrained_weights'])

    ###############################
    #   Start the training process
    ###############################
    print("training ... ")
    yolo.train(train_imgs=train_imgs,
               valid_imgs=valid_imgs,
               train_times=config['train']['train_times'],
               valid_times=config['valid']['valid_times'],
               nb_epoch=config['train']['nb_epoch'],
               learning_rate=config['train']['learning_rate'],
               batch_size=config['train']['batch_size'],
               warmup_epochs=config['train']['warmup_epochs'],
               object_scale=config['train']['object_scale'],
               no_object_scale=config['train']['no_object_scale'],
               coord_scale=config['train']['coord_scale'],
               class_scale=config['train']['class_scale'],
               log_dir=config['train']['log_dir'],
               saved_weights_name=config['train']['saved_weights_name'],
               debug=config['train']['debug'])


if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
