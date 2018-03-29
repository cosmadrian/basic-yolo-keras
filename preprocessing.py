import os
import cv2
import copy
import numpy as np
from imgaug import augmenters as iaa
from keras.utils import Sequence
import json
import random
from utils import BoundBox, bbox_iou


def parse_annotation(annotation_file, image_dir, input_size=224, size=-1):
    """
        There is only one type of label: "human".
        Annotation file format:
        {
            'filename': [[x, y, w, h]]
        }
    """
    all_imgs = []
    with open(annotation_file, 'rt') as f:
        annotations = json.load(f)

    seen_labels = {'human': 0}
    for name, boxes in annotations.items():
        all_imgs.append({
            'filename': os.path.abspath(image_dir) + '/' + name,
            'width': input_size,
            'height': input_size,
            'object': [{
                'name': 'human',
                'xmin': int(x),
                'xmax': int(x + w),
                'ymin': int(y),
                'ymax': int(y + h)
            } for x, y, w, h in boxes]
        })
        seen_labels['human'] += len(boxes)

    if size == -1:
        return all_imgs, seen_labels

    selected_images = random.sample(all_imgs, size)

    seen_labels['human'] = 0
    for image in selected_images:
        seen_labels['human'] += len(image['object'])

    print("Selected images:")
    for image in selected_images:
        print(image['filename'])

    return selected_images, seen_labels


class BatchGenerator(Sequence):
    def __init__(self, images,
                 config,
                 shuffle=True,
                 jitter=True,
                 norm=None):
        self.generator = None

        self.images = images
        self.config = config

        self.shuffle = shuffle
        self.jitter = jitter
        self.norm = norm

        self.anchors = [BoundBox(0, 0, config['ANCHORS'][2*i], config['ANCHORS'][2*i+1])
                        for i in range(int(len(config['ANCHORS'])//2))]

        self.aug_pipe = iaa.Sequential([
            iaa.Sometimes(0.3, [
                iaa.OneOf([
                    iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
                    iaa.Grayscale(alpha=(0.0, 1.0)),
                    iaa.Add((-10, 10), per_channel=0.5),
                    ])
                ]),
            ])


        if shuffle:
            np.random.shuffle(self.images)

    def __len__(self):
        return int(np.ceil(float(len(self.images))/self.config['BATCH_SIZE']))

    def __getitem__(self, idx):
        l_bound = idx * self.config['BATCH_SIZE']
        r_bound = (idx + 1) * self.config['BATCH_SIZE']

        if r_bound > len(self.images):
            r_bound = len(self.images)
            l_bound = r_bound - self.config['BATCH_SIZE']

        instance_count = 0

        # input images
        x_batch = np.zeros((r_bound - l_bound, self.config['IMAGE_H'], self.config['IMAGE_W'], 3))
        # list of self.config['TRUE_BOX_BUFFER'] GT boxes
        b_batch = np.zeros((r_bound - l_bound, 1, 1, 1, self.config['TRUE_BOX_BUFFER'], 4))
        # desired network output
        y_batch = np.zeros((r_bound - l_bound, self.config['GRID_H'],  self.config['GRID_W'], self.config['BOX'], 4 + 1 + self.config['CLASS']))

        for train_instance in self.images[l_bound:r_bound]:
            # augment input image and fix object's position and size
            img, all_objs = self.aug_image(train_instance, jitter=self.jitter)
            if img is None:
                raise Exception("Image not found. Exiting ... ")

            # construct output from object's x, y, w, h
            true_box_index = 0

            for obj in all_objs:
                if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin'] and obj['name'] in self.config['LABELS']:
                    center_x = .5*(obj['xmin'] + obj['xmax'])
                    center_x = center_x / (float(self.config['IMAGE_W']) / self.config['GRID_W'])
                    center_y = .5*(obj['ymin'] + obj['ymax'])
                    center_y = center_y / (float(self.config['IMAGE_H']) / self.config['GRID_H'])

                    grid_x = int(np.floor(center_x))
                    grid_y = int(np.floor(center_y))

                    if grid_x < self.config['GRID_W'] and grid_y < self.config['GRID_H']:
                        obj_indx = self.config['LABELS'].index(obj['name'])

                        center_w = (obj['xmax'] - obj['xmin']) / (float(self.config['IMAGE_W']) / self.config['GRID_W'])  # unit: grid cell
                        center_h = (obj['ymax'] - obj['ymin']) / (float(self.config['IMAGE_H']) / self.config['GRID_H'])  # unit: grid cell

                        box = [center_x, center_y, center_w, center_h]

                        # find the anchor that best predicts this box
                        best_anchor = -1
                        max_iou = -1

                        shifted_box = BoundBox(0, 0, center_w, center_h)

                        for i in range(len(self.anchors)):
                            anchor = self.anchors[i]
                            iou = bbox_iou(shifted_box, anchor)

                            if max_iou < iou:
                                best_anchor = i
                                max_iou = iou

                        # assign ground truth x, y, w, h, confidence and class probs to y_batch
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 0:4] = box
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 4] = 1.
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 5 + obj_indx] = 1

                        # assign the true box to b_batch
                        b_batch[instance_count, 0, 0, 0, true_box_index] = box

                        true_box_index += 1
                        true_box_index = true_box_index % self.config['TRUE_BOX_BUFFER']

            # assign input image to x_batch
            x_batch[instance_count] = img

            # increase instance counter in current batch
            instance_count += 1

        return [x_batch, b_batch], y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.images)

    def aug_image(self, train_instance, jitter):
        image_name = train_instance['filename']
        image = cv2.imread(image_name)

        if image is None:
            print('Cannot find ', image_name)
            return None, []

        h, w, c = image.shape
        all_objs = copy.deepcopy(train_instance['object'])
        # TESTING
        return image, all_objs

        if jitter:
            image = self.aug_pipe.augment_image(image)

            flip = np.random.binomial(1, .5)
            if flip > 0.5:
                image = cv2.flip(image, 1)

        if jitter and flip > 0.5:
            for obj in all_objs:
                obj['xmin'] = self.config['IMAGE_W'] - obj['xmax']
                obj['xmax'] = self.config['IMAGE_W'] - obj['xmin']

        return image, all_objs
