import keras
from keras_compressor import custom_layers

class BaseFeatureExtractor(object):
    # to be defined in each subclass
    def __init__(self, input_size):
        raise NotImplementedError()

    # to be defined in each subclass
    def normalize(self, image):
        raise NotImplementedError()

    def get_output_shape(self):
        return self.feature_extractor.get_output_shape_at(-1)[1:3]

    def extract(self, input_image):
        return self.feature_extractor(input_image)


class TinySqueezeResNet(BaseFeatureExtractor):
    def __init__(self, input_size):
        # TODO
        pass

    def normalize(self, image):
        # trained without normalization
        return image
