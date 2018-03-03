import keras
from keras_compressor import custom_layers
from keras.applications.resnet50 import ResNet50

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


class ResNet50Features(BaseFeatureExtractor):
    def __init__(self, input_size):
        resnet50 = ResNet50(input_shape=(input_size, input_size, 3), include_top=False)
        resnet50.layers.pop()

        self.feature_extractor = keras.models.Model(resnet50.layers[0].input, resnet50.layers[-1].output)
        self.feature_extractor.summary()

    def normalize(self, image):
        return image


class TinySqueezeResNet(BaseFeatureExtractor):
    def __init__(self, **kwargs):
        resnet50 = ResNet50(input_shape=(input_size, input_size, 3), include_top=False)
        resnet50.layers.pop()

        self.feature_extractor = Model(resnet50.layers[0].input, resnet50.layers[-1].output)

    def normalize(self, image):
        # trained without normalization
        return image
