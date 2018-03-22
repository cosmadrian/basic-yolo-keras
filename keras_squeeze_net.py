"""SqueezeNet Model Defined in Keras."""
import functools
from functools import partial, reduce

from keras.layers import Conv2D, MaxPooling2D, Concatenate, add, Dropout, Lambda, Activation, AveragePooling2D, Flatten, Input, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
import numpy as np

def compose(*funcs):
    return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)

# Partial wrapper for Convolution2D with static default argument.
_SqueezeConv2D = partial(Conv2D, padding='same')

@functools.wraps(Conv2D)
def SqueezeConv2D(*args, **kwargs):
    """Wrapper to set SqueezeNet weight regularizer for Convolution2D."""
    squeezenet_conv_kwargs = {'kernel_regularizer': l2(5e-4), 'kernel_initializer':'lecun_normal'}
    squeezenet_conv_kwargs.update(kwargs)
    return _SqueezeConv2D(*args, **squeezenet_conv_kwargs)

def BN_Leaky():
    return compose(
        BatchNormalization(),
        LeakyReLU(0.3)
        )

def SqueezeConv2D_BN_Leaky(*args, **kwargs):
    """SqueezeNet Convolution2D followed by BatchNormalization and LeakyReLU."""
    bias_kwargs = {'use_bias': True}
    bias_kwargs.update(kwargs)
    return compose(
        SqueezeConv2D(*args, **bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.3))

def fire(expand1, expand2, squeeze):
    def pad(x):
        import tensorflow as tf
        padding = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        return tf.pad(x, padding, "CONSTANT")
    def _fire(_input):
        x = SqueezeConv2D_BN_Leaky(squeeze, (1, 1), strides=(1, 1))(_input)

        _squeeze = SqueezeConv2D_BN_Leaky(expand1, (1, 1), strides=(1, 1))(x)
        # padded = Lambda(pad)(x)
        _expand = SqueezeConv2D_BN_Leaky(expand2, (3, 3), strides=(1, 1), padding='same')(x)
        return Concatenate(axis=3)([_squeeze, _expand])
    return _fire


def squeeze_net_body(input):
    x = SqueezeConv2D_BN_Leaky(96, (7, 7), strides=(1, 1))(input)
    x = MaxPooling2D((2, 2))(x)
    skip1 = fire(squeeze=16, expand1=64, expand2=64)(x)
    x = fire(squeeze=16, expand1=64, expand2=64)(skip1)
    x = add([x, skip1])
    x = BN_Leaky()(x)
    x = fire(squeeze=32, expand1=128, expand2=128)(x)
    skip2 = MaxPooling2D((2, 2))(x)
    x = fire(squeeze=32, expand1=128, expand2=128)(skip2)
    x = add([x, skip2])
    x = BN_Leaky()(x)
    skip3 = fire(squeeze=48, expand1=192, expand2=192)(x)
    x = fire(squeeze=48, expand1=192, expand2=192)(skip3)
    x = add([x, skip3])
    x = BN_Leaky()(x)
    x = fire(squeeze=64, expand1=256, expand2=256)(x)
    skip4 = MaxPooling2D((2, 2))(x)
    x = fire(squeeze=64, expand1=256, expand2=256)(skip4)
    x = add([x, skip4])
    x = BN_Leaky()(x)
    x = Dropout(0.3)(x)
    return x


def squeeze_net(options):
    input_image = Input(shape=(options['IMAGE_H'], options['IMAGE_W'], 3))
    true_boxes  = Input(shape=(1, 1, 1, options['TRUE_BOX_BUFFER'] , 4))
    body = squeeze_net_body(input_image)
    x = Conv2D(options['BOX'] * (5 + options['CLASS']), (1, 1), strides=(1,1), padding='same')(body)
    grid_h, grid_w = x.shape[1:3]
    grid_h, grid_w = int(grid_h), int(grid_w)

    output = Reshape((grid_h, grid_w, options['BOX'], 5 + options['CLASS']))(x)
    output = Lambda(lambda args: args[0])([output, true_boxes])

    return Model([input_image, true_boxes], output), true_boxes, grid_h, grid_w
