from keras_squeeze_net import squeeze_net

model = squeeze_net({
	'IMAGE_W': 224,
	'IMAGE_H': 224,
	'TRUE_BOX_BUFFER': 10,
	'CLASS': 1,
	'BOX': 10,
	})

model.summary()

