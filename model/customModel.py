import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import SeparableConv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

# import the necessary models
from keras.models import Model, load_model
from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import MobileNetV2


# define a dictionary that maps model names to their classes
# inside Keras
MODELS = {
	"vgg16": VGG16,
	"vgg19": VGG19,
	"inception": InceptionV3,
	"resnet": ResNet50,
	"mobilenet": MobileNetV2,
	"customModel" : "customModel"
}

class CustomModel:
	@staticmethod
	def build(width, height, depth, classes):
		model=Sequential()
		shape=(height, width, depth)
		channelDim=-1

		if K.image_data_format()=="channels_first":
			shape=(depth, height, width)
			channelDim=1

		model.add(SeparableConv2D(32, (3,3), padding="same",input_shape=shape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=channelDim))
		model.add(MaxPooling2D(pool_size=(2,2)))

		model.add(SeparableConv2D(64, (3,3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=channelDim))
		model.add(SeparableConv2D(64, (3,3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=channelDim))
		model.add(MaxPooling2D(pool_size=(2,2)))


		model.add(SeparableConv2D(128, (3,3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=channelDim))
		model.add(SeparableConv2D(128, (3,3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=channelDim))
		model.add(SeparableConv2D(128, (3,3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=channelDim))
		model.add(MaxPooling2D(pool_size=(2,2)))

		model.add(Flatten())
		model.add(Dense(256))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))

		model.add(Dense(classes))
		model.add(Activation("sigmoid"))

		return model

def add_top(base_model, shape, classes):


	headModel = base_model.output
	#headModel = GlobalAveragePooling2D()(top_layers)#Flatten()(top_layers)
	headModel = Flatten()(headModel)
	headModel = Dense(256, activation='relu')(headModel)
	headModel = Dropout(0.2)(headModel)

	predictions = Dense(classes, activation='sigmoid')(headModel)

	model = Model(inputs=base_model.input, outputs=predictions)
	
	return model
	


def get_Network_Model(args, height, width, depth, classes):
	
	# esnure a valid model name was supplied via command line argument
	if args["model"] not in MODELS.keys():
		raise AssertionError("The --model command line argument should "
			"be a key in the `MODELS` dictionary")

	
	Network = MODELS[args["model"]]

	print("You choosed  ", args["model"] ," as Network \n")

	
	shape=(height, width, depth)
	if K.image_data_format()=="channels_first":
				shape=(depth, height, width)
				channelDim=1
	

	if Network == "customModel":
		base_model =  CustomModel.build(width=width, height=height, depth=depth, classes=classes)
		return base_model, base_model
	else:
		weights="imagenet"   # for now just using imageNet for transfer learning, it will take a while to download weights for the first time
		base_model = Network(weights=weights, include_top = False, input_shape = shape)
		base_model.trainable = False #freeze the convolution base
		# loop over all layers in the base model and freeze them so they will
		#  not be updated during the first training process
		for layer in base_model.layers:
			layer.trainable = False

	model = add_top(base_model, shape, classes)

	return model, base_model
