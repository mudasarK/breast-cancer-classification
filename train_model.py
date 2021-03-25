'''
# The task to predict the presence of inasive ductal carcinoma given a tissue patch is a binary classification. 
# A common loss for this problem is the binary cross entropy function. 
'''


import matplotlib
matplotlib.use("Agg") #non-GUI backend

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import np_utils
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from model.customModel import CustomModel, get_Network_Model

from config import config
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

NUM_EPOCHS = 25; INIT_LR=1e-2; BATCH_SIZE = 32
WIDTH = 48; HEIGHT = 48; DEPTH = 3; CLASSES = 1


# initialize the input image shape (48x48 pixels) along with
inputShape = (HEIGHT, WIDTH)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-model", "--model", type=str, default="customModel",
	help="name of pre-trained network to use")
args = vars(ap.parse_args())

# determine the total number of image paths in training, validation,
# and testing directories
train_data_len = len(list(paths.list_images(config.TRAIN_PATH)))
val_data_len = len(list(paths.list_images(config.VAL_PATH)))
test_data_len = len(list(paths.list_images(config.TEST_PATH)))



def get_data():
#training data generator
	trainAug = ImageDataGenerator(
		rescale=1/255.0,
		rotation_range=20,
		zoom_range=0.05,
		width_shift_range=0.1,
		height_shift_range=0.1,
		shear_range=0.05,
		horizontal_flip=True,
		vertical_flip=True,
		fill_mode="nearest")

	#validation data generator
	valAug=ImageDataGenerator(rescale=1 / 255.0)

	trainGen = trainAug.flow_from_directory(
		config.TRAIN_PATH,
		class_mode="binary",  #we have two classes so it is binary classification problem
		target_size=(HEIGHT, WIDTH),
		color_mode="rgb",
		shuffle=True,
		batch_size=BATCH_SIZE)
	valGen = valAug.flow_from_directory(
		config.VAL_PATH,
		class_mode="binary",
		target_size=(HEIGHT, WIDTH),
		color_mode="rgb",
		shuffle=True,
		batch_size=BATCH_SIZE)
	testGen = valAug.flow_from_directory(
		config.TEST_PATH,
		class_mode="binary",
		target_size=(HEIGHT, WIDTH),
		color_mode="rgb",
		shuffle=True,
		batch_size=BATCH_SIZE)

	

	return trainGen, valGen, testGen




def plot_results(pred_indices, H, name):
	pred_indices=np.argmax(pred_indices,axis=1)

	print(classification_report(testGen.classes, pred_indices, target_names=testGen.class_indices.keys()))

	cm 			= confusion_matrix(testGen.classes, pred_indices)
	total 		= sum(sum(cm))
	accuracy 	= (cm[0,0]+cm[1,1])/total
	specificity = cm[1,1]/(cm[1,0]+cm[1,1])
	sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
	print(cm)
	
	print(f'Accuracy: {accuracy}')
	print(f'Specificity: {specificity}')
	print(f'Sensitivity: {sensitivity}')


	epochlen = len(H.history["loss"])
	print()

	plt.style.use("ggplot")
	figure = plt.figure()
	plt.plot(np.arange(0, epochlen), H.history["loss"], 	label="train_loss")
	plt.plot(np.arange(0, epochlen), H.history["val_loss"], label="val_loss")
	plt.plot(np.arange(0, epochlen), H.history["acc"], 	 	label="train_acc")
	plt.plot(np.arange(0, epochlen), H.history["val_acc"],  label="val_acc")
	plt.title("Training Loss and Accuracy on the IDC Dataset")
	plt.xlabel("Epoch No.")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	figure.savefig(args["model"]+name+"_network.png", dpi=figure.dpi)

	return


if __name__ == "__main__":


	# Prepare the data
	trainGen, valGen, testGen = get_data()

	model, base_model = get_Network_Model(args, HEIGHT, WIDTH, DEPTH, CLASSES)


	model.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.0001), metrics=["accuracy"])
	#model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
	model.summary()



	checkpoint = ModelCheckpoint("model_weights/"+args["model"]+"_weights.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='max', period=1)
	#consider an improvement that is a specific increment, such as 1 unit for mean squared error or 1% for accuracy.  This can be specified via the “min_delta” argument.
	es = EarlyStopping(monitor='val_acc', mode='max', patience=15, verbose=1, min_delta=1) 
	
	# Fit the model 
	H = model.fit_generator(
						trainGen,
						steps_per_epoch=train_data_len//BATCH_SIZE,
						validation_data=valGen,
						validation_steps=val_data_len//BATCH_SIZE,
						epochs=NUM_EPOCHS,
						callbacks=[es, checkpoint])


	testGen.reset()
	pred_indices = model.predict_generator(testGen, steps=(test_data_len//BATCH_SIZE)+1)

	plot_results(pred_indices, H, "_frozen")

	if args["model"] != "customModel":
		print(" /n /n /t Fine Tunning the model /n/n")
	# Now let’s proceed to unfreeze the final set of CONV layers in the base model layers:
		# now that the head FC layers have been trained/initialized, lets
		# unfreeze the final set of CONV layers and make them trainable
		for layer in base_model.layers:
			layer.trainable = True

		# reset our data generators
		trainGen.reset()
		valGen.reset()
		testGen.reset()
		
		base_model.trainable = True #unfreeze the convolution base

		model.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.0001), metrics=["accuracy"])
		model.summary()

		#This might be more useful when fine tuning a model.
		checkpoint = ModelCheckpoint("model_weights/"+args["model"]+"_weights.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min', period=1)
		ES = EarlyStopping(monitor='val_loss', mode='min', patience=15, baseline=0.2, verbose=1)
		h = model.fit_generator(
							trainGen,
							steps_per_epoch=train_data_len//BATCH_SIZE,
							validation_data=valGen,
							validation_steps=val_data_len//BATCH_SIZE,
							epochs=NUM_EPOCHS,
							callbacks=[ES, checkpoint])

		print(" /n /n /t Evaluating the model /n/n")

		testGen.reset()
		pred_indices = model.predict_generator(testGen, steps=(test_data_len//BATCH_SIZE)+1)

		plot_results(pred_indices, h, "_unfrozen")



