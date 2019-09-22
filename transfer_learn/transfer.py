# filter warnings
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.applications.xception import Xception, preprocess_input
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.applications.densenet import DenseNet121, preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.nasnet import NASNetMobile, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.models import model_from_json
from keras.layers import Input
from sklearn.preprocessing import LabelEncoder
import numpy as np
import glob
import cv2
import h5py
import os
import json
import datetime
import time

{
	"model"           : "xception",
	"weights"         : "imagenet",
	"include_top"     : false,

	"train_path"      : "/Users/parthsharma/Desktop/flower-recognition/dataset/train",
	"test_path"		  : "/Users/parthsharma/Desktop/flower-recognition/test-client",
	"features_path"   : "/Users/parthsharma/Desktop/flower-recognition/output/defacement/xception/features.h5",
	"labels_path"     : "/Users/parthsharma/Desktop/flower-recognition/output/defacement/xception/labels.h5",
	"results"         : "/Users/parthsharma/Desktop/flower-recognition/output/defacement/xception/results.txt",
	"classifier_path" : "/Users/parthsharma/Desktop/flower-recognition/output/defacement/xception/classifier.pickle",
	"model_path"	  : "/Users/parthsharma/Desktop/flower-recognition/output/defacement/xception/model",

	"test_size"       : 0.25,
	"seed"            : 9,
	"num_classes"     : 2
}
class TransferLearning():
	"""docstring for TransferLearning"""
	def __init__(self, model="mobilenet", weights="imagenet", include_top=True, test_size=0.20, seed=0, num_classes=2):
		
		
		if model_name == "vgg16":
			self.base_model = VGG16(weights=weights)
			self.model = Model(input=self.base_model.input, output=self.base_model.get_layer('fc1').output)
			self.image_size = (224, 224)
		elif model_name == "vgg19":
			self.base_model = VGG19(weights=weights)
			self.model = Model(input=self.base_model.input, output=self.base_model.get_layer('fc1').output)
			self.image_size = (224, 224)
		elif model_name == "resnet50":
			self.base_model = ResNet50(weights=weights)
			self.base_model.summary()
			self.model = Model(inputs=self.base_model.input, outputs=self.base_model.get_layer('fc1000').output)
			self.image_size = (224, 224)
		elif model_name == "inceptionv3":
			self.base_model = InceptionV3(include_top=include_top, weights=weights, input_tensor=Input(shape=(299,299,3)))
			self.model = Model(input=self.base_model.input, output=self.base_model.get_layer('custom').output)
			self.image_size = (299, 299)
		elif model_name == "inceptionresnetv2":
			self.base_model = InceptionResNetV2(include_top=include_top, weights=weights, input_tensor=Input(shape=(299,299,3)))
			self.model = Model(inputs=self.base_model.input, outputs=self.base_model.get_layer('custom').output)
			self.image_size = (299, 299)
		elif model_name == "mobilenet":
			self.base_model = MobileNet(include_top=include_top, weights=weights, input_tensor=Input(shape=(224,224,3)), input_shape=(224,224,3))
			self.model = Model(inputs=self.base_model.input, outputs=self.base_model.get_layer('conv_pw_13_relu').output)
			self.image_size = (224, 224)
		elif model_name == "mobilenetv2":
			self.base_model = MobileNeV2(include_top=include_top, weights=weights, input_tensor=Input(shape=(224,224,3)), input_shape=(224,224,3))
			self.base_model.summary()
			self.model = Model(inputs=self.base_model.input, outputs=self.base_model.get_layer('conv_pw_13_relu').output)
			self.image_size = (224, 224)
		elif model_name == "xception":
			self.base_model = Xception(weights=weights, , input_tensor=Input(shape=(299,299,3)))
			self.model = Model(inputs=self.base_model.input, outputs=self.base_model.get_layer('avg_pool').output)
			self.image_size = (299, 299)
		elif model_name == "densenet":
			self.base_model = DenseNet121(include_top=include_top, weights=weights, input_tensor=Input(shape=(224,224,3)), input_shape=(224,224,3))
			self.model = Model(inputs=self.base_model.input, outputs=self.base_model.get_layer('avg_pool').output)
			self.image_size = (224, 224)
		elif model_name == "xception":
			self.base_model = NASNetMobile(include_top=include_top, weights=weights, input_tensor=Input(shape=(224,224,3)), input_shape=(224,224,3))
			self.model = Model(inputs=self.base_model.input, outputs=self.base_model.get_layer('avg_pool').output)
			self.image_size = (224, 224)
		else:
			self.base_model = None
		self.features_path = ''
		self.model_path = ''
		self.features = []
		self.labels = []

	def list_directory(self, path=None):

		train_labels = os.listdir(path)

		return 


	def label_encoder(self, train_labels=None):

		le = LabelEncoder()
		le.fit(train_labels)
		self.le_labels = le.transform(self.labels)


		return le, self.le_labels


	def extract_feature(self, features_path='', train_path=path, train_labels=train_labels):

		# loop over all the labels in the folder
		count = 0
		for i, label in enumerate(train_labels):
			cur_path = train_path + "/" + label
			for image_path in glob.glob(cur_path + "/*.png"):
				flat = process_image(image_path)
				features.append(flat)
				labels.append(label)
				count += 1
		print ("[INFO] processed - " + str(count))
		print ("[STATUS] training labels: {}".format(le_labels))
		print ("[STATUS] training labels shape: {}".format(le_labels.shape))

	def process_image(self, image_path=path):

		# extract features from images
		img = image.load_img(image_path, target_size=self.image_size)
		img = image.img_to_array(img)
		img = np.expand_dims(img, axis=0)
		img = preprocess_input(img)
		feature = model.predict(img)
		flat = feature.flatten()

		return flat

	def save(self):

		# save features and labels
		h5f_data = h5py.File(self.features_path, 'w')
		h5f_data.create_dataset('dataset_1', data=np.array(self.features))

		h5f_label = h5py.File(labels_path, 'w')
		h5f_label.create_dataset('dataset_1', data=np.array(self.le_labels))

		h5f_data.close()
		h5f_label.close()

		# save model and weights
		model_json = self.model.to_json()
		with open(self.model_path + str(test_size) + ".json", "w") as json_file:
			json_file.write(model_json)

		# save weights
		model.save_weights(self.model_path + str(test_size) + ".h5")
		








