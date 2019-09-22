from keras.models import model_from_json
import numpy as np
import h5py
import json

def save_model(model=model, feature_path=features_path, labels_path=labels_path):

	# save features and labels
	h5f_data = h5py.File(features_path, 'w')
	h5f_data.create_dataset('dataset_1', data=np.array(features))

	h5f_label = h5py.File(labels_path, 'w')
	h5f_label.create_dataset('dataset_1', data=np.array(le_labels))

	h5f_data.close()
	h5f_label.close()

	# save model and weights
	model_json = model.to_json()
	with open(model_path + ".json", "w") as json_file:
		json_file.write(model_json)

	# save weights
	model.save_weights(model_path + str(test_size) + ".h5")


def load_model(self, model=model_path, weights=weights_path feature_path=feature_path, label_path=label_path):


	h5f_data  = h5py.File(features_path, 'r')
	h5f_label = h5py.File(labels_path, 'r')


	features_string = h5f_data['dataset_1']
	labels_string   = h5f_label['dataset_1']

	features = np.array(features_string)
	labels   = np.array(labels_string)

	h5f_data.close()
	h5f_label.close()

	# load json and create model
	json_file = open(model, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights(weights)

	return loaded_model, features, labels
		