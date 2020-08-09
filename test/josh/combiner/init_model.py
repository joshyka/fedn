import keras
#from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense
from keras.models import Sequential
import os

from scaleout.project import Project

import tempfile

def create_seed_model():
	model = Sequential()
	model.add(Dense(16, input_dim=2, kernel_initializer='normal', activation='relu'))
	model.add(Dense(32, activation='relu'))
	model.add(Dense(64, activation='relu'))
	model.add(Dense(64, activation='relu'))
	model.add(Dense(1, activation='linear'))
	model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])

	return model

if __name__ == '__main__':

	# Create a seed model and push to Minio
	model = create_seed_model()
	fod, outfile_name = tempfile.mkstemp(suffix='.h5') 
	model.save(outfile_name)

	project = Project()
	from scaleout.repository.helpers import get_repository
	storage = get_repository(project.config['Alliance']['Repository'])

	model_id = storage.set_model(outfile_name,is_file=True)
	os.unlink(outfile_name)
	print("Created seed model with id: {}".format(model_id))

	

