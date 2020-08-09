from __future__ import print_function
import sys
import tensorflow as tf
import tensorflow.keras as keras 
import tensorflow.keras.models as krm
from random import sample
import numpy

import pickle
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from read_data import read_data


def train(model,data,sample_fraction):
    print("-- RUNNING TRAINING --")

    batch_size = 32
    epochs = 1

    # The data, split between train and test sets
    (x_train, y_train, classes) = read_data(data,sample_fraction=sample_fraction)
    """
    num = 3 # Number of Clients
    ran_order = sample(range(0, x_train.shape[0]), x_train.shape[0])
    local_size=int(x_train.shape[0]/num)
    partitionedX=[]
    partitionedY=[]
    for i in range(0,num):
        partitionedX.append(x_train[ran_order[i*local_size:(i+1)*local_size]])
        partitionedY.append(y_train[ran_order[i*local_size:(i+1)*local_size]])
    X = numpy.array(partitionedX)
    Y = numpy.array(partitionedY)
    """

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

    print("-- TRAINING COMPLETED --")
    return model

if __name__ == '__main__':
    print("#####################################################################################################################")
    print("#####################################################################################################################")
    print("#####################################################################################################################")
    print("#####################################################################################################################")
    print("#####################################################################################################################")

    model = krm.load_model(sys.argv[1])
    model = train(model,'../data/train.csv',sample_fraction=0.1)
    model.save(sys.argv[2])


