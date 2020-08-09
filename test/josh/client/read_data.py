import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import keras
import numpy

def read_data(filename, sample_fraction):
    """ Helper function to read and preprocess data for training with Keras. """

    data = pd.read_csv(filename, header=None)
    new_header = data.iloc[0]
    data = data[1:]
    data.columns = new_header
    data.drop(['Formatted Date','Summary','Precip Type','Apparent Temperature (C)','Pressure (millibars)','Daily Summary','Loud Cover','Wind Bearing (degrees)','Visibility (km)'], axis=1, inplace=True)
    data = data.astype(float)
    
    X=data[['Humidity','Wind Speed (km/h)']]
    y=data['Temperature (C)']
    
    #X = np.array(y).reshape((-1,1))
    y = np.array(y).reshape((-1,1))
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    print(scaler_x.fit(X))
    xscale=scaler_x.transform(X)
    print(scaler_y.fit(y))
    yscale=scaler_y.transform(y)

    print("######### I m reading")

    print("############# plain")

    print(X)
    print(y)
    print("############# shape ")
    print(X.shape)
    print(y.shape)
    print("############# scale")

    print(xscale)
    print(yscale)
    print("############# scale shape")

    print(xscale.shape)
    print(yscale.shape)
    
    if sample_fraction < 1.0:
        foo, X, bar, y = train_test_split(xscale, yscale, test_size=sample_fraction)
    classes = range(10)
    return  (X, y, classes)

