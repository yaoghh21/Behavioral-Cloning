import csv
import cv2
import numpy as np

lines =[]
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
images = []
measurements = []
for line in lines:
    for i in range(3):
        source_path = line[0]
        filename = source_path.split('/')[-1]
        current_path = 'data/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        measurement = line[3]
        measurements.append(measurement)
    del measurements[0]
    del images[0]

X_train = np.array(images)
Y_train = np.array(measurements, dtype=float)

augmented_images, augmented_measurements = [], []
for image,measurement in zip(X_train, Y_train):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)

X_train = np.array(images)
Y_train = np.array(measurements, dtype=float)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(filters=24, kernel_size=(5, 5), subsample=(2,2),activation='relu'))
model.add(Convolution2D(filters=36, kernel_size=(5, 5), subsample=(2,2),activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(filters=48, kernel_size=(5, 5), subsample=(2,2),activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#model.add(Flatten())
#model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, epochs=25)

model.save('model4.h5')

    