'''
In summary, this is our directory structure:
```
data/
    train/
        rotated_left/
            001.jpg
            002.jpg
            ...
        rotated_right/
            001.jpg
            002.jpg
            ...
        
```
'''

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Flatten, Dense
from keras import backend as K
import os

# dimensions of our images.
img_width, img_height = 64, 64

path = os.getcwd()
train_data_dir = path + '/train'
validation_data_dir = path + '/test'
nb_train_samples = 48896
epochs = 3
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# create model
model = Sequential()

# add model layers
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(64,64,3)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(4, activation='softmax'))

# compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size
    )

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs
    )

model.save_weights('try1.h5')
