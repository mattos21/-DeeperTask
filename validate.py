from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Flatten, Dense
from pandas import DataFrame
import numpy as np
import time
import cv2
import os

path = os.getcwd()
img_width, img_height = 64, 64

path = os.getcwd()


def create_model():
    # create model
    model = Sequential()

    # add model layers
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(64, 64, 3)))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(4, activation='softmax'))

    # compile model 
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


model = create_model()
model.load_weights(path + "/try1.h5")

labels = ['rotated_left', 'rotated_right',  'upright', 'upside_down']

def load_test_images():
    path = os.getcwd()
    files = os.listdir(path + "/test/")
    time.sleep(45)
    return files

def corret_image(file, label):
    image = cv2.imread(path + '/test/{}'.format(file))

    if labels[0] in label:
        # Rotate 90 Degrees Counter Clockwise
        cv2.transpose(image, image)
        cv2.flip(image, 0, image)
        cv2.imwrite(path + "/corrected_orientations/{}".format(file), image)

    elif labels[1] in label:
        # Rotate 90 Degrees Clockwise
        cv2.transpose(image, image);
        cv2.flip(image, +1, image)
        cv2.imwrite(path + "/corrected_orientations/{}".format(file), image)

    elif labels[2] in label:
        # Do nothing
        cv2.imwrite(path + "/corrected_orientations/{}".format(file), image)

    elif labels[3] in label:
        # Rotates 180 Degress
        cv2.flip(image, -1, image)
        cv2.imwrite(path + "/corrected_orientations/{}".format(file), image)

files = load_test_images()
images, label = [], []

for i in range(len(files)):
    x_test = []
    img = load_img(path + '/test/{}'.format(files[i]))

    x_test = img_to_array(img)

    x_test = np.array(x_test)
    x_test = x_test.reshape(1, 64, 64, 3)

    softm = model.predict(x_test)
    index_max = np.argmax(softm[0])

    corret_image(files[i], labels[index_max])

    images.append(files[i])
    label.append(labels[index_max])

def export_csv(images, label):

    Results = {'images': images,
                'labels': label
            }

    df = DataFrame(Results, columns= ['images', 'labels'])
    export_csv = df.to_csv (path + "/test.preds.csv", index = None, header=True)

export_csv(images, label)


