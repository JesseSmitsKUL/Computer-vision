from lxml import etree
import numpy as np
import os
from skimage import io
from skimage.transform import resize

import keras
import numpy as np
from keras.models import Model
from keras.optimizers import RMSprop
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose,Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
from keras import backend as K
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import time


# parameters that you should set before running this script
filter = ['aeroplane', 'car', 'chair', 'dog', 'bird']       # select class, this default should yield 1489 training and 1470 validation images
voc_root_folder = "/home/ubuntu/Documents/vision/Autoencoders/VOCdevkit"  # please replace with the location on your laptop where you unpacked the tarball
image_size = 224    # image size that you will use for your network (input images will be resampled to this size), lower if you have troubles on your laptop (hint: use io.imshow to inspect the quality of the resampled images before feeding it into your network!)


# step1 - build list of filtered filenames
annotation_folder = os.path.join(voc_root_folder, "VOC2009/Annotations/")
annotation_files = os.listdir(annotation_folder)
filtered_filenames = []
for a_f in annotation_files:
    tree = etree.parse(os.path.join(annotation_folder, a_f))
    if np.any([tag.text == filt for tag in tree.iterfind(".//name") for filt in filter]):
        filtered_filenames.append(a_f[:-4])

# step2 - build (x,y) for TRAIN/VAL (classification)
classes_folder = os.path.join(voc_root_folder, "VOC2009/ImageSets/Main/")
classes_files = os.listdir(classes_folder)
train_files = [os.path.join(classes_folder, c_f) for filt in filter for c_f in classes_files if filt in c_f and '_train.txt' in c_f]
val_files = [os.path.join(classes_folder, c_f) for filt in filter for c_f in classes_files if filt in c_f and '_val.txt' in c_f]


def build_classification_dataset(list_of_files):
    """ build training or validation set

    :param list_of_files: list of filenames to build trainset with
    :return: tuple with x np.ndarray of shape (n_images, image_size, image_size, 3) and  y np.ndarray of shape (n_images, n_classes)
    """
    temp = []
    train_labels = []
    for f_cf in list_of_files:
        with open(f_cf) as file:
            lines = file.read().splitlines()
            temp.append([line.split()[0] for line in lines if int(line.split()[-1]) == 1])
            label_id = [f_ind for f_ind, filt in enumerate(filter) if filt in f_cf][0]
            train_labels.append(len(temp[-1]) * [label_id])
    train_filter = [item for l in temp for item in l]

    image_folder = os.path.join(voc_root_folder, "VOC2009/JPEGImages/")
    image_filenames = [os.path.join(image_folder, file) for f in train_filter for file in os.listdir(image_folder) if
                       f in file]
    x = np.array([resize(io.imread(img_f), (image_size, image_size, 3)) for img_f in image_filenames]).astype(
        'float32')
    # changed y to an array of shape (num_examples, num_classes) with 0 if class is not present and 1 if class is present
    y_temp = []
    for tf in train_filter:
        y_temp.append([1 if tf in l else 0 for l in temp])
    y = np.array(y_temp)

    return x, y


x_train, y_train = build_classification_dataset(train_files)
print('%i training images from %i classes' %(x_train.shape[0], y_train.shape[1]))
x_val, y_val = build_classification_dataset(val_files)
print('%i validation images from %i classes' %(x_val.shape[0],  y_train.shape[1]))

# from here, you can start building your model
# you will only need x_train and x_val for the autoencoder
# you should extend the above script for the segmentation task (you will need a slightly different function for building the label images)


def get_labels(class_probs, threshold=0.6):
    labels = []
    probs = np.array(class_probs)
    indices = np.argwhere(probs >= threshold)

    maxprob = np.amax(probs)

    print(indices)

    if len(indices) > 0:
        adjusted_indices = np.argwhere(np.abs(probs-maxprob) < 0.03)
        labels = [filter[index[0]] for index in adjusted_indices]
    else:
        # take index of max probability
        index = np.argmax(probs)
        labels = [filter[index]]
    return labels


train = False
test = False
trainClass = False
testClass = True

import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

if test:
    from keras.models import load_model

    model = load_model('cae_model_400_1556127610.9818544_GOOD_MORE_EPOCHS.h5')

    test_val = x_val[:100]

    print(y_val)

    pred = model.predict(np.array(test_val))

    for p, origim in zip(pred, test_val):
        io.imshow(p)
        io.show()

if train:
    input_img = Input(shape = (image_size, image_size, 3))

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (28, 28, 32) i.e. 28*28*32-dimensional

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])

    autoencoder.summary()

    epochs = 400

    autoencoder_train = autoencoder.fit(x_train, x_train,epochs=epochs,verbose=1,validation_split=0.2)

    autoencoder.save('cae_model_{epochs}_{id}.h5'.format(epochs=epochs, id=time.time()))

if testClass:
    from keras.models import load_model

    model = load_model('class_model_50_1556195320.1027088.h5')

    test_val = x_val
    np.random.shuffle(test_val)

    print(y_val)

    pred = model.predict(np.array(test_val))

    for l, im in zip(pred, test_val):
        io.imshow(im)
        print('LABEL:', l)
        labels = get_labels(l, threshold=0.3)
        print(labels)
        io.show()

if trainClass:
    from keras.models import load_model

    cae_model = load_model('cae_model_400_1556127610.9818544_GOOD_MORE_EPOCHS.h5')
    for layer in cae_model.layers:
        layer.trainable = False

    layer_output = cae_model.get_layer("max_pooling2d_3").output
    x = Flatten()(layer_output)
    x = Dropout(0.5)(x)
    x = Dense(512 ,activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(512 ,activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(256 ,activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(128 ,activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(len(filter),activation="softmax")(x)

    model = Model(inputs=[cae_model.input], outputs=[x])

    model.compile(loss='binary_crossentropy', optimizer='adadelta',metrics=['accuracy'])
    model.summary()

    epochs = 50

    print(y_train.shape)

    train_x, val_x, train_y, val_y = train_test_split(x_train, y_train, test_size=0.35, random_state=42)

    model_train = model.fit(train_x, train_y,epochs=epochs, batch_size=128, verbose=1,validation_data=(val_x, val_y))

    model.save('class_model_{epochs}_{id}.h5'.format(epochs=epochs, id=time.time()))
