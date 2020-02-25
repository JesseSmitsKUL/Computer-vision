from lxml import etree
import numpy as np
import os
from skimage import io
from skimage.transform import resize


import keras
import numpy as np
from keras.models import Model
from keras.optimizers import RMSprop
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
from keras import backend as K
from keras.utils import to_categorical
import time


# parameters that you should set before running this script
filter = ['aeroplane', 'car', 'chair', 'dog', 'bird']       # select class, this default should yield 1489 training and 1470 validation images
voc_root_folder = "/home/ubuntu/Desktop/vision/VOCdevkit"  # please replace with the location on your laptop where you unpacked the tarball
image_size = 224   # image size that you will use for your network (input images will be resampled to this size), lower if you have troubles on your laptop (hint: use io.imshow to inspect the quality of the resampled images before feeding it into your network!)


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
train_files = [os.path.join(classes_folder, c_f) for c_f in classes_files for filt in filter if filt in c_f and '_train.txt' in c_f]
val_files = [os.path.join(classes_folder, c_f) for c_f in classes_files for filt in filter if filt in c_f and '_val.txt' in c_f]


def build_classification_dataset(list_of_files):
    """ build training or validation set

    :param list_of_files: list of filenames to build trainset with
    :return: tuple with x np.ndarray of shape (n_images, image_size, image_size, 3) and  y np.ndarray of shape (n_images, )
    """
    temp = []
    train_labels = []
    for f_cf in list_of_files:
        with open(f_cf) as file:
            lines = file.read().splitlines()
            temp.append([line.split()[0] for line in lines if int(line.split()[-1]) == 1])
            label_id = [f_ind for f_ind, filt in enumerate(filter) if filt in f_cf][0]
            print(label_id)
            train_labels.append(len(temp[-1]) * [label_id])
    train_filter = [item for l in temp for item in l]
    y = np.array([item for l in train_labels for item in l])

    image_folder = os.path.join(voc_root_folder, "VOC2009/JPEGImages/")
    image_filenames = [os.path.join(image_folder, file) for file in os.listdir(image_folder) for f in train_filter if
                       f in file]
    x = np.array([resize(io.imread(img_f), (image_size, image_size, 3)) for img_f in image_filenames]).astype(
        'float32')

    #print(list(zip(image_filenames, y)))
    return x, y

x_train, y_train = build_classification_dataset(train_files)
#io.imshow(x_train[0])
#io.show()
print('%i training images from %i classes' %(x_train.shape[0], len(np.unique(y_train))))
x_val, y_val = build_classification_dataset(val_files)
print(len(x_val), len(y_val))
print('%i validation images from %i classes' %(x_val.shape[0], len(np.unique(y_val))))



# from here, you can start building your model
# you will only need x_train and x_val for the autoencoder
# you should extend the above script for the segmentation task (you will need a slightly different function for building the label images)

train = False

import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

if not train:
	from keras.models import load_model

	model = load_model('cae_model_1556021244.9334974_GOOD.h5')

	test_val = x_val[:100]

	print(y_val)

	pred = model.predict(np.array(test_val))

	for p, origim in zip(pred, test_val):
		io.imshow(p)
		io.show()

else:
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

    autoencoder_train = autoencoder.fit(x_train, x_train,epochs=200,verbose=1,validation_split=0.2)

    autoencoder.save('cae_model_{id}.h5'.format(id=time.time()))
