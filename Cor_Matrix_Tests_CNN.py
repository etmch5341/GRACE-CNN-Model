import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import random
from tensorflow.keras.layers import Dense, BatchNormalization
from numpy.random import seed
import os, sys, math
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
from numpy.random import seed

seed_value = 7
os.environ['PYTHONASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
os.environ['PYTHONASHSEED']=str(seed)

NPZPath = '/Users/VSOP/Desktop/matrixData/'
NPZFname = 'SA_monthly_trainingTesting_numpy.npz'
#imageFname = 'SA_AOHISMonthlyTraining_GRACEmonthlyTest_SEESArch_numpy_052021.png'

with np.load(NPZPath+NPZFname) as data:
	trainData = data['trainingData']
	trainLabels = data['trainingLabels']
	
	testData = data['testingData']
	testLabels = data['testingLabels']


#.npz files do weird things to data size, shape, & type- reformatting so works with tensorflow environment
print('shape of trainData')
print(np.shape(trainData))
mSize = trainData.shape[1] * trainData.shape[2]
numTrainImages = trainData.shape[0]
numTestImages = testData.shape[0]
#trainData = trainData.reshape(trainData.shape[0], trainData.shape[1] * trainData.shape[2]).astype("float32")
#testData = testData.reshape(testData.shape[0], testData.shape[1] * testData.shape[2]).astype("float32")
print('shape of trainData')
print(np.shape(trainData))

#sys.exit()

all_train_images = []
for i in range(numTrainImages):
	img = trainData[i,:,:]
	width = img.shape[0]
	height = img.shape[1]
	img = img.reshape([width, height, 1]).astype("float32")
	all_train_images.append(img)


all_test_images = []
for i in range(numTestImages):
	img = testData[i,:,:]
	width = img.shape[0]
	height = img.shape[1]
	img = img.reshape([width, height, 1]).astype("float32")
	all_test_images.append(img)

#sys.exit()
trainLabels = trainLabels.astype("float32")
testLabels = testLabels.astype("float32")

#print(max(trainLabels), min(trainLabels))
trainLabels = trainLabels 
testLabels = testLabels 
print(max(trainLabels), min(trainLabels)) #making sure labels are between 0 & 11 for monthly sorting

#makes categories out of the labels 
trainLabels = tf.keras.utils.to_categorical(trainLabels,12)
testLabels = tf.keras.utils.to_categorical(testLabels,12)

#train_dataset = tf.data.Dataset.from_tensor_slices((trainData, trainLabels))
#test_dataset = tf.data.Dataset.from_tensor_slices((testData, testLabels))

train_dataset = tf.data.Dataset.from_tensor_slices((all_train_images, trainLabels))
test_dataset = tf.data.Dataset.from_tensor_slices((all_test_images, testLabels))
print(' train_dataset')
print(train_dataset)


batchSize = 16
train_dataset = train_dataset.batch(batchSize)
test_dataset = test_dataset.batch(batchSize)
'''
#Model1 -- Control w/ re-arranged preprocessing & NO Batch Norm 
model = models.Sequential()
model.add(layers.Conv2D(16, (2, 2), padding='same', activation='relu', input_shape=(trainData.shape[1], trainData.shape[2], 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu')) 

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(12, activation='softmax'))
'''

#Model1 -- Control - NO Batch Norm 
# SA Small matrices, Middle East Matrices, Australia, Greenland, SA
model = models.Sequential()
#model.add(BatchNormalization(input_shape=(trainData.shape[1], trainData.shape[2], 1)))
model.add(layers.Conv2D(16, (2, 2), padding='same', activation='relu', input_shape=(trainData.shape[1], trainData.shape[2], 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu')) 

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(12, activation='softmax'))
#epochs = 30, batchsize=16

'''
#Model2 -- control & NO Batch Norm (but with batchnorm for preprocess
model = models.Sequential()
model.add(layers.BatchNormalization()
model.add(layers.Conv2D(16, (2, 2), padding='same', activation='relu', input_shape=(trainData.shape[1], trainData.shape[2], 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(12, activation='softmax'))
model.summary
''' 
'''
#Model3 Architecture #Stride/downsampling & 1 batchNorm & GAP
model = models.Sequential()
model.add(layers.Conv2D(16, (2, 2),strides=2,  activation='relu', input_shape=(trainData.shape[1], trainData.shape[2], 1)))
model.add(BatchNormalization())#
model.add(layers.MaxPooling2D((2, 2)))#1
model.add(layers.Conv2D(32, (3, 3),strides=2, activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))#1
model.add(layers.Conv2D(64, (3, 3),strides=2,  activation='relu'))
model.add(layers. MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3),strides=2, padding='same', activation='relu'))

model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(12, activation='softmax'))

model.summary()

'''
'''
#Model3 Architecture #Stride/downsampling & 1 batchNorm & GAP
#SA 
model = models.Sequential()
model.add(layers.Conv2D(16, (2, 2),strides=2,  activation='relu', input_shape=(trainData.shape[1], trainData.shape[2], 1)))
model.add(BatchNormalization())#
model.add(layers.MaxPooling2D((2, 2)))#1
model.add(layers.Conv2D(32, (3, 3),strides=2, activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))#1
model.add(layers.Conv2D(64, (3, 3),strides=2, activation='relu'))
#model.add(layers. MaxPooling2D((2, 2)))
#model.add(layers.Conv2D(128, (3, 3),strides=2, padding='same', activation='relu'))

model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(12, activation='softmax'))

model.summary()
'''
'''
#Model 4 -- Model 3 but w/ batchnorm 
model = models.Sequential()
model.add(BatchNormalization()
model.add(layers.Conv2D(16, (2, 2),strides=2,  activation='relu', input_shape=(trainData.shape[1], trainData.shape[2], 1)))
model.add(BatchNormalization())#
model.add(layers.MaxPooling2D((2, 2)))#1
model.add(layers.Conv2D(32, (3, 3),strides=2,  activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))#1
model.add(layers.Conv2D(64, (3, 3),strides=2,  padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))#1
model.add(layers.Conv2D(128, (3, 3),strides=2, padding='same', activation='relu'))

model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(12, activation='softmax'))
model.summary() 
'''
'''
#Model 4 -- Model 3 but w/ batchnorm 
model = models.Sequential()
model.add(BatchNormalization(input_shape=(trainData.shape[1], trainData.shape[2], 1)))
model.add(layers.Conv2D(16, (2, 2),strides=2,  activation='relu'))
model.add(BatchNormalization())#
model.add(layers.MaxPooling2D((2, 2)))#1
model.add(layers.Conv2D(32, (3, 3),strides=2, activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))#1
model.add(layers.Conv2D(64, (3, 3),strides=2, padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))#1
model.add(layers.Conv2D(128, (3, 3),strides=2, padding='same', activation='relu'))

model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(12, activation='softmax'))
model.summary() 
'''
'''
#Model 5 -- Model Architecture #Stride/downsampling & 1 batchNorm & Orig Flatten & FCL
model = models.Sequential()
model.add(layers.Conv2D(16, (2, 2),strides=2,  activation='relu', input_shape=(trainData.shape[1], trainData.shape[2], 1)))
model.add(BatchNormalization())#
model.add(layers.MaxPooling2D((2, 2)))#1
model.add(layers.Conv2D(32, (3, 3),strides=2,  activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))#1
model.add(layers.Conv2D(64, (3, 3),strides=2,  padding='same', activation='relu')) #added the padding = 'same' to prevent against matrix from getting too small????
model.add(layers.MaxPooling2D((2, 2)))#1
model.add(layers.Conv2D(128, (3, 3),strides=2, padding='same', activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(12, activation='softmax'))
'''

'''
#Model 6 -- Model 5 but w/ batchnorm for preprocessing 
model = models.Sequential()
model.add(BatchNormalization(input_shape=(trainData.shape[1], trainData.shape[2], 1)))
model.add(layers.Conv2D(16, (2, 2),strides=2,  activation='relu', input_shape=(trainData.shape[1], trainData.shape[2], 1)))
model.add(BatchNormalization())#
model.add(layers.MaxPooling2D((2, 2)))#1
model.add(layers.Conv2D(32, (3, 3),strides=2,  activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))#1
model.add(layers.Conv2D(64, (3, 3),strides=2,  padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))#1
model.add(layers.Conv2D(128, (3, 3),strides=2, padding='same', activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(12, activation='softmax'))
'''

model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'], experimental_run_tf_function=False)

numEpochs=30
history = model.fit(train_dataset, epochs=numEpochs, validation_data=test_dataset, batch_size =batchSize )

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label = 'Testing Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()