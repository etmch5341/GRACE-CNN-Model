 #Import programs, packages, and libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import matplotlib.ticker as ticker
import numpy as np
import os, sys, math
from PIL import Image
from numpy.random import seed
import random
from matplotlib.ticker import MaxNLocator

#Check for correct tensorflow update version 
print(tf.version.VERSION) #I ran with 2.4.0 successfully
historyDir = '/Users/etch5/Desktop/GRACE Python/NNhistories/'
checkpoint_path = "/Users/etch5/Desktop/GRACE Python/NNCheckpointTest_022721/"
modelsDir = '/Users/etch5/Desktop/GRACE Python/NNModels/'

seed_value = 7

os.environ['PYTHONASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

os.environ['PYTHONASHSEED']=str(seed)

#Identify and load data sets
pathToSampleImage = '/Users/etch5/Desktop/GRACE Python/SEES_Training_Imgs/SA/5/DDK2_AOHIS_199905_avg_AOHISZero_120_0_WH_trainingImg.SA.jpg'
sampleImage = Image.open(pathToSampleImage)
print('Image size:')
print(sampleImage.size)
ImgWidth=sampleImage.size[0]
ImgHeight=sampleImage.size[1]

ImgWidth_resample = math.floor(ImgWidth/10)
ImgHeight_resample = math.floor(ImgHeight/10)
batchSize = 32

#Training phase of neural network
train_dataset= \
    tf.keras.preprocessing.image_dataset_from_directory('/Users/etch5/Desktop/GRACE Python/SEES_Training_Imgs/SA', #change for each user
     labels='inferred', #data in directory are sorted into folders for each class
     label_mode='categorical', #in multiple categories,change to int if u put numbers
     class_names=['1','2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],
     color_mode='rgb',
     batch_size=batchSize,
     image_size=(ImgHeight_resample, ImgWidth_resample)
     )

#Testing phase of neural network
test_dataset=tf.keras.preprocessing.image_dataset_from_directory('/Users/etch5/Desktop/GRACE Python/SEES_Testing_Imgs/SA', #change for each user
     labels='inferred', #data in directory are sorted into folders for each class
     label_mode='categorical', #in multiple categories,change to int if u put numbers
     class_names=['1','2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],
     color_mode='rgb',
     batch_size=batchSize,
     image_size=(ImgHeight_resample, ImgWidth_resample)
     )


print('Testdataset:', test_dataset)

#In this we will load ALL images
#Define elements of Neural Network (Layers, etc.)
def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (2, 2), padding='same', activation='relu', input_shape=(ImgHeight_resample, ImgWidth_resample, 3)))
    model.add(layers.experimental.preprocessing.Rescaling(1./255))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))

    model.add(layers.AveragePooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(12, activation = 'softmax'))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'], experimental_run_tf_function=False)
    return model


#Create basic model instance
model = create_model()

#Display model architecture
model.summary()
numEpochs=3
history = model.fit(train_dataset, epochs=numEpochs, validation_data=test_dataset, batch_size =batchSize )
model.save(modelsDir+'my_model')


trainingHistory_arr = np.array(history.history['accuracy'])
testingHistory_arr = np.array(history.history['val_accuracy'])
np.save(historyDir+'trainingHistory_p1Model.npy',trainingHistory_arr,allow_pickle=True,fix_imports=True)
np.save(historyDir+'testingHistory_p1Model.npy',testingHistory_arr,allow_pickle=True,fix_imports=True)

epochsXAxis = np.arange(1,numEpochs+1)
np.save(historyDir+'epochs_p1Model.npy', epochsXAxis,allow_pickle=True,fix_imports=True)


f,ax = plt.subplots(1)
plt.plot(epochsXAxis,trainingHistory_arr,'o-',color='indigo',label='Training Accuracy')
plt.plot(epochsXAxis,testingHistory_arr, 'o-',color='darkgreen',label = 'Testing Accuracy') #Change Information in this line
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.7)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
plt.ylim([0, 1])
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.legend(loc='lower right')
plt.savefig(historyDir+'savedModel.png')


plt.show()


#################################################
#################################################
#################################################
#################################################




