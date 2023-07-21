import scipy
from keras.layers import Convolution2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
test_datagen =ImageDataGenerator(rescale = 1)

x_train = train_datagen.flow_from_directory('C:\\Users\\prakash\\Downloads\\Brain tumor\\Brain tumor\\dataset\\dataset\\Brain_Tumor_Train_Test_Folders\\train_set',target_size = (64,64),batch_size = 32, class_mode = 'binary')
x_test =  test_datagen.flow_from_directory('C:\\Users\\prakash\\Downloads\\Brain tumor\\Brain tumor\\dataset\\dataset\\Brain_Tumor_Train_Test_Folders\\test_set',target_size = (64,64),batch_size = 32, class_mode = 'binary')

print(x_train.class_indices)
print(x_test.class_indices)
model = Sequential()
model.compile(optimizer='adam', loss='binary_crossentropy')
model.add(Convolution2D(32,(3,3),input_shape = (64,64,3),activation = 'relu')) 
#no. of feature detectors, size of feature detector, input shape and activation function
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Flatten())
model.add(Dense(units= 40 ,activation = 'relu'))
model.add(Dense(units = 1,activation = 'softmax'))
model.fit(x_train)
model.save("braintumor1.h5")