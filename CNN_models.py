from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Convolution2D,MaxPooling2D
from keras.optimizers import SGD
import data_gen
import os

class Network():
    def __init__(self,model_type=0):
        curdir = os.path.abspath(os.curdir)
        self.inp_w=128
        self.inp_h=128
        self.inp_c=3
        self.inp_shape=(self.inp_w,self.inp_h,self.inp_c)
        self.classes=['RG','R','PR','PRG']
        print(self.classes)
        self.classes_num = len(self.classes)
        self.model_type=model_type
        self.model = simple_convolution_model(self.inp_shape, self.classes_num)

    def init_model(self):
        if self.model_type<1:
            self.model=simple_convolution_model(self.inp_shape,self.classes_num)
        elif self.model_type==1:
            self.model=vgg_mod(self.inp_shape,self.classes_num)
        else:
            self.model=vgg16_mod(self.inp_shape,self.classes_num)

    def predict_classes(self,dat):
        print(dat.shape)
        print(self.model)
        print(self.model.predict_classes(dat))
        return self.model.predict_classes(dat)

    def predict_classes_as_text(self,inp_arrays):
        print (inp_arrays.shape)
        pr_classes= self.model.predict_classes(inp_arrays)
        result = []
        print(self.classes)
        for cl in pr_classes:
            result.append(self.classes[cl])
        return result

    def load_data(self,data_path,batch_size=64,batches_num=1):
        return data_gen.get_data(data_path,(self.inp_w,self.inp_h),batch_size,batches_num)

    def load_weights(self,path):
        self.model.load_weights(path)

    def train_model(self,data_path,batch_size=64,nb_epoch=25):
        (self.train_dat_X,self.train_dat) = self.load_data(data_path,batch_size,nb_epoch)
        (self.val_dat_X,self.val_dat_Y) = self.load_data(data_path,batch_size,nb_epoch)
        if( (len(self.train_dat_X)>0) and (len(self.train_dat_Y)>0)):
            self.model.fit(self.cust_img_X,self.cust_img_Y,batch_size=batch_size,nb_epoch=nb_epoch)
        else:
            print 'no data loaded'

    def save_weights(self,path):
         self.model.save_weights(path)

    def evaluate_model(self,data_path):
        (test_dat_X,test_dat_Y) = self.load_data(data_path)
        print(self.model.predict_classes(test_dat_X))
        scores = self.model.evaluate(test_dat_X, test_dat_Y, verbose=0)
        print("Tested accuracy: %.2f%%" % (scores[1]*100))


#2 x (2 x Convolution -> MaxPooling -> Dropout(0.25)) -> MPL(512) -> Dropout(0.5)-> softmax
def simple_convolution_model(input_shape,nb_classes):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same',input_shape=input_shape,dim_ordering='tf',activation='relu'))
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(nb_classes, activation='softmax'))

    sgd = SGD(lr=0.001, decay=1e-7, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
    return model

def vgg_mod(input_shape, nb_classes):
    model = Sequential()
    model.add(Convolution2D(32,3,3, border_mode='valid',input_shape =input_shape,dim_ordering='tf'))
    model.add(Activation('relu'))
    model.add(Convolution2D(32,3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64,3,3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64,3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    sgd= SGD (lr=0.001,decay=1e-6,momentum=0.9,nesterov=True)
    model.compile(loss='categorical_crossentropy',optimizer=sgd, metrics=['accuracy'])
    return model

def vgg16_mod(inp_shape,nb_classes):
    model=Sequential()
    # Block 1
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1',input_shape=inp_shape,dim_ordering='tf'))
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1'))
    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block 3
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1'))
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2'))
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block 4
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1'))
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2'))
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    # Block 5
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1'))
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2'))
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

    # Classification block
    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, activation='relu', name='fc1'))
    model.add(Dense(4096, activation='relu', name='fc2'))
    model.add(Dense(nb_classes, activation='softmax', name='predictions'))

    sgd= SGD (lr=0.001,decay=1e-6,momentum=0.9,nesterov=True)
    model.compile(loss='categorical_crossentropy',optimizer=sgd, metrics=['accuracy'])
    return model

#def Inception_mod
