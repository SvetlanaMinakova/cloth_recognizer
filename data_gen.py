from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)


def get_data(path,size,batch_size,batches_num):
        train_generator = train_datagen.flow_from_directory(
        path,
        target_size=size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=get_classes(path))

        dat = train_generator.next()
        X=dat[0]
        Y=dat[1]
        if(batches_num>1):
                for i in range(batches_num-1):
                        dat = train_generator.next()
                        X=np.vstack((X,dat[0]))
                        Y=np.vstack((Y,dat[1]))
        return (X,Y)


def get_classes(path):
        return os.listdir(path)