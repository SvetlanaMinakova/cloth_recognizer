# cloth_recognizer
The project solves the problem of pattern recognition on the example of recognition of business clothes.
In total, 3 types of clothes (jacket, shirt, tie) and 4 class sets of clothes (jacket + shirt + tie, jacket + shirt, shirt + tie, jacket + tie) are recognized
The CNN_Models file contains implementations of convolutional neural networks that can be used to solve the recognition problem. Models of convolutional neural networks are implemented using the Keras framework avaliable on https://keras.io/
The data_gen file implements data augmentation for Training/Validation/Test DataSets creation from the sample dataset, part of which is avaliable in the img_pack archive.
The problem of localization of the desired clothes on the whole image is solved on the basis of the selectivesearch project avaliable on https://github.com/AlpacaDB/selectivesearch 
GUI is implemented on the basis of PyQt5
