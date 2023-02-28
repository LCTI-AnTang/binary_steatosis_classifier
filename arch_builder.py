"""
@author: pvianna
"""

from keras.applications import ResNet50, VGG16, InceptionV3
from keras import optimizers
from keras import backend as K
from keras.layers import Dropout
from keras.models import Model

def import_arch(architecture, in_shape, transfer_learning, classes):
    '''
    architecture: a string indicating the pre-trained architecture to use.
    in_shape: a tuple representing the input shape of the data.
    transfer_learning: a string indicating the file path of pre-trained weights to use for transfer learning (optional).
    classes: an integer representing the number of output classes.
    '''
    
    K.set_image_data_format('channels_last')
    if architecture=='InceptionV3':
        model = InceptionV3(include_top=True, weights=None, input_shape = in_shape, classes=classes)

    if architecture=='Resnet50':
        model = ResNet50(include_top=True, weights=None, input_shape = in_shape, classes=classes)
       
    if architecture=='VGG16':
        model = VGG16(include_top=True, weights=None, input_shape = in_shape, classes=classes)
    
    if architecture=='VGG16-dropout':
        # Adding two dropout layers before prediction
        model = VGG16(include_top=True, weights=None, input_shape = in_shape, classes=classes)
        fc1 = model.layers[-3]
        fc2 = model.layers[-2]
        pred = model.layers[-1]
        dropout1 = Dropout(0.5)
        dropout2 = Dropout(0.5)
        x = dropout1(fc1.output)
        x = fc2(x)
        x = dropout2(x)
        predictors = pred(x)
        model = Model(input=model.input, output=predictors)
        
    if transfer_learning:
        # Importing trained weights
        K.set_image_data_format('channels_first')
        model.load_weights(transfer_learning,by_name=True)
        print('Transfer learning weights were loaded to the model.')

    opt = optimizers.SGD(lr=1e-4)
    loss_f = 'binary_crossentropy'
    model.compile(loss=loss_f, optimizer = opt, metrics = ['accuracy'])
    return model
