"""
@author: pvianna
"""
import glob
import os
import pandas as pd
from PIL import Image
import numpy as np
import keras
import math
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from arch_builder import import_arch
import saveimgs #custom function

def preprocess_data(X, source):
    """
    The function computes the mean and standard deviation of the source data. 
    It then normalizes the input data X by subtracting the mean and dividing by the standard deviation.
    This function takes two input arguments:
    
    X:  input data that needs to be preprocessed.
    source: source data used to compute the mean and standard deviation for normalization.
    """
    mean = source.mean()
    std = source.std()
    X_p = X - mean
    X_p = X_p/std
    return X_p

def train_and_val(path_to_set, architecture, tl, n_task, images_dir, resize_shape = False):
    
    print('Training and Validation')
    print('Dataset selected:', path_to_set)
    dataset = pd.read_csv(path_to_set, delimiter=',')
    dataset = dataset.dropna(subset=['steatosis']) #removing images without steatosis grades

    # The dataset needs a "filename" column,
    # A "Subset" column defining training and validation images,
    # And a "steatosis" columns with grades in an ordinal scale from 0 to 3
    train_set =  dataset.loc[dataset['Subset']=='train']
    val_set =  dataset.loc[dataset['Subset']=='val']

    train_labels = train_set['steatosis']
    val_labels = val_set['steatosis']

    #Importing all images to arrays: 
    train_images = []
    val_images = []
    
    print('Loading images...')
    for i in range(len(train_set)):
        image_temp = Image.open(images_dir+train_set.iloc[i]['filename']).convert('RGB')
        if resize_shape:
            image_temp = np.array(image_temp.resize(resize_shape))
        else:
            image_temp = np.array(image_temp)

        #Normalizing each individual image
        image_temp -= image_temp.min()
        image_temp = image_temp / image_temp.max()

        train_images.append(image_temp)
        
    for j in range(len(val_set)):
        image_temp = Image.open(images_dir+val_set.iloc[j]['filename']).convert('RGB')
        if resize_shape:
            image_temp = np.array(image_temp.resize(resize_shape))
        else:
            image_temp = np.array(image_temp)

        image_temp -= image_temp.min()
        image_temp = image_temp / image_temp.max()

        val_images.append(image_temp)  

    x_train = np.array(train_images)
    x_val = np.array(val_images)
        
    print('Train dataset shape:', x_train.shape,
          '\nValidation dataset shape:', x_val.shape)

    #Standardizing training and validation sets with training set statistics
    X_train = preprocess_data(x_train, x_train)
    X_val = preprocess_data(x_val, x_train)

    n_task = [n_task]
    if n_task[0]==9: n_task=[0,1,2]

    for task in n_task:
    #Transforming steatosis labels into binary classes
        if task==0: #0 vs 1-2-3
            y_train = train_labels.replace({1.0:1, 2.0:1, 3.0:1})
            y_val = val_labels.replace({1.0:1, 2.0:1, 3.0:1})
            label0 = 'S0'
            label1 = '>=S1'
            
        if task==1: #0-1 vs 2-3
            y_train = train_labels.replace({1.0:0, 2.0:1, 3.0:1})
            y_val = val_labels.replace({1.0:0, 2.0:1, 3.0:1})
            label0 = '<=S1'
            label1 = '>=S2'
            
        if task==2: #0-1-2 vs 3
            y_train = train_labels.replace({1.0:0, 2.0:0, 3.0:1})
            y_val = val_labels.replace({1.0:0, 2.0:0, 3.0:1})
            label0 = '<=S2'
            label1 = 'S3'
    
        print('Classification task: %s vs %s' %(label0,label1))
    
        #Transforming steatosis labels vector to categorical
        classes = 2 #binary classification
        Y_train = keras.utils.to_categorical(y_train,classes)
        Y_val= keras.utils.to_categorical(y_val,classes)        
    
        #Defining transfer_learning
        tl_weights = None
        if tl != 'None':
            tl_weights = tl
            
        # Importing the model with given parameters
        network = import_arch(architecture, in_shape=x_train.shape[1:], transfer_learning=tl_weights, classes=classes)
        print(network.summary())
    
        # Setting up the output directory and file names for saved models and logs
        saved_model_path = '/output/'
        saved_model_filename = saved_model_path + architecture + '_Task%i' %(task) + '_{epoch:02d}_{val_acc:.4f}.hdf5'
        csv_logger_training = saved_model_path + architecture + '_Task%i' %(task) + '.csv'
        
        # Setting up the training parameters
        batch_size = 32
        number_of_epochs = 200
        patience = 10 #for early stopping
        if resize_shape == False:
            batch_size = batch_size//4 #to avoid memory allocation errors when images are too large
            
        # Defining the callbacks to be used during training
        model_checkpoint = ModelCheckpoint(saved_model_filename, monitor='val_loss', save_best_only=True, verbose=1)
        csv_logger = CSVLogger(csv_logger_training,append = True, separator=';')
        early_stopping = EarlyStopping(monitor='val_loss',patience = patience)
        
        # Training and validation
        history = network.fit(X_train, Y_train, batch_size=batch_size, epochs=number_of_epochs, verbose=2,
                            shuffle=True, callbacks= [model_checkpoint, csv_logger, early_stopping], validation_data = (X_val, Y_val))
    
        tr_acc = history.history['acc']
        tr_loss = history.history['loss']
        vl_acc = history.history['val_acc']
        vl_loss = history.history['val_loss']
    
        training_metrics = [tr_acc, tr_loss]
        validation_metrics = [vl_acc, vl_loss]
        name = architecture + '_Task%i' %(task)
        saveimgs.plot_acc_loss(training_metrics, validation_metrics, name) #custom function to plot loss and accuracy curves
    
        print('Training is complete. Evaluating...')
    
        
        #Loading model with lowest validation loss
        trained_models = glob.glob(saved_model_path+'/*.hdf5')
        best_model = sorted(trained_models,key=os.path.getmtime)[-1]
        network.load_weights(best_model)
        
        # Get validation results
        predictions = network.predict(X_val, verbose=0)
        probs = np.array(predictions)[:,1]
        
        # Getting ROC curves
        no_skill = [0 for _ in range(len(y_val))] #random guesser ROC
        ns_fpr, ns_tpr, _ = roc_curve(y_val, no_skill)
        fpr, tpr, th = roc_curve(y_val,probs)
        roc_set = pd.DataFrame([th,fpr,tpr]).T
        roc_set.to_csv(saved_model_path+name+'_ROC_Results.csv')
            
        model_auc = roc_auc_score(y_val, probs)
        plt.plot(ns_fpr, ns_tpr, linestyle='--')
        plt.plot(fpr, tpr, marker='.')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Plot for ' + name )
        figure = plt.gcf()
        figure.set_size_inches(16,9)
        plt.savefig(saved_model_path+'ROC_'+name+'.png', dpi=300)
        plt.close()

        print('Area Under the ROC Curve:', round(model_auc,4))               
        print('Threshold: Highest Youden index') #difference between the true positive rate (sensitivity) and the false positive rate (1 - specificity)
        J = np.argmax(tpr-fpr)
        ix = th[J]
        y_pred = []
        print('Sensitivity and Specificity from the ROC Curve:', round(tpr[J],4), round(1-fpr[J],4))

        # Assigning binary predictions based on highest Youden index threshold
        for values in range(len(probs)):
            y_pred.append(math.floor(probs[values]) if probs[values]<ix else math.ceil(probs[values]))
                
        #Creating a new file with the outputs of the network
        dummy_set = val_set.copy()
        dummy_set['predictions'] = y_pred
        dummy_set['Label1_conf'] = probs
        dummy_set['task_labels'] = y_val
        dummy_set = dummy_set.fillna('no_info') #for empty rows
        dummy_set.to_csv(saved_model_path+name+'_Output_set.csv')
        
        #Calculating classification metrics and saving them to a csv file
        report = classification_report(y_val,y_pred,target_names=[label0,label1], output_dict=True)
        confmat = confusion_matrix(y_val,y_pred)
    
        accuracy = report['accuracy']
        f1score = report[label1]['f1-score']
        sens = report[label1]['recall'] #the same as recall
        spec = report[label0]['recall']
        tn, fp, fn, tp = confmat.ravel()
        ppv = tp/(tp+fp) #the same as precision
        npv = tn/(fn+tn)
        results_file = pd.DataFrame(index=['accuracy', 'f1score', 'sens', 'spec', 'ppv', 'npv', 'auc'])
        metrics = [accuracy, f1score, sens, spec, ppv, npv, model_auc]
        results_file[architecture+'_Task%i' %(task)] = metrics
        
        results_file.to_csv(saved_model_path+name+'_Results.csv')
        print('Results saved.')
        
    return results_file
