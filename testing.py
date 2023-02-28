"""
@author: pvianna
Code adapted from steatosis_dl.py
"""

import pandas as pd
from PIL import Image
import numpy as np
import math
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from arch_builder import import_arch #custom function
from steatosis_dl import preprocess_data #custom function

def testing_steatosis(path_to_set, architecture, tl, n_task, images_dir, resize_shape = False):
    
    print('Testing trained models')
    print('Dataset selected:', path_to_set)
    dataset = pd.read_csv(path_to_set, delimiter=',')
    dataset = dataset.dropna(subset=['steatosis']) #removing images without steatosis grades

    # Training images are loaded again in order to preprocess the test set
    train_set =  dataset.loc[dataset['Subset']=='train']
    test_set =  dataset.loc[dataset['Subset']=='test']

    test_labels = test_set['steatosis']

    #Importing all images to arrays: 
    train_images = []
    test_images = []
    
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
        
    for j in range(len(test_set)):
        image_temp = Image.open(images_dir+test_set.iloc[j]['filename']).convert('RGB')
        if resize_shape:
            image_temp = np.array(image_temp.resize(resize_shape))
        else:
            image_temp = np.array(image_temp)

        image_temp -= image_temp.min()
        image_temp = image_temp / image_temp.max()

        test_images.append(image_temp)  

    x_train = np.array(train_images)
    x_test = np.array(test_images)
        
    print('Test dataset shape:', x_test.shape)

    #Standardizing test set with training set statistics
    #This is the reason why we reloaded the training data
    X_test = preprocess_data(x_test, x_train)

    n_task = [n_task]
    if n_task[0]==9: n_task=[0,1,2]

    for task in n_task:
    #Transforming steatosis labels into binary classes
        if task==0: #0 vs 1-2-3
            y_test = test_labels.replace({1.0:1, 2.0:1, 3.0:1})
            label0 = 'S0'
            label1 = '>=S1'
            
        if task==1: #0-1 vs 2-3
            y_test = test_labels.replace({1.0:0, 2.0:1, 3.0:1})
            label0 = '<=S1'
            label1 = '>=S2'
            
        if task==2: #0-1-2 vs 3
            y_test = test_labels.replace({1.0:0, 2.0:0, 3.0:1})
            label0 = '<=S2'
            label1 = 'S3'
    
        print('Classification task: %s vs %s' %(label0,label1))
    
        #Defining network         
        classes = 2 #binary classification
        network = import_arch(architecture, in_shape=x_test.shape[1:], transfer_learning=None, classes=classes)
        print(network.summary())
    
        saved_model_path = '/output/'
        name = 'Test_' + architecture + '_Task%i' %(task)

        #Loading model weights
        if tl == 'None': print('No weights loaded for test')
        network.load_weights(tl)
        
        #Get test results
        predictions = network.predict(X_test, verbose=0)
        probs = np.array(predictions)[:,1]
        
        no_skill = [0 for _ in range(len(y_test))] #random guesser ROC
        ns_fpr, ns_tpr, _ = roc_curve(y_test, no_skill)
        fpr, tpr, th = roc_curve(y_test,probs)
        roc_set = pd.DataFrame([th,fpr,tpr]).T
        roc_set.to_csv(saved_model_path+name+'_ROC_Results.csv')
            
        model_auc = roc_auc_score(y_test, probs)
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
        print('Threshold: Highest Youden index')
        J = np.argmax(tpr-fpr)
        ix = th[J]
        y_pred = []
        print('Sensitivity and Specificity from the ROC Curve:', round(tpr[J],4), round(1-fpr[J],4))

        for values in range(len(probs)):
            y_pred.append(math.floor(probs[values]) if probs[values]<ix else math.ceil(probs[values]))
                
        #Creating a new file with the outputs of the network
        dummy_set = test_set.copy()
        dummy_set['predictions'] = y_pred
        dummy_set['Label1_conf'] = probs
        dummy_set['task_labels'] = y_test
        dummy_set = dummy_set.fillna('no_info') #for empty rows
        dummy_set.to_csv(saved_model_path+name+'_Output_set.csv')
     
        report = classification_report(y_test,y_pred,target_names=[label0,label1], output_dict=True)
        confmat = confusion_matrix(y_test,y_pred)
    
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
