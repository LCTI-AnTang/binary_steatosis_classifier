"""
@author: pvianna
"""
import matplotlib.pyplot as plt

def plot_acc_loss(training_metrics, validation_metrics, name):
    colours = ['b','r']
    markers = ['o','v'] 
    path_to_save = '/output/'

    tr_acc = training_metrics[0]
    tr_loss = training_metrics[1]
    plt.plot(tr_acc, colours[0]+markers[0],label='Acc')
    plt.plot(tr_loss, colours[1]+markers[0], label='Loss')
    plt.legend() 
    plt.xlabel('Epochs')
    plt.ylabel('Loss or Accuracy')
    plt.title('Training')
    figure = plt.gcf()
    figure.set_size_inches(16,9)
    plt.savefig(path_to_save+name+'_Training.png', dpi=300)
    plt.clf()
    
    vl_acc = validation_metrics[0]
    vl_loss = validation_metrics[1]
    plt.plot(vl_acc, colours[0]+markers[0],label='Acc')
    plt.plot(vl_loss, colours[1]+markers[0], label='Loss')
    plt.legend() 
    plt.xlabel('Epochs')
    plt.ylabel('Loss or Accuracy')
    plt.title('Validation')
    figure = plt.gcf()
    figure.set_size_inches(16,9)
    plt.savefig(path_to_save+name+'_Validation.png', dpi=300)
    plt.close()
    
    plt.plot(tr_acc, colours[0]+markers[0],label='Train - Acc')
    plt.plot(tr_loss, colours[1]+markers[0], label='Train - Loss')
    plt.plot(vl_acc, colours[0]+markers[1],label='Val - Acc')
    plt.plot(vl_loss, colours[1]+markers[1], label='Val - Loss')
    plt.legend() 
    plt.xlabel('Epochs')
    plt.ylabel('Loss or Accuracy')
    plt.title('Training and Validation')
    figure = plt.gcf()
    figure.set_size_inches(16,9)
    plt.savefig(path_to_save+name+'_Training and Validation.png', dpi=300)
    plt.close()
