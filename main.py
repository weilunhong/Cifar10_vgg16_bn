import sys
import os
import argparse
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QPlainTextEdit,QLineEdit,QLabel,QMessageBox
import numpy as np
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from six.moves import cPickle 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import vgg
import train


class Window(QWidget):

    def __init__(self, *args, **kwargs):
        super(Window, self).__init__(*args, **kwargs)
        self.setGeometry(400, 400, 330, 400)
        self.setWindowTitle("Training Cifar10 Classifier Using VGG16")
        layout = QVBoxLayout(self) #set objects with vertical layout
        layout.addWidget(QPushButton('1. Show Train Images', self, objectName='Btn1', pressed=self.onPressed1))
        layout.addWidget(QPushButton('2. Show Hyperparameters', self, objectName='Btn2', pressed=self.onPressed2))
        layout.addWidget(QPushButton('3. Show Model Structure', self, objectName='Btn3', pressed=self.onPressed3))
        layout.addWidget(QPushButton('4. Show Accuracy', self, objectName='Btn4', pressed=self.onPressed4))
        #layout.addWidget(QLabel('Input number to choose the test image:', self, objectName='Label'))
        self.InputNum = QLineEdit('', self, objectName='InputNum')
        self.InputNum.setPlaceholderText('Input number to choose the test image:')
        layout.addWidget(self.InputNum)
        layout.addWidget(QPushButton('5. Test', self, objectName='Btn5', pressed=self.onPressed5))
       
        self.classes = ('plane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck')
        
    def onPressed1(self):
        #Show Train Images
        data_path = str('./data/cifar-10-batches-py/data_batch_')+str(np.random.choice(5)+1)
        #print(data_path)
        f = open(data_path, 'rb')
        datadict = cPickle.load(f,encoding='latin1')
        f.close()
        X = datadict["data"] 
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
        Y = np.array(Y)
        #Visualizing
        fig, axes1 = plt.subplots(1,10)
        for j in range(10):
            i = np.random.choice(range(len(X)))
            axes1[j].set_axis_off()
            axes1[j].imshow(X[i:i+1][0])
            axes1[j].set_title(self.classes[Y[i]])
        plt.tight_layout()
        plt.show()        
                
    def onPressed2(self):
        #Show Hyperparameters
        print("Hyperparameters:")
        train.Show_Hyperparameters()
        
    def onPressed3(self):
        #Show Model Structure
        train.Show_Model_Structure()

    def onPressed4(self):
        #Show Accuracy
        train.Show_Accuracy()
    
    def onPressed5(self):
        #Show test image & classify
        number = self.InputNum.text()
        if number.isdigit() == 0:
            print ("Plz input number in the bar")
            QMessageBox.information(self, "Something Wrong!", "Plz input number!!", QMessageBox.Yes | QMessageBox.Yes)
            return 0
        print(number)
        number = int(number) 
        if (number < 10000 and number >= 0):
            train.show_test_image(number,self.classes)
            return 0
        print("Input should < 10000 and >= 0")
        QMessageBox.information(self, "Something Wrong!", "Input should\n>=0 and <10000", QMessageBox.Yes | QMessageBox.Yes)
        return 0    
        

if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())