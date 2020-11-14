"# Cifar10_vgg16_bn" 

Open **hyperparameter.txt** to set number of epochs ,training batchsize and learning rate.\n
Enter **python train.py** to train (you can also add **--resume** to continue the last time training\n
Enter **python main.py** to get following functions:\n
    1.Show Cifar10 training dataset and randomly show 10 images and labels respectively\n
    2.Print out training hyperparameters \n
    3.Show model structure\n
    4.Show training loss and accuracy\n
    5.Choose one image from test images then show it and estimate it\n
\n
Environment:\n
python                    3.6.12\n
pytorch                   1.3.0\n
cudatoolkit               10.1\n
torchvision               0.4.2\n
matplotlib                3.1.1\n
numpy                     1.19.3\n
