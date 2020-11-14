'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import argparse
from utils import progress_bar
import vgg  

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

f = open("./hyperparameter.txt",mode = 'rt')
name = []
para = np.zeros(3)
for i in range(3):
    line = f.readline()
    name,para[i] = line.split()
    #print(name,para[i])
f.close()
# Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = vgg.vgg16_bn()
net = net.to(device)
if device == 'cuda':
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=para[1],
                      momentum=0.9, weight_decay=5e-4)
#optimizer = optim.Adam(net.parameters(),lr = para[1])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(para[2]/4), gamma=0.2)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    print (train_loss/(batch_idx+1),(100.*correct/total))
    return  train_loss/(batch_idx+1),(100.*correct/total)
    

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
       
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'optimizer': optimizer, #
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
    return acc
    
    
def Show_Hyperparameters():
    print("Batch size: ",int(para[0]))
    print("Optimizer: ",optimizer)
    
def Show_Model_Structure():
    print(net)
    
    
def Show_Accuracy():
    train_acc_array = np.load("train_acc.npy")
    test_acc_array = np.load("test_acc.npy")
    loss_array = np.load("loss.npy")
    fig, ax = plt.subplots(2,1)
    ax[0].set_title("Accuracy")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("%")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("loss")
    ax[0].plot(range(0,len(train_acc_array)),train_acc_array,label = "Train Accuracy")
    ax[0].plot(range(0,len(test_acc_array)),test_acc_array,label = "Test Accuracy")
    ax[1].plot(range(0,len(loss_array)),loss_array,label = "Loss")
    ax[0].legend()
    ax[1].legend()
    plt.tight_layout()
    plt.savefig("./accuracy.png",dpi=1000)
    plt.show()

def show_test_image(number,classes):
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    transform_test = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=False, transform=transform_test)
    img, label = testset[number]
    img_show = img.permute(1, 2, 0)
    img = transforms.Normalize(mean = (0.4914, 0.4822, 0.4465), std = (0.2023, 0.1994, 0.2010))(img)
    img = img.unsqueeze(0).cuda()
    net.eval()

    with torch.no_grad():           
        outputs = net(img)
        soft = torch.nn.Softmax(dim=1)
        soft_out = soft(outputs)
        prediction = soft_out.argmax(dim=1).item()
        prob = soft_out[0].cpu().tolist()
        #print(outputs)
        #print(soft_out)
        
    fig, axes2 = plt.subplots(1,2)
    #print(prediction)
    axes2[0].imshow(img_show)
    axes2[0].set_axis_off()
    true_label = str("Label: "+classes[label])
    axes2[0].set_title(true_label)
    axes2[1].bar(classes,prob, width=0.5, bottom=None, align='center', color=['darkblue'])
    pred_label = str("Prediction: "+classes[prediction])
    axes2[1].set_title(pred_label)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__': ##multiporcessing should be done in main function in windows os
   
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    end_epoch = start_epoch+int(para[2])
    batch = int(para[0])

    # Data
    #print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch, shuffle=True, num_workers=1) #num_workers: cpu use how many thread to load data

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=20, shuffle=False, num_workers=1)

    classes = ('plane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck')
   
    loss_array = np.zeros([end_epoch],float)
    train_acc_array = np.zeros([end_epoch],float)
    test_acc_array = np.zeros([end_epoch],float)
###train &test with stepLR
    for epoch in range(start_epoch, end_epoch):
        scheduler.step()
        loss_array[epoch],train_acc_array[epoch] = train(epoch)
        test_acc_array[epoch] = test(epoch)
        
    fig, ax = plt.subplots(2,1)
    print(train_acc_array)
    print(test_acc_array)
    print(loss_array)
    np.save("train_acc",train_acc_array)
    np.save("test_acc",test_acc_array)
    np.save("loss", loss_array)
    ax[0].set_title("Accuracy")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("%")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("loss")
    ax[0].plot(range(0,end_epoch),train_acc_array,label = "Train Accuracy")
    ax[0].plot(range(0,end_epoch),test_acc_array,label = "Test Accuracy")
    ax[1].plot(range(0,end_epoch),loss_array,label = "Loss")
    ax[0].legend()
    ax[1].legend()
    plt.tight_layout()
    plt.savefig("./accuracy.png",dpi=1000)
    plt.show()
