#%% Preprocess and download datasets

# Variables
MODEL = 'LeNet'

DATASET_PATH = './data/'
RESULT_PATH = './result_{}/'.format(MODEL)

EPOCHS = 150
BATCH_SIZE = 128

LR = 1e-7
MOMENTUM = 0.9

# Import the necessary libraries
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
import time
from tqdm import tqdm
import torch.utils.data as data

# Loading the MNIST dataset
from torchvision import datasets, transforms
import cupy as cp

# Define a transform to normalize the data
if MODEL == 'LeNet' or MODEL == 'VGG16' : # 28x28
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5), (0.5))
                                    ])
else: # 224x224
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((224, 224)),
                                    transforms.Normalize((0.5), (0.5))
                                    ])

# Download and load the training data
random_seed = 42
torch.manual_seed(random_seed)

trainset = datasets.MNIST(DATASET_PATH, download = True, train = True, transform = transform)
testset = datasets.MNIST(DATASET_PATH, download = True, train = False, transform = transform)

# trainset_size = len(trainset)
# testset_size = len(testset)

# indices = torch.randperm(trainset_size)[:trainset_size // 10]  # 取百分之一的大小
# indices2 = torch.randperm(testset_size)[:testset_size // 1]

# trainset = data.Subset(trainset, indices)
# testset = data.Subset(testset, indices2)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE, shuffle = True)
testloader = torch.utils.data.DataLoader(testset, batch_size = BATCH_SIZE, shuffle = True)


#%% Define the model and the optimizer
from alexnet import AlexNet
from lenet import LeNet
from resnet import resnet18
from optim import SGD
from vgg16 import VGG16




if MODEL == 'LeNet':
    LR = 2e-6
    model = LeNet(input_channel=1, output_class=10)
elif MODEL == 'AlexNet':
    LR = 1e-8
    model = AlexNet(input_channel=1, output_class=10)
elif MODEL == 'ResNet18':
    LR = 5e-5
    model = resnet18(input_channel=1, output_class=10)
elif MODEL == 'VGG16':
    LR = 1e-5
    model = VGG16(input_channel=1, output_class=10)
else:
    print('No such model {}, please indicate another'.format(MODEL))
optimizer = SGD(model, lr=LR, momentum=MOMENTUM)

#%% Training and evaluating process

# Initiate the timer to instrument the performance
timer_start = time.process_time_ns()
epoch_times = [timer_start]

train_losses, test_losses, accuracies = [], [], []

for e in range(EPOCHS):
    running_loss = 0
    print("Epoch: {:03d}/{:03d}..".format(e+1, EPOCHS))

    # Training pass
    print("Training pass:")
    for data in (tbar := tqdm(trainloader, total=len(trainloader))):
        images, labels = data[0].numpy(), data[1].numpy()
        images = cp.asarray(images)
        labels = cp.asarray(labels)

        prob, loss = model.forward(images, labels, train_mode=True)
        model.backward(loss)
        optimizer.step()
        
        totloss = sum(loss)
        running_loss += totloss

        tbar.set_description("Running loss {:.2f}".format(totloss))
    
    # Testing pass
    print("Validation pass:")
    test_loss = 0
    accuracy = 0
    for data in tqdm(testloader, total=len(testloader)):
        images, labels = data[0].numpy(), data[1].numpy()
        images = cp.asarray(images)
        labels = cp.asarray(labels)
        prob, loss = model.forward(images, labels, train_mode=False)
        test_loss += cp.sum(loss)
        
        top_class = cp.argmax(prob, axis=1)
        equals = (top_class == labels)
        accuracy += cp.mean(equals.astype(float))

    train_losses.append(running_loss/len(trainloader))
    test_losses.append(test_loss/len(testloader))
    accuracies.append(accuracy/len(testloader))

    
    epoch_times.append(time.process_time_ns())
    print("Train loss: {:.3f}..".format(running_loss/len(trainloader)),
          "Test loss: {:.3f}..".format(test_loss/len(testloader)),
          "Test Accuracy: {:.3f}".format(accuracy/len(testloader)),
          "Cur time(ns): {}".format(epoch_times[-1]))


#%% Evaluation

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(train_losses, label="Train loss")
ax.plot(test_losses, label="Validation loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Cross Entropy Loss")
ax.legend()
ax2 = ax.twinx()
ax2.plot(np.array(accuracies)*100, label="Accuracy", color='g')
ax2.set_ylabel("Accuracy (%)")
plt.title("Training procedure")
plt.savefig(RESULT_PATH + 'training_proc.png', dpi = 100)

proc_results = {
    'train_losses': train_losses,
    'test_losses': test_losses,
    'epoch_times': epoch_times,
    'accuracies': accuracies,
}

# print(proc_results)
with open(RESULT_PATH + 'torch_results.json', 'w+') as f:
    json.dump(proc_results, f)

