
# coding=utf-8
# Developer: Mert Karacam
import numpy as np
import time
import json
import torch
import argparse

from PIL import Image
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from collections import OrderedDict

# Give user the option to choose data directory, whether to use the GPU or not, and the model architecture
parser = argparse.ArgumentParser(description='Artificial Neural Networks Train Model')
parser.add_argument('--data_directory', type=str, default='flowers', help='Directory where training and testing images come from')
parser.add_argument('--save_directory', type=str, default='my_checkpoint.pth', help='Directory where the checkpoints will be saved')
parser.add_argument('--gpu', type=bool, default=False, help='Wherther to use GPU for training or not')
parser.add_argument('--arch', type=str, default='VGG', help='Choose the architecture, VGG or densenet')
parser.add_argument('--lr', type=float, default=0.0002, help='Choose the learining rate for the model')
parser.add_argument('--hidden_unit', type=int, default=500, help='Choose the number of hiddent units in the model')
parser.add_argument('--epochs', type=int, default=3, help='Choose the training epochs for the model')

# Store user options in the "args" variable
args = parser.parse_args()   

data_dir = args.data_directory
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
transforms_training = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

transforms_validation = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

transforms_testing = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder 
train_data = datasets.ImageFolder(train_dir, transform=transforms_training)
validate_data = datasets.ImageFolder(valid_dir, transform=transforms_validation)
test_data = datasets.ImageFolder(test_dir, transform=transforms_testing)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(validate_data, batch_size=32)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    
# TODO: Build the model use the chosen architecture
if args.arch == 'VGG':
    model = models.vgg19(pretrained=True)
    # Get number of input nodes
    num_ftrs = model.classifier[0].in_features
else:
    model = models.densenet161(pretrained=True)
    num_ftrs = model.classifier.in_features
    
# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

# Build classifier
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(num_ftrs, 2000)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(p=0.3)),
                          ('fc2', nn.Linear(2000, args.hidden_unit)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(p=0.3)),
                          ('fc3', nn.Linear(args.hidden_unit, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
model.classifier = classifier

# Set negative log loss as the criteria
criterion = nn.NLLLoss()
# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr) # Need to give the option to change learn rate

# Train the network

epochs = args.epochs
steps = 0
running_loss = 0
print_every = 40

device = torch.device('cuda:0' if args.gpu else 'cpu')

model.to(device)

for e in range(epochs):
    # Model in training mode, dropout is on
    model.train()
    for inputs, labels in iter(trainloader):
        steps += 1
        
        # Wrap images and labels in Variables so we can calculate gradients
        inputs = Variable(inputs)
        targets = Variable(labels)
        
        # Move input and label tensors to the GPU
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        output = model.forward(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            # Model in inference mode, dropout is off
            model.eval()
            
            accuracy = 0
            test_loss = 0
            for ii, (inputs, labels) in enumerate(validloader):
                
                # Set volatile to True so we don't save the history
                inputs = Variable(inputs, volatile=True)
                labels = Variable(labels, volatile=True)
                
                inputs, labels = inputs.to(device), labels.to(device)
                
                output = model.forward(inputs)
                test_loss += criterion(output, labels).data[0]
                
                ## Calculating the accuracy 
                # Model's output is log-softmax, take exponential to get the probabilities
                ps = torch.exp(output).data
                # Class with highest probability is our predicted class, compare with true label
                equality = (labels.data == ps.max(1)[1])
                # Accuracy is number of correct predictions divided by all predictions, just take the mean
                accuracy += equality.type_as(torch.FloatTensor()).mean()
                
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                  "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
            
            running_loss = 0
            
            # Make sure dropout is on for training
            model.train()        
                
# TODO: Do validation on the test set
model.eval()

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

# TODO: Save the checkpoint
model.class_to_idx = train_data.class_to_idx

checkpoint = {'arch':args.arch,
              'input':num_ftrs,
              'output':102,
              'epochs':args.epochs,
              'learning_rate':args.lr,
              'dropout':0.3,
              'batch_size':32,
              'classifier':classifier,
              'state_dict':model.state_dict(),
              'optimizer':optimizer.state_dict(),
              'class_to_idx': model.class_to_idx}
torch.save(checkpoint, args.save_directory)