#!/usr/bin/env python
# coding: utf-8

# In[1]:


#imports
import torch
from glob import glob
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.dataset import Dataset
from torchvision import models, transforms
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from livelossplot import PlotLosses
import time


# In[2]:


def freeze_all(model_params):
    '''
    Freeze entire network - taken from Skuldur.
    '''
    for param in model_params:
        param.requires_grad = False
        
def unfreeze_all(model_params):
    '''
    Unfreeze entire network - taken from Skuldur.
    '''
    for param in model_params:
        param.requires_grad = True
        
def load_image(filename) :
    '''
    Load an image using PIL - taken from Skuldur.
    '''
    img = Image.open(filename)
    img = img.convert('RGB')
    return img


# In[3]:


#get the data - taken from Skuldur.
filenames = glob('./datasets/images/*.jpg')
classes = set()

data = []
labels = []

# Load the images and get the classnames from the image path
for image in filenames:
    class_name = image.rsplit("/", 1)[1].rsplit('_', 1)[0]
    classes.add(class_name)
    img = load_image(image)

    data.append(img)
    labels.append(class_name)

# convert classnames to indices
class2idx = {cl: idx for idx, cl in enumerate(classes)}        
labels = torch.Tensor(list(map(lambda x: class2idx[x], labels))).long()

data = list(zip(data, labels))


# In[4]:


class PetDataset(Dataset):
    '''
    Dataset to serve individual images to our model - taken from Skuldur.
    '''
    
    def __init__(self, data, transforms=None):
        self.data = data
        self.len = len(data)
        self.transforms = transforms
    
    def __getitem__(self, index):
        img, label = self.data[index]
        
        if self.transforms:
            img = self.transforms(img)
            
        return img, label
    
    def __len__(self):
        return self.len


# Since the data is not split into train and validation datasets we have to 
# make sure that when splitting between train and val that all classes are represented in both
class Databasket:
    '''
    Helper class to ensure equal distribution of classes
    in both train and validation datasets - taken from Skuldur.
    '''
    
    def __init__(self, data, num_cl, val_split=0.2, train_transforms=None, val_transforms=None):
        class_values = [[] for x in range(num_cl)]
        
        # create arrays for each class type
        for d in data:
            class_values[d[1].item()].append(d)
            
        self.train_data = []
        self.val_data = []
        
        # put (1-val_split) of the images of each class into the train dataset
        # and val_split of the images into the validation dataset
        for class_dp in class_values:
            split_idx = int(len(class_dp)*(1-val_split))
            self.train_data += class_dp[:split_idx]
            self.val_data += class_dp[split_idx:]
            
        self.train_ds = PetDataset(self.train_data, transforms=train_transforms)
        self.val_ds = PetDataset(self.val_data, transforms=val_transforms)


# In[5]:


#data transforms - taken from Skuldur.
# Apply transformations to the train dataset
train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# apply the same transformations to the validation set, with the exception of the
# randomized transformation. We want the validation set to be consistent
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])

databasket = Databasket(data, len(classes), val_split=0.2, train_transforms=train_transforms,
                        val_transforms=val_transforms)


# In[6]:


def decimate(tensor, m):
    """
    Decimate a tensor by a factor 'm', i.e. downsample by keeping every 'm'th value.
    This is used when we convert FC layers to equivalent Convolutional layers, BUT of a smaller size.
    Taken from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection.
    :param tensor: tensor to be decimated
    :param m: list of decimation factors for each dimension of the tensor; None if not to be decimated along a dimension
    :return: decimated tensor
    """
    assert tensor.dim() == len(m)
    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor = tensor.index_select(dim=d,
                                         index=torch.arange(start=0, end=tensor.size(d), step=m[d]).long())

    return tensor

def make_fcn_classifier(in_features, num_classes, convert_from_dense=False):
    '''
    This function converts a VGG network into a fully-convolutional classifier
    by replacing the linear head with a convolutional one.
    Inspired by:
    https://stackoverflow.com/questions/44146655/how-to-convert-pretrained-fc-layers-to-conv-layers-in-pytorch.
    '''
    model = models.vgg11_bn(pretrained=True)
    features = model.features
    fcLayers = nn.Sequential(
        # stop at last layer group
        *list(model.classifier.children())[:-1]
    )
    fc = fcLayers[0].state_dict()
    in_ch = in_features
    out_ch = fc["weight"].size(0)
    assert out_ch == 4096
    p = 0.4
    conv1 = nn.Conv2d(in_ch, 1024, kernel_size=3)
    #if we are converting a dense layer into convolutional,
    #then we reshape and decimate so that the shape fits.
    #this is the same procedure as in the SSD paper.
    if convert_from_dense:
        conv_fc6_weight = fc["weight"].view(out_ch, in_ch, 7, 7)
        conv_fc6_bias = fc["bias"]
        conv1_weights = decimate(conv_fc6_weight, m=[4, None, 3, 3])  # (1024, 512, 3, 3)
        conv1_bias = decimate(conv_fc6_bias, m=[4])  # (1024)
        conv1.load_state_dict({"weight": conv1_weights,
                               "bias": conv1_bias})
        
    return nn.Sequential(
        conv1,
        nn.BatchNorm2d(1024),
        nn.ReLU(inplace=True),
        nn.Dropout2d(p),
        nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        nn.Dropout2d(p),
        nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.Dropout2d(p),
        nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=5),
        nn.Flatten()
    )
    
def requires_grad(layer):
    '''
    Determines whether 'layer' requires gradients - taken from Skuldur.
    '''
    ps = list(layer.parameters())
    if not ps: return None
    return ps[0].requires_grad

def cnn_model(model, nc, convert_from_dense=False, init=nn.init.kaiming_normal_):
    '''
    This function has been modified from Skuldur to fit our purposes -
    it takes the VGG backbone and appends to it the new convolutional head,
    and then freezes all the pretrained layers 
    (potentially including the converted convolution) and initializes the rest.
    '''
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    
    # remove dense and freeze everything
    body = nn.Sequential(*list(model.children())[:-2])
    head = make_fcn_classifier(512, nc, convert_from_dense=convert_from_dense)
    
    model = nn.Sequential(body, head)
    
    # freeze the base of the model
    freeze_all(model[0].parameters())
    
    # initialize the weights of the head
    for i, child in enumerate(model[1].children()):
        if i == 0 and convert_from_dense:
            freeze(child)
            continue
        if isinstance(child, nn.Module) and (not isinstance(child, bn_types)) and requires_grad(child): 
            init(child.weight)
    
    return model


# In[7]:


#data setup - taken from Skuldur.
train_indices = list(range(len(databasket.train_ds)))
test_indices = list(range(len(databasket.val_ds)))

train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

bs = 32

# Basic dataloader to retrieve mini-batches from the datasets
trainloader = DataLoader(databasket.train_ds, batch_size=bs,
                          sampler=train_sampler, shuffle=False, num_workers=0)
testloader = DataLoader(databasket.val_ds, batch_size=bs,
                         sampler=test_sampler, shuffle=False, num_workers=0)

dataloaders = {
    "train": trainloader,
    "val": testloader
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# In[8]:


def train_model(model, criterion, optimizer, scheduler, num_epochs):
    '''
    This function takes in a model, a loss function,
    an optimizer, a learning-rate scheduler, and
    a number of epochs, and runs the train loop
    while plotting live training results.
    Returns best model at the end of training.
    Based on liveloss example.
    '''
    liveloss = PlotLosses()
    model = model.to(device)
    since = time.time()

    for epoch in range(num_epochs):
        logs = {}
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                #load data and labels
                inputs = inputs.to(device)
                labels = labels.to(device)
    
                #run data through model
                outputs = model(inputs)

                loss = criterion(outputs, labels)

                if phase == 'train':

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.detach() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.float() / len(dataloaders[phase].dataset)
            
            prefix = ''
            if phase == 'val':
                prefix = 'val_'

            logs[prefix + 'log loss'] = epoch_loss.item()
            logs[prefix + 'accuracy'] = epoch_acc.item()
            
        scheduler.step()
        
        liveloss.update(logs)
        liveloss.draw()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

