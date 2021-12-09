#References 
#https://medium.com/secure-and-private-ai-writing-challenge/loading-image-using-pytorch-c2e2dcce6ef2

import os
import json
from random import randrange
import random


import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.models as models
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import tqdm 
from tqdm import trange
import numpy as np

from helper import imshow


import helper
import label_tags
EPOCHS = 1
BATCH_SIZE = 1

device =None
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'



class AdversarialNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fixed_input = nn.Parameter(torch.ones((200,200,3)), requires_grad = True)
        self.vgg16 = models.vgg16(pretrained = True)
    
    def forward(self,x):

        x = x.to(device)
        patch_size = randrange(120,200)
        split_upper = 224 - patch_size
        self.patch_map = torch.ones((patch_size,patch_size,3)).to(device)
        for i in range(patch_size):
            for c in range(patch_size):
                if ((i-(patch_size/2.0))**2) + ((c-(patch_size/2.0))**2) > ((patch_size/2.0)**2):
                    self.patch_map[i,c,:] = 0


        v_split_index = randrange(0,split_upper)
        v_before_padding = v_split_index
        v_after_padding = 224 - (v_split_index + patch_size)

        h_split_index = randrange(0,split_upper)
        h_before_padding = h_split_index
        h_after_padding = 224 - (h_split_index + patch_size)


        x = x.swapaxes(1,3)
        resized_fixed_input =  torchvision.transforms.functional.rotate(torchvision.transforms.functional.resize(self.fixed_input.swapaxes(0,2), (patch_size, patch_size)), randrange(-45,45),interpolation =torchvision.transforms.InterpolationMode.NEAREST)
        padded_patch = torchvision.transforms.functional.pad(resized_fixed_input.swapaxes(0,2).view(1,patch_size,patch_size,3).swapaxes(1,3),padding = (h_before_padding,v_before_padding,h_after_padding,v_after_padding)).swapaxes(1,3)
        sample_patch_map = torchvision.transforms.functional.pad(self.patch_map.view(1,patch_size,patch_size,3).swapaxes(1,3),padding = (h_before_padding,v_before_padding,h_after_padding,v_after_padding)).swapaxes(1,3)
        x = torch.where(sample_patch_map > 0, padded_patch , x) 
        
        x = torchvision.transforms.functional.adjust_brightness(x.swapaxes(1,3),random.uniform(0.8,1.2))
        x = torchvision.transforms.functional.adjust_contrast(x,random.uniform(0.5,2))
        x = torchvision.transforms.functional.adjust_saturation(x,random.uniform(0.2,1.3))
        x = torchvision.transforms.functional.adjust_hue(x,random.uniform(-0.1,0.1)).swapaxes(1,3)


        x = x.swapaxes(1,3)
        x = self.vgg16(x)
        return x


def n_most_likely(output):
    prediction_list = output.view(1000).detach().numpy()
    largest_indices = np.flip(np.argsort(prediction_list))
    for index in largest_indices[:10]:
        print(str(label_tags.labels[index]) + " " + str(prediction_list[index]))





# freeze gradients of model, may be unnecessary but maybe not
def freeze_model(model):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False

if __name__ == "__main__":

    transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor()])

    dataset = datasets.ImageFolder(os.path.join('training_images'), transform = transform )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True)

    images, labels = next(iter(dataloader))

    
    net = AdversarialNetwork().to(device)
   

    net.train(mode = True)
    freeze_model(net.vgg16)
    optimizer = optim.Adam(net.parameters(), lr=.01)




    t = trange(EPOCHS)
    for i in t:

        target_label = torch.ones((BATCH_SIZE, 1000))
        target_label = torch.mul(target_label, -1000)
        
        # Follow this format to specify which label you want to train the patch to target.
        # You can see which label corresponds to which index of target_label in label_tags.py
        
        #target_label[:,859] = 1000 # toaster
        #target_label[:, 145] = 1000 # king penguin
        target_label[:,724] = 1000 # pirate, pirate ship
        
        
        count = 0
        for data in dataloader:
            count += 1
            X = data[0]
            y = target_label.to(device)
            net.zero_grad()
            output = net(X)
            loss = torch.max(functional.mse_loss(output, y))
            loss.backward()
            optimizer.step()
        t.set_postfix(loss = str(loss))
    net.eval()        
    

    adversarial_patch = net.fixed_input.view(3,200,200).detach().to('cpu').numpy().tolist()
    
    

    with open('adversarial_patch.json', 'w') as json_file:
        json.dump( {"image" : adversarial_patch}, json_file)


    helper.imshow(net.fixed_input.view(200,200,3).swapaxes(0,2).detach().to('cpu'), normalize = False)
    plt.show()

