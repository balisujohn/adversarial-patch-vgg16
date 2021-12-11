import json
from random import randrange
import random
import os


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



import helper
import label_tags
from train_adversarial_patch import n_most_likely, freeze_model


SAMPLES = 100
#CORRECT_CLASS = 859 # toaster
#CORRECT_CLASS = 145 # king penguin
CORRECT_CLASS = 74 # garden spider
RENDER = True

image = None
transform = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])

dataset = datasets.ImageFolder(os.path.join('training_images'), transform = transform )

dataloader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle = True)

images, labels = next(iter(dataloader))



with open("adversarial_patch.json", 'r') as json_file:
    patch = torch.Tensor(np.array(json.load(json_file)['image'])).view(200,200,3).swapaxes(0,2)
    patch_map = torch.ones((3,200,200))
    
    if RENDER:
        helper.imshow(patch, normalize=False)
        plt.show()


    vgg16 = models.vgg16(pretrained=True)
    freeze_model(vgg16)
    vgg16.eval()

    correct_count = 0


    for i in range(SAMPLES):
        print(f"Sample {str(i)}")
        images, _ = next(iter(dataloader))
        x = torch.clone(images[0].view(1,3,224,224))
       

        if RENDER:
            helper.imshow(x.view(3,224,224).detach(), normalize = False)
            plt.show()
            
        result = vgg16.forward(x)
        print("predictions before patch added:")
        n_most_likely(result)



        patch_size = randrange(120,200)
        split_upper = 224 - patch_size
        patch_map = torch.ones((patch_size,patch_size,3))
        for i in range(patch_size):
            for c in range(patch_size):
                if ((i-(patch_size/2.0))**2) + ((c-(patch_size/2.0))**2) > ((patch_size/2.0)**2):
                    patch_map[i,c,:] = 0


        v_split_index = randrange(0,split_upper)
        v_before_padding = v_split_index
        v_after_padding = 224 - (v_split_index + patch_size)

        h_split_index = randrange(0,split_upper)
        h_before_padding = h_split_index
        h_after_padding = 224 - (h_split_index + patch_size)


        x = x.swapaxes(1,3)
        resized_fixed_input =  torchvision.transforms.functional.rotate(torchvision.transforms.functional.resize(patch, (patch_size, patch_size)), randrange(-45,45),interpolation =torchvision.transforms.InterpolationMode.NEAREST)
        padded_patch = torchvision.transforms.functional.pad(resized_fixed_input.swapaxes(0,2).view(1,patch_size,patch_size,3).swapaxes(1,3),padding = (h_before_padding,v_before_padding,h_after_padding,v_after_padding)).swapaxes(1,3)
        sample_patch_map = torchvision.transforms.functional.pad(patch_map.view(1,patch_size,patch_size,3).swapaxes(1,3),padding = (h_before_padding,v_before_padding,h_after_padding,v_after_padding)).swapaxes(1,3)
        x = torch.where(sample_patch_map > 0, padded_patch , x) 

       
    

        if RENDER:
            helper.imshow(x.swapaxes(1,3).view(3,224,224).detach(), normalize = False)
            plt.show()

        x = torchvision.transforms.functional.adjust_brightness(x.view(224,224,3).swapaxes(0,2),random.uniform(0.8,1.2))
        x = torchvision.transforms.functional.adjust_contrast(x,random.uniform(0.5,2))
        x = torchvision.transforms.functional.adjust_saturation(x,random.uniform(0.2,1.3))
        x = torchvision.transforms.functional.adjust_hue(x,random.uniform(-0.1,0.1)).view(1,3,224,224)



        result = vgg16.forward(x)
        print("predictions after patch added:")
        n_most_likely(result)
        if np.argmax(result).item() == CORRECT_CLASS:
            correct_count += 1.0

        if RENDER:
            helper.imshow(x.view(3,224,224).detach(), normalize = False)
            plt.show()
    print("Percent Correct: " + str(100.0 * (correct_count/SAMPLES)))
