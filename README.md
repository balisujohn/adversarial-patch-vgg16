### Adversarial Patch Implementation for VGG-16


This project is a free and open source PyTorch implementation of the adversarial patch attack presented in the 
paper Brown et. al 2018 (https://arxiv.org/abs/1712.09665) The attack is performed against the default PyTorch pretrained instance of VGG-16, as proposed in the paper Simonyan et. al 2014 (https://arxiv.org/abs/1409.1556)


### Python Environment Setup

#### CPU only

Python Version: 3.6

Platforms tested: Ubuntu 18.04(64 bit)

Create a python3.6 virtual environment, then install the python package dependencies in requirements.txt via pip. You can refer to this tutorial if this part is unclear https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/

#### GPU enabled


Python Version: 3.9

Platforms tested: Windows 10(64 bit)


First, make sure cuda and Conda are installed, then run the following conda commands to create your conda virtual env

````
conda create --name patch-env python=3.9 --file  requirements.yml
conda activate patch-env
conda install -c pytorch torchvision 
````

This should give you a cuda-enabled python 3.9 environment called patch-env. To use the environment, use
the command 

````
conda activate patch-env
````

### Try out some pre-made patches!

(Make sure your have sourced your python virtual environment)

To try out premade patches, move one of the .json files with the label class you want to the top level folder of the directory, rename the file to adversarial_patch.json then run the python script "test_adversarial_patch.py". By default, visualization is enabled for each virtual example. You can toggle visualization with the "RENDER" flag. "test_adversarial_patch.py" will also tell you on what percent of test examples the target class is predicted as most likely by VGG-16. You can set the target class manually with the global variable "CORRECT_CLASS"


### Create your own patches!

To generate an adversarial patch, first follow the instructions for obtaining a suitable image base training set found in the README.md file in the "training_images" directory. 

Once this is done, you can select which label class you would like the system to create a patch for on with the TARGET_CLASS flag in "train_adversarial_patch.py". You can see which integer corresponds to which label in "label_tags.py." 

After this, run train_adversarial_patch.py. It will save the patch in a json format once it is done training, and display a figure which you can save as an image file. You can print out the image of the adversarial mask then use it to trying VGG-16(for example if you are holding the patch up to your webcam). You can also verify for yourself that a patch works on virtual examples by trying it out with "test_adversarial_patch.py" as previously detailed.



### Contribute
Contribute whatever you want. Maybe you could improve the efficiency of adversarial training or efficacy of the attack, by employing a different loss function or changing the architecture. Maybe you could improve the python style practices or add unit tests. Maybe you could fill in gaps in the instructions or documentation based on sticking points you encounter when using the code. The sky's the limit! 

### License

\*all included files are are covered by the below license unless another license is specified in their header.

MIT License

Copyright 2021 John Balis

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

