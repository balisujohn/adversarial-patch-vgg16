### Adversarial Patch Implementation for VGG-16


This project is a free and open source PyTorch implementation of the adversarial patch attack presented in the 
paper Brown et. al 2018 (https://arxiv.org/abs/1712.09665) 


### python environment setup

Create a python3.6 virtual environment, then install the python package dependencies via pip. You can refer to this tutorial if this part is unclear https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/


### Try out some pre-made patches!

(Make sure your have sourced your python virtual environment)

To try out premade patches, move one of the .json files with the label class you want to the top level folder of the directory, rename the file to adversarial_patch.json then run the python script "test_adversarial_patch.py". By default, visualization is enabled for each virtual example. You can toggle visualization with the "RENDER" flag. "test_adversarial_patch.py" will also tell you on what percent of test examples the target class is predicted as most likely by VGG-16. You can set the target class manually with the global variable "CORRECT_CLASS"


### Create your own patches!

To generate an adversarial patch, first follow the instructions for obtaining a suitable image base training set found in the README.md file in the "training_images" directory. 

Once this is done, you can select which label class you would like the system to create a patch for on like 132 of "train_adversarial_patch.py". You can see which integer corresponds to which label in "label_tags.py." 

After this, run train_adversarial_patch.py. It will save the patch in a json format once it is done training, and display a figure which you can save as an image file. You can print out the image of the adversarial mask then use it to trying VGG-16(for example if you are holding the patch up to your webcam). You can also verify for yourself that a patch works on virtual examples by trying it out with "test_adversarial_patch.py" as previously detailed.



### Contribute
Contribute whatever you want. Maybe you could improve the efficiency of adversarial training or efficacy of the attack, by employing a different loss function or changing the architecture. Maybe you could improve the python style practices or add unit tests. Maybe you could fill in gaps in the instructions or documentation based on sticking points you encounter when using the code. The sky's the limit!