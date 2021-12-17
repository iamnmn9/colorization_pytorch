Dataset Link: https://drive.google.com/file/d/15jprd8VTdtIQeEtQj6wbRx6seM8j0Rx5/view?usp=sharing



# Image Colorization Starter Code
The objective is to produce color images given grayscale input image. 

## Setup Instructions
Create a conda environment with pytorch, cuda. 

`$ conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia`

For systems without a dedicated gpu, you may use a CPU version of pytorch.
`$ conda install pytorch torchvision torchaudio cpuonly -c pytorch`

## Dataset
Use the zipfile provided as your dataset. You are expected to split your dataset to create a validation set for initial testing. Your final model can use the entire dataset for training. Note that this model will be evaluated on a test dataset not visible to you.





#######################.............NOTE1..............#####################
Run train_val_split.py to split the images in val and train folder.
(Make sure to put landscape_images folder in the same directory)

10-12-2021  01:36             1,330 		basic_model.py
13-12-2021  14:32             1,294 		colorize_data.py
14-12-2021  02:23              <DIR>          	landscape_images
10-12-2021  01:36             1,853 		README.md
14-12-2021  01:59             1,137 		train.py
13-12-2021  14:05               602 		train_val_split.py

#####################..............NOTE2................######################
In colorize_data.py:

 def __getitem__(self, index: int) -> Tuple(torch.Tensor, torch.Tensor):
changed to:
 def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:

#####################..............NOTE3..................######################
I coded in google colab because of GPU need. I added code cell to respective python files.

#####################.............NOTE4...................######################
model.pth is trained model with 30 epochs (2hours of training with colab GPU)
samsung_naman.ipynb is the ipynb googlecolab ipynb file.
##################################################################################################

## Code Guide
Baseline Model: A baseline model is available in `basic_model.py` You may use this model to kickstart this assignment. We use 256 x 256 size images for this problem.
-	Fill in the dataloader, (colorize_data.py)
-	Fill in the loss function and optimizer. (train.py)
-	Complete the training loop, validation loop (train.py)
-	Determine model performance using appropriate metric. Describe your metric and why the metric works for this model? 
- Prepare an inference script that takes as input grayscale image, model path and produces a color image. 

## Additional Tasks 
- The network available in model.py is a very simple network. How would you improve the overall image quality for the above system? (Implement)
- You may also explore different loss functions here.

## Bonus
You are tasked to control the average color/mood of the image that you are colorizing. What are some ideas that come to your mind? (Bonus: Implement)

## Solution
- Document the things you tried, what worked and did not. 
- Update this README.md file to add instructions on how to run your code. (train, inference). 
- Once you are done, zip the code, upload your solution.  
