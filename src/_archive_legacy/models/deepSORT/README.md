# Project Kestrel: Deep-SORT
________________________
* Deep-SORT is a MOT algorithm that improves on SORT's ability to handle occlusion by using deep learning. A convolutional neural network is used to extract appearance based vectors from each object, then the data association algorithm can associate apperance vectors to tracks.
## Adam Mouedden's Work Explained:
* CNN_model.py: Contains the CNN model.
* CNN_Data_Processing.py: Contains the data loader, trained on MOT 16.
* CNN_Training_Script.py: Contains the training loop.
* Data Association.py: The skeleton code for deep-SORT's data association algorithm.
  * Cost functions implemented from research paper: https://www.mdpi.com/2076-3417/12/3/1319