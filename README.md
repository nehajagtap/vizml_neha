# vizml_neha
**Introduction:**
Extracting valuable insights from a dataset is a crucial part of any domain study.
For this purpose visualization plays an important role. Creating effective visualizations with
expressive visualization tools often requires users to manually specify views, which can be a
daunting task for those with limited statistical knowledge and coding skills. As a solution,
visualization recommender systems have emerged, with the goal of lowering the barrier to
exploratory data analysis by automating the process of identifying and recommending appro-
priate visualizations. Our work presents a novel approach to visualization recommendation,
leveraging an end-to-end trainable model to automatically generate visualization recommen-
dations.

**Dependencies:**
This repository uses python 3.7.3, depends on the packages listed in packages.txt.

**Components:**

**1. Data**:
 Raw data can be found in the folder clean_data. Data for extracting semantic features is places in final_data folder. Extracted features and respective graound truths are places in a folder named Extracted Features and GTs in .pkl files which can be directly used for training or testing.
 
** 2. Feature Extraction**
  For just using feature extraction module and test features for any excel file, run feature_exct.py file. Inside references to individual files for each type of feature extraction can be found. For semantic feature extraction please refer sato folder.
  
 **3. Neural Network**
 For training neural network please run the file neural_net.py. Logs will be stored in seperate folder named Logs. 
 FOr testing on different available classifiers use new_classifier.py and get test results. For this purpose you can use .pkl files as mentioned above.
 
 
