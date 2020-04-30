# DeepMicro
DeepMicro is a deep representation learning framework exploiting various autoencoders to learn robust low-dimensional representations from high-dimensional data and training classification models based on the learned representation.

## Quick Setup Guide

**Step 1:** Change the current working directory to the location where you want to install `DeepMicro`.

**Step 2:** Clone the repository using git command
```
~$ git clone https://github.com/minoh0201/DeepMicro
~$ cd DeepMicro
```
**Step 3:** Create virtual environment using Anaconda3 ([Read Anaconda3 install guide](https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart)) and activate the virtual environment
```
~$ conda create --name deep_env python=3.6
```
```
~$ conda activate deep_env
```
**Step 4:** Install required packages, then install tensorflow.
```
~$ pip install --upgrade pip && pip install numpy==1.16.2 && pip install pandas==0.24.2 && pip install scipy==1.2.1 && pip install scikit-learn==0.20.3 && pip install matplotlib==3.0.3 && pip install psutil==5.6.1 && pip install keras==2.2.4
```
* If your machine is *not* equipped with GPU, install tensorflow CPU version 
  ```
  ~$ pip install tensorflow==1.13.1
  ```
* If it is equipped with GPU, then install tensorflow GPU version
  ```
  ~$ pip install tensorflow-gpu==1.13.1
  ```
**Step 5:** Run DeepMicro, printing out its usage.
```
~$ python DM.py -h
```

## Quick Start Guide
*Make sure you have already gone through the **Quick Setup Guide** above.*
### Learning representation with your own data
__1. Copy your data under the `/data` directory.__ Your data should be a comma separated file without header and index, where each row represents a sample and each column represents a microbe. We are going to assume that your file name is `UserDataExample.csv` which is already provided.

__2. Check your data can be successfully loaded and verify its shape with the following command.__
```
~$ python DM.py -r 1 --no_clf -cd UserDataExample.csv
```
The output will show the number of rows and columns right next to `X_train.shape`. Our data `UserDataExample.csv` contains 80 rows and 200 columns.
```
Using TensorFlow backend.
Namespace(act='relu', ae=False, ae_lact=False, ae_oact=False, aeloss='mse', cae=False, custom_data='UserDataExample.csv', custom_data_labels=None, data=None, dataType='float64', data_dir='', dims='50', max_epochs=2000, method='all', no_clf=True, numFolds=5, numJobs=-2, patience=20, pca=False, repeat=1, rf_rate=0.1, rp=False, save_rep=False, scoring='roc_auc', seed=0, st_rate=0.25, svm_cache=1000, vae=False, vae_beta=1.0, vae_warmup=False, vae_warmup_rate=0.01)
X_train.shape:  (80, 200)
Classification task has been skipped.
```
    
__3. Suppose that we want to reduce the number of dimensions of our data to 20 from 200 using a *shallow autoencoder*.__ Note that `--save_rep` argument will save your representation under the `/results` folder.
```
~$ python DM.py -r 1 --no_clf -cd UserDataExample.csv --ae -dm 20 --save_rep
```
    
__4. Suppose that we want to use *deep autoencoder* with 2 hidden layers which has 100 units and 40 units, respectively.__ Let the size of latent layer to be 20. We are going to see the structure of deep autoencoder first.
```
~$ python DM.py -r 1 --no_clf -cd UserDataExample.csv --ae -dm 100,40,20 --no_trn
```
It looks fine. Now, run the model and get the learned representation.
```    
~$ python DM.py -r 1 --no_clf -cd UserDataExample.csv --ae -dm 100,40,20 --save_rep
```
__5. We can try *variational autoencoder* and * convolutional autoencoder* as well.__ Note that you can see detailed argument description by using `-h` argument.
```
~$ python DM.py -r 1 --no_clf -cd UserDataExample.csv --vae -dm 100,20 --save_rep
```
```
~$ python DM.py -r 1 --no_clf -cd UserDataExample.csv --cae -dm 100,50,1 --save_rep
```

### Conducting binary classification after Learning representation with your own data
__1. Copy your *data file* and *label file* under the `/data` directory.__ Your data file should be a comma separated value (CSV) format without header and index, where each row represents a sample and each column represents a microbe. __Your label file should contain a binary value (0 or 1) in each line and the number of lines should be equal to that in your data file.__ We are going to assume that your data file name is `UserDataExample.csv` and label file name is `UserLabelExample.csv` which are already provided.

__2. Check your data can be successfully loaded and verify its shape with the following command.__
```
~$ python DM.py -r 1 --no_clf -cd UserDataExample.csv -cl UserLabelExample.csv
```
Our data `UserDataExample.csv` consists of 80 samples each of which has 200 features. The data will be split into the training set and the test set (in 8:2 ratio). The output will show the number of rows and columns for each data set.
```
Namespace(act='relu', ae=False, ae_lact=False, ae_oact=False, aeloss='mse', cae=False, custom_data='UserDataExample.csv', custom_data_labels='UserLabelExample.csv', data=None, dataType='float64', data_dir='', dims='50', max_epochs=2000, method='all', no_clf=True, no_trn=False, numFolds=5, numJobs=-2, patience=20, pca=False, repeat=1, rf_rate=0.1, rp=False, save_rep=False, scoring='roc_auc', seed=0, st_rate=0.25, svm_cache=1000, vae=False, vae_beta=1.0, vae_warmup=False, vae_warmup_rate=0.01)
X_train.shape:  (64, 200)
y_train.shape:  (64,)
X_test.shape:  (16, 200)
y_test.shape:  (16,)
Classification task has been skipped.
```

__3. Suppose that we want to directly apply SVM algorithm on our data without representation learning.__  Remove `--no_clf` command and specify classification method with `-m svm` argument (If you don't specify classification algorithm, all three algorithms will be running). 
```
~$ python DM.py -r 1 -cd UserDataExample.csv -cl UserLabelExample.csv -m svm
```
The result will be saved under `/results` folder as a `UserDataExample_result.txt`. The resulting file will be growing as you conduct more experiments.

__4. You can learn representation first, and then apply SVM algorithm on the learned representation.__
```
~$ python DM.py -r 1 -cd UserDataExample.csv -cl UserLabelExample.csv --ae -dm 20 -m svm
```

__5. You can repeat the same experiment by changing seeds for random partitioning of training and test set.__  Suppose we want to repeat classfication task five times. You can do it by put 5 into `-r` argument.
```
~$ python DM.py -r 5 -cd UserDataExample.csv -cl UserLabelExample.csv --ae -dm 20 -m svm
```

### Reproducing the experiments described in our paper
__1. Unzip `abundance.zip` and `marker.zip` files under the `/data` directory.__ 
```
~$ cd data
~$ unzip abundance.zip && unzip marker.zip
~$ cd ..
```
__2. Specify dataset name to run.__ Choose dataset you want to run. You can choose one of the followings: `abundance_Cirrhosis`, `abundance_Colorectal`, `abundance_IBD`, `abundance_Obesity`, `abundance_T2D`, `abundance_WT2D`, `marker_Cirrhosis`, `marker_Colorectal`, `marker_IBD`, `marker_Obesity`, `marker_T2D`, `marker_WT2D`. Note that WT2D indicates European Women cohort (EW-T2D) and T2D indicates Chinese cohort (C-T2D).

__3. Run experiments, specifying autoencoder details.__ 
Suppose we are going to run the best representation model on marker profile of EW-T2D dataset as shown in Table S1. Then, all three classification algorithms are trained and evaluated. We are going to repeat this process 5 times with the following command:
```
~$ python DM.py -d marker_WT2D --ae -dm 256
```
Note that if you don't specify `-r` argument, it will repeat five times by default. We can use all available CPU cores when we train classification models by introducing `-t -1` argument.

Here are another examples using a single classification algorithm.
```
~$ python DM.py -d marker_T2D --cae -dm 4,2 -m mlp
```
```
~$ python DM.py -d abundance_Obesity --cae -dm 4,2 -m rf
```
```
~$ python DM.py -d marker_Colorectal --dae -dm 512,256,128 -m mlp
```

The result will be saved under `/results` folder in a file whose name is ended with `_results.txt` (e.g. `marker_WT2D_result.txt`)

## Citation
Oh, Min, and Liqing Zhang. "DeepMicro: deep representation learning for disease prediction based on microbiome data." Scientific reports 10.1 (2020): 1-9.
