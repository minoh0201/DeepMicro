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
~$ conda create --name deep_env python=3.5
```
```
~$ conda activate deep_env
```
**Step 4:** Install required packages, then install tensorflow.
```
~$ pip install --upgrade pip && pip install numpy==1.16.2 && pip install pandas==0.24.2 && pip install scipy==1.2.1 && pip install sklearn==0.20.3 && pip install scikit-learn==0.20.3 && pip install matplotlib==3.0.3 && pip install psutil==5.6.1 && pip install keras==2.2.4
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
**1. Copy your data under the `/data` directory.** Your data should be a comma separated file without header and index, where each row represents a sample and each column represents a microbe. We are going to assume that your file name is `UserDataExample.csv` which is already provided.
**2. Check your data can be successfully loaded and verify its shape with the following command.**
    ```
    ~$ python DM.py -r 1 --no_clf -cd UserDataExample.csv
    ```
    The output will show the number of rows and columns right next to `X_train.shape`. Our data `UserDataExample.csv` contains 100 rows and 30 columns.
    ```
    Using TensorFlow backend.
    Namespace(act='relu', ae=False, ae_lact=False, ae_oact=False, aeloss='mse', cae=False, custom_data='UserDataExample.csv', custom_data_labels=None, data=None, dataType='float64', data_dir='', dims='50', max_epochs=2000, method='all', no_clf=True, numFolds=5, numJobs=-2, patience=20, pca=False, repeat=1, rf_rate=0.1, rp=False, save_rep=False, scoring='roc_auc', seed=0, st_rate=0.25, svm_cache=1000, vae=False, vae_beta=1.0, vae_warmup=False, vae_warmup_rate=0.01)
    X_train.shape:  (100, 30)
    Classification task has been skipped.
    ```
**3. Suppose that we want to reduce the number of dimensions of our data to 3 from 30 using a *shallow autoencoder*.** `--save_rep` command will save your representation under the `/results` folder.
    ```
    ~$ python DM.py -r 1 --no_clf -cd UserDataExample.csv --ae -dm 3 --save_rep
    ```
**4. Suppose that we want to use *deep autoencoder* with 2 hidden layers which has 10 units and 5 units, respectively.** Let the size of latent layer to be 3. We are going to see the structure of deep autoencoder first.
    ```
    ~$ python DM.py -r 1 --no_clf -cd UserDataExample.csv --ae -dm 10,5,3 --no_trn
    ```
    It looks fine. Now, run the model and get the learned representation.
    ```
    ~$ python DM.py -r 1 --no_clf -cd UserDataExample.csv --ae -dm 10,5,3 --save_rep
    ```
