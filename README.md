# DMKformer: A Dual-Routing Transformer with Hybrid Mamba and Kan for Medical Time Series


## Datasets
### Data preprocessing
[APAVA](https://osf.io/jbysn/) 

[ADFD](https://openneuro.org/datasets/ds004504/versions/1.0.6) 

[PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) 

### Processed data
Download the raw data from the links above and run notebooks in the folder `data_preprocessing/` for each raw dataset to get the processed dataset. 
The folder for processed datasets has two directories: `Feature/` and `Label/`. 
The folder `Feature/` contains files named in the format `feature_ID.npy` files for all the subjects, where ID is the patient ID. 
Each`feature_ID.npy` file contains samples belonging to the same subject and stacked into a 3-D array with shape [N-sample, T, C], where N-sample denotes the number of samples in the subject with XX ID, T denotes the timestamps for a sample, and C denotes the number of channels. 
Notice that different subjects may have different numbers of samples.
The folder `Label/` has a file named `label.npy`. This label file is a 2-D array with shape [N-subject, 2], where N-subject denotes the number of subjects. The first column is the subject's label(e.g., healthy or AD), and the second column is the subject ID, ranging from 1 to N-subject.  

The processed data should be put into `dataset/DATA_NAME/` so that each subject file can be located by `dataset/DATA_NAME/Feature/feature_ID.npy`, and the label file can be located by `dataset/DATA_NAME/Label/label.npy`.  

The processed datasets can be manually downloaded at the following links.
* APAVA dataset: https://drive.google.com/file/d/1FKvUnB8qEcHng7J9CfHaU7gqRLGeS7Ea/view?usp=drive_link
* ADFD dataset: https://drive.google.com/file/d/1QcX_M58IQUBn3lDBlVVL0SDN7_QI1vWe/view?usp=drive_link
* PTB-XL dataset: https://drive.google.com/file/d/1whskRvTZUNb1Qph2SeXEdpcU2rQY0T1E/view?usp=drive_link

Since TDBrain need permission to get access first, we do not provide a download link here. 
Users need to request permission to download the raw data on the TDBrain official website and preprocess it with the jupyter notebook we provided.


## Experimental setups
We evaluate our model in two settings: subject-dependent and subject-independent.
In the subject-dependent setup, samples from the same subject can appear in both the training and test sets, causing information leakage. 
In a subject-independent setup, samples from the same subject are exclusively in either the training or test set, which is
more challenging and practically meaningful but less studied.

We compare with 10 time series transformer baselines: 
[Autoformer](https://github.com/thuml/Autoformer),
[Crossformer](https://github.com/Thinklab-SJTU/Crossformer),
[FEDformer](https://github.com/MAZiqing/FEDformer),
[Informer](https://github.com/zhouhaoyi/Informer2020),
[iTransformer](https://github.com/thuml/iTransformer),
[MTST](https://github.com/networkslab/MTST),
[Nonstationary_Transformer](https://github.com/thuml/Nonstationary_Transformers),
[PatchTST](https://github.com/yuqinie98/PatchTST),
[Reformer](https://github.com/lucidrains/reformer-pytorch),
[Transformer](https://github.com/jadore801120/attention-is-all-you-need-pytorch). 
[Medformer](https://github.com/DL4mHealth/Medformer).



## Requirements  
  
The recommended requirements are specified as follows:  
* Python 3.8  
* Jupyter Notebook  
* einops==0.4.0
* matplotlib==3.7.0
* numpy==1.23.5
* pandas==1.5.3
* patool==1.12
* reformer-pytorch==1.4.4
* scikit-learn==1.2.2
* scipy==1.10.1
* sktime==0.16.1
* sympy==1.11.1
* torch==2.0.0
* tqdm==4.64.1
* wfdb==4.1.2
* neurokit2==0.2.9
* mne==1.6.1 
  
The dependencies can be installed by:  
```bash  
pip install -r requirements.txt
```


## Run Experiments
Before running, make sure you have all the processed datasets put under `dataset/`. For Linux users, run each method's shell script in `scripts/classification/`. 
You could also run all the experiments by running the `meta-run.py/` file, which the method included in _skip_list_ will not be run.
For Windows users, see jupyter notebook `experimenets.ipynb`. All the experimental scripts are provided cell by cell. 
The gpu device ids can be specified by setting command line `--devices` (e,g, `--devices 0,1,2,3`). 
You also need to change the visible gpu devices in script file by setting `export CUDA_VISIBLE_DEVICES` (e,g, `export CUDA_VISIBLE_DEVICES=0,1,2,3`). 
The gpu devices specified by commend line should be a subset of visible gpu devices.


After training and evaluation, the saved model can be found in`checkpoints/classification/`; 
and the results can be found in  `results/classification/`. 
You can modify the parameters by changing the command line. 
The meaning and explanation of each parameter in command line can be found in `run.py` file. 
Since APAVA is the smallest dataset and faster to run, 
it is recommended to run and test our code with the APAVA dataset to get familiar with the framework.



