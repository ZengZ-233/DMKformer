import os
import wfdb
import math
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy import signal
import neurokit2 as nk
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# sampling rate: 1000Hz -> resampling rate: 500Hz
# segmentation: 2s -> 1000 timestamps
# the length of the ECG signal recording of each subject can be different
# 15-lead ECGs (12 standard + Frank XYZ leads)
# 290 subjects with 9 diagnostic classes (drop n/a class)

# root path 
root_path = '/home/zzz/pycharm_project_49/ptb-diagnostic-ecg-database-1.0.0'

# only select the patients with myocardial infarction disease label
li_slc_sub = []
for sub in os.listdir(root_path):
    sub_path = os.path.join(root_path, sub)
    if os.path.isdir(sub_path):
        for tri in os.listdir(sub_path):
            if '.dat' in tri:
                tri_path = os.path.join(sub_path, tri)
                label = wfdb.rdsamp(record_name=tri_path[:-4])[1]['comments'][4].split(':')[-1].strip()
                if (label == 'Myocardial infarction')|(label == 'Healthy control'):
                    li_slc_sub.append(sub_path)
                    break

print(len(li_slc_sub))
# resampling to 250Hz
def resampling(array, freq, kind='linear'):
    t = np.linspace(1, len(array), len(array))
    f = interpolate.interp1d(t, array, kind=kind)
    t_new = np.linspace(1, len(array), int(len(array)/freq * 250))
    new_array = f(t_new)
    return new_array

# standard normalization 
def normalize(data):
    scaler = StandardScaler()
    data_norm = scaler.fit_transform(data)
    return data_norm
    
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    norm = scaler.fit_transform(df.values)
    df_norm = pd.DataFrame(norm)
    return df_norm
    
    """
    

"""
# segmentation with no overlapping (1000 timestamps)
# start from the beginning
def segment(df, window_size=500*10):
    res = []
    index = 0
    while index <= df.shape[0] - window_size:
        res.append(df.iloc[index: index+window_size, :])
        index += window_size
    return res
    
"""


# function of R peaks of a resampled trial
def R_Peaks(ecg_data):
    # get R Peak positions
    pos = []
    # get R Peak intervals
    trial_interval = []
    for ch in range(ecg_data.shape[1]):
        cleaned_ecg = nk.ecg_clean(ecg_data[:, ch], sampling_rate=250, method='neurokit')
        signals, info = nk.ecg_peaks(cleaned_ecg, sampling_rate=250, correct_artifacts=False)
        peaks = signals[signals['ECG_R_Peaks']==1].index.to_list()
        pos.append(peaks)
        channel_interval = []
        for i in range(len(peaks)-1):
            channel_interval.append(peaks[i+1] - peaks[i])
        trial_interval.append(channel_interval)
        
    df_peaks = pd.DataFrame(pos) # [C=15, num of the R-Peaks of a channel]
    df = pd.DataFrame(trial_interval).T
    med = df.median()
    return df, med, df_peaks

# get median R-Peak intervals for all trials
med_intervals = []
li_abnormal_trial = []
for sub_path in li_slc_sub:
    for tri in os.listdir(sub_path):
        if '.dat' in tri:
            tri_path = os.path.join(sub_path, tri)
            ecg_data = wfdb.rdsamp(record_name=tri_path[:-4])[0]
            trial = []
            for ch in range(ecg_data.shape[1]):
                data = resampling(ecg_data[:,ch], freq=1000, kind='linear')
                trial.append(data)
            trial = np.array(trial).T
            trial_norm = normalize(trial)
            try:
                _, med, _ = R_Peaks(trial_norm)
                med_intervals.append(med.to_list())
            except IndexError:
                print('The trial is invalid with trial path {}'.format(tri_path))
                li_abnormal_trial.append(tri_path)
                pass
            
li_abnormal_trial = list(set(li_abnormal_trial))
print(li_abnormal_trial) # no abnormal trial
df_med_intervals = pd.DataFrame(med_intervals).T

# set max_duration
all_med = df_med_intervals.median()
print(all_med[all_med<=300].shape)
print(all_med[all_med<=300].max())
max_duration = 300

# update li_abnormal_trial (invalid + outlier)
med_intervals = []
li_abnormal_trial = []
for sub_path in li_slc_sub:
    for tri in os.listdir(sub_path):
        if '.dat' in tri:
            tri_path = os.path.join(sub_path, tri)
            ecg_data = wfdb.rdsamp(record_name=tri_path[:-4])[0]
            trial = []
            for ch in range(ecg_data.shape[1]):
                data = resampling(ecg_data[:,ch], freq=1000, kind='linear')
                trial.append(data)
            trial = np.array(trial).T
            trial_norm = normalize(trial)
            try:
                _, med, _ = R_Peaks(trial_norm)
                if med.median() <= max_duration: 
                    med_intervals.append(med.to_list())
                else:
                    print('The trial is an outlier with trial path {}'.format(tri_path))
                    li_abnormal_trial.append(tri_path)
            except IndexError:
                print('The trial is invalid with trial path {}'.format(tri_path))
                li_abnormal_trial.append(tri_path)
                pass
            
li_abnormal_trial = list(set(li_abnormal_trial))
print(li_abnormal_trial) # no abnormal trial
df_med_intervals = pd.DataFrame(med_intervals).T



# split resampled trial to sample level(single heartbeat)
def sample(ecg_data, max_duration=300):
    samples = []
    _, med, df_peaks = R_Peaks(ecg_data)
    trial_med = med.median()
    for i in range(df_peaks.shape[1]):
        RP_pos = df_peaks.iloc[:, i].median()
        ini_beat = ecg_data[max(0,int(RP_pos)-int(trial_med/2)):min(int(RP_pos)+int(trial_med/2),ecg_data.shape[0]), :]
        left_zero_num = int((int(max_duration)-ini_beat.shape[0])/2)
        padding_left = np.zeros([left_zero_num, ecg_data.shape[1]])
        padding_right = np.zeros([int(max_duration)-left_zero_num-ini_beat.shape[0], ecg_data.shape[1]])
        beat = np.concatenate([padding_left, ini_beat, padding_right], axis=0)
        samples.append(beat)
    return samples 


# concat samples to segmentations
def sample2seg(samples, seg_size=10):
    segmentations = []
    index = 0
    while index <= len(samples)-seg_size:
        beat = samples[index]
        for i in range(index+1, index+seg_size):
            beat = np.vstack((beat, samples[i]))
        segmentations.append(beat)
        index += seg_size
    return segmentations



# main
feature_path = './Feature'
if not os.path.exists(feature_path):
    os.mkdir(feature_path)

dict_label = {}
sub_id = 1
for sub_path in li_slc_sub:
    li_sub_segs = []
    for tri in os.listdir(sub_path):
        if 'dat' in tri:
            tri_path = os.path.join(sub_path, tri)
            if tri_path not in li_abnormal_trial:
                label = wfdb.rdsamp(record_name=tri_path[:-4])[1]['comments'][4].split(':')[-1].strip() # label
                if label == 'Myocardial infarction':
                    dict_label['{}'.format(sub_id)] = 1
                if label == 'Healthy control':
                    dict_label['{}'.format(sub_id)] = 0
                ecg_data = wfdb.rdsamp(record_name=tri_path[:-4])[0] # data
                trial = []
                for ch in range(ecg_data.shape[1]):
                    data = resampling(ecg_data[:,ch], freq=1000, kind='linear')
                    trial.append(data)
                trial = np.array(trial).T
                trial_norm = normalize(trial)
                samples = sample(trial_norm, max_duration=300)
                segmentations = sample2seg(samples, seg_size=1) # several samples concat into a segmentation, seg_size=1 if one sample only
                for seg in segmentations:
                    li_sub_segs.append(seg)
                    print(seg.shape)
                    
    if li_sub_segs != list(): # Not None list
        array_sub = np.array(li_sub_segs)
        print(array_sub.shape)
        print('\n')
        np.save(feature_path + '/feature_{:03d}.npy'.format(sub_id), array_sub)
        sub_id += 1
    else:
        print('The subject is None after preprocessing with the path {}'.format(tri_path))              



# test feature_X_Y.npy
np.load('./Feature/feature_183.npy').shape


# label.npy
label_path = './Label'
if not os.path.exists(label_path):
    os.mkdir(label_path)

df_label = pd.DataFrame([dict_label]).T
df_label = df_label.reset_index().astype('int64')
labels = df_label[[0, 'index']].values
np.save(label_path + '/label.npy', labels)

# test label.npy
np.load('./Label/label.npy')