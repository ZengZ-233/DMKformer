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


# match the patient id, label, and the ecg data id
root = '/home/zzz/pycharm_project_49/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'
# root="ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
records_path = root + 'records500'
info = pd.read_csv(root + 'ptbxl_database.csv', index_col=None)
info = info[['ecg_id', 'scp_codes', 'patient_id']]
# choose diagnosis result with the highest probability as the final result 
def final_scp(codes):
    dict_scp = {}
    ls = codes.strip('{').strip('}').split(',')
    for code in ls:
        dict_scp[code.split(':')[0].replace("'",'').replace(' ','')] = float(code.split(':')[-1])
    max_v = max(dict_scp.values())
    scp = [k for k, v in dict_scp.items() if v == max_v][0]
    if scp == 'NDT':
        scp = 'STTC'
    elif scp == 'NST_':
        scp = 'STTC'
    elif scp == 'DIG':
        scp = 'STTC'
    elif scp == 'ISC_':
        scp = 'STTC'
    elif scp == 'ISCAL':
        scp = 'STTC' 
    elif scp == 'LNGQT':
        scp = 'STTC'
    elif scp == 'ISCIN':
        scp = 'STTC'
    elif scp == 'ISCIL':
        scp = 'STTC'
    elif scp == 'ISCAS':
        scp = 'STTC'
    elif scp == 'ISCLA':
        scp = 'STTC'
    elif scp == 'ANEUR':
        scp = 'STTC'
    elif scp == 'EL':
        scp = 'STTC'
    elif scp == 'ISCAN':
        scp = 'STTC'
    elif scp == 'NORM':
        scp = 'NORM'
    elif scp == 'IMI':
        scp = 'MI'
    elif scp == 'ASMI':
        scp = 'MI'
    elif scp == 'ILMI':
        scp = 'MI'
    elif scp == 'AMI':
        scp = 'MI'
    elif scp == 'ALMI':
        scp = 'MI'
    elif scp == 'INJAS':
        scp = 'MI'
    elif scp == 'LMI':
        scp = 'MI'
    elif scp == 'INJAL':
        scp = 'MI'
    elif scp == 'IPLMI':
        scp = 'MI'
    elif scp == 'IPMI':
        scp = 'MI'
    elif scp == 'INJIN':
        scp = 'MI'
    elif scp == 'INJLA':
        scp = 'MI'
    elif scp == 'PMI':
        scp = 'MI'
    elif scp == 'INJIL':
        scp = 'MI'
    elif scp == 'LVH':
        scp = 'HYP'
    elif scp == 'LAO/LAE':
        scp = 'HYP'
    elif scp == 'RVH':
        scp = 'HYP'  
    elif scp == 'RAO/RAE':
        scp = 'HYP'
    elif scp == 'SEHYP':
        scp = 'HYP'
    elif scp == 'LAFB':
        scp = 'CD'
    elif scp == 'IRBBB':
        scp = 'CD'
    elif scp == '1AVB':
        scp = 'CD'
    elif scp == 'IVCD':
        scp = 'CD'
    elif scp == 'CRBBB':
        scp = 'CD'
    elif scp == 'CLBBB':
        scp = 'CD'
    elif scp == 'LPFB':
        scp = 'CD'
    elif scp == 'WPW':
        scp = 'CD'
    elif scp == 'ILBBB':
        scp = 'CD'
    elif scp == '3AVB':
        scp = 'CD'
    elif scp == '2AVB':
        scp = 'CD'
    else:
        scp = 'others'
    return scp  

info['scp_codes'] = info['scp_codes'].apply(lambda x: final_scp(x))

# drop patients with different diagnosis results for multiple trials 
id_dict = {}
order = 1
group = info.groupby('patient_id', sort=True)
for _, df in group:
    scps = df['scp_codes'].to_list()
    if ('others' not in set(scps)) & (len(set(scps))==1):
        id_dict['{:05d}'.format(order)] = [df['ecg_id'].to_list(), scps]
        order += 1

print(len(id_dict))


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




# main
feature_path = './Feature'
if not os.path.exists(feature_path):
    os.mkdir(feature_path)

for pid in id_dict.keys():
    sub = []
    for folder in os.listdir(records_path):
        folder_path = os.path.join(records_path, folder)
        for tri in os.listdir(folder_path):
            if ('.hea' in tri) & (int(tri.split('.')[0].split('_')[0]) in id_dict[pid][0]):
                tri_path = os.path.join(folder_path, tri.split('.')[0])
                ecg_data, field = wfdb.rdsamp(tri_path)
                trial = []
                for ch in range(ecg_data.shape[1]):
                    data = resampling(ecg_data[:,ch], freq=500, kind='linear')
                    trial.append(data)
                trial = np.array(trial).T
                trial_norm = normalize(trial)
                sub.append(trial_norm)
    sub = np.array(sub)
    sub = sub.reshape(-1, 250, sub.shape[-1])  # split 10s trial into 1s sample
    # sub = sub.reshape(-1, 1250, sub.shape[-1])  # split 10s trial into 5s sample
    # print(sub.shape)
    np.save(feature_path + '/feature_{}.npy'.format(pid), sub)




# label.npy

label_path = './Label'
if not os.path.exists(label_path):
    os.mkdir(label_path)
    
label = []
for k, v in id_dict.items():
    if 'NORM' in set(v[-1]):
        diag = 0
    elif  'MI' in set(v[-1]):
        diag = 1
    elif 'STTC' in set(v[-1]):
        diag = 2
    elif 'CD' in set(v[-1]):
        diag = 3
    else:
        diag = 4
    label.append([int(diag), int(k)])
label = np.array(label)
print(label)
np.save(label_path + '/label.npy', label)


np.load('./Feature/feature_01003.npy').shape


