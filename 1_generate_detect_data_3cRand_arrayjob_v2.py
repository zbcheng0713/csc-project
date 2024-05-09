# %load generate_detect_data_3cRand.py
#     preprocessing the data 
# processes all detection data given earthquake and noise events. 
#It creates the files for the training and test data for training the detection model.

import pandas as pd
import numpy as np
from obspy import read
import os
import matplotlib.pyplot as plt
import pdb
import torch


from sklearn.model_selection import train_test_split

pts_train_dir = "../../pts/det_train"
os.makedirs(pts_train_dir, exist_ok=True)
pts_test_dir = "../../pts/det_test"
os.makedirs(pts_test_dir, exist_ok=True)
 
#========Load raw trace data=====================================
# Load trace data given a SAC file
def load_data(filename, see_stats=False, bandpass=False):
    st = read(filename)
    tr = st[0]
    if bandpass:
        tr.filter(type='bandpass', freqmin=5.0, freqmax=40.0)
    tr.taper(0.02)
    if see_stats:
        print(tr.stats)
        tr.plot()
    return tr

#========Find all events=====================================
# Inputs a path and returns all events (directories) is list     
def find_all_events(path):
    dirs = []
    for r, d, f in os.walk(path):
        for name in d:
            dirs.append(os.path.join(r, name))
    return dirs

def find_all_SAC(path):
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if '.SAC' in file:
                files.append(os.path.join(r, file))
    return files

#========Get all station traces for a  given event=====================================
# Given a path, find all station traces. If a station did not record an event, zero-fill      
def get_event(path, station_no, showPlots=False):
    sample_size = 2500
    channel_size = 3
    event_array = torch.zeros(channel_size, station_no,sample_size)
    sorted_event_array = torch.zeros(channel_size, station_no,sample_size)
    max_amp_idx = np.ones(station_no) * sample_size
    snr = []
    ii=0
    for r, d, f in os.walk(path):
        if len(f)!=165:
            continue
        if f == []:
            return []
        for filename in sorted(f):
            i = ii // 3
            j = ii % 3
            tr = load_data(os.path.join(r,filename), False)
            if len(tr.data) < sample_size:
                print('ERROR '+filename+' '+str(len(tr.data)))
            else:
                event_array[j,i,:] = torch.from_numpy(tr.data[:sample_size])
                peak_amp = max(abs(event_array[j,i,:]))
                if tr.stats['network'] != 'FG':
                    event_array[j,i,:] = event_array[j,i,:] / peak_amp
                    if tr.stats['channel'] == 'HHZ' or tr.stats['channel'] == 'EHZ':
                        max_amp_idx[i] = np.argmax(abs(event_array[j,i,:])).numpy()
                else:
                    event_array[j,i,:] = event_array[j,i,:] * 0
            ii+=1
            if i == station_no:
                break

    # sort traces in order of when their maximum amplitude arrives
    idx = np.argsort(max_amp_idx)
    sorted_event_array = event_array[:,idx,:]

    # sorted and absolute
    event_array = abs(sorted_event_array)

    # Include option to visualize traces for each event
    if (showPlots):
        fig, axs = plt.subplots(station_no, sharey="col")
        fig.suptitle(path)
        for i in range(station_no):
            axs[i].plot(sorted_event_array[2,i,:])
            axs[i].axis('off')
        plt.show()
    return event_array

# def sub_folders(base_path, start_index, end_index):
#     processed_dirs = []
#     current_index = 0

#     for entry in os.scandir(base_path):
#         if entry.is_dir():
#             if start_index <= current_index < end_index:
#                 processed_dirs.append(entry.path)
#             current_index += 1
#             if current_index >= end_index:
#                 break

#     return processed_dirs


if __name__ == "__main__":
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"Running on Device {device}")    
    event_dirlist= '../data_dir_index/event_dirs.txt'
    noise_dirlist= '../data_dir_index/noise_dirs.txt'
    start_index = int(os.getenv("START_INDEX"))  # defalut 0
    end_index = int(os.getenv("END_INDEX"))   # defalut 5000
    print(f"process {start_index}-{end_index} events")
    pos_path = "../../data/data_prep_events"
    # all_pos_dir= [entry.path for entry in os.scandir(pos_path) if entry.is_dir()]
    neg_path = "../../data/data_prep_noises_all_checked_220_arrayjob"
    # all_neg_dir= [entry.path for entry in os.scandir(neg_path) if entry.is_dir()]
    with open(event_dirlist, 'r') as f:
        all_folders = [line.strip() for line in f.readlines()]
        pos_dirs = all_folders[start_index:end_index]
    with open(noise_dirlist, 'r') as f:
        all_folders = [line.strip() for line in f.readlines()]
        neg_dirs = all_folders[start_index:end_index]
    print(len(pos_dirs)) # number of earthquake events
    print(len(neg_dirs)) # number of noise events
    sample_size = 2500
    station_no = 55
    channel_size=3
    X_all = torch.zeros(len(pos_dirs)+len(neg_dirs), channel_size, station_no, sample_size)
    y_all = torch.zeros(len(pos_dirs)+len(neg_dirs))

if len(pos_dirs)>0:
    for i,dirname in enumerate(pos_dirs):    ###         earthquake events lable     ####
        print(dirname)
        event_array = get_event(dirname, station_no)
        X_all[i,:,:] = event_array
        y_all[i] = torch.tensor(1)
if len(neg_dirs)>0:
    for i,dirname in enumerate(neg_dirs):  ###         noise lable     ####
        print(dirname)
        event_array = get_event(dirname, station_no)
        X_all[i+len(pos_dirs),:,:] = event_array
        y_all[i+len(pos_dirs)] = torch.tensor(0)
if len(pos_dirs) + len(neg_dirs) > 0:
    # Split all data randomly into a 75-25 training/test set
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size = 0.25, random_state=42)
    # Save all processed data into training and test files
    
    torch.save((X_train, y_train), os.path.join(pts_train_dir, f"{start_index}-{end_index}_detect_train_data_sortedAbs50s.pt"))
    torch.save((X_test, y_test), os.path.join(pts_test_dir,f"{start_index}-{end_index}_detect_test_data_sortedAbs50s.pt"))
    # torch.save((X_train, y_train), '/Volumes/jd/data.hawaii/pts/detect_train_data_sortedAbs50s.pt')
    # torch.save((X_test, y_test), '/Volumes/jd/data.hawaii/pts/detect_test_data_sortedAbs50s.pt')