# %load generate_event_array3c4d.py
#processes all earthquake event data and their labels. 
#It creates the files for the training and test data for training the localization model.
from turtle import st
import pandas as pd
import numpy as np
from obspy import read
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pdb
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

pts_train_dir = "../pts/loc_train"
os.makedirs(pts_train_dir, exist_ok=True)
pts_test_dir = "../pts/loc_test"
os.makedirs(pts_test_dir, exist_ok=True)
  
#========Load raw trace data=====================================
# Load trace data given a SAC file
def load_data(filename, see_stats=False, bandpass=False):
    st = read(filename)
    tr = st[0]
    if bandpass:
        tr.filter(type='bandpass', freqmin=5.0, freqmax=40.0)
    tr.taper(0.2)
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


#========Get all station traces for a  given event=====================================
# Given a path, find all station traces. If a station did not record an event, zero-fill      
def get_event(path, station_no, showPlots=False):
    sample_size = 2500
    channel_size = 3
    event_array = torch.zeros(channel_size,station_no,sample_size)
    max_amp_idx = []
    max_amp = []
    snr = []
    ii=0
    for r, d, f in os.walk(path):
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
                    event_array[j,i,:] = event_array[j,i,:] / peak_amp     # FG stands for funcgen, a random time series generated by SAC
                else:
                    event_array[j,i,:] = event_array[j,i,:] * 0

            ii+=1
            if i == station_no:
                break

    # Include option to visualize traces for each event
    if (showPlots):
        fig, axs = plt.subplots(station_no, sharey="col")
        fig.suptitle(path)	
        for i in range(station_no):
            axs[i].plot(event_array[2,i,:])
            axs[i].axis('off')
        plt.show()
    return event_array	

#========Get the true coordinates for each event=====================================
# Given a directory name with event information, return event coordinates
def get_coords(dirname):
    # Includes path to CSV that includes all earthquake information
    # earthquake_df = pd.read_csv('/Users/yshen/Proj.ML/EQinfo/Lin_2020_reloc_withTBO.csv')
    #earthquake_df = pd.read_csv('../catalog/reloc_2012to2018_tbo.csv')
    earthquake_df = pd.read_csv('../catalog/test_2017_tbo.csv')
    

    uniqueID = dirname[-17:]  ##  e.g. 20171225035917-01 ###
    earthquake_df = earthquake_df.set_index(earthquake_df['event cut id'].str[:17])
    match = uniqueID
    return torch.tensor([earthquake_df['latR'][match], earthquake_df['lonR'][match], earthquake_df['depR'][match], earthquake_df['time before origin'][match]])

def sub_folders(base_path, start_index, end_index):
    processed_dirs = []
    current_index = 0

    for entry in os.scandir(base_path):
        if entry.is_dir():
            if start_index <= current_index < end_index:
                processed_dirs.append(entry.path)
            current_index += 1
            if current_index >= end_index:
                break

    return processed_dirs

if __name__ == "__main__":
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"Running on Device {device}")

    start_index = int(os.getenv("START_INDEX"))  # defalut 0
    end_index = int(os.getenv("END_INDEX"))   # defalut 5000
    print(f"process {start_index}-{end_index} events")

    # pos_path = "/Volumes/jd/data.hawaii/data_prepared_LinReloc"
    pos_path = "../data/data_prep_events_1000"

    pos_dirs= sub_folders(pos_path, start_index, end_index)

    sample_size = 2500
    # pos_dirs = find_all_events(pos_path)
    print(len(pos_dirs))
    station_no = 55
    channel_size = 3
    
    X_all = torch.zeros(len(pos_dirs),channel_size,station_no, sample_size)
    y_all = torch.zeros(len(pos_dirs),4)
    if len(pos_dirs)>0:
        for i,dirname in enumerate(pos_dirs):
            print(dirname)
            event_coordref = (19.5,-155.5,0.0,0.0)
            event_norm = (1.0,1.0,50.0,10.0)
        
            event_array = get_event(dirname, station_no)
            event_coordinates = get_coords(dirname)
        
            # normalize location and time
            event_coordinates = np.subtract(event_coordinates, event_coordref)
            event_coordinates = np.divide(event_coordinates, event_norm) 	
            X_all[i,:,:,:] = event_array
            y_all[i,:] = event_coordinates
        X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size = 0.25, random_state=42)
    
        torch.save((X_train, y_train),os.path.join(pts_train_dir, f"{start_index}-{end_index}_train_data3c4d.pt"))
        torch.save((X_test, y_test), os.path.join(pts_test_dir,f"{start_index}-{end_index}_test_data3c4d.pt"))
        # torch.save((X_train, y_train), '/Volumes/jd/data.hawaii/pts/train_data3c4d_NotAbs2017Mcut50sLin.pt')
        # torch.save((X_test, y_test), '/Volumes/jd/data.hawaii/pts/test_data3c4d_NotAbs2017Mcut50sLin.pt')

