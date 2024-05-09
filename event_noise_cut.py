#################################### prepare events random cuts as sac file ############################

import pandas as pd
import numpy as np
import os
import glob
import time
from obspy import read, Stream, Trace, UTCDateTime



stations = pd.read_csv('../catalog/sta_info.csv')
catalog = pd.read_csv('../catalog/reloc_2012to2018_tbo.csv')
source = "../data/data_mseedtosac"
target_event_dir = "../data/data_prep_events"
os.makedirs(target_event_dir, exist_ok=True)
target_noise_dir = "../data/data_prep_noises"
os.makedirs(target_noise_dir, exist_ok=True)
# each event
for index, event in catalog.iterrows():
    event_cut_id = event['event cut id']
    eid=event['eid']
    year=event['yr']
    event_time = UTCDateTime(event['Time'])
    event_time_str=event['Time'].translate(str.maketrans('', '', '-:T.Z'))[:-6]
    time_before_origin = event['time before origin']
    event_cut_start=event_time-time_before_origin
    event_cut_end=event_time-time_before_origin+50
    
    noise_cut_start=event_cut_start-50
    noise_cut_end=event_cut_start

    event_source_path = os.path.join(source, f"{year}/{event_time_str}_{eid}")
    event_target_path = os.path.join(target_event_dir, event_cut_id)
    os.makedirs(event_target_path, exist_ok=True)
    noise_target_path= os.path.join(target_noise_dir, event_cut_id)
    os.makedirs(noise_target_path, exist_ok=True)

    # each station
    for index, station in stations.iterrows():
        channel_prefix = station['chan']
        network = station['ntwk']
        station = station['stnm']
        # check each channel
        for channel_suffix in ['E', 'N', 'Z']:
            channel = channel_prefix + channel_suffix
            sac_file_name = f"{network}.{station}..{channel}.M.{event_time_str}.SAC"
            sac_file_path = os.path.join(event_source_path, sac_file_name)
            # if event sac exits
            if os.path.isfile(sac_file_path):
                st = read(sac_file_path)
                st.detrend("linear").detrend("demean").taper(max_percentage=0.05, type="hann")
                st.filter("bandpass", freqmin=3, freqmax=20)
                st.decimate(factor=2, no_filter=False)
                noi=st.copy()
                st.trim(starttime=event_cut_start, endtime=event_cut_end)
                st.write(os.path.join(event_target_path, sac_file_name), format="SAC")
                noi.trim(starttime=noise_cut_start, endtime=noise_cut_end)
                noi.write(os.path.join(noise_target_path, sac_file_name), format="SAC")
            # creat zero-filled event sac
            else:
                npts=2501
                eq = Trace(data=np.zeros(npts), header={
                    'network': "FG",
                    'station': station,
                    'channel': channel,
                    'starttime': event_cut_start,
                    'delta': 0.02
                })
                eq.write(os.path.join(event_target_path, f"{network}.{station}..{channel}.zf.SAC"), format="SAC")
                noi=Trace(data=np.random.randn(npts), header={
                    'network': "FG",
                    'station': station,
                    'channel': channel,
                    'starttime': noise_cut_start,
                    'delta': 0.02
                })
                noi.write(os.path.join(noise_target_path, f"{network}.{station}..{channel}.zf.SAC"), format="SAC")
