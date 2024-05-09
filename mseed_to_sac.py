###############  mseed to sac,remove instument response############

import pandas as pd
import numpy as np
import os
import glob
from obspy import read, Stream, Trace, UTCDateTime, read_inventory


# sta_info_df = pd.read_csv(r'F:\URI_Work\HPC project\newcode\sta_info\sta_info.csv')
catalog = pd.read_csv('../catalog/eq_subset2012.csv')

# Define the source and target directories
source = "../data/data_mseed"
target = "../data/test_mseedtosac"
# inv=read_inventory(r"F:\URI_Work\HPC project\newcode\data_download\stationXML\*.xml")
inv=read_inventory("../data/stationXML/ALL_StationXML.xml")

# Make sure the target directory exists
os.makedirs(target, exist_ok=True)

for index, event in catalog.iterrows():
    eid=event['eid']
    year=event['yr']
    event_time = event['Time'].translate(str.maketrans('', '', '-:T.Z'))[:-6]
    
    event_source_path = os.path.join(source, f"{year}/{event_time}_{eid}")
    event_target_path = os.path.join(target, f"{year}/{event_time}_{eid}")
    os.makedirs(event_target_path, exist_ok=True)
# 转换格式并重命名
    st = read(event_source_path+'/*')
    st.merge(fill_value=0)
    
            # st.write(filepath, format='MSEED')
    
    for tr in st:
        try:
            tr.remove_response(inventory=inv, pre_filt=[0.04, 0.05, 40, 45], output="VEL",water_level=None) 
        except ValueError:
            print(f"Warning: No matching response information found for {tr.id}. Proceeding without response removal.")
        new_filename = f"{tr.stats.network}.{tr.stats.station}..{tr.stats.channel}.M.{event_time}.SAC"
        new_filepath = os.path.join(event_target_path, new_filename)
        tr.write(new_filepath, format='SAC')