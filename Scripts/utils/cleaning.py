import numpy as np
import pandas as pd
import argparse
import os
import glob


# & helper func - clean signal files metadata
def clean_signal_files(df, value_type='integer'):
    df = df.iloc[5: , :].copy()
    
    col_name = df.columns[0]
    
    df['Time'] = df[col_name].str.split(';').str[0]
    df['Value'] = df[col_name].str.split(';').str[1]
    
    df = df.drop([col_name], axis=1)
    
    df = df.set_index('Time')
    df.index = pd.to_datetime(df.index, format="%d.%m.%Y %H:%M:%S,%f")

    if value_type == 'integer':
        df['Value'] = pd.to_numeric(df['Value'].astype(str).str.strip(), errors='coerce')
    elif value_type == 'string':
        df['Value'] = df['Value'].astype(str).str.strip()
        
    
    return df
    

# & helper func - clean event file metadata
def clean_event_file(flow_events):
    flow_events = flow_events.iloc[3: , :].copy()
    
    col_name = flow_events.columns[0]
    
    flow_events['Time'] = flow_events[col_name].str.split(';').str[0]
    flow_events['Value'] = flow_events[col_name].str.split(';').str[1]
    flow_events['Disorder'] = flow_events[col_name].str.split(';').str[2]
    flow_events['Stage'] = flow_events[col_name].str.split(';').str[3]


    flow_events[['start_str', 'end_str']] = flow_events['Time'].str.split('-', expand=True)
    flow_events['date_part'] = flow_events['start_str'].str.split(' ').str[0]
    flow_events['end_str'] = flow_events['date_part'] + ' ' + flow_events['end_str']
    flow_events['start_time'] = pd.to_datetime(flow_events['start_str'], format="%d.%m.%Y %H:%M:%S,%f")
    flow_events['end_time'] = pd.to_datetime(flow_events['end_str'], format="%d.%m.%Y %H:%M:%S,%f")
    flow_events.drop([col_name, 'Time', 'start_str', 'end_str', 'date_part'], axis=1, inplace=True)
        
    
    return flow_events

# ^ clean data and make it ready for visulization
def clean_data(flow_events, thorac, spo2, sleep_profile, nasal):
    nasal = clean_signal_files(nasal)
    thorac = clean_signal_files(thorac)
    spo2 = clean_signal_files(spo2)
    sleep_profile = clean_signal_files(sleep_profile, value_type='string')
    flow_events = clean_event_file(flow_events)
    
    return flow_events, thorac, spo2, sleep_profile, nasal