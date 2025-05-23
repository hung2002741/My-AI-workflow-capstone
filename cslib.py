# cslib.py
import os
import pandas as pd
import json
import numpy as np
import re

def load_and_process_data(data_dir):
    """Load and process JSON files into a DataFrame."""
    if not os.path.isdir(data_dir):
        raise Exception("specified data dir does not exist")
    if not len(os.listdir(data_dir)) > 0:
        raise Exception("specified data dir does not contain any files")

    file_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if re.search("\.json", f)]
    correct_columns = ['country', 'customer_id', 'day', 'invoice', 'month',
                       'price', 'stream_id', 'times_viewed', 'year']

    all_months = {}
    for file_name in file_list:
        df = pd.read_json(file_name)
        all_months[os.path.split(file_name)[-1]] = df

    for f, df in all_months.items():
        cols = set(df.columns.tolist())
        if 'StreamID' in cols:
            df.rename(columns={'StreamID': 'stream_id'}, inplace=True)
        if 'TimesViewed' in cols:
            df.rename(columns={'TimesViewed': 'times_viewed'}, inplace=True)
        if 'total_price' in cols:
            df.rename(columns={'total_price': 'price'}, inplace=True)

        cols = df.columns.tolist()
        if sorted(cols) != correct_columns:
            raise Exception("column names could not be matched to correct columns")

    df = pd.concat(list(all_months.values()), sort=True)
    years, months, days = df['year'].values, df['month'].values, df['day'].values
    dates = ["{}-{}-{}".format(years[i], str(months[i]).zfill(2), str(days[i]).zfill(2)) for i in range(df.shape[0])]
    df['invoice_date'] = np.array(dates, dtype='datetime64[D]')
    df['invoice'] = [re.sub("\D+", "", i) for i in df['invoice'].values]

    df.sort_values(by='invoice_date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df

def convert_to_ts(df_orig, country=None):
    """Convert DataFrame to daily time-series DataFrame."""
    if country:
        if country not in np.unique(df_orig['country'].values):
            raise Exception("country not found")
    
        mask = df_orig['country'] == country
        df = df_orig[mask]
    else:
        df = df_orig
        
    invoice_dates = df['invoice_date'].values
    start_month = '{}-{}'.format(df['year'].values[0], str(df['month'].values[0]).zfill(2))
    stop_month = '{}-{}'.format(df['year'].values[-1], str(df['month'].values[-1]).zfill(2))
    df_dates = df['invoice_date'].values.astype('datetime64[D]')
    days = np.arange(start_month, stop_month, dtype='datetime64[D]')
    
    purchases = np.array([np.where(df_dates == day)[0].size for day in days])
    invoices = [np.unique(df[df_dates == day]['invoice'].values).size for day in days]
    streams = [np.unique(df[df_dates == day]['stream_id'].values).size for day in days]
    views = [df[df_dates == day]['times_viewed'].values.sum() for day in days]
    revenue = [df[df_dates == day]['price'].values.sum() for day in days]
    year_month = ["-".join(re.split("-", str(day))[:2]) for day in days]

    df_time = pd.DataFrame({
        'date': days,
        'purchases': purchases,
        'unique_invoices': invoices,
        'unique_streams': streams,
        'total_views': views,
        'year_month': year_month,
        'revenue': revenue
    })
    return df_time