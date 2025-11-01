import numpy as np
import pandas as pd
import os

def standardize_colname(c):
    c = c.strip().lower().replace('*','').replace(' ','_').replace('-','_')
    c = c.replace('marketcap', 'market_cap')
    return c

def coerce_numeric(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return x
    s = str(x).replace(',', '').replace('$', '').strip()
    try:
        return float(s)
    except Exception:
        return np.nan

def load_and_clean_csv(path):
    df = pd.read_csv(path, delimiter=';')
    df.columns = [standardize_colname(c) for c in df.columns]

    # Handle schemas with timestamp or timeOpen columns
    if 'timestamp' in df.columns:
        df['date'] = pd.to_datetime(df['timestamp'], errors='coerce')
    elif 'timeopen' in df.columns:
        df['date'] = pd.to_datetime(df['timeopen'], errors='coerce')
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    else:
        raise ValueError(f'No recognizable date column in {path}')

    # Rename columns to unified form
    rename_map = {
        'marketcap': 'market_cap',
        'market_capitalization': 'market_cap',
        'timeclose': 'time_close',
        'timehigh': 'time_high',
        'timelow': 'time_low',
    }
    df = df.rename(columns=rename_map)

    # Drop irrelevant columns
    if 'name' in df.columns:
        df = df.drop(columns=['name'])

    # Coerce numerics
    for col in ['open','high','low','close','volume','market_cap']:
        if col in df.columns:
            df[col] = df[col].apply(coerce_numeric)

    df = df.dropna(subset=['date']).sort_values('date')
    df['source_file'] = os.path.basename(path)
    return df