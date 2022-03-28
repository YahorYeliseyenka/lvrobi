import os
import pandas as pd
import numpy as np
import gpxpy

from tqdm import tqdm
from .helper import distance


def remove_substandard_trips(dataframe):
    df = dataframe.copy()

    tripids4removing = df[(df['latitude'] == 0.0) | (df['latitude'].isna()) 
                            | (df['longitude'] == 0.0) | (df['longitude'].isna()) 
                            | (df['timestamp'] == 0.0) | (df['timestamp'].isna())]['tripid'].unique()
    df = df[~df['tripid'].isin(tripids4removing)]
    df.reset_index(inplace=True, drop=True)
    print(f'Removed {len(tripids4removing)} substandard trips.')
    return df

def get_trip_start(df_lat_lon):
    df = df_lat_lon.copy()
    tqdm.pandas(desc='start')
    df['start'] = pd.concat([df['tripid'].shift().rename('tripid0'),
                                df['tripid'].rename('tripid1')], axis=1
                                ).progress_apply(lambda row: False if row[0] == row[1] else True, axis=1, raw=True)
    return df

def get_trip_end(df_lat_lon):
    df = df_lat_lon.copy()
    tqdm.pandas(desc='end')
    df['end'] = pd.concat([df['tripid'].shift(-1).rename('tripid0'),
                                df['tripid'].rename('tripid1')], axis=1
                                ).progress_apply(lambda row: False if row[0] == row[1] else True, axis=1, raw=True)
    df.loc[df.index[-1], 'end'] = True
    return df

def df_calc_basic(dataframe):
    df = dataframe.copy()

    tqdm.pandas(desc='distance')
    df['distance'] = pd.concat([df['latitude'].shift(-1).rename('x0'), 
                                    df['latitude'].rename('x1'), 
                                    df['longitude'].shift(-1).rename('y0'), 
                                    df['longitude'].rename('y1'),
                                    df['tripid'].shift(-1),
                                    df['tripid']], 
                                    axis=1).progress_apply(lambda row: 0.0 if row[4] != row[5] else distance(row[0], row[2], row[1], row[3]), axis=1, raw=True)

    tqdm.pandas(desc='duration')
    df['duration'] = pd.concat([df['timestamp'].shift(-1).rename('ts0'), 
                                    df['timestamp'].rename('ts1'), 
                                    df['tripid'].shift(-1),
                                    df['tripid']], 
                                    axis=1).progress_apply(lambda row: 0.0 if row[2] != row[3] else row[0]-row[1], axis=1, raw=True)

    df['speed'] = df['distance'] / (df['duration'] / 60 / 60)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna({'speed': 0.0}, inplace=True)

    ''' Removing points with 0 distance passed '''
    df = df[(df['distance'] != 0)]

    df = get_trip_start(df)
    df = get_trip_end(df)

    tqdm.pandas(desc='stop')
    df['stop'] = pd.concat([df['timestamp'].shift(-1), 
                                    df['timestamp'], 
                                    df['start'],
                                    df['end'],
                                    df['duration']], 
                                    axis=1).progress_apply(lambda row: 0.0 if row[2] | row[3] | (row[0]-row[1] == row[4]) else (row[0]-row[1]-row[4]), axis=1, raw=True)

    df.reset_index(inplace=True, drop=True)

    return df


def df_join_generic_with_gps(df_generic, df_gps):
    df_context = df_generic.copy()

    df_context.set_index('tripid', inplace=True)
    df_context = df_context[~df_context.index.duplicated()]
    df_context = pd.concat([df_context, calc_context(df_gps)], axis=1, join="inner")

    df_context.drop_duplicates(subset=list(set(df_context.columns.tolist()) - set(['startts', 'endts'])), keep='first', inplace=True)

    df_context.reset_index(inplace=True)

    return df_context


def calc_context(df_gps):
    df = df_gps.groupby('tripid').agg({'timestamp': ['min', 'max'], 'distance': 'sum', 'speed': ['min', 'max', 'mean']})
    df.columns = [''.join(col).strip() for col in df.columns.values]
    df.rename({'timestampmin': 'startts', 'timestampmax': 'endts', 'distancesum': 'distance'}, axis=1, inplace=True)
    df['speedavg_real'] = df['distance'] / ((df['endts']-df['startts'])/60/60)

    return df


def read_gpx(path, prefix):
    fpaths = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        for file in filenames:
            if file.endswith(".gpx"):
                fpaths.append(os.path.join(dirpath, file))

    id_ad, name, email =[], [], []
    id, lat, lon, alt, ts = [], [], [], [], []
    for fpath in tqdm(fpaths):
        tripid = prefix + ''.join(fpath.split('/')[-1]).split('.')[0]

        gpx_file = open(fpath, 'r')
        gpx = gpxpy.parse(gpx_file)
        
        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    id.append(tripid)
                    lat.append(point.latitude)
                    lon.append(point.longitude)
                    alt.append(point.elevation)
                    ts.append(point.time.timestamp())
                if segment != []:
                    id_ad.append(tripid)
                    name.append(gpx.author_name)
                    email.append(gpx.author_email)

    df_context = pd.DataFrame(np.array([id_ad, name, email]).T, columns=['tripid', 'name', 'email'])

    df_main = pd.DataFrame(np.array([id, lat, lon, alt, ts]).T, columns=['tripid', 'latitude', 'longitude', 'altitude', 'timestamp'])
    df_main = df_main.astype({'latitude': 'float', 'longitude': 'float', 'altitude': 'float', 'timestamp': 'float'})

    df_main['timestamp'] = round(df_main['timestamp'])

    return df_main, df_context


def get_df_detail_final(df_detail, df_generic):
    df = df_detail.copy()
    df = df[['tripid', 'latitude', 'longitude', 'timestamp', 'stop', 'distance', 'duration']]
    if ~df_generic.empty:
        df = df[df['tripid'].isin(df_generic['tripid'].tolist())]
    return df


def get_df_generic_final(df_generic, columns):
    df = df_generic.copy()
    df = df[['tripid', 'speedmin', 'speedmax', 'speedmean', 'speedavg_real', 'distance', 'startts', 'endts'] + columns]
    return df