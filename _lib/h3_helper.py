import h3
import pandas as pd
import osmnx as ox
import shapely
import sys
import numpy as np

from tqdm import tqdm
from .data_preparation import get_trip_start


def get_points_as_hexes(df_lat_lon, resolution):
    df = df_lat_lon.copy()
    df_temp = pd.DataFrame({'count' : df.groupby(['latitude', 'longitude']).size()}).reset_index()
    tqdm.pandas(desc=f'Converting distinct points 2 hexagons with resolution {resolution}')
    df_temp['hexid'] = df_temp.progress_apply(lambda row: h3.geo_to_h3(row[0], row[1], resolution), axis=1, raw=True)
    df = df.merge(df_temp, on=['latitude', 'longitude'], how='left')[['tripid', 'timestamp', 'distance', 'start', 'stop', 'hexid']]
    return df


def get_trips_as_hexes(df, resolution):
    df = get_points_as_hexes(df, resolution)

    tqdm.pandas(desc='Searching 4 distinct hexagons over trips')
    df['leave'] = pd.concat([df['hexid'].shift().rename('hex0'),
                           df['hexid'].rename('hex1'),
                           df['start']], axis=1).progress_apply(lambda row: True if row[2] else row[0] != row[1], axis=1, raw=True)
    
    print('Hexagons count / Points count :', df[df['leave']].shape[0], '/', df.shape[0])

    distance_sum, stop_sum, first_timestamp, last_timestamp = 0, 0, df['timestamp'][0], 0
    list_res = []

    for row in tqdm(df[1:].itertuples(), total=df[1:].shape[0], file=sys.stdout):
        if row.leave:
            list_res.extend([distance_sum, stop_sum, last_timestamp-first_timestamp if row.start else row.timestamp-first_timestamp])
            
            first_timestamp = row.timestamp
            distance_sum, stop_sum = 0, 0

        distance_sum += row.distance
        stop_sum += row.stop
        last_timestamp = row.timestamp

    list_res.extend([distance_sum, stop_sum, last_timestamp-first_timestamp])

    distance_sum = list_res[0::3]
    stop_sum = list_res[1::3]
    duration = list_res[2::3]

    df = df[df['leave']]
    df['distance_sum'] = distance_sum
    df['stop_sum'] = stop_sum
    df['duration'] = duration
    
    df = df[['tripid', 'hexid', 'timestamp', 'distance_sum', 'stop_sum', 'duration']]

    return df


def get_city_polygon(city_name):
    ox.config(use_cache=True, log_console=True)

    # define the place query
    # query = {'city': city_name}

    # get the boundaries of the place
    gdf = ox.geocode_to_gdf(city_name)
    geometry = gdf.geometry[0]

    if type(geometry) == shapely.geometry.multipolygon.MultiPolygon:
        geometry = geometry[0]

    # swap polygon coordinates
    coords = [(coord[1], coord[0]) for coord in geometry.exterior.coords]

    return coords


def get_trips_inside_city(df_trips_hex, city_shape, resolution):
    df_trips_hex_count = pd.DataFrame({'count' : df_trips_hex.groupby('tripid').size()}).reset_index()

    hexagons_inside_city_set = h3.polyfill_polygon(city_shape, resolution)
    df_hexagons_inside_city = pd.DataFrame(hexagons_inside_city_set, columns=['hexid'])
    
    df_trips_hex_inside_city = pd.merge(df_trips_hex, df_hexagons_inside_city, on='hexid')
    df_trips_hex_inside_city_count = pd.DataFrame({'count' : df_trips_hex_inside_city.groupby('tripid').size()}).reset_index()
    
    df_trips2save = pd.merge(df_trips_hex_inside_city_count, df_trips_hex_count, on=['tripid', 'count'])
    df_trips2save.drop(columns=['count'], inplace=True)

    df_trips_hex2save = pd.merge(df_trips_hex, df_trips2save, on=['tripid'])
    
    return df_trips_hex2save