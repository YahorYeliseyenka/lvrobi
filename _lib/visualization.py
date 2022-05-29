import folium
import json
import branca.colormap as cm
import pandas as pd
import h3

from geojson.feature import *


def generate_base_map(default_location=[51.107771, 17.038651], default_zoom_start=13):
    base_map = folium.Map(location=default_location, 
                          control_scale=True, 
                          zoom_start=default_zoom_start, 
                          tiles='cartodbpositron'
                         )
    return base_map


def hexagons_dataframe_to_geojson(df, value=True):
    '''Produce the GeoJSON representation containing all geometries in a dataframe
     based on a column in geojson format (geometry_field)'''
    
    list_features = df.apply(lambda row: Feature(geometry=row['geom'], 
                                                 id=row['hexid'], 
                                                 properties={"value": row['count']} if value else {}
                                                ), axis=1).tolist()

    return json.dumps(FeatureCollection(list_features))


def choropleth_map(df, value=True, initial_map=None, border_color='black', fill_opacity=0.5, legend=True):
    '''Plots a choropleth map with folium'''

    if initial_map is None:
        base_coord = df['geom'][0]['coordinates'][0][0]
        initial_map = generate_base_map((base_coord[1], base_coord[0]), 13)
    
    geojson_data = hexagons_dataframe_to_geojson(df, value)

    if value:
        custom_cm = cm.LinearColormap(['green', 'yellow', 'red'],
                                    vmin = df['count'].min(),
                                    vmax = df['count'].max())

        folium.GeoJson(
            geojson_data,
            style_function=lambda feature: {
                'fillColor': custom_cm(feature['properties']['value']),
                'color': border_color,
                'weight': 1,
                'fillOpacity': fill_opacity
            }
        ).add_to(initial_map)

        if legend:
            custom_cm.add_to(initial_map)

    else:
        folium.GeoJson(
            geojson_data,
            style_function=lambda feature: {
                'fillColor': 'green',
                'color': border_color,
                'weight': 1,
                'fillOpacity': fill_opacity
            }
        ).add_to(initial_map)

    return initial_map


def draw_csv(fpath, min_perc=0):
    df = pd.read_csv(fpath, sep=';')

    df = df.groupby(['hexid']).agg({'tripid': 'count'})
    df.rename(columns={'tripid': 'count'}, inplace=True)
    df.sort_values(by='count', ascending=False, inplace=True)
    df.reset_index(inplace=True)
    
    min_count = df['count'].max() * min_perc / 100
    df = df[df['count'] > min_count]

    df['geom'] = df['hexid'].apply(lambda hexid: {'type': 'Polygon',
                                                  'coordinates': [h3.h3_to_geo_boundary(h=hexid, geo_json=True)]
                                                 })
    return choropleth_map(df)