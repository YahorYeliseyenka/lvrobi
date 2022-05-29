import osmnx as ox
import geopandas as gpd
import pandas as pd
import h3
import json

from pandas.core.frame import DataFrame
from shapely.geometry import Point, Polygon, geo, MultiPolygon, mapping
from shapely import wkt
from geopandas import GeoDataFrame
from typing import Union, Dict, List
from pathlib import Path
from tqdm import tqdm
from shapely.geometry import Polygon
from .settings import DATA_INTERIM_DIR, DATA_RAW_DIR, FILTERS_DIR, DATA_PROCESSED_DIR, TOP_LEVEL_OSM_TAGS, SELECTED_TAGS


def download_whole_city(city_name: Union[str, List[str]], save_path: Path, timeout: int = 10000):
    if type(city_name) == str:
        name = city_name
    else:
        name = city_name[0]
    print(name)
    area_path = save_path.joinpath(name)
    area_path.mkdir(parents=True, exist_ok=True)
    for tag in tqdm(SELECTED_TAGS):
        tag_path = area_path.joinpath(tag + ".pkl")
        if not tag_path.exists():
            tag_gdf = download_whole_osm_tag(city_name, tag, timeout)
            if tag_gdf.empty:
                print(f"Tag: {tag} empty for city: {city_name}")
            else:
                tag_gdf.to_pickle(tag_path)
        else:
            print(f"Tag: {tag} exists for city: {city_name}")


def download_whole_osm_tag(
    area_name: Union[str, Dict[str, str]], tag: str, timeout: int = 10000
) -> GeoDataFrame:
    return download_specific_tags(area_name, {tag: True}, timeout)


def download_specific_tags(
    area_name: Union[str, Dict[str, str]],
    tags: Dict[str, Union[str, bool]],
    timeout: int = 10000,
) -> GeoDataFrame:
    ox.config(timeout=timeout)
    geometries_df = ox.geometries_from_place(area_name, tags=tags)
    geometries_df = ensure_geometry_type(geometries_df)
    return geometries_df


def ensure_geometry_type(
    df: GeoDataFrame, geometry_column: str = "geometry"
) -> GeoDataFrame:
    def ensure_geometry_type_correct(geometry):
        if type(geometry) == str:
            return wkt.loads(geometry)
        else:
            return geometry

    if geometry_column in df.columns:
        df[geometry_column] = df[geometry_column].apply(ensure_geometry_type_correct)
    return df


def prepare_city_path(base_path: Path, city: str) -> Path:
    city_path = base_path.joinpath(city)
    city_path.mkdir(parents=True, exist_ok=True)
    return city_path


def get_bounding_gdf(city: str) -> GeoDataFrame:
    # query = {"city": city}
    query = city
    gdf = ox.geocode_to_gdf(query)
    return gdf


def get_buffered_place_for_h3(place: GeoDataFrame, resolution: int) -> GeoDataFrame:
    twice_edge_length = 2 * h3.edge_length(resolution=resolution, unit="m")
    buffered = place.copy()
    buffered['geometry'] = place.to_crs(epsg=3395).buffer(int(twice_edge_length)).to_crs(epsg=4326)
    return buffered


def h3_to_polygon(hex: int) -> Polygon:
    boundary = h3.h3_to_geo_boundary(hex)
    boundary = [[y, x] for [x, y] in boundary]
    h3_polygon = Polygon(boundary)
    return h3_polygon


def get_hexes_for_place(
    area_gdf: GeoDataFrame, resolution: int, return_gdf=False
) -> Union[List[int], GeoDataFrame]:

    original = area_gdf
    buffered = get_buffered_place_for_h3(original, resolution)
    geojson = mapping(buffered)

    all_hexes = []
    all_hex_polygons = []
    for feature in geojson['features']:
        geom = feature['geometry']
        geom["coordinates"] = [
            [[j[1], j[0]] for j in i] for i in geom["coordinates"]
        ]
        hexes = list(h3.polyfill(geom, resolution))
        all_hexes.extend(hexes)
        hex_polygons = list(map(h3_to_polygon, hexes))
        all_hex_polygons.extend(hex_polygons)

    hexes_gdf = GeoDataFrame(
        pd.DataFrame({"h3": all_hexes, "geometry": all_hex_polygons}), crs="EPSG:4326"
    )

    intersecting_hexes_gdf = gpd.sjoin(hexes_gdf, original)
    intersecting_hexes_gdf = intersecting_hexes_gdf[['h3', 'geometry']]
    intersecting_hexes_gdf.drop_duplicates(inplace=True)

    if return_gdf:
        return intersecting_hexes_gdf
    else:
        intersecting_hexes = intersecting_hexes_gdf.h3.tolist()
        return intersecting_hexes


def get_hexes_polygons_for_city(
    city: Union[str, List[str]], resolution: int, use_cache=False
) -> GeoDataFrame:
    if type(city) == str:
        city_name = city
    else:
        city_name = city[0]
    city_path = prepare_city_path(DATA_RAW_DIR, city_name)
    cache_file = city_path.joinpath(f"h3_{resolution}.geojson")

    if use_cache and cache_file.exists() and cache_file.is_file():
        return gpd.read_file(cache_file)
    bounding_gdf = get_bounding_gdf(city)

    if type(bounding_gdf.geometry[0]) == MultiPolygon:
        bounding_gdf = bounding_gdf.explode().reset_index(drop=True)

    hexes_gdf = get_hexes_for_place(bounding_gdf, resolution, return_gdf=True)
    hexes_gdf.to_file(cache_file, driver="GeoJSON")
    return hexes_gdf


def load_gdf(path: Path, crs="EPSG:4326") -> GeoDataFrame:
    df = pd.read_pickle(path)
    gdf = GeoDataFrame(df, crs="EPSG:4326")
    return gdf


def filter_gdf(gdf: GeoDataFrame, tag: str, filter_values: Dict[str, str]) -> GeoDataFrame:
    if filter_values is not None:
        selected_tag_values = set(filter_values[tag])
        gdf = gdf[gdf[tag].isin(selected_tag_values)]
    return gdf
    

def load_city_tag(city: str, tag: str, split_values=True, filter_values: Dict[str, str] = None) -> GeoDataFrame:
    path = DATA_RAW_DIR.joinpath(city, f"{tag}.pkl")
    if path.exists():
        gdf = load_gdf(path)
        if split_values:
            gdf[tag] = gdf[tag].str.split(';')
            gdf = gdf.explode(tag)
            gdf[tag] = gdf[tag].str.strip()
        gdf = filter_gdf(gdf, tag, filter_values)
        return gdf
    else:
        return None


def add_h3_indices_to_city(city: Union[str, List[str]], resolution: int):
    if type(city) == str:
        city_name = city
    else:
        city_name = city[0]

    city_destination_path = prepare_city_path(DATA_INTERIM_DIR, city_name)
    hexes_polygons_gdf = get_hexes_polygons_for_city(city, resolution)
    for tag in TOP_LEVEL_OSM_TAGS:
        tag_gdf = load_city_tag(city_name, tag)
        if tag_gdf is not None:
            tag_gdf = tag_gdf[[tag, 'geometry']]
            h3_gdf = gpd.sjoin(tag_gdf, hexes_polygons_gdf, how="inner", predicate="intersects")
            result_path = city_destination_path.joinpath(f"{tag}_{resolution}.pkl")
            h3_gdf.to_pickle(result_path)
        else:
            print(f"Tag {tag} doesn't exist for city {city}, skipping...")


def load_filter(filter_name: str, values_to_drop: Dict[str, List[str]] = None) -> Dict[str, List[str]]:
    filter_file_path = FILTERS_DIR.joinpath(filter_name)
    if filter_file_path.exists() and filter_file_path.is_file():
        with filter_file_path.open(mode='rt') as f:
            filter_values = json.load(f)
            if values_to_drop is not None:
                for key, values in values_to_drop.items():
                    for value in values:
                        filter_values[key].remove(value)
            return filter_values
    else:
        available_filters = [f.name for f in FILTERS_DIR.iterdir() if f.is_file()]
        raise FileNotFoundError(f"Filter {filter_name} not found. Available filters: {available_filters}")


def load_city_tag_h3(city: str, tag: str, resolution: int, filter_values: Dict[str, str] = None) -> GeoDataFrame:
    path = DATA_INTERIM_DIR.joinpath(city, f"{tag}_{resolution}.pkl")
    if path.exists():
        gdf = load_gdf(path)
        gdf = filter_gdf(gdf, tag, filter_values)
        return gdf
    else:
        return None


def group_df_by_tag_values(df, tag: str):
    tags = df.reset_index(drop=True)[['h3', tag]]   
    indicators = tags[[tag]].pivot(columns=tag, values=tag)
    indicators[indicators.notnull()] = 1
    indicators.fillna(0, inplace = True)
    indicators = indicators.add_prefix(f"{tag}_")
    result = pd.concat([tags, indicators], axis=1).groupby('h3').sum().reset_index()
    return result


def group_city_tags(city: str, resolution: int, tags=TOP_LEVEL_OSM_TAGS, filter_values: Dict[str, str] = None, fill_missing=True) -> pd.DataFrame:
    dfs = []
    for tag in tags:
        df = load_city_tag_h3(city, tag, resolution, filter_values)
        if df is not None and not df.empty:
            tag_grouped = group_df_by_tag_values(df, tag)
        else:
            tag_grouped = pd.DataFrame()
        if fill_missing and filter_values is not None:
            columns_names = [f"{tag}_{value}" for value in filter_values[tag]]
            for c_name in columns_names:
                if c_name not in tag_grouped.columns:
                    tag_grouped[c_name] = 0
        dfs.append(tag_grouped)

    results = pd.concat(dfs, axis=0)
    results = results.fillna(0).groupby('h3').sum().reset_index()
    
    city_destination_path = prepare_city_path(DATA_PROCESSED_DIR, city)
    file_path = city_destination_path.joinpath(f"{resolution}.pkl")
    results.to_pickle(file_path)
    return results


def load_grouped_city(city: str, resolution: int) -> DataFrame:
    city_df_path = DATA_PROCESSED_DIR.joinpath(city).joinpath(f"{resolution}.pkl")
    return pd.read_pickle(city_df_path)


def group_cities(cities: str, resolution: int, add_city_column=True) -> pd.DataFrame:
    dfs = []
    for city in cities:
        df = load_grouped_city(city, resolution)
        print(city)
        print(df.isna().sum().sum())
        print(len(df.columns))
        if add_city_column:
            df['city'] = city
        dfs.append(df)
    
    all_cities = pd.concat(dfs, axis=0, ignore_index=True).set_index('h3')
    print(all_cities.isna().sum().sum())
    all_cities.to_pickle(DATA_PROCESSED_DIR.joinpath(f"{resolution}.pkl"))
    return all_cities


def load_processed_dataset(resolution: int, select_cities: List[str]=None, drop_cities: List[str]=None,
    select_tags: List[str]=None) -> DataFrame:
    dataset_path = DATA_PROCESSED_DIR.joinpath(f"{resolution}.pkl")
    df = pd.read_pickle(dataset_path)
    if select_cities is not None:
        df = df[df['city'].isin(select_cities)]
    if drop_cities is not None:
        df = df[~df['city'].isin(drop_cities)]
    if select_tags is not None:
        df = df[[*df.columns[df.columns.str.startswith(tuple(select_tags))], 'city']]
    df = df[~(df.drop(columns='city') == 0).all(axis=1)]
    return df