import os
import unicodedata

from math import sin, cos, sqrt, atan2, radians


def distance(x1, y1, x2, y2):
    lat1 = radians(x1)
    lon1 = radians(y1)
    lat2 = radians(x2)
    lon2 = radians(y2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return 6373 * c


def val2year(value):
    year = float('nan')
    digs = [c for c in str(value) if c.isdigit()]
    if len(digs) >= 4:
        if 1900 <= int(''.join(digs[-4:])) <= 2021:
            year = ''.join(digs[-4:])
        elif 1900 <= int(''.join(digs[:4])) <= 2021:
            year = ''.join(digs[:4])
    elif len(digs) == 2:
        year = f"19{''.join(digs)}"

    if type(year) == float:
        arr = str(value).replace('_', '.').replace('/', '.').replace('-', '.').replace(' ', '')
        arr = arr[:-1] if arr[-1] == '.' else arr
        arr = arr.split('.')
        if len(arr) == 3:
            year = f'19{arr[-1]}'
    return year


def val2zip(value, maxlen=5):
    zip = float('nan')
    digs = [c for c in str(value) if c.isdigit()]
    if len(digs) >= maxlen:
        zip = ''.join(digs[:maxlen])
    return zip


def val2utf8(value):
    normalized = unicodedata.normalize('NFD', value).encode('ascii', 'ignore').decode("utf-8").lower()
    if normalized.replace('-', '') == '':
        normalized = float('nan')
    return normalized


def get_file_paths(dpath, format, includes=[]):
    fpaths = []
    for (dirpath, dirnames, filenames) in os.walk(dpath):
        for file in filenames:
            if file.endswith(format):
                if includes != [] and not any(fname in file for fname in includes):
                    continue
                fpaths.append(os.path.join(dirpath, file))
    fpaths.sort()
    
    return fpaths