from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR.joinpath("data")
DATA_ORIGIN_DIR = DATA_DIR.joinpath("_trips_origin")
DATA_AFTER_PREPARATION_DIR = DATA_DIR.joinpath("trips_prepared")
DATA_TRIPS_AS_HEXES_DIR = DATA_DIR.joinpath("trips_as_hexes")
DATA_W2V_DIR = DATA_DIR.joinpath("word2vec")
DATA_FLAIR_DIR = DATA_DIR.joinpath("flair")

DATA_OSM_CITIES_DIR = DATA_DIR.joinpath("cities_data")
DATA_RAW_DIR = DATA_OSM_CITIES_DIR.joinpath("raw")
DATA_INTERIM_DIR = DATA_OSM_CITIES_DIR.joinpath("interim")
DATA_PROCESSED_DIR = DATA_OSM_CITIES_DIR.joinpath("processed")

FILTERS_DIR = DATA_OSM_CITIES_DIR.joinpath("filters")

DATA_ORIGIN_AMIENS_DIR = DATA_ORIGIN_DIR.joinpath("fr_amiens")
DATA_ORIGIN_WROCLAW_DIR = DATA_ORIGIN_DIR.joinpath("pl_wroclaw")
DATA_ORIGIN_OREBRO_DIR = DATA_ORIGIN_DIR.joinpath("se_orebro")
DATA_ORIGIN_OLDENBURG_DIR = DATA_ORIGIN_DIR.joinpath("de_oldenburg")
DATA_ORIGIN_BERLIN_DIR = DATA_ORIGIN_DIR.joinpath("de_berlin")
DATA_ORIGIN_GDANSK_DIR = DATA_ORIGIN_DIR.joinpath("pl_gdansk")
DATA_ORIGIN_SODERTALIE_DIR = DATA_ORIGIN_DIR.joinpath("sw_sodertalje")

DATA_TRIPS_AS_HEXES_GRAPH_DIR = DATA_TRIPS_AS_HEXES_DIR.joinpath('graph')

DATA_W2V_TESTS_DIR = DATA_W2V_DIR.joinpath("optuna_tests")
DATA_W2V_VECTORS_DIR = DATA_W2V_DIR.joinpath("vectors")
DATA_W2V_KEYED_VECTORS_DIR = DATA_W2V_DIR.joinpath("keyed_vectors")

DATA_FLAIR_TRIPS_DIR = DATA_FLAIR_DIR.joinpath("trips")
DATA_FLAIR_CORPUS_DIR = DATA_FLAIR_DIR.joinpath("corpus")
DATA_FLAIR_TESTS_DIR = DATA_FLAIR_DIR.joinpath("tests")

CITY_NAMES = {  
                'ami': 'Amiens, France',
                'wro': 'Wroclaw, Poland',
                'ore': 'Orebro, Sweden',
                # 'bei': 'Beijing, China',
                'old': 'Oldenburg, Germany',
                
                'ber': 'Berlin, Germany',
                'gda': 'Gdansk, Poland',
                'sod': 'Sweden, Sodertalje'
            }

SELECTED_RESOLUTIONS = [8, 9, 10]
HEX_RESOLUTIONS = SELECTED_RESOLUTIONS + [11, 12]

SELECTED_CITIES = [
    "Vienna, Austria",
    "Minsk, Belarus",
    "Sofia, Bulgaria",
    "Zagreb, Croatia",
    "Prague, Czech Republic",
    "Tallinn, Estonia",
    "Helsinki, Finland",
    "Paris, France",
    "Reykjavík, Iceland",
    "Dublin, Ireland",
    "Rome, Italy",
    "Nur-Sultan, Kazakhstan",
    "Latvia, Riga",
    "Vilnius, Lithuania",
    "Luxembourg City, Luxembourg",
    "Amsterdam, Netherlands",
    "Oslo, Norway",
    "Warszawa, PL",
    "Kraków, PL",
    "Łódź, PL",
    "Poznań, PL",
    "Lisbon, Portugal",
    ['Moscow, Russia', 'Zelenogradsky Administrative Okrug', 'Western Administrative Okrug', 'Novomoskovsky Administrative Okrug', 'Troitsky Administrative Okrug'],
    "Belgrade, Serbia",
    "Bratislava, Slovakia",
    "Ljubljana, Slovenia",
    "Madrid, Spain",
    "Stockholm, Sweden",
    "Bern, Switzerland",
    ["London, United Kingdom", "City of London"],
    "New York City, USA",
    "Chicago, USA",
    "San Francisco, USA",
] + list(CITY_NAMES.values())

SELECTED_TAGS = [
    "aeroway",
    "amenity",
    "building",
    "healthcare",
    "historic",
    "landuse",
    "leisure",
    "military",
    "natural",
    "office",
    "shop",
    "sport",
    "tourism",
    "water",
    "waterway"
]

TOP_LEVEL_OSM_TAGS = [
    "aerialway",
    "barrier",
    "boundary",
    "craft",
    "emergency",
    "geological",
    "highway",
    "man_made",
    "place",
    "power",
    "public_transport",
    "railway",
    "route",
    "telecom"
] + SELECTED_TAGS