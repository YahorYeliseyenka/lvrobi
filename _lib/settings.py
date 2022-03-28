from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR.joinpath("data")
DATA_ORIGIN_DIR = DATA_DIR.joinpath("_origin")
DATA_AFTER_PREPARATION_DIR = DATA_DIR.joinpath("prepared")
DATA_TRIPS_AS_HEXES_DIR = DATA_DIR.joinpath("trips_as_hexes")


DATA_ORIGIN_AMIENS_DIR = DATA_ORIGIN_DIR.joinpath("fr_amiens")
DATA_ORIGIN_WROCLAW_DIR = DATA_ORIGIN_DIR.joinpath("pl_wroclaw")
DATA_ORIGIN_OREBRO_DIR = DATA_ORIGIN_DIR.joinpath("se_orebro")
DATA_ORIGIN_OLDENBURG_DIR = DATA_ORIGIN_DIR.joinpath("de_oldenburg")
DATA_ORIGIN_BERLIN_DIR = DATA_ORIGIN_DIR.joinpath("de_berlin")
DATA_ORIGIN_GDANSK_DIR = DATA_ORIGIN_DIR.joinpath("pl_gdansk")
DATA_ORIGIN_SODERTALIE_DIR = DATA_ORIGIN_DIR.joinpath("sw_sodertalje")

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

HEX_RESOLUTIONS = [8, 9, 10, 11, 12]