import requests
from datetime import datetime, timedelta
from requests_futures.sessions import FuturesSession
from os import listdir
from os.path import isfile, join



CYCLES = ["00", "12"]
FORECAST_LENGTH = 36 #hours

PRODUCTS = ["mean", "sprd", "pmmn"]

BASE_URL = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hiresw/prod"
GRIB_DIR = "href_prod/grib"
IMAGE_DIR = "href_prod/images"

grib_download_session = FuturesSession(max_workers=2)


def select_cycle():
    utc_hour = datetime.utcnow().hour
    if utc_hour > 14 or utc_hour < 3:
        return "12"
    else:
        return "00"


def should_get_yesterday():
    utc_hour = datetime.utcnow().hour
    if utc_hour < 3:
        return True
    else:
        return False


def yesterday_date():
    return datetime.utcnow() - timedelta(1)


def latest_day_and_cycle():
    cycle = select_cycle()
    if should_get_yesterday():
        date = yesterday_date()
    else:
        date = datetime.utcnow()

    return date.strftime('%Y%m%d'), cycle


def format_url(product, day_of_year, cycle, fhour):
    """day_of_year = 'yyyymmdd'"""
    fhour = str(fhour).zfill(2)
    url = f"{BASE_URL}/href.{day_of_year}/ensprod/href.t{cycle}z.conus.{product}.f{fhour}.grib2"
    return url

def list_files(path):
    return [f for f in listdir(path) if isfile(join(path, f))]


def grib_filename(product, cycle, fhour):
    fhour = str(fhour).zfill(2)
    return f"href.t{cycle}z.conus.{product}.f{fhour}.grib2"


def download_latest_grib():
    date, cycle = latest_day_and_cycle()

    futures = []
    for prod in PRODUCTS:
        for fhour in range(1, FORECAST_LENGTH + 1):
            url = format_url(prod, date, cycle, fhour)
            fname = grib_filename(prod, cycle, fhour)
            future  = grib_download_session.get(url)
            futures.append((future, fname, cycle))

    for future, fname, cycle in futures:
        with open(f'{GRIB_DIR}/{cycle}z/{fname}', 'wb') as f:
            print(fname)
            f.write(future.result().content)
