import requests
from datetime import datetime, timedelta
from requests_futures.sessions import FuturesSession
from os import listdir
from os.path import isfile, join


import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from metpy.units import masked_array, units
#from netCDF4 import Dataset
import numpy as np


import xarray as xr
import cfgrib

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader


CYCLES = ["00", "12"]
FORECAST_LENGTH = 36 #hours

PRODUCTS = ["mean", "sprd", "pmmn"]

BASE_URL = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hiresw/prod"
GRIB_DIR = "href_prod/grib"
IMAGE_DIR = "href_prod/images"

grib_download_session = FuturesSession(max_workers=2)


COLORAO_EXTENT = [-109.5, -103.1, 35.4, 42.2]
DOMAIN_EXTENT = [-113, -103.1, 35.4, 42.2]

COUNTY_SHAPEFILE = 'resources/countyp010g.shp'
STATE_SHAPEFILE = 'resources/statesp010g.shp'
WATER_SHAPEFILE =  'resources/wtrbdyp010g.shp'
CAIC_PRECIP_CLEVS = [ 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5, 2, 3, 4, 5]
CAIC_SNOW_CLEVS = [ 0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50]
CAIC_PRECIP_CMAP_DATA = np.array([
    (215,215,255),
    (189,190,255),
    (126,126,255),
    (76,76,255),
    (0,7,255),
    (190,255,190),
    (24,255,0),
    (6,126,0),
    (190,189,0),
    (255,255,0),
    (255,203,152),
    (255,126,0),
    (255,0,0),
    (126,0,0),
    (255,126,126),
    (254, 126, 255),
    (126,1,126),
    (255,255,255)]) / 255.0

WEATHERBELL_PRECIP_CLEVS = [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5,
 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 9,]

WEATHERBELL_PRECIP_CMAP_DATA = np.array([
    (190,190,190),
    (170,165,165),
    (130,130,130),
    (110,110,110),
    (180,250,170),
    (150,245,140),
    (120,245,115),
    (80,240,80),
    (30,180,30),
    (15,160,15),
    (20,100,210),
    (40,130,240),
    (80,165,245),
    (150,210,230),
    (180,240,250),
    (255,250,170),
    (255,232,120),
    (255,192,60),
    (253,159,0),
    (255,96,0),
    (253,49,0),
    (225,20,20),
    (191,0,0),
    (165,0,0),
    (135,0,0),
    (99,59,59),
    (139,99,89),
    (179,139,129),
    (199,159,149),
    (240,240,210),]) / 255.0


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


def create_feature(shapefile, projection=ccrs.PlateCarree()):
	reader = shpreader.Reader(shapefile)
	feature = list(reader.geometries())
	return cfeature.ShapelyFeature(feature, projection)


def basic_surface_plot(x,y,z, extent=None, colormap = None, color_levels = None, num_colors = 15):

    fig = plt.figure()
    projection=ccrs.AlbersEqualArea()
    #projection=ccrs.LambertConformal()
    #ax = plt.axes(projection=ccrs.PlateCarree())
    ax = plt.axes(projection=projection)

    if extent is not None:
        ax.set_extent(extent)
    counties = create_feature(COUNTY_SHAPEFILE, projection=projection)
    states = create_feature(STATE_SHAPEFILE, projection=projection)
    ax.add_feature(counties, facecolor='none', edgecolor='gray')
    ax.add_feature(states, facecolor='none', edgecolor='black')
    #ax.add_feature(cfeature.LAKES, facecolor='none', edgecolor='blue')


    if (colormap is not None) and (color_levels is not None):
        cmap = mcolors.ListedColormap(colormap)
        norm = mcolors.BoundaryNorm(color_levels, cmap.N)
        cs = ax.contourf(x, y, z, color_levels, cmap=cmap, norm=norm,)
    else:
        cs = ax.contourf(x,y,z, num_colors)

    cbar = plt.colorbar(cs, orientation='vertical')
    cbar.set_label(z.units)
    return fig


def href_precip(dataset):
    #nc = cfgrib.open_dataset("href_prod/grib/00z/href.t00z.conus.pmmn.f01.grib2", backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface'}})
    nc = cfgrib.open_dataset(dataset, backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface'}})


    WATER_DENSITY = 997 * units('kg/m^3')

    data = nc['tp'].values * units('kg/m^2')
    #x = nc['tp'].longitude - 360 #convert to proper longitude
    #y = nc['tp'].latitude

    precip =  data / WATER_DENSITY
    precip = precip.to('in')

    return precip


def accumulate_precip(product, cycle):

    cycle = str(cycle).zfill(2)

    total_precip = 0
    for fhour in range(1, FORECAST_LENGTH + 1):
        fname = grib_filename(product, cycle, fhour)
        dataset = f'{GRIB_DIR}/{cycle}z/{fname}'
        total_precip += href_precip(dataset)

    return total_precip


#nc = cfgrib.open_dataset("href_prod/grib/00z/href.t00z.conus.pmmn.f01.grib2", backend_kwargs={'parallel':True,'filter_by_keys': {'typeOfLevel': 'surface'}})
#nc = xr.open_dataset("href_prod/grib/00z/href.t00z.conus.pmmn.f01.grib2", engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface'}})
#x = nc['tp'].longitude - 360 #convert to proper longitude
#y = nc['tp'].latitude

#p = accumulate_precip('mean', 0)
#p = href_precip("href_prod/grib/00z/href.t00z.conus.pmmn.f01.grib2")

#f = basic_surface_plot(x,y,p, extent=None, colormap=WEATHERBELL_PRECIP_CMAP_DATA, color_levels=WEATHERBELL_PRECIP_CLEVS)
#f = basic_surface_plot(x,y,p, extent=None, colormap=WEATHERBELL_PRECIP_CMAP_DATA)
#plt.show()
