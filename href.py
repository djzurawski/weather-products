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
from cartopy.feature import NaturalEarthFeature


CYCLES = ["00", "12"]
FORECAST_LENGTH = 36 #hours

PRODUCTS = ["mean", "sprd", "pmmn"]

BASE_URL = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hiresw/prod"
GRIB_DIR = "href_prod/grib"
IMAGE_DIR = "href_prod/images"

grib_download_session = FuturesSession(max_workers=2)


COLORADO_EXTENT = [-109.5, -103.1, 35.4, 42.2]
DOMAIN_EXTENT = [-113, -103.1, 35.4, 42.2]
CONUS_EXTENT= [-120, -74, 23, 51]

COUNTY_SHAPEFILE = 'resources/cb_2018_us_county_5m.shp'

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
            print("Downloading ", fname)
            f.write(future.result().content)


def create_feature(shapefile, projection=ccrs.PlateCarree()):
    reader = shpreader.Reader(shapefile)
    feature = list(reader.geometries())
    return cfeature.ShapelyFeature(feature, projection)


class HrefSurfaceForecast:
    NS_TO_MINUTE = 1E-9 / 60
    def __init__(self, dataset):
        self.forecast = cfgrib.open_dataset(dataset, backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface'}})
        self.model = "HREF"
        self.lats = self.forecast.latitude
        self.lons = self.forecast.longitude
        self.initialized = self.forecast.time
        self.fhour = int(self.forecast.step * (self.NS_TO_MINUTE / 60))
        self.fminute = int(self.forecast.step * self.NS_TO_MINUTE)
        self.valid = self.initialized + np.timedelta64(self.fminute,'m')

    def total_precip(self):
        WATER_DENSITY = 997 * units('kg/m^3')
        data = self.forecast.tp * units('kg/m^2')
        precip_m =  data / WATER_DENSITY
        precip_in = precip_m / (0.0254 * units('m/in'))
        return precip_in


class SurfacePlot:
    def __init__(self,
                 x,y,z,
                 extent=CONUS_EXTENT,
                 colormap = None,
                 color_levels = None,
                 num_colors = 15,
                 figsize=(18,10),
                 central_longitude=-96,
                 display_counties = False):
        self.x = x
        self.y = y
        self.z = z
        self.extent = extent
        self.colormap=colormap
        self.color_levels=color_levels
        self.num_colors=num_colors
        self.figsize=figsize
        self.central_longitude=central_longitude
        self.display_counties = False
        self.plot = None

    def create_plot(self):

        fig = plt.figure(figsize=self.figsize)
        data_projection=ccrs.LambertConformal(central_longitude=self.central_longitude)
        ax = plt.axes(projection=data_projection)

        if self.extent is not None:
            ax.set_extent(self.extent)

        if self.display_counties:
            counties = create_feature(COUNTY_SHAPEFILE)
            ax.add_feature(counties, facecolor='none', edgecolor='gray',)

        states = NaturalEarthFeature(category="cultural", scale="50m",
                                     facecolor="none",
                                     edgecolor='black',
                                     name="admin_1_states_provinces_shp")
        lakes = NaturalEarthFeature('physical', 'lakes', '50m',
                                    edgecolor='blue',
                                    facecolor='none')
        ax.add_feature(lakes, facecolor='none', edgecolor='blue', linewidth=0.5)
        ax.add_feature(states, facecolor='none', edgecolor='black')
        ax.coastlines('50m', linewidth=0.8)


        if (self.colormap is not None) and (self.color_levels is not None):
            cmap = mcolors.ListedColormap(self.colormap)
            norm = mcolors.BoundaryNorm(self.color_levels, cmap.N)
            cs = ax.contourf(self.x, self.y, self.z, self.color_levels, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())

        else:
            cs = ax.contourf(self.x, self.y, self.z, self.num_colors, transform=ccrs.PlateCarree())

        cbar = plt.colorbar(cs, orientation='vertical')
        cbar.set_label(self.z.data.units)
        self.plot = fig

    def show_plot(self):
        self.create_plot()
        self.plot.show()

    def save_plot(self, path):
        self.create_plot()
        self.plot.savefig(path, bbox_inches='tight')
        plt.close()


def save_accumulated_precip_plots(product, cycle):

    cycle = str(cycle).zfill(2)

    total_precip = 0
    for fhour in range(1, FORECAST_LENGTH + 1):
        fname = grib_filename(product, cycle, fhour)
        fhour = str(fhour).zfill(2)
        dataset = f'{GRIB_DIR}/{cycle}z/{fname}'
        forecast = HrefSurfaceForecast(dataset)
        total_precip += forecast.total_precip()
        plot = SurfacePlot(forecast.lons, forecast.lats, total_precip,
                           colormap=WEATHERBELL_PRECIP_CMAP_DATA,
                           color_levels=WEATHERBELL_PRECIP_CLEVS,)
        plot.save_plot(f"href_prod/images/{product}-{cycle}-{fhour}.png")

    return total_precip

#f = HrefSurfaceForecast("href_prod/grib/12z/href.t12z.conus.mean.f36.grib2")
