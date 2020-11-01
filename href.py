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
from metpy.interpolate import inverse_distance_to_points
#from netCDF4 import Dataset
import numpy as np


import xarray as xr
import cfgrib

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from cartopy.feature import NaturalEarthFeature
import basemap

from multiprocessing import Pool

from metpy.plots import USCOUNTIES
import haversine


CYCLES = ["00", "12"]
FORECAST_LENGTH = 36 #hours

PRODUCTS = ["mean", "sprd", "pmmn"]

BASE_URL = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hiresw/prod"
FILTERED_BASE_URL = "https://nomads.ncep.noaa.gov/cgi-bin/filter_hrefconus.pl"
GRIB_DIR = "href_prod/grib"
IMAGE_DIR = "href_prod/images"

grib_download_session = FuturesSession(max_workers=4)


COLORADO_EXTENT = [-109.5, -103.1, 35.4, 42.2]
DOMAIN_EXTENT = [-113, -103.1, 35.4, 42.2]
CONUS_EXTENT= [-120, -74, 23, 51]
WASHINGTON_EXTENT = [-126, -116, 45, 50.5]


COUNTY_SHAPEFILE = 'resources/cb_2018_us_county_20m.shp'

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

PRECIP_CLEVS = [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5,
 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 9,]

PRECIP_CMAP_DATA = np.array([
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
    cycle = str(cycle).zfill(2)
    #url = f"{FILTERED_BASE_URL}?file=href.t{cycle}z.conus.{product}.f{fhour}.grib2&lev_surface=on=&leftlon=-128&rightlon=-100&toplat=51&bottomlat=30&dir=%2Fhref.{day_of_year}%2Fensprod"
    url = f"{FILTERED_BASE_URL}?file=href.t{cycle}z.conus.{product}.f{fhour}.grib2&var_APCP=on&subregion=&leftlon=-128&rightlon=-100&toplat=51&bottomlat=30&dir=%2Fhref.{day_of_year}%2Fensprod"

    #url = f"{BASE_URL}/href.{day_of_year}/ensprod/href.t{cycle}z.conus.{product}.f{fhour}.grib2"
    return url

def list_files(path):
    return [f for f in listdir(path) if isfile(join(path, f))]


def grib_filename(product, cycle, fhour):
    fhour = str(fhour).zfill(2)
    return f"href.t{cycle}z.conus.{product}.f{fhour}.grib2"


def download_gribs(date,cycle):
    cycle = str(cycle).zfill(2)
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


def download_latest_grib():
    date, cycle = latest_day_and_cycle()
    download_gribs(date,cycle)


def create_feature(shapefile, projection=ccrs.PlateCarree()):
    reader = shpreader.Reader(shapefile)
    feature = list(reader.geometries())
    return cfeature.ShapelyFeature(feature, projection)


class HrefSurfaceForecast:
    NS_TO_MINUTE = 1E-9 / 60
    def __init__(self, dataset):
        self.forecast = cfgrib.open_dataset(dataset,
                                            backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface'}}).metpy.parse_cf()
        self.model = "HREF"
        self.lats = self.forecast.latitude
        self.lons = self.forecast.longitude
        self.initialized = self.forecast.time
        self.fhour = int(self.forecast.step * (self.NS_TO_MINUTE / 60))
        self.fminute = int(self.forecast.step * self.NS_TO_MINUTE)
        self.valid = self.initialized + np.timedelta64(self.fminute,'m')
        self.forecast.close()


    def total_precip(self):
        WATER_DENSITY = 997 * units('kg/m^3')
        data = self.forecast.tp * units('kg/m^2')
        precip_m =  data / WATER_DENSITY
        precip_in = precip_m / (0.0254 * units('m/in'))
        return precip_in


def plot_title(init,
               valid,
               fhour,
               field_name,
               model_name="",
               field_units=""):

    init = np.datetime_as_string(init, unit='h', timezone='UTC')
    valid = np.datetime_as_string(valid, unit='h', timezone='UTC')
    fhour = str(fhour).zfill(2)

    return f"{model_name}    Init: {init}     Valid: {valid}     {field_name} ({field_units})    Hour: {fhour}"


class SurfacePlot:
    def __init__(self,
                 x,y,z,
                 extent=CONUS_EXTENT,
                 colormap = None,
                 color_levels = None,
                 num_colors = 15,
                 figsize=(18,10),
                 central_longitude=-96,
                 display_counties = False,
                 title = None,
                 units = None):
        self.x = x
        self.y = y
        self.z = z
        self.extent = extent
        self.colormap=colormap
        self.color_levels=color_levels
        self.num_colors=num_colors
        self.figsize=figsize
        self.central_longitude=central_longitude
        self.display_counties = display_counties
        self.title = title
        self.plot = None
        self.units = units


    def create_plot(self):

        fig = plt.figure(figsize=self.figsize)
        #data_projection=ccrs.LambertConformal(central_longitude=self.central_longitude)
        #data_projection=ccrs.Alb(central_longitude=self.central_longitude)
        data_projection=ccrs.Robinson(central_longitude=self.central_longitude)
        ax = plt.axes(projection=data_projection)

        if self.extent is not None:
            ax.set_extent(self.extent)

        if self.display_counties:
            counties = create_feature(COUNTY_SHAPEFILE)
            ax.add_feature(counties, facecolor='none', edgecolor='gray',)


        border_scale = '50m'
        states = NaturalEarthFeature(category="cultural", scale=border_scale,
                                     facecolor="none",
                                     edgecolor='black',
                                     name="admin_1_states_provinces_shp")
        lakes = NaturalEarthFeature('physical', 'lakes', border_scale,
                                    edgecolor='blue',
                                    facecolor='none')
        ax.add_feature(lakes, facecolor='none', edgecolor='blue', linewidth=0.5)
        ax.add_feature(states, facecolor='none', edgecolor='black')
        ax.coastlines(border_scale, linewidth=0.8)


        if (self.colormap is not None) and (self.color_levels is not None):
            cmap = mcolors.ListedColormap(self.colormap)
            norm = mcolors.BoundaryNorm(self.color_levels, cmap.N)
            cs = ax.contourf(self.x, self.y, self.z, self.color_levels, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
        else:
            cs = ax.contourf(self.x, self.y, self.z, self.num_colors, transform=ccrs.PlateCarree())

        if isinstance(self.color_levels, list):
            cbar = plt.colorbar(cs, orientation='vertical', ticks=self.color_levels)
        else:
            cbar = plt.colorbar(cs, orientation='vertical')

        if self.units:
            cbar.set_label(self.units)

        if self.title is not None:
            plt.title(self.title)


        self.plot = fig
        self.ax = ax

    def show_plot(self):
        self.create_plot()
        self.plot.show()

    def save_plot(self, path):
        self.create_plot()
        self.plot.savefig(path, bbox_inches='tight')
        plt.close()


def save_accumulated_precip_plots(product, cycle):

    cycle = str(cycle).zfill(2)
    areas = [basemap.ROBINSON_COLORADO, basemap.ROBINSON_WASATCH]

    total_precip = 0
    for fhour in range(1, FORECAST_LENGTH + 1):
        print('Processing', product, fhour)
        fname = grib_filename(product, cycle, fhour)
        fhour = str(fhour).zfill(2)
        dataset = f'{GRIB_DIR}/{cycle}z/{fname}'
        forecast = HrefSurfaceForecast(dataset)
        total_precip += forecast.total_precip()
        title = plot_title(forecast.initialized,
                           forecast.valid,
                           forecast.fhour,
                           field_name = f"Accumulated Precip {product}",
                           field_units = 'in',
                           model_name = "HREF")

        for area in areas:
            plot = SurfacePlot(forecast.lons, forecast.lats, total_precip,
                               colormap=PRECIP_CMAP_DATA,
                               color_levels=PRECIP_CLEVS,
                               extent=area.extent,
                               central_longitude=area.central_longitude,
                               display_counties=area.display_counties,
                               title=title,
                               units='in')
            plot.save_plot(f"href_prod/images/{area.name}-{cycle}z-{product}-{fhour}.png")

    return total_precip

def generate_all_plots(cycle):

    pool = Pool(len(PRODUCTS))
    cycles = [cycle] * len(PRODUCTS)
    args = [pair for pair in zip(PRODUCTS, cycles)]

    pool.starmap(save_accumulated_precip_plots, args)


def load_grib_surface(f):
    return cfgrib.open_dataset(f, backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface'}}).metpy.parse_cf()


def nearest_point(ds, lon, lat):
    def haver(ds_lat,ds_lon):
        if ds_lon > 180 or ds_lon < 0:
            ds_lon = ds_lon - 360
        return haversine.haversine((lat, lon), (ds_lat, ds_lon))

    vectorized_haver = np.vectorize(haver)

    distances = vectorized_haver(ds.latitude, ds.longitude)
    #https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmin.html - example with np.unravel_index
    lat_idx, lon_idx = np.unravel_index(np.argmin(distances, axis=None), distances.shape)
    nearest = ds.sel(x=lon_idx, y=lat_idx)
    nearest_lat = float(nearest.latitude.values)
    nearest_lon = float(nearest.longitude.values)
    if nearest_lon > 180 or nearest_lon < 0:
            nearest_lon = nearest_lon - 360

    return ((lon_idx, lat_idx), (nearest_lon, nearest_lat))


def load_full_forecast(cycle, product):
    cycle = str(cycle).zfill(2)
    fname = grib_filename(product, cycle, 1)

    datasets = []
    for fhour in range(1, FORECAST_LENGTH+1):
        fname = grib_filename(product, cycle, fhour)
        f = f'{GRIB_DIR}/{cycle}z/{fname}'
        ds = load_grib_surface(f)
        datasets.append(ds)
    return xr.concat(datasets, dim='step')


def precip_in_inches(ds):
    WATER_DENSITY = 997 * units('kg/m^3')
    data = ds.tp * units('kg/m^2')
    precip_m =  data / WATER_DENSITY
    precip_in = precip_m / (0.0254 * units('m/in'))
    return precip_in


def cumulate_precip(ds):
        return precip_in_inches(ds).cumsum(axis=0)

def get_datetime64_hour(dt):
    SEC_TO_NS = 1e-9
    return datetime.utcfromtimestamp(s.time.astype(object) * SEC_TO_NS).hour


def save_accumulated_precip_plots(self, full_forecast, product_name,):

    areas = [basemap.ROBINSON_COLORADO, basemap.ROBINSON_WASATCH]

    total_precip = 0
    for step in full_forecast.step:
        hourly_forecast = full_forecast.sel(step=step)
        fhour = int(hourly_forecast.step / np.timedelta64(1, 'h'))
        total_precip += precip_in_inches(hourly_forecast)
        cycle = str(get_datetime64_hour(hourly_forecast.time)).zfill(2)
        title = plot_title(hourly_forecast.time,
                           hourly_forecast.valid_time,
                           fhour,
                           field_name = f"Accumulated Precip {product_name}",
                           field_units = 'in',
                           model_name = "HREF")

        for area in areas:
            plot = SurfacePlot(hourly_forecast.longitude - 360,
                               hourly_forecast.latitude,
                               total_precip,
                               colormap=PRECIP_CMAP_DATA,
                               color_levels=PRECIP_CLEVS,
                               extent=area.extent,
                               central_longitude=area.central_longitude,
                               display_counties=area.display_counties,
                               title=title,
                               units='in')
            plot.save_plot(f"href_prod/images/{area.name}-{cycle}z-{product_name}-{fhour}.png")


    return total_precip


def load_all_products(cycle):
    return dict([(prod,load_full_forecast(cycle, prod)) for prod in PRODUCTS])


def plot_point_precipitation(means, pmmns, sprds, lon, lat, location_name=None):

    initialized = np.datetime_as_string(means.time, unit='m', timezone='UTC')

    ((x_idx,y_idx),(nearest_lon, nearest_lat)) = nearest_point(means, lon,lat)

    cum_means = cumulate_precip(means).sel(x=x_idx, y=y_idx)
    cum_pmmns = cumulate_precip(pmmns).sel(x=x_idx, y=y_idx)
    cum_sprds = cumulate_precip(sprds).sel(x=x_idx, y=y_idx)

    high = cum_means + cum_sprds
    low = cum_means - cum_sprds
    low = low.values.clip(min=0) #cant have negative precip


    plt.rcParams.update({'font.size': 14})
    fig = plt.figure(figsize=(12,7))
    plt.plot(means.valid_time, cum_means, color='r', label='mean')
    plt.plot(pmmns.valid_time, cum_pmmns, color='b', label='probability matched mean')
    plt.fill_between(means.valid_time.values,low ,high , color='k', alpha=0.2, label='stdv')
    plt.title("HREF Initialized: " + initialized, loc='left', fontsize=18)
    if location_name is not None:
        location_title = "{} ({}, {})".format(location_name, round(nearest_lat,3), round(nearest_lon,3))
        plt.title(location_title, loc='right', fontsize=18)
    plt.grid(linestyle='--')
    plt.xlabel("Datetime (UTC)")
    plt.ylabel("Accumulated Precipitation (in)")
    plt.legend()
    plt.show()



class HrefSurfaceForecast2:

    def __init__(self,cycle):
        self.means = load_full_forecast(cycle, 'mean')
        self.pmmns = load_full_forecast(cycle, 'pmmn')
        self.sprds = load_full_forecast(cycle, 'sprd')
        self.initialized = np.datetime_as_string(self.means.time, unit='m', timezone='UTC')



    def plot_point_precipitation(self, lon, lat, location_name=None):
        ((x_idx,y_idx),(nearest_lon, nearest_lat)) = nearest_point(self.means, lon,lat)

        cum_means = cumulate_precip(self.means).sel(x=x_idx, y=y_idx)
        cum_pmmns = cumulate_precip(self.pmmns).sel(x=x_idx, y=y_idx)
        cum_sprds = cumulate_precip(self.sprds).sel(x=x_idx, y=y_idx)

        high = cum_means + cum_sprds
        low = cum_means - cum_sprds
        low = low.values.clip(min=0) #cant have negative precip


        plt.rcParams.update({'font.size': 14})
        fig = plt.figure(figsize=(12,7))
        plt.plot(self.means.valid_time, cum_means, color='r', label='mean')
        plt.plot(self.pmmns.valid_time, cum_pmmns, color='b', label='probability matched mean')
        plt.fill_between(self.means.valid_time.values,low ,high , color='k', alpha=0.2, label='stdv')
        plt.title("HREF Initialized: " + self.initialized, loc='left', fontsize=18)
        if location_name is not None:
            location_title = "{} ({}, {})".format(location_name, round(nearest_lat,3), round(nearest_lon,3))
            plt.title(location_title, loc='right', fontsize=18)
        plt.grid(linestyle='--')
        plt.xlabel("Datetime (UTC)")
        plt.ylabel("Accumulated Precipitation (in)")
        plt.legend()
        plt.show()


#generate_all_plots(12)
#save_accumulated_precip_plots(12)
#f = HrefSurfaceForecast("href_prod/grib/12z/href.t12z.conus.mean.f36.grib2")
#ds1 = cfgrib.open_dataset("href_prod/grib/12z/href.t12z.conus.mean.f01.grib2")

#download_latest_grib()
#download_gribs("20200203", 00)
#forecasts.plot_point_precipitation(-105.764, 39.892, location_name="Winter Park")

