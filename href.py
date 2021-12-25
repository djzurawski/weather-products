import requests
import subprocess
from datetime import datetime, timedelta
from requests_futures.sessions import FuturesSession
import os
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
FORECAST_LENGTH = 48  # hours

PRODUCTS = ["mean", "sprd", "pmmn"]

#BASE_URL = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hiresw/prod"
BASE_URL = "https://para.nomads.ncep.noaa.gov/pub/data/nccf/com/hiresw/para/"
FILTERED_BASE_URL = "https://nomads.ncep.noaa.gov/cgi-bin/filter_hrefconus.pl"
GRIB_DIR = "href_prod/grib"
IMAGE_DIR = "href_prod/images"

grib_download_session = FuturesSession(max_workers=2)


COLORADO_EXTENT = [-109.5, -103.1, 35.4, 42.2]
DOMAIN_EXTENT = [-113, -103.1, 35.4, 42.2]
CONUS_EXTENT = [-120, -74, 23, 51]
WASHINGTON_EXTENT = [-126, -116, 45, 50.5]


COUNTY_SHAPEFILE = 'resources/cb_2018_us_county_20m.shp'
#COUNTY_SHAPEFILE = 'resources/countyp010g.shp'

CAIC_PRECIP_CLEVS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3,
                     0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5, 2, 3, 4, 5]
CAIC_SNOW_CLEVS = [0.1, 0.2, 0.5, 1, 2, 3, 4,
                   5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50]
CAIC_PRECIP_CMAP_DATA = np.array([
    (215, 215, 255),
    (189, 190, 255),
    (126, 126, 255),
    (76, 76, 255),
    (0, 7, 255),
    (190, 255, 190),
    (24, 255, 0),
    (6, 126, 0),
    (190, 189, 0),
    (255, 255, 0),
    (255, 203, 152),
    (255, 126, 0),
    (255, 0, 0),
    (126, 0, 0),
    (255, 126, 126),
    (254, 126, 255),
    (126, 1, 126),
    (255, 255, 255)]) / 255.0

PRECIP_CLEVS = [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5,
                0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 9, ]

PRECIP_CMAP_DATA = np.array([
    (190, 190, 190),
    (170, 165, 165),
    (130, 130, 130),
    (110, 110, 110),
    (180, 250, 170),
    (150, 245, 140),
    (120, 245, 115),
    (80, 240, 80),
    (30, 180, 30),
    (15, 160, 15),
    (20, 100, 210),
    (40, 130, 240),
    (80, 165, 245),
    (150, 210, 230),
    (180, 240, 250),
    (255, 250, 170),
    (255, 232, 120),
    (255, 192, 60),
    (253, 159, 0),
    (255, 96, 0),
    (253, 49, 0),
    (225, 20, 20),
    (191, 0, 0),
    (165, 0, 0),
    (135, 0, 0),
    (99, 59, 59),
    (139, 99, 89),
    (179, 139, 129),
    (199, 159, 149),
    (240, 240, 210), ]) / 255.0


def select_cycle():
    utc_hour = datetime.utcnow().hour
    if utc_hour > 15 or utc_hour < 3:
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
    url = f"{FILTERED_BASE_URL}?file=href.t{cycle}z.conus.{product}.f{fhour}.grib2&lev_surface=on=&leftlon=-128&rightlon=-100&toplat=51&bottomlat=30&dir=%2Fhref.{day_of_year}%2Fensprod"
    #url = f"{BASE_URL}/href.{day_of_year}/ensprod/href.t{cycle}z.conus.{product}.f{fhour}.grib2"
    #url = f"{BASE_URL}/href.{day_of_year}_expv3/href.t{cycle}z.conus.{product}.f{fhour}.grib2"
    return url


def list_files(path):
    return [f for f in listdir(path) if isfile(join(path, f))]


def grib_filename(product, cycle, fhour):
    fhour = str(fhour).zfill(2)
    return f"href.t{cycle}z.conus.{product}.f{fhour}.grib2"


def download_gribs(date, cycle):
    cycle = str(cycle).zfill(2)
    futures = []

    dir = f'{GRIB_DIR}/{cycle}z/'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

    for prod in PRODUCTS:
        for fhour in range(1, FORECAST_LENGTH + 1):
            url = format_url(prod, date, cycle, fhour)
            fname = grib_filename(prod, cycle, fhour)
            print("Downloading ", fname)
            r = requests.get(url)
            with open(f'{GRIB_DIR}/{cycle}z/{fname}', 'wb') as f:
                f.write(r.content)

    """
    for prod in PRODUCTS:
        for fhour in range(1, FORECAST_LENGTH + 1):
            url = format_url(prod, date, cycle, fhour)
            fname = grib_filename(prod, cycle, fhour)
            future = grib_download_session.get(url)
            futures.append((future, fname, cycle))

    for future, fname, cycle in futures:
        with open(f'{GRIB_DIR}/{cycle}z/{fname}', 'wb') as f:
            print("Downloading ", fname)
            f.write(future.result().content)
    """

    for prod in PRODUCTS:
        task = subprocess.call(
            f'cat {GRIB_DIR}/{cycle}z/*{prod}* > {GRIB_DIR}/{cycle}z/{prod}_combined.grib2', shell=True)


def download_latest_grib():
    date, cycle = latest_day_and_cycle()
    download_gribs(date, cycle)
    return date, cycle


def combine_grib_tp(f):
    "Load all 48 hours of href v3 into one dataset"
    first = cfgrib.open_dataset(f, backend_kwargs={'filter_by_keys': {'totalNumber': 10, 'typeOfLevel': 'surface'}})
    middle = cfgrib.open_dataset(f, backend_kwargs={'filter_by_keys': {'totalNumber': 8, 'typeOfLevel': 'surface'}})
    end = cfgrib.open_dataset(f, backend_kwargs={'filter_by_keys': {'totalNumber': 7, 'typeOfLevel': 'surface'}})

    res = first.tp.combine_first(middle.tp)
    res = res.combine_first(end.tp)
    return res


def create_feature(shapefile, projection=ccrs.PlateCarree()):
    reader = shpreader.Reader(shapefile)
    feature = list(reader.geometries())
    return cfeature.ShapelyFeature(feature, projection)


def precip_mass_to_in(precip_m):
    WATER_DENSITY = 997 * units('kg/m^3')
    data = precip_m * units('kg/m^2')
    precip_m = data / WATER_DENSITY
    precip_in = precip_m / (0.0254 * units('m/in'))
    return precip_in


def cumulate_precip(tp_ds):
    return precip_mass_to_in(tp_ds).cumsum(axis=0)


def get_datetime64_hour(dt):
    SEC_TO_NS = 1e-9
    return datetime.utcfromtimestamp(int(dt.astype(object) * SEC_TO_NS)).hour


def plot_title(init,
               valid,
               fhour,
               field_name,
               model_name="",
               field_units=""):

    try:
        init = np.datetime64(init.values)
        valid = np.datetime64(valid.values)
    except:
        init = np.datetime64(init)
        valid = np.datetime64(valid)

    init = np.datetime_as_string(init, unit='h', timezone='UTC')
    valid = np.datetime_as_string(valid, unit='h', timezone='UTC')
    fhour = str(fhour).zfill(2)

    return f"{model_name}    Init: {init}     Valid: {valid}     {field_name} ({field_units})    Hour: {fhour}"


class SurfacePlot:
    def __init__(self,
                 x, y, z,
                 extent=None,
                 colormap=None,
                 color_levels=None,
                 num_colors=15,
                 figsize=(18, 10),
                 central_longitude=-96,
                 display_counties=False,
                 title=None,
                 units=None,
                 labels=[]):
        self.x = x
        self.y = y
        self.z = z
        self.extent = extent
        self.colormap = colormap
        self.color_levels = color_levels
        self.num_colors = num_colors
        self.figsize = figsize
        self.central_longitude = central_longitude
        self.display_counties = display_counties
        self.title = title
        self.plot = None
        self.units = units
        self.labels = labels

    def create_plot(self):

        fig = plt.figure(figsize=self.figsize)
        # data_projection=ccrs.LambertConformal(central_longitude=self.central_longitude)
        # data_projection=ccrs.Alb(central_longitude=self.central_longitude)
        data_projection = ccrs.Robinson(
            central_longitude=self.central_longitude)
        ax = plt.axes(projection=data_projection)

        if self.extent is not None:
            ax.set_extent(self.extent)
        else:
            left, right, top, bottom = np.min(self.x), np.max(
                self.x), np.max(self.y), np.min(self.y)
            ax.set_extent([left, right, bottom, top])

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
        ax.add_feature(lakes, facecolor='none',
                       edgecolor='blue', linewidth=0.5)
        ax.add_feature(states, facecolor='none', edgecolor='black')
        ax.coastlines(border_scale, linewidth=0.8)

        if (self.colormap is not None) and (self.color_levels is not None):
            cmap = mcolors.ListedColormap(self.colormap)
            norm = mcolors.BoundaryNorm(self.color_levels, cmap.N)
            cs = ax.contourf(self.x, self.y, self.z, self.color_levels,
                             cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
            #cs = ax.pcolormesh(self.x, self.y, self.z, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
        else:
            cs = ax.contourf(self.x, self.y, self.z,
                             self.num_colors, transform=ccrs.PlateCarree())
            #cs = ax.pcolormesh(self.x, self.y, self.z, transform=ccrs.PlateCarree())

        for label in self.labels:
            text, coords = label
            lon, lat = coords
            transform = ccrs.PlateCarree()._as_mpl_transform(ax)
            #ax.annotate(text, (lon,lat), xycoords=transform)
            ax.text(lon, lat, text, horizontalalignment='left',
                    transform=transform)
            ax.plot(lon, lat, markersize=2, marker='o',
                    color='k', transform=ccrs.PlateCarree())

        if isinstance(self.color_levels, list):
            cbar = plt.colorbar(cs, orientation='vertical',
                                ticks=self.color_levels)
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


def crop_dataset(forecast, domain):
    west, east, south, north = domain.extent
    west = west % 360
    east = east % 360

    mask_lon = (forecast.longitude >= west) & (forecast.longitude <= east)
    mask_lat = (forecast.latitude >= south) & (forecast.latitude <= north)

    cropped = forecast.where(mask_lon & mask_lat, drop=True)
    return cropped


def save_accumulated_precip_plots(forecast, product='mean',):
    areas = [basemap.NORCO, basemap.COTTONWOODS]

    precip_in = precip_mass_to_in(forecast)

    acc_precip = precip_in.cumsum(axis=0)

    init_time = forecast.time
    cycle = str(get_datetime64_hour(init_time)).zfill(2)
    for i, step in enumerate(acc_precip.step):
        valid_time = forecast.time + step
        print('saving', product, i + 1)
        fhour = str(i+1).zfill(2)

        title = plot_title(init_time,
                           valid_time,
                           i + 1,
                           field_name=f"Accumulated Precip {product}",
                           field_units='in',
                           model_name="HREF")

        for area in areas:
            cropped = crop_dataset(acc_precip[i], area)
            plot = SurfacePlot(cropped.longitude, cropped.latitude, cropped,
                               colormap=PRECIP_CMAP_DATA,
                               color_levels=PRECIP_CLEVS,
                               extent=area.extent,
                               central_longitude=area.central_longitude,
                               display_counties=area.display_counties,
                               title=title,
                               units='in',
                               labels=area.labels)

            plot.save_plot(
                f"href_prod/images/{cycle}z/{area.name}-{cycle}z-{product}-{fhour}.png")


def nearest_point(ds, lon, lat):
    def haver(ds_lat, ds_lon):
        if ds_lon > 180 or ds_lon < 0:
            ds_lon = ds_lon - 360
        return haversine.haversine((lat, lon), (ds_lat, ds_lon))

    vectorized_haver = np.vectorize(haver)

    distances = vectorized_haver(ds.latitude, ds.longitude)
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmin.html - example with np.unravel_index
    lat_idx, lon_idx = np.unravel_index(
        np.argmin(distances, axis=None), distances.shape)
    nearest = ds.sel(x=lon_idx, y=lat_idx)
    nearest_lat = float(nearest.latitude.values)
    nearest_lon = float(nearest.longitude.values)
    if nearest_lon > 180 or nearest_lon < 0:
        nearest_lon = nearest_lon - 360

    return ((lon_idx, lat_idx), (nearest_lon, nearest_lat))


def plot_point_precipitation(means, pmmns, sprds, lon, lat, location_name='location', show=False):

    initialized = np.datetime_as_string(means.time, unit='m', timezone='UTC')

    ((x_idx, y_idx), (nearest_lon, nearest_lat)) = nearest_point(means, lon, lat)

    cum_means = cumulate_precip(means).sel(x=x_idx, y=y_idx)
    cum_pmmns = cumulate_precip(pmmns).sel(x=x_idx, y=y_idx)
    cum_sprds = cumulate_precip(sprds).sel(x=x_idx, y=y_idx)

    high = cum_means + (0.5 * cum_sprds)
    low = cum_means - (0.5 * cum_sprds)
    low = low.values.clip(min=0)  # cant have negative precip

    means_times = [(means.time + step).data for step in means.step]
    pmmns_times = [(pmmns.time + step).data for step in pmmns.step]

    plt.rcParams.update({'font.size': 14})
    fig = plt.figure(figsize=(12, 7))
    plt.plot(means_times, cum_means, color='r', label='mean')
    plt.plot(pmmns_times, cum_pmmns, color='b',
             label='probability matched mean')
    plt.fill_between(means_times, low, high, color='k',
                     alpha=0.2, label='sprd')
    plt.title("HREF Initialized: " + initialized, loc='left', fontsize=18)
    if location_name is not None:
        location_title = "{} ({}, {})".format(
            location_name, round(nearest_lat, 3), round(nearest_lon, 3))
        plt.title(location_title, loc='right', fontsize=18)
    plt.grid(linestyle='--')
    plt.xlabel("Datetime (UTC)")
    plt.ylabel("Accumulated Precipitation (in)")
    plt.legend()
    if show:
        plt.show()
    else:
        plt.savefig(
            f'{IMAGE_DIR}/{cycle}z/{location_name}-{cycle}z-meteogram.png', bbox_inches='tight')


def plot_point_precipitation2(means, pmmns, sprds, coords, location_names, show=False):

    initialized = np.datetime_as_string(means.time, unit='m', timezone='UTC')

    cum_means = cumulate_precip(means)
    cum_pmmns = cumulate_precip(pmmns)
    cum_sprds = cumulate_precip(sprds)

    for ((lon,lat), location_name) in zip(coords, location_names):
        ((x_idx, y_idx), (nearest_lon, nearest_lat)) = nearest_point(means, lon, lat)

        loc_cum_means = cum_means.sel(x=x_idx, y=y_idx)
        loc_cum_pmmns = cum_pmmns.sel(x=x_idx, y=y_idx)
        loc_cum_sprds = cum_sprds.sel(x=x_idx, y=y_idx)

        high = loc_cum_means + (0.5 * loc_cum_sprds)
        low = loc_cum_means - (0.5 * loc_cum_sprds)
        low = low.values.clip(min=0)  # cant have negative precip

        means_times = [(means.time + step).data for step in means.step]
        pmmns_times = [(pmmns.time + step).data for step in pmmns.step]

        plt.rcParams.update({'font.size': 14})
        fig = plt.figure(figsize=(12, 7))
        plt.plot(means_times, loc_cum_means, color='r', label='mean')
        plt.plot(pmmns_times, loc_cum_pmmns, color='b',
                 label='probability matched mean')
        plt.fill_between(means_times, low, high, color='k',
                         alpha=0.2, label='sprd')
        plt.title("HREF Initialized: " + initialized, loc='left', fontsize=18)
        if location_name is not None:
            location_title = "{} ({}, {})".format(
                location_name, round(nearest_lat, 3), round(nearest_lon, 3))
            plt.title(location_title, loc='right', fontsize=18)
        plt.grid(linestyle='--')
        plt.xlabel("Datetime (UTC)")
        plt.ylabel("Accumulated Precipitation (in)")
        plt.legend()
        if show:
            plt.show()
        else:
            plt.savefig(
                f'{IMAGE_DIR}/{cycle}z/{location_name}-{cycle}z-meteogram.png', bbox_inches='tight')


if __name__ == "__main__":
    _, cycle = download_latest_grib()
    #cycle = "12"

    mean = combine_grib_tp(f'{GRIB_DIR}/{cycle}z/mean_combined.grib2')
    pmmn = combine_grib_tp(f'{GRIB_DIR}/{cycle}z/pmmn_combined.grib2')
    sprd = combine_grib_tp(f'{GRIB_DIR}/{cycle}z/sprd_combined.grib2')

    save_accumulated_precip_plots(mean, 'mean')
    save_accumulated_precip_plots(pmmn, 'pmmn')
    save_accumulated_precip_plots(sprd, 'sprd')

    coords = [(-105.777, 39.798)]
    location_names = ["BerthoudPass"]
    for domain in [basemap.NORCO, basemap.COTTONWOODS]:
        for label in domain.labels:
            name, coord = label
            coords.append(coord)
            location_names.append(name)
    plot_point_precipitation2(mean, pmmn, sprd, coords, location_names)

    """
    for domain in [basemap.NORCO, basemap.COTTONWOODS]:
        for label in domain.labels:
            name, coords = label
            lon, lat = coords
            plot_point_precipitation(mean, pmmn, sprd, lon, lat, name)
    """


#f  = cfgrib.open_dataset('href_prod/grib/12z/mean_combined.grib2', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface'}}).metpy.parse_cf()
#f  = cfgrib.open_dataset('href_prod/grib/12z/mean_combined.grib2', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface'}})
#save_accumulated_precip_plots(f, 'mean')
#f  = cfgrib.open_dataset('href_prod/grib/12z/href.t12z.conus.mean.f48.grib2', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface'}})
