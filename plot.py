import matplotlib.pyplot as plt
import numpy as np

from wrf import (getvar, interplevel, to_np, latlon_coords, get_cartopy,
                 cartopy_xlim, cartopy_ylim)

from netCDF4 import Dataset
from cartopy.feature import NaturalEarthFeature
import cartopy.crs as crs
import cartopy.io.shapereader as shpreader
import cartopy.feature as cfeature
from matplotlib.cm import get_cmap
from dateutil import parser
from datetime import datetime, timedelta
import matplotlib.colors as mcolors

ds = Dataset('/home/dan/Documents/wrfprd/wrfout_d01_2020-11-13_12:00:00')
COUNTY_SHAPEFILE = 'resources/cb_2018_us_county_20m.shp'

VORT_CMAP = np.array([
    (255, 255, 255),
    (190, 190, 190),
    (151, 151, 151),
    (131, 131, 131),
    (100, 100, 100),
    (0, 255, 255),
    (0, 231, 205),
    (0, 203, 126),
    (0, 179, 0),
    (126, 205, 0),
    (205, 231, 0),
    (255, 255, 0),
    (255, 205, 0),
    (255, 153, 0),
    (255, 102, 0),
    (255, 0, 0),
    (205, 0, 0),
    (161, 0, 0),
    (141, 0, 0),
    (121, 0, 0),
    (124, 0, 102),
    (145, 0, 155),
    (163, 0, 189),
    (255, 0, 231),
    (255, 201, 241)]) / 255.0

VORT_LEVELS = [0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 14,
               16, 18, 20, 22, 25, 30, 35, 40, 45, 50, 55, 60, 85]


def coriolis_parameter(lat_degrees):
    lat_rads = lat_degrees * (np.pi / 180)
    f = 2 * 7.2921e-5 * np.sin(lat_rads)
    return f


def create_feature(shapefile, projection=crs.PlateCarree()):
    reader = shpreader.Reader(shapefile)
    feature = list(reader.geometries())
    return cfeature.ShapelyFeature(feature, projection)


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

    return f"{model_name}   Init: {init}    Valid: {valid}    {field_name} ({field_units})   Hour: {fhour}"


def rh_700(ds):

    init_time = parser.parse(ds.START_DATE.replace('_', ' '))
    cycle = str(init_time.hour).zfill(2)
    fhour = int(ds.variables['XTIME'][0] / 60)
    fhour_str = str(fhour).zfill(2)
    valid_time = init_time + timedelta(hours=fhour)

   # Extract the pressure, geopotential height, and wind variables
    p = getvar(ds, "pressure")
    z = getvar(ds, "z", units="dm")
    rh = getvar(ds, 'rh')
    ua = getvar(ds, "ua", units="kt")
    va = getvar(ds, "va", units="kt")
    wspd = getvar(ds, "wspd_wdir", units="kts")[0, :]

    # Interpolate geopotential height, u, and v winds to 500 hPa
    ht_700 = interplevel(z, p, 700)
    u_700 = interplevel(ua, p, 700)
    v_700 = interplevel(va, p, 700)
    wspd_500 = interplevel(wspd, p, 700)
    rh_700 = interplevel(rh, p, 700)

    lats, lons = latlon_coords(ht_700)

    # Get the map projection information
    cart_proj = get_cartopy(ht_700)

    fig = plt.figure(figsize=(18, 10))
    ax = plt.axes(projection=cart_proj)

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
    counties = create_feature(COUNTY_SHAPEFILE)
    ax.add_feature(counties, facecolor='none', edgecolor='gray',)

    ht_levels = np.arange(180, 420, 3)
    contours = plt.contour(to_np(lons), to_np(lats), to_np(ht_700),
                           levels=ht_levels, colors="black",
                           transform=crs.PlateCarree())
    plt.clabel(contours, inline=1, fontsize=10, fmt="%i")

    rh_levels = [0, 1, 2, 3, 5, 10, 15, 20, 25, 30, 40,
                 50, 60, 65, 70, 75, 80, 85, 90, 95, 99, 100]
    rh_contours = plt.contourf(to_np(lons), to_np(lats), to_np(rh_700),
                               levels=rh_levels,
                               cmap=get_cmap("BrBG"),
                               transform=crs.PlateCarree())
    plt.colorbar(rh_contours, ax=ax, orientation="vertical", pad=.05)

    plt.barbs(to_np(lons[::7, ::7]), to_np(lats[::7, ::7]),
              to_np(u_700[::7, ::7]), to_np(v_700[::7, ::7]),
              transform=crs.PlateCarree(), length=5)

    # Set the map bounds
    ax.set_xlim(cartopy_xlim(ht_700))
    ax.set_ylim(cartopy_ylim(ht_700))

    title = plot_title(init_time, valid_time, fhour,
                       'Relative Humidity', 'danwrf', '%')
    plt.title(title)

    return fig


def vort_500(ds):

    init_time = parser.parse(ds.START_DATE.replace('_', ' '))
    cycle = str(init_time.hour).zfill(2)
    fhour = int(ds.variables['XTIME'][0] / 60)
    fhour_str = str(fhour).zfill(2)
    valid_time = init_time + timedelta(hours=fhour)

   # Extract the pressure, geopotential height, and wind variables
    p = getvar(ds, "pressure")
    z = getvar(ds, "z", units="dm")
    abs_vort = getvar(ds, 'avo')
    ua = getvar(ds, "ua", units="kt")
    va = getvar(ds, "va", units="kt")
    wspd = getvar(ds, "wspd_wdir", units="kts")[0, :]

    # Interpolate geopotential height, u, and v winds to 500 hPa
    ht_500 = interplevel(z, p, 500)
    u_500 = interplevel(ua, p, 500)
    v_500 = interplevel(va, p, 500)
    wspd_500 = interplevel(wspd, p, 500)
    abs_vort_500 = interplevel(abs_vort, p, 500)  # in 10^-5

    abs_vort_500 = np.clip(abs_vort_500, a_min=0, a_max=200)

    lats, lons = latlon_coords(ht_500)

    rel_vort_500 = abs_vort_500 - (coriolis_parameter(lats) * 10**5)

    # Get the map projection information
    cart_proj = get_cartopy(ht_500)

    fig = plt.figure(figsize=(18, 10))
    ax = plt.axes(projection=cart_proj)

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

    ht_levels = np.arange(492, 594, 3)
    contours = plt.contour(to_np(lons), to_np(lats), to_np(ht_500),
                           levels=ht_levels, colors="black",
                           transform=crs.PlateCarree())
    plt.clabel(contours, inline=1, fontsize=10, fmt="%i")

    cmap = mcolors.ListedColormap(VORT_CMAP)
    norm = mcolors.BoundaryNorm(VORT_LEVELS, cmap.N)

    vort_contours = plt.contourf(to_np(lons), to_np(lats), to_np(rel_vort_500),
                                 VORT_LEVELS,
                                 norm=norm,
                                 levels=VORT_LEVELS,
                                 cmap=cmap,
                                 transform=crs.PlateCarree())
    plt.colorbar(vort_contours, ax=ax, orientation="vertical", pad=.05)

    plt.barbs(to_np(lons[::7, ::7]), to_np(lats[::7, ::7]),
              to_np(u_500[::7, ::7]), to_np(v_500[::7, ::7]),
              transform=crs.PlateCarree(), length=5)

    # Set the map bounds
    ax.set_xlim(cartopy_xlim(ht_500))
    ax.set_ylim(cartopy_ylim(ht_500))

    title = plot_title(init_time, valid_time, fhour,
                       'Rel Vort', 'danwrf', '10^5 s^-1')
    plt.title(title)

    return fig
