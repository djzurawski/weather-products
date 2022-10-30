import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import cartopy.crs as crs
from cartopy.feature import NaturalEarthFeature
from metpy.plots import USCOUNTIES, USSTATES
import xarray as xr
import matplotlib.colors as mcolors
import numpy as np

from netCDF4 import Dataset
from dateutil import parser
from datetime import timedelta

from wrf import (
    getvar,
    interplevel,
    to_np,
    latlon_coords,
    get_cartopy,
    cartopy_xlim,
    cartopy_ylim,
)


M_PER_S_TO_KT = 1.94384
MM_TO_IN = 0.03937008

VORT_COLORS = (
    np.array(
        [
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
            (255, 201, 241),
        ]
    )
    / 255.0
)

VORT_LEVELS = [
    0.5,
    1,
    1.5,
    2,
    3,
    4,
    5,
    6,
    8,
    10,
    12,
    14,
    16,
    18,
    20,
    22,
    25,
    30,
    35,
    40,
    45,
    50,
    55,
    60,
    85,
]

PRECIP_CLEVS = [
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1,
    1.2,
    1.4,
    1.6,
    1.8,
    2.0,
    2.5,
    3,
    3.5,
    4,
    5,
    6,
    7,
    8,
    9,
]

PRECIP_CMAP_DATA = (
    np.array(
        [
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
            (240, 240, 210),
        ]
    )
    / 255.0
)

RH_LEVELS = [
    0,
    1,
    2,
    3,
    5,
    10,
    15,
    20,
    25,
    30,
    40,
    50,
    60,
    65,
    70,
    75,
    80,
    85,
    90,
    95,
    99,
    100,
]


def get_barb_interval(domain):
    if domain == "d02":
        return 6
    else:
        return 8


def coriolis_parameter(lat_degrees):
    lat_rads = lat_degrees * (np.pi / 180)
    f = 2 * 7.2921e-5 * np.sin(lat_rads)
    return f


def make_title_str(init_dt, valid_dt, fhour, field_name, model_name="", field_units=""):

    date_format = "%Y-%m-%dT%HZ"
    init_str = init_dt.strftime(date_format)
    valid_str = valid_dt.strftime(date_format)
    fhour = str(fhour).zfill(2)

    return f"{model_name}   Init: {init_str}    Valid: {valid_str}    {field_name} ({field_units})   Hour: {fhour}"


def add_title(
    fig, ax, init_dt, valid_dt, fhour, field_name, model_name="", field_units=""
):
    text = make_title_str(init_dt, valid_dt, fhour, field_name, model_name, field_units)

    fig.title(text)
    return fig, ax


def create_basemap(projection=crs.PlateCarree()):

    fig, ax = plt.subplots(figsize=(18, 10), subplot_kw={"projection": projection})
    # fig = plt.figure(figsize=(18, 10))
    # ax = plt.axes(projection=projection)

    border_scale = "50m"
    county_scale = "20m"

    ax.coastlines(border_scale, linewidth=0.8)

    lakes = NaturalEarthFeature(
        "physical", "lakes", border_scale, edgecolor="blue", facecolor="none"
    )
    ax.add_feature(lakes, facecolor="none", edgecolor="blue", linewidth=0.5)
    ax.add_feature(USCOUNTIES.with_scale(county_scale), edgecolor="gray")
    ax.add_feature(USSTATES.with_scale(county_scale), edgecolor="black")

    return fig, ax


def add_contour(
    fig,
    ax,
    lons,
    lats,
    data,
    levels=None,
):

    contours = ax.contour(
        lons,
        lats,
        data,
        levels=levels,
        colors="black",
        transform=crs.PlateCarree(),
    )
    ax.clabel(contours, inline=1, fontsize=10, fmt="%i")
    return fig, ax


def add_contourf(
    fig,
    ax,
    lons,
    lats,
    data,
    levels=None,
    colors=None,
    cmap=None,
):

    if colors is not None and levels is not None:
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(levels, cmap.N)
    elif cmap is not None and levels is not None:
        norm = mcolors.BoundaryNorm(levels, cmap.N)
    else:
        cmap = None
        norm = None

    contours = ax.contourf(
        lons,
        lats,
        data,
        levels=levels,
        norm=norm,
        cmap=cmap,
        transform=crs.PlateCarree(),
    )

    """
    contours = ax.pcolormesh(
        lons,
        lats,
        data,
        # levels,
        # levels=levels,
        norm=norm,
        cmap=cmap,
        transform=crs.PlateCarree(),
    )
    """
    fig.colorbar(contours, ax=ax, orientation="vertical", pad=0.05)
    return fig, ax


def add_wind_barbs(
    fig,
    ax,
    lons,
    lats,
    u,
    v,
    barb_length=5.5,
    # barb_interval=8,
):
    # step = barb_interval
    step = int(lons.shape[0] // 20)

    u = np.array(u)
    v = np.array(v)

    ax.barbs(
        lons[::step, ::step],
        lats[::step, ::step],
        u[::step, ::step],
        v[::step, ::step],
        transform=crs.PlateCarree(),
        length=barb_length,
    )
    return fig, ax


def add_label_markers(fig, ax, labels):
    """labels: ('text', (lon, lat))"""
    for label in labels:
        text, coords = label
        lon, lat = coords
        ax.text(lon, lat, text, horizontalalignment="left", transform=crs.PlateCarree())
        ax.plot(
            lon, lat, markersize=2, marker="o", color="k", transform=crs.PlateCarree()
        )

    return fig, ax


def plot_precip(lons, lats, precip_in, **kwargs):
    projection = kwargs.get("projection", crs.PlateCarree())

    fig, ax = create_basemap(projection=projection)

    fig, ax = add_contourf(
        fig,
        ax,
        lons,
        lats,
        precip_in,
        PRECIP_CLEVS,
        PRECIP_CMAP_DATA,
    )

    if "u10" in kwargs and "v10" in kwargs:
        u10 = kwargs["u10"]
        v10 = kwargs["v10"]
        fig, ax = add_wind_barbs(fig, ax, lons, lats, u10, v10)

    return fig


def plot_swe(lons, lats, swe_in, **kwargs):
    projection = kwargs.get("projection", crs.PlateCarree())

    fig, ax = create_basemap(projection=projection)

    fig, ax = add_contourf(
        fig,
        ax,
        lons,
        lats,
        swe_in,
        PRECIP_CLEVS,
        PRECIP_CMAP_DATA,
    )

    if "u10" in kwargs and "v10" in kwargs:
        u10 = kwargs["u10"]
        v10 = kwargs["v10"]
        fig, ax = add_wind_barbs(fig, ax, lons, lats, u10, v10)

    if "labels" in kwargs:
        labels = kwargs["labels"]
        fig, ax = add_label_markers(fig, ax, labels)

    return fig, ax


def plot_500_vorticity(lons, lats, hgt_500, vort_500, u_500, v_500, **kwargs):
    projection = kwargs.get("projection", crs.PlateCarree())

    fig, ax = create_basemap(projection=projection)

    hgt_500_levels = np.arange(492, 594, 3)

    fig, ax = add_contour(fig, ax, lons, lats, hgt_500, hgt_500_levels)

    fig, ax = add_contourf(
        fig,
        ax,
        lons,
        lats,
        vort_500,
        levels=VORT_LEVELS,
        colors=VORT_COLORS,
    )

    fig, ax = add_wind_barbs(fig, ax, lons, lats, u_500, v_500)

    if "title" in kwargs:
        ax.set_title(kwargs["title"])

    return fig, ax


def plot_700_rh(lons, lats, hgt_700, rh_700, u_700, v_700, **kwargs):
    projection = kwargs.get("projection", crs.PlateCarree())
    fig, ax = create_basemap(projection=projection)

    hgt_700_levels = np.arange(180, 420, 3)

    rh_clevels = [
        0,
        1,
        2,
        3,
        5,
        10,
        15,
        20,
        25,
        30,
        40,
        50,
        60,
        65,
        70,
        75,
        80,
        85,
        90,
        95,
        99,
        100,
    ]

    fig, ax = add_contour(
        fig,
        ax,
        lons,
        lats,
        hgt_700,
        hgt_700_levels,
    )

    fig, ax = add_contourf(
        fig,
        ax,
        lons,
        lats,
        rh_700,
        levels=rh_clevels,
        cmap=get_cmap("BrBG"),
    )

    fig, ax = add_wind_barbs(
        fig,
        ax,
        lons,
        lats,
        u_700,
        v_700,
    )

    if "title" in kwargs:
        ax.set_title(kwargs["title"])

    return fig, ax


def test500():
    # ds = xr.open_dataset("/home/dan/Documents/weather/wrfprd/d01_08")
    ds = Dataset("/home/dan/Documents/weather/wrfprd/d01_08")

    init_time = parser.parse(ds.START_DATE.replace("_", " "))
    fhour = int(ds.variables["XTIME"][0] / 60)
    valid_time = init_time + timedelta(hours=fhour)

    title = make_title_str(
        init_time, valid_time, fhour, "Rel Vort", "Danwrf", "10^5 s^-1"
    )

    p = getvar(ds, "pressure")
    z = getvar(ds, "z", units="dm")
    abs_vort = getvar(ds, "avo")
    ua = getvar(ds, "ua", units="kt")
    va = getvar(ds, "va", units="kt")

    # Interpolate geopotential height, u, and v winds to 500 hPa
    ht_500 = interplevel(z, p, 500)
    u_500 = interplevel(ua, p, 500)
    v_500 = interplevel(va, p, 500)
    abs_vort_500 = interplevel(abs_vort, p, 500)  # in 10^-5

    # abs_vort_500 = np.clip(abs_vort_500, a_min=0, a_max=200)

    lats, lons = latlon_coords(ht_500)
    lats_wind, lons_wind = latlon_coords(u_500)

    rel_vort_500 = abs_vort_500 - (coriolis_parameter(lats) * 10**5)

    fig, ax = plot_500_vorticity(
        lons_wind, lats_wind, ht_500, rel_vort_500, u_500, v_500, title=title
    )

    fig.show()


def test700():
    # ds = xr.open_dataset("/home/dan/Documents/weather/wrfprd/d01_08")
    ds = Dataset("/home/dan/Documents/weather/wrfprd/d01_08")

    init_time = parser.parse(ds.START_DATE.replace("_", " "))
    fhour = int(ds.variables["XTIME"][0] / 60)
    valid_time = init_time + timedelta(hours=fhour)

    title = title_str(init_time, valid_time, fhour, "Rel Vort", "Danwrf", "10^5 s^-1")

    p = getvar(ds, "pressure")
    z = getvar(ds, "z", units="dm")
    # ua = getvar(ds, "ua", units="kt")
    # va = getvar(ds, "va", units="kt")
    ua = getvar(ds, "U")
    va = getvar(ds, "V")

    # Interpolate geopotential height, u, and v winds to 700 hPa
    ht_700 = interplevel(z, p, 700)
    u_700 = interplevel(ua, p, 700)
    v_700 = interplevel(va, p, 700)
    rh = getvar(ds, "rh")
    rh_700 = interplevel(rh, p, 700)

    # abs_vort_700 = np.clip(abs_vort_700, a_min=0, a_max=200)

    lats, lons = latlon_coords(ht_700)

    fig, ax = plot_700_rh(lons, lats, ht_700, rh_700, u_700, v_700, title=title)

    fig.show()


def testbarbs():
    # ds = xr.open_dataset("/home/dan/Documents/weather/wrfprd/d01_08")
    ds = Dataset("/home/dan/Documents/weather/wrfprd/d01_08")

    init_time = parser.parse(ds.START_DATE.replace("_", " "))
    fhour = int(ds.variables["XTIME"][0] / 60)
    valid_time = init_time + timedelta(hours=fhour)

    title = make_title_str(
        init_time, valid_time, fhour, "Rel Vort", "Danwrf", "10^5 s^-1"
    )

    p = getvar(ds, "pressure")
    z = getvar(ds, "z", units="dm")
    abs_vort = getvar(ds, "avo")
    # ua = getvar(ds, "ua", units="kt")
    # va = getvar(ds, "va", units="kt")

    ua = getvar(ds, "ua", units="kt")
    va = getvar(ds, "va", units="kt")

    uv = getvar(ds, "uvmet", units="kt")
    ua = uv[0]
    va = uv[1]

    # Interpolate geopotential height, u, and v winds to 500 hPa
    ht_500 = interplevel(z, p, 500)
    u_500 = np.array(interplevel(ua, p, 500))
    v_500 = np.array(interplevel(va, p, 500))
    abs_vort_500 = interplevel(abs_vort, p, 500)  # in 10^-5

    # abs_vort_500 = np.clip(abs_vort_500, a_min=0, a_max=200)

    lats, lons = latlon_coords(ht_500, as_np=True)
    projection = get_cartopy(ht_500)
    # projection = crs.PlateCarree()
    import pyproj

    lam_x, lam_y = pyproj.Proj(projection)(lons, lats)

    rel_vort_500 = abs_vort_500 - (coriolis_parameter(lats) * 10**5)

    fig, ax = create_basemap(projection)

    step = 6
    barb_length = 5.5

    # lats_wind, lons_wind = latlon_coords(u_500)

    """
    ax.barbs(
        lam_x[::step, ::step],
        lam_y[::step, ::step],
        u_500[::step, ::step],
        v_500[::step, ::step],
        length=barb_length,
    )
    """

    ax.barbs(
        lons[::step, ::step],
        lats[::step, ::step],
        u_500[::step, ::step],
        v_500[::step, ::step],
        transform=crs.PlateCarree(),
        length=barb_length,
    )

    hgt_500_levels = np.arange(492, 594, 3)
    fig, ax = add_contour(fig, ax, lons, lats, ht_500, hgt_500_levels)

    fig.show()


def plot_terrain():

    CO_LABELS = [
        ("Boulder", (-105.27, 40.01)),
        ("WinterPark", (-105.77, 39.867)),
        ("Abasin", (-105.876, 39.63)),
        ("Copper", (-106.15, 39.48)),
        ("Eldora", (-105.6, 39.94)),
        ("Steamboat", (-106.75, 40.45)),
        ("Vail", (-106.37, 39.617)),
    ]

    # ds = Dataset("geo_em.d01.nc")
    ds = Dataset("4km.nc")

    h = ds["HGT_M"][0] * 3.281
    lats = ds["XLAT_M"][0]
    lons = ds["XLONG_M"][0]

    projection = crs.LambertConformal(central_longitude=-105, central_latitude=40)

    fig, ax = create_basemap(projection)

    levels = np.linspace(4000, 13000, 100)
    cmap = get_cmap("BrBG")
    norm = mcolors.BoundaryNorm(levels, cmap.N)
    ax.pcolormesh(lons, lats, h, cmap=cmap, norm=norm, transform=crs.PlateCarree())

    fig, ax = add_label_markers(fig, ax, CO_LABELS)

    fig.show()
