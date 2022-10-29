from netCDF4 import Dataset
from datetime import datetime, timedelta
import os
from dateutil import parser
import numpy as np
import multiprocessing as mp

import basemap as bmap
import matplotlib.pyplot as plt
import plot
import plot2

import cartopy.crs as crs

from wrf import (
    getvar,
    interplevel,
    to_np,
    latlon_coords,
    get_cartopy,
    cartopy_xlim,
    cartopy_ylim,
)


UT_NC_DIR = "/home/dan/uems/runs/wasatch/wrfprd"
CO_NC_DIR = "/home/dan/uems/runs/colorado3nest/wrfprd"

# NC_DIR = '/home/dan/Documents/wrfprd'
# NC_DIR = '/home/dan/Documents/wrfprd_ut'
# IMAGE_DIR = "wrf_prod/images"

MM_TO_IN = 0.03937008


def coriolis_parameter(lat_degrees):
    lat_rads = lat_degrees * (np.pi / 180)
    f = 2 * 7.2921e-5 * np.sin(lat_rads)
    return f


def domain_netcdf_files(domain="d02", path=UT_NC_DIR):
    domain_files = sorted([f for f in os.listdir(path) if domain in f])
    return domain_files


def accumulated_swe_plots(
    nc_dir=UT_NC_DIR,
    domain="d02",
    labels=[],
    extent=None,
    central_longitude=-110,
):

    for nc_file in domain_netcdf_files(path=nc_dir, domain=domain):

        ds = Dataset(nc_dir + "/" + nc_file)

        swe_in = ds.variables["SNOWNC"][0] * MM_TO_IN
        lons = np.array(ds.variables["XLONG"][0])
        lats = np.array(ds.variables["XLAT"][0])

        u_10 = ds.variables["U10"][0]
        v_10 = ds.variables["V10"][0]

        init_time = parser.parse(ds.START_DATE.replace("_", " "))
        cycle = str(init_time.hour).zfill(2)
        fhour = int(ds.variables["XTIME"][0] / 60)
        fhour_str = str(fhour).zfill(2)
        valid_time = init_time + timedelta(hours=fhour)

        title = plot2.make_title_str(
            init_time, valid_time, fhour, "swe", "danwrf", "in"
        )
        print("saving swe", domain, cycle, fhour)

        mid_lon = np.median(lons)
        mid_lat = np.median(lats)
        projection = crs.LambertConformal(
            central_latitude=mid_lat, central_longitude=mid_lon
        )

        fig, ax = plot2.plot_swe(
            lons, lats, swe_in, u10=u_10, v10=v_10, labels=labels, projection=projection
        )

        ax.set_title(title)
        fig.savefig(f"wrf_prod/images/{cycle}z/{domain}-{cycle}z-swe-{fhour_str}.png")
        plt.close(fig)


def accumulated_precip_plots(
    nc_dir=UT_NC_DIR,
    domain="d02",
    labels=[],
    extent=None,
    central_longitude=-110,
):

    for nc_file in domain_netcdf_files(path=nc_dir, domain=domain):

        ds = Dataset(nc_dir + "/" + nc_file)

        swe_in = ds.variables["RAINNC"][0] * MM_TO_IN
        lons = np.array(ds.variables["XLONG"][0])
        lats = np.array(ds.variables["XLAT"][0])

        u_10 = ds.variables["U10"][0]
        v_10 = ds.variables["V10"][0]

        init_time = parser.parse(ds.START_DATE.replace("_", " "))
        cycle = str(init_time.hour).zfill(2)
        fhour = int(ds.variables["XTIME"][0] / 60)
        fhour_str = str(fhour).zfill(2)
        valid_time = init_time + timedelta(hours=fhour)

        title = plot2.make_title_str(
            init_time, valid_time, fhour, "swe", "danwrf", "in"
        )
        print("saving swe", domain, cycle, fhour)

        mid_lon = np.median(lons)
        mid_lat = np.median(lats)
        projection = crs.LambertConformal(
            central_latitude=mid_lat, central_longitude=mid_lon
        )

        fig, ax = plot2.plot_swe(
            lons, lats, swe_in, u10=u_10, v10=v_10, labels=labels, projection=projection
        )

        ax.set_title(title)
        fig.savefig(f"wrf_prod/images/{cycle}z/{domain}-{cycle}z-swe-{fhour_str}.png")
        plt.close(fig)


def vort_500_plot(nc_path, domain):
    ds = Dataset(nc_path)
    init_time = parser.parse(ds.START_DATE.replace("_", " "))
    cycle = str(init_time.hour).zfill(2)
    fhour = int(ds.variables["XTIME"][0] / 60)
    valid_time = init_time + timedelta(hours=fhour)
    fhour_str = str(fhour).zfill(2)

    p = getvar(ds, "pressure")
    z = getvar(ds, "z", units="dm")
    abs_vort = getvar(ds, "avo")
    # Seems more correct than wrf-python examples
    uv = getvar(ds, "uvmet", units="kt")
    ua = uv[0]
    va = uv[1]

    # Interpolate geopotential height, u, and v winds to 500 hPa
    ht_500 = interplevel(z, p, 500)
    u_500 = interplevel(ua, p, 500)
    v_500 = interplevel(va, p, 500)
    abs_vort_500 = interplevel(abs_vort, p, 500)  # in 10^-5

    abs_vort_500 = np.clip(abs_vort_500, a_min=0, a_max=200)

    lats, lons = latlon_coords(ht_500, as_np=True)

    rel_vort_500 = abs_vort_500 - (coriolis_parameter(lats) * 10**5)

    title = plot2.make_title_str(
        init_time, valid_time, fhour, "Rel Vort", "Danwrf", "10^5 s^-1"
    )

    projection = get_cartopy(ht_500)

    print("saving vort 500", domain, cycle, fhour)
    fig, ax = plot2.plot_500_vorticity(
        lons,
        lats,
        ht_500,
        rel_vort_500,
        u_500,
        v_500,
        title=title,
        projection=projection,
    )

    fig.savefig(
        f"wrf_prod/images/{cycle}z/{domain}-{cycle}z-vort500-{fhour_str}.png",
        bbox_inches="tight",
    )

    plt.close(fig)


def vort_500_plots(nc_dir=UT_NC_DIR, domain="d01"):
    nc_paths = [
        nc_dir + "/" + nc_file
        for nc_file in domain_netcdf_files(path=nc_dir, domain=domain)
    ]

    for nc_path in nc_paths:
        vort_500_plot(nc_path, domain)


def rh_700_plot(nc_path, domain):
    ds = Dataset(nc_path)
    init_time = parser.parse(ds.START_DATE.replace("_", " "))
    cycle = str(init_time.hour).zfill(2)
    fhour = int(ds.variables["XTIME"][0] / 60)
    fhour_str = str(fhour).zfill(2)
    valid_time = init_time + timedelta(hours=fhour)

    # Extract the pressure, geopotential height, and wind variables
    p = getvar(ds, "pressure")
    z = getvar(ds, "z", units="dm")
    rh = getvar(ds, "rh")
    # Seems more correct than wrf-python examples
    uv = getvar(ds, "uvmet", units="kt")
    ua = uv[0]
    va = uv[1]

    # Interpolate geopotential height, u, and v winds to 700 hPa
    ht_700 = interplevel(z, p, 700)
    u_700 = interplevel(ua, p, 700)
    v_700 = interplevel(va, p, 700)
    rh_700 = interplevel(rh, p, 700)

    lats, lons = latlon_coords(ht_700, as_np=True)

    title = plot2.make_title_str(
        init_time, valid_time, fhour, "Rel Vort", "Danwrf", "10^5 s^-1"
    )

    projection = get_cartopy(ht_700)

    print("saving rh 700", domain, cycle, fhour)
    fig, ax = plot2.plot_700_rh(
        lons, lats, ht_700, rh_700, u_700, v_700, title=title, projection=projection
    )

    fig.savefig(
        f"wrf_prod/images/{cycle}z/{domain}-{cycle}z-rh700-{fhour_str}.png",
        bbox_inches="tight",
    )

    plt.close(fig)


def rh_700_plots(nc_dir=UT_NC_DIR, domain="d01"):

    nc_paths = [
        nc_dir + "/" + nc_file
        for nc_file in domain_netcdf_files(path=nc_dir, domain=domain)
    ]

    for nc_path in nc_paths:
        rh_700_plot(nc_path, domain)


def tst():
    d = "/home/dan/Documents/weather/wrfprd"
    rh_700_plots(d, domain="d02")


def tst2():
    ds = "/home/dan/Documents/weather/wrfprd/d02_08"
    # vort_500_plot(ds, domain="d02")
    # vort_50_plot(ds, domain="d02")


def tst3():
    accumulated_swe_plots(
        nc_dir="/home/dan/Documents/weather/wrfprd/",
        domain="d02",
        labels=[],
        central_longitude=-106.5,
    )


def test_swe():
    f = "/home/dan/Documents/weather/wrfprd/d03_08"
    domain = "d03"

    ds = Dataset(f)

    swe_in = ds.variables["SNOWNC"][0] * MM_TO_IN
    lons = np.array(ds.variables["XLONG"][0])
    lats = np.array(ds.variables["XLAT"][0])
    u_10 = ds.variables["U10"][0]
    v_10 = ds.variables["V10"][0]

    init_time = parser.parse(ds.START_DATE.replace("_", " "))
    cycle = str(init_time.hour).zfill(2)
    fhour = int(ds.variables["XTIME"][0] / 60)
    fhour_str = str(fhour).zfill(2)
    valid_time = init_time + timedelta(hours=fhour)

    title = plot2.make_title_str(init_time, valid_time, fhour, "swe", "danwrf", "in")
    print("saving swe", domain, cycle, fhour)

    mid_lon = np.median(lons)
    mid_lat = np.median(lats)
    projection = crs.LambertConformal(
        central_latitude=mid_lat, central_longitude=mid_lon
    )

    fig, ax = plot2.plot_swe(
        lons, lats, swe_in, u10=u_10, v10=v_10, labels=[], projection=projection
    )

    ax.set_title(title)
    fig.savefig(f"wrf_prod/images/{cycle}z/{domain}-{cycle}z-swe-{fhour_str}.png")


if __name__ == "__main__":

    CO_NC_DIR = "/home/dan/Documents/weather/wrfprd/"
    """

    accumulated_swe_plots(
        nc_dir=CO_NC_DIR,
        domain="d03",
        labels=bmap.CO_D2.labels,
        central_longitude=-106.5,
    )

    accumulated_swe_plots(
        nc_dir=CO_NC_DIR,
        domain="d02",
        labels=[],
        central_longitude=-106.5,
    )

    accumulated_swe_plots(
        nc_dir=CO_NC_DIR,
        domain="d01",
        labels=[],
        central_longitude=-106.5,
    )

    rh_700_plots(nc_dir=CO_NC_DIR, domain="d01")
    rh_700_plots(nc_dir=CO_NC_DIR, domain="d02")

    vort_500_plots(nc_dir=CO_NC_DIR, domain="d01")
    vort_500_plots(nc_dir=CO_NC_DIR, domain="d02")

    """
    # Do it this way because mp.Pool() freezes computer when using after calling
    # accumulated_swe_plots()
    with mp.Pool() as pool:
        res_swe_d01 = pool.apply_async(accumulated_swe_plots, (CO_NC_DIR, "d01"))
        res_swe_d02 = pool.apply_async(accumulated_swe_plots, (CO_NC_DIR, "d02"))
        res_swe_d03 = pool.apply_async(accumulated_swe_plots, (CO_NC_DIR, "d03"))

        res_precip_d01 = pool.apply_async(accumulated_precip_plots, (CO_NC_DIR, "d01"))
        res_precip_d02 = pool.apply_async(accumulated_precip_plots, (CO_NC_DIR, "d02"))
        res_precip_d03 = pool.apply_async(accumulated_precip_plots, (CO_NC_DIR, "d03"))

        res_rh_700_d01 = pool.apply_async(rh_700_plots, (CO_NC_DIR, "d01"))
        res_rh_700_d02 = pool.apply_async(rh_700_plots, (CO_NC_DIR, "d02"))
        res_rh_700_d03 = pool.apply_async(rh_700_plots, (CO_NC_DIR, "d03"))

        res_vort_500_d01 = pool.apply_async(vort_500_plots, (CO_NC_DIR, "d01"))
        res_vort_500_d02 = pool.apply_async(vort_500_plots, (CO_NC_DIR, "d02"))
        res_vort_500_d03 = pool.apply_async(vort_500_plots, (CO_NC_DIR, "d03"))

        pool.close()
        pool.join()
