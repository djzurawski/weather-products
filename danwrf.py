from netCDF4 import Dataset
from datetime import timedelta
import os
from dateutil import parser
import numpy as np
import multiprocessing as mp
import argparse


import matplotlib.pyplot as plt

import plot2

import cartopy.crs as crs

from wrf import (
    getvar,
    interplevel,
    latlon_coords,
    get_cartopy,
)


UT_NC_DIR = "/home/dan/uems/runs/wasatch/wrfprd"
CO_NC_DIR = "/home/dan/uems/runs/colorado5km/wrfprd"

CO_LABELS = [
    ("Boulder", (-105.27, 40.01)),
    ("WinterPark", (-105.77, 39.867)),
    ("Abasin", (-105.876, 39.63)),
    ("Copper", (-106.15, 39.48)),
    ("Eldora", (-105.6, 39.94)),
    ("Steamboat", (-106.75, 40.45)),
    ("Vail", (-106.37, 39.617)),
]

# NC_DIR = '/home/dan/Documents/wrfprd'
# NC_DIR = '/home/dan/Documents/wrfprd_ut'
# IMAGE_DIR = "wrf_prod/images"

MM_TO_IN = 0.03937008


def k_to_f(k):
    f = (k - 273.15) * (9/5) + 32
    return f


def coriolis_parameter(lat_degrees):
    lat_rads = lat_degrees * (np.pi / 180)
    f = 2 * 7.2921e-5 * np.sin(lat_rads)
    return f


def domain_netcdf_files(wrf_domain="d02", path=UT_NC_DIR):
    domain_files = sorted([f for f in os.listdir(path) if wrf_domain in f])
    return domain_files


def accumulated_swe_plots(
    wrfprd_dir,
    domain_name,
    wrf_domain="d02",
    labels=[],
):

    for nc_file in domain_netcdf_files(path=wrfprd_dir, wrf_domain=wrf_domain):

        ds = Dataset(wrfprd_dir + "/" + nc_file)

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
        print("saving swe", domain_name, cycle, fhour)

        mid_lon = np.median(lons)
        mid_lat = np.median(lats)
        projection = crs.LambertConformal(
            central_latitude=mid_lat, central_longitude=mid_lon
        )

        fig, ax = plot2.plot_swe(
            lons, lats, swe_in, u10=u_10, v10=v_10, labels=labels, projection=projection
        )

        ax.set_title(title)
        fig.savefig(
            f"wrf_prod/images/{cycle}z/{domain_name}-{cycle}z-swe-{fhour_str}.png",
            bbox_inches="tight",
        )
        plt.close(fig)


def accumulated_precip_plots(wrfprd_dir, domain_name, wrf_domain="d02", labels=[]):

    for nc_file in domain_netcdf_files(path=wrfprd_dir, wrf_domain=wrf_domain):

        ds = Dataset(wrfprd_dir + "/" + nc_file)

        precip_in = ds.variables["RAINNC"][0] * MM_TO_IN
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
            init_time, valid_time, fhour, "precip", "danwrf", "in"
        )
        print("saving precip", domain_name, cycle, fhour)

        mid_lon = np.median(lons)
        mid_lat = np.median(lats)
        projection = crs.LambertConformal(
            central_latitude=mid_lat, central_longitude=mid_lon
        )

        fig, ax = plot2.plot_swe(
            lons,
            lats,
            precip_in,
            u10=u_10,
            v10=v_10,
            labels=labels,
            projection=projection,
        )

        ax.set_title(title)
        fig.savefig(
            f"wrf_prod/images/{cycle}z/{domain_name}-{cycle}z-precip-{fhour_str}.png",
            bbox_inches="tight",
        )
        plt.close(fig)


def temp_2m_plot(nc_path, domain_name):
    ds = Dataset(nc_path)
    init_time = parser.parse(ds.START_DATE.replace("_", " "))
    cycle = str(init_time.hour).zfill(2)
    fhour = int(ds.variables["XTIME"][0] / 60)
    valid_time = init_time + timedelta(hours=fhour)
    fhour_str = str(fhour).zfill(2)

    temp_k = ds.variables["T2"][0]
    temp_f = k_to_f(temp_k)
    lons = np.array(ds.variables["XLONG"][0])
    lats = np.array(ds.variables["XLAT"][0])

    u_10 = ds.variables["U10"][0]
    v_10 = ds.variables["V10"][0]

    mid_lon = np.median(lons)
    mid_lat = np.median(lats)
    projection = crs.LambertConformal(
        central_latitude=mid_lat, central_longitude=mid_lon
    )

    title = plot2.make_title_str(init_time, valid_time, fhour, "temp", "danwrf", "F")

    fig, ax = plot2.plot_temp_2m(
        lons,
        lats,
        temp_f,
        u10=u_10,
        v10=v_10,
        projection=projection,
    )

    ax.set_title(title)
    fig.savefig(
        f"wrf_prod/images/{cycle}z/{domain_name}-{cycle}z-2m_temp-{fhour_str}.png",
        bbox_inches="tight",
    )
    plt.close(fig)


def vort_500_plot(nc_path, domain_name):
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

    print("saving vort 500", domain_name, cycle, fhour)
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
        f"wrf_prod/images/{cycle}z/{domain_name}-{cycle}z-vort500-{fhour_str}.png",
        bbox_inches="tight",
    )

    plt.close(fig)


def temp_2m_plots(wrfprd_dir, domain_name, wrf_domain="d01"):
    nc_paths = [
        wrfprd_dir + "/" + nc_file
        for nc_file in domain_netcdf_files(path=wrfprd_dir, wrf_domain=wrf_domain)
    ]

    for nc_path in nc_paths:
        temp_2m_plot(nc_path, domain_name)


def vort_500_plots(wrfprd_dir, domain_name, wrf_domain="d01"):
    nc_paths = [
        wrfprd_dir + "/" + nc_file
        for nc_file in domain_netcdf_files(path=wrfprd_dir, wrf_domain=wrf_domain)
    ]

    for nc_path in nc_paths:
        vort_500_plot(nc_path, domain_name)


def rh_700_plot(nc_path, domain_name):
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

    print("saving rh 700", domain_name, cycle, fhour)
    fig, ax = plot2.plot_700_rh(
        lons, lats, ht_700, rh_700, u_700, v_700, title=title, projection=projection
    )

    fig.savefig(
        f"wrf_prod/images/{cycle}z/{domain_name}-{cycle}z-rh700-{fhour_str}.png",
        bbox_inches="tight",
    )

    plt.close(fig)


def rh_700_plots(wrfprd_dir, domain_name, wrf_domain):

    nc_paths = [
        wrfprd_dir + "/" + nc_file
        for nc_file in domain_netcdf_files(path=wrfprd_dir, wrf_domain=wrf_domain)
    ]

    for nc_path in nc_paths:
        rh_700_plot(nc_path, domain_name)


def tst():
    d = "/home/dan/Documents/weather/wrfprd"
    rh_700_plots(d, wrf_domain="d02")


def tst2():
    ds = "/home/dan/Documents/weather/wrfprd/d02_08"
    # vort_500_plot(ds, wrf_domain="d02")
    # vort_50_plot(ds, wrf_domain="d02")


def tst3():
    accumulated_swe_plots(
        wrfprd_dir="/home/dan/Documents/weather/wrfprd/",
        wrf_domain="d02",
        labels=[],
        central_longitude=-106.5,
    )


def error_callback(e):
    print(e)


def main(wrfprd_path, domain_name, wrf_domain="d01", labels=[]):

    # Do it this way because mp.Pool() freezes computer when using after calling
    # accumulated_swe_plots()
    with mp.Pool() as pool:
        pool.apply_async(
            accumulated_swe_plots,
            (wrfprd_path, domain_name, wrf_domain, labels),
            error_callback=error_callback,
        )
        pool.apply_async(
            accumulated_precip_plots,
            (wrfprd_path, domain_name, wrf_domain, labels),
            error_callback=error_callback,
        )
        pool.apply_async(
            rh_700_plots,
            (wrfprd_path, domain_name, wrf_domain),
            error_callback=error_callback,
        )
        pool.apply_async(
            vort_500_plots,
            (wrfprd_path, domain_name, wrf_domain),
            error_callback=error_callback,
        )

        pool.apply_async(
            temp_2m_plots,
            (wrfprd_path, domain_name, wrf_domain),
            error_callback=error_callback,
        )

        pool.close()
        pool.join()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Danwrf plot generator")
    argparser.add_argument("-p", "--wrfprd-path", type=str, required=True)
    argparser.add_argument("-d", "--domain-name", type=str, required=True)
    argparser.add_argument("-n", "--num-nests", type=int, default=1)

    args = argparser.parse_args()

    domain_name = args.domain_name
    wrfprd_path = args.wrfprd_path
    num_nests = args.num_nests

    wrf_domains = ["d0" + str(i) for i in range(1, num_nests + 1)]
    domain_names = [f"{domain_name}-{d}" for d in wrf_domains]

    for domain_name, wrf_domain in zip(domain_names, wrf_domains):
        print(wrfprd_path, domain_name, wrf_domain)
        main(wrfprd_path, domain_name, wrf_domain)
