from netCDF4 import Dataset
from datetime import datetime, timedelta
import os
from dateutil import parser
import numpy as np

from href import (
    SurfacePlot,
    plot_title,
    PRECIP_CLEVS,
    PRECIP_CMAP_DATA,
    CAIC_PRECIP_CLEVS,
    CAIC_PRECIP_CMAP_DATA,
)
import basemap as bmap
import matplotlib.pyplot as plt
import plot


UT_NC_DIR = "/home/dan/uems/runs/wasatch/wrfprd"
CO_NC_DIR = "/home/dan/uems/runs/colorado/wrfprd"

# NC_DIR = '/home/dan/Documents/wrfprd'
# NC_DIR = '/home/dan/Documents/wrfprd_ut'
# IMAGE_DIR = "wrf_prod/images"

MM_TO_IN = 0.03937008


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
        lons = ds.variables["XLONG"][0]
        lats = ds.variables["XLAT"][0]

        init_time = parser.parse(ds.START_DATE.replace("_", " "))
        cycle = str(init_time.hour).zfill(2)
        fhour = int(ds.variables["XTIME"][0] / 60)
        fhour_str = str(fhour).zfill(2)
        valid_time = init_time + timedelta(hours=fhour)

        title = plot_title(init_time, valid_time, fhour, "swe", "danwrf", "in")
        print("saving swe", domain, cycle, fhour)

        plot = SurfacePlot(
            lons,
            lats,
            swe_in,
            extent=extent,
            colormap=PRECIP_CMAP_DATA,
            color_levels=PRECIP_CLEVS,
            central_longitude=central_longitude,
            labels=labels,
            display_counties=True,
            title=title,
        )

        plot.save_plot(
            f"wrf_prod/images/{cycle}z/{domain}-{cycle}z-swe-{fhour_str}.png"
        )


def vort_500_plots(nc_dir=UT_NC_DIR, domain="d01"):

    for nc_file in domain_netcdf_files(path=nc_dir, domain=domain):
        ds = Dataset(nc_dir + "/" + nc_file)

        init_time = parser.parse(ds.START_DATE.replace("_", " "))
        cycle = str(init_time.hour).zfill(2)
        fhour = int(ds.variables["XTIME"][0] / 60)
        fhour_str = str(fhour).zfill(2)
        valid_time = init_time + timedelta(hours=fhour)

        print("saving vort 500", domain, cycle, fhour)
        vort_500_plot = plot.vort_500(ds)

        vort_500_plot.savefig(
            f"wrf_prod/images/{cycle}z/{domain}-{cycle}z-vort500-{fhour_str}.png",
            bbox_inches="tight",
        )
        plt.close(vort_500_plot)


def rh_700_plots(nc_dir=UT_NC_DIR, domain="d01"):

    for nc_file in domain_netcdf_files(path=nc_dir, domain=domain):
        ds = Dataset(nc_dir + "/" + nc_file)

        init_time = parser.parse(ds.START_DATE.replace("_", " "))
        cycle = str(init_time.hour).zfill(2)
        fhour = int(ds.variables["XTIME"][0] / 60)
        fhour_str = str(fhour).zfill(2)
        valid_time = init_time + timedelta(hours=fhour)

        print("saving rh 700", domain, cycle, fhour)
        rh_700_plot = plot.rh_700(ds)

        rh_700_plot.savefig(
            f"wrf_prod/images/{cycle}z/{domain}-{cycle}z-rh700-{fhour_str}.png",
            bbox_inches="tight",
        )
        plt.close(rh_700_plot)


if __name__ == "__main__":
    """
    accumulated_swe_plots(nc_dir=UT_NC_DIR,
                          domain='d02',
                          labels=[('Alta', (-111.62, 40.574))],
                          central_longitude=-111.75,
                          domain_name="UT2.6km")
    accumulated_swe_plots(nc_dir=UT_NC_DIR,
                          domain='d01',
                          labels=[('Alta', (-111.62, 40.574))],
                          central_longitude=-111.75,
                          domain_name="UT8km")

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
    """
    accumulated_swe_plots(nc_dir=UT_NC_DIR,
                          domain='d02',
                          domain_name=bmap.COTTONWOODS.name,
                          central_longitude=bmap.COTTONWOODS.central_longitude,
                          labels=bmap.COTTONWOODS.labels,
                          extent=bmap.COTTONWOODS.extent)
    """
    # rh_700_plots(nc_dir=UT_NC_DIR, domain_name='UT8km')
    rh_700_plots(nc_dir=CO_NC_DIR, domain="d01")
    rh_700_plots(nc_dir=CO_NC_DIR, domain="d02")

    # vort_500_plots(nc_dir=UT_NC_DIR, domain_name='UT8km')
    vort_500_plots(nc_dir=CO_NC_DIR, domain="d01")
    vort_500_plots(nc_dir=CO_NC_DIR, domain="d02")

    # accumulated_swe_plots(bmap=basemap.UT_D2, nc_dir=UT_NC_DIR)
    # accumulated_swe_plots(bmap=basemap.CO_D2, nc_dir=CO_NC_DIR)

    # accumulated_swe_plots(bmap=None, nc_dir=UT_NC_DIR, domain='d01')
    # accumulated_swe_plots(bmap=None, nc_dir=NC_DIR, domain='d01')
    # accumulated_swe_plots(nc_dir=NC_DIR, domain='d01', domain_name="CO8km")
