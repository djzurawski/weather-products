from netCDF4 import Dataset
from datetime import datetime, timedelta
import os
from dateutil import parser
import numpy as np

from href import SurfacePlot, plot_title, PRECIP_CLEVS, PRECIP_CMAP_DATA, CAIC_PRECIP_CLEVS, CAIC_PRECIP_CMAP_DATA
import basemap
import matplotlib.pyplot as plt

UT_NC_DIR = "/home/dan/uems/runs/wasatch/wrfprd"
CO_NC_DIR = "/home/dan/uems/runs/colorado/wrfprd"
#NC_DIR = '/home/dan/Documents/wrfprd'
#NC_DIR = '/home/dan/Documents/wrfprd_ut'
#IMAGE_DIR = "wrf_prod/images"

MM_TO_IN = 0.03937008


def domain_netcdf_files(domain='d02', path=UT_NC_DIR):
    domain_files = sorted([f for f in os.listdir(path) if domain in f])
    return domain_files


def accumulated_swe_plots(domain='d02', bmap=basemap.COTTONWOODS, nc_dir=UT_NC_DIR):
    for nc_file in domain_netcdf_files(path=nc_dir):
        ds = Dataset(nc_dir + '/' + nc_file)

        swe_in = ds.variables['SNOWNC'][0] * MM_TO_IN
        lons = ds.variables['XLONG'][0]
        lats = ds.variables['XLAT'][0]

        init_time = parser.parse(ds.START_DATE.replace('_', ' '))
        cycle = str(init_time.hour).zfill(2)
        fhour = int(ds.variables['XTIME'][0] / 60)
        fhour_str = str(fhour).zfill(2)
        valid_time = init_time + timedelta(hours=fhour)

        title = plot_title(init_time, valid_time, fhour, 'swe', 'danwrf', 'in')
        print('saving', cycle, fhour)

        plot = SurfacePlot(lons, lats, swe_in,
                           extent=bmap.extent,
                           colormap=PRECIP_CMAP_DATA,
                           color_levels=PRECIP_CLEVS,
                           central_longitude=bmap.central_longitude,
                           labels=bmap.labels,
                           display_counties=True,
                           title = title)

        plot.save_plot(f'wrf_prod/images/{cycle}z/{bmap.name}-{cycle}z-swe-{fhour_str}.png')

if __name__ == "__main__":
    accumulated_swe_plots(bmap=basemap.COTTONWOODS, nc_dir=UT_NC_DIR)
    accumulated_swe_plots(bmap=basemap.UT_D2, nc_dir=UT_NC_DIR)
    accumulated_swe_plots(bmap=basemap.CO_D2, nc_dir=CO_NC_DIR)
