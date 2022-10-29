#!/bin/bash

sudo apt-get install -y libproj-dev build-essential gfortran g++ gcc python3-dev libgeos-dev

mkdir -p $(dirname $0)/href_prod/grib/12z
mkdir -p $(dirname $0)/href_prod/grib/00z
mkdir -p $(dirname $0)/href_prod/images

curl -s  https://prd-tnm.s3.amazonaws.com/StagedProducts/Small-scale/data/Boundaries/countyp010g.shp_nt00934.tar.gz | tar xvz -C  $(dirname $0)/../resources
curl -s https://prd-tnm.s3.amazonaws.com/StagedProducts/Small-scale/data/Hydrography/wtrbdyp010g.shp_nt00886.tar.gz | tar xvz -C  $(dirname $0)/../resources
curl -s https://prd-tnm.s3.amazonaws.com/StagedProducts/Small-scale/data/Boundaries/statesp010g.shp_nt00938.tar.gz | tar xvz -C  $(dirname $0)/../resources

wget https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_county_20m.zip
unzip cb_2018_us_county_20m.zip
mv cb_* $(dirname $0)/../resources/
