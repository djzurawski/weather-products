#!/bin/bash

curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
source $HOME/.poetry/env

sudo apt-get install -y python3-venv libproj-dev

poetry install

pip uninstall shapely
pip install --no-binary :all: shapely

curl -s  https://prd-tnm.s3.amazonaws.com/StagedProducts/Small-scale/data/Boundaries/countyp010g.shp_nt00934.tar.gz | tar xvz -C  $(dirname $0)/../resources

#curl -s https://prd-tnm.s3.amazonaws.com/StagedProducts/Small-scale/data/Hydrography/wtrbdyp010g.shp_nt00886.tar.gz | tar xvz -C  $(dirname $0)/../resources

curl -s https://prd-tnm.s3.amazonaws.com/StagedProducts/Small-scale/data/Boundaries/statesp010g.shp_nt00938.tar.gz | tar xvz -C  $(dirname $0)/../resources
