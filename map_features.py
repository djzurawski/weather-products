import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader


def create_feature(shapefile, projection=ccrs.PlateCarree()):
	reader = shpreader.Reader(shapefile)
	feature = list(reader.geometries())
	return cfeature.ShapelyFeature(feature, projection)
