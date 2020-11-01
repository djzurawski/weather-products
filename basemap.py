class Basemap:
    def __init__(self,
                 extent,
                 central_longitude = -96,
                 display_counties = False,
                 name=''):
        self.name = name
        self.extent = extent
        self.central_longitude = central_longitude
        self.display_counties = display_counties


WASHINGTON = Basemap([-126, -116, 45, 50.5],
                     -121,
                     True,)

COLORADO = Basemap([-109.5, -103.1, 35.4, 42.2],
                     -106,
                     True,
                   "Colorado")

WASATCH = Basemap([-115.5, -108.5, 37.5, 42.6],
                     -112.5,
                     True,
                  "wasatch")

CONUS = Basemap([-120, -74, 23, 51])

DANWRF = Basemap([-113, -103.1, 35.4, 42.2],
                 -107,
                 True)
