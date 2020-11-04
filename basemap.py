class Basemap:
    def __init__(self,
                 extent,
                 central_longitude = -96,
                 display_counties = False,
                 name='',
                 labels=[]):
        self.name = name
        self.extent = extent
        self.central_longitude = central_longitude
        self.display_counties = display_counties
        self.labels = labels


WASHINGTON = Basemap([-126, -116, 45, 50.5],
                     -121,
                     True,)

COLORADO = Basemap([-109.5, -103.1, 35.4, 42.2],
                   -106,
                   True,
                   "colorado",
                   labels = [('Boulder', (-105.27, 40.01)),
                             ('Wnter Park', (-105.77, 39.867)),
                             ('Abasin', (-105.876, 39.63)),
                             ('Copper', (-106.15, 39.48)),
                             ('Eldora', (-105.6, 39.94)),])

WASATCH = Basemap([-113, -109, 39.15, 42.6],
                  -111,
                  True,
                  "wasatch",
                  labels= [('Alta', (-111.64, 40.57)),
                           ('Solitude', (-111.6, 40.61)),])

CONUS = Basemap([-120, -74, 23, 51])

DANWRF = Basemap([-113, -103.1, 35.4, 42.2],
                 -107,
                 True)
