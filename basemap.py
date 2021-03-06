class Basemap:
    def __init__(self,
                 extent,
                 central_longitude=-96,
                 display_counties=False,
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
                   labels=[('Boulder', (-105.27, 40.01)),
                           ('WinterPark', (-105.77, 39.867)),
                           ('Abasin', (-105.876, 39.63)),
                           ('Copper', (-106.15, 39.48)),
                           ('Eldora', (-105.6, 39.94)),
                           ('Steamboat', (-106.75, 40.45)), ])

NORCO = Basemap([-107.9, -104.63, 38.25, 40.9],
                -106,
                True,
                "norco",
                labels=[('Boulder', (-105.27, 40.01)),
                        ('WinterPark', (-105.77, 39.867)),
                        ('Abasin', (-105.876, 39.63)),
                        ('Copper', (-106.15, 39.48)),
                        ('Eldora', (-105.6, 39.94)),
                        ('Steamboat', (-106.75, 40.45)),
                        ('Vail', (-106.37, 39.617))])

WASATCH = Basemap([-113, -109, 39.15, 42.6],
                  -111,
                  True,
                  "wasatch",
                  labels=[('Alta', (-111.62, 40.574)),
                          ('Solitude', (-111.6, 40.61)), ])

UT_D2 = Basemap([-113.39, -110.0, 39.3, 41.8],
                -111.75,
                True,
                "UT2.6km",
                labels=[('Alta', (-111.62, 40.574)), ])

CO_D2 = Basemap([-107.9, -104.6, 38.26, 40.76],
                -105.5,
                True,
                "CO2.6km",
                labels=[('Boulder', (-105.27, 40.01)),
                        ('WinterPark', (-105.77, 39.867)),
                        ('Abasin', (-105.876, 39.63)),
                        ('Copper', (-106.15, 39.48)),
                        ('Eldora', (-105.6, 39.94)),
                        ('Steamboat', (-106.75, 40.45)),
                        ('Vail', (-106.37, 39.617))])

COTTONWOODS = Basemap([-112.75, -111.2, 40.00, 41.2],
                      -112,
                      True,
                      "cottonwoods",
                      labels=[('Alta', (-111.62, 40.574)),
                              ('Solitude', (-111.6, 40.61)),
                              ('ParkCity', (-111.5, 40.63)), ])

CONUS = Basemap([-120, -74, 23, 51])

DANWRF = Basemap([-113, -103.1, 35.4, 42.2],
                 -107,
                 True)
