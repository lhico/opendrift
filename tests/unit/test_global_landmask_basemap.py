import numpy as np
import pytest
from . import *
from opendrift.readers import reader_basemap_landmask
from opendrift.readers import reader_global_landmask

@pytest.mark.skip(reason="too different probably due to different masking")
def test_matches():
    print("setting up global landmask")
    reader_global = reader_global_landmask.Reader(
                        llcrnrlon=18.64, llcrnrlat=69.537,
                        urcrnrlon=19.37, urcrnrlat=69.81)

    print("setting up basemap")
    reader_basemap = reader_basemap_landmask.Reader(
                        llcrnrlon=18.64, llcrnrlat=69.537,
                        urcrnrlon=19.37, urcrnrlat=69.81, resolution = 'f')



    x = np.linspace(18.641, 19.369, 10)
    y = np.linspace(69.538, 69.80, 10)

    xx, yy = np.meshgrid(x,y)
    xx = xx.ravel()
    yy = yy.ravel()

    print ("masking against basemap")
    b = reader_basemap.__on_land__(xx,yy)

    print ("masking against global")
    c = reader_global.__on_land__(xx,yy)

    print ("checking")
    np.testing.assert_array_equal(b,c)

@pytest.mark.slow
def test_plot(tmpdir):
    print("setting up global landmask")
    reader_global = reader_global_landmask.Reader(
                        llcrnrlon=18.64, llcrnrlat=69.537,
                        urcrnrlon=19.37, urcrnrlat=69.81)

    # reader_basemap = reader_basemap_landmask.Reader(
    #                     llcrnrlon=18.64, llcrnrlat=69.537,
    #                     urcrnrlon=19.37, urcrnrlat=69.81, resolution = 'f')

    x = np.linspace(18.641, 19.369, 10)
    y = np.linspace(69.538, 69.80, 10)

    xx, yy = np.meshgrid(x,y)
    shp = xx.shape
    xx = xx.ravel()
    yy = yy.ravel()

    print ("points:", len(xx))

    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy

    reader = cartopy.feature.GSHHSFeature(scale = 'f')

    plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    c = reader_global.__on_land__(xx,yy).reshape(shp)
    # c = reader_basemap.__on_land__(xx,yy).reshape(shp)
    print (c)
    ex = [18.641, 19.369, 69.538, 69.80]
    plt.imshow(c, extent = ex, transform = ccrs.PlateCarree())
    ax.coastlines()
    # ax.set_global()
    # plt.show()
    plt.savefig('%s/cartplot.png' % tmpdir)


def test_performance_global(benchmark):
    print("setting up global landmask")
    reader_global = reader_global_landmask.Reader(
                        llcrnrlon=18.64, llcrnrlat=69.537,
                        urcrnrlon=19.37, urcrnrlat=69.81)


    x = np.linspace(18.641, 19.369, 100)
    y = np.linspace(69.538, 69.80, 100)

    xx, yy = np.meshgrid(x,y)
    xx = xx.ravel()
    yy = yy.ravel()

    print ("points:", len(xx))

    # warmup
    reader_global.__on_land__(xx, yy)

    print ("masking against cartopy")
    benchmark(reader_global.__on_land__, xx,yy)

@pytest.mark.slow
def test_performance_basemap(benchmark):
    print("setting up basemap")
    reader_basemap = reader_basemap_landmask.Reader(
                        llcrnrlon=18.64, llcrnrlat=69.537,
                        urcrnrlon=19.37, urcrnrlat=69.81, resolution = 'f')


    x = np.linspace(18.641, 19.369, 100)
    y = np.linspace(69.538, 69.80, 100)

    xx, yy = np.meshgrid(x,y)
    xx = xx.ravel()
    yy = yy.ravel()

    # warmup
    reader_basemap.__on_land__(xx, yy)

    print ("masking against basemap")
    benchmark(reader_basemap.__on_land__, xx, yy)
