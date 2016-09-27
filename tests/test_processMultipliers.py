"""
Unit test suite for processMultipliers.py


"""

import sys
import os
from os.path import join as pjoin, exists
import unittest
import tempfile

from numpy.testing import assert_almost_equal
import numpy as np

from osgeo import osr, gdal
from osgeo.gdalconst import *
from netCDF4 import Dataset

import processMultipliers as pM
from pathLocator import getRootDirectory

class TestProcessMultipliers(unittest.TestCase):

    """
    TestProcessMultipliers: test the processMultipliers script
    """

    def setUp(self):
        self.array = np.arange(100, dtype=float).reshape((10, 10))
        self.x = np.arange(120, 130)
        self.y = np.arange(10, 20)
        self.dx = 1
        self.dy = -1
        self.testRasterFile = "testRaster.tif"
        self.reprojectRaster = "testReprojection.tif" # is this used?
        self.projectedDataset = pM.createRaster(np.ones((2000, 5000)),
                                                np.arange(15000, 55000, 20),
                                                np.arange(20000, 120000, 20),
                                                dx=20, dy=-20, epsg=3123)

        self.uu = np.array([-1., 1., 1., -1.])
        self.vv = np.array([-1., -1., 1., 1.])
        self.bearing = np.array([45., 315., 225., 135.])

    def test_createRaster(self):
        """Test createRaster returns a gdal dataset"""

        result = pM.createRaster(self.array, self.x, self.y,
                                 self.dx, self.dy,
                                 filename=self.testRasterFile)
        self.assertEqual(type(result), gdal.Dataset)
        assert exists(self.testRasterFile)

    def test_createRasterII(self):
        """Test can create img files"""
        file_is = 'test.img'
        result = pM.createRaster(self.array, self.x, self.y,
                                 self.dx, self.dy,
                                 filename=file_is)
        self.assertEqual(type(result), gdal.Dataset)
        assert exists(file_is)

    def test_loadRasterFile(self):
        """Test loadRasterFile correctly loads data"""

        result = pM.loadRasterFile(self.testRasterFile, -9999)
        self.assertEqual(type(result), np.ndarray)
        assert_almost_equal(result, self.array[::np.sign(self.dy) * 1])

    def test_calculateBearing(self):
        """Test the correct bearings are returned"""
        bb = pM.calculateBearing(self.uu, self.vv)
        assert_almost_equal(bb, self.bearing)

    def test_reprojectDataset(self):
        """Test a dataset is correctly reprojected"""

        pM.reprojectDataset(self.testRasterFile, self.projectedDataset,
                            self.reprojectRaster)
        assert exists(self.reprojectRaster)
        prjDataset = gdal.Open(self.reprojectRaster)
        prjBand = prjDataset.GetRasterBand(1)
        prj_data = prjBand.ReadAsArray()
        # Check shape of arrays:
        self.assertEqual((2000, 5000), prj_data.shape)

        # Check geographic transform:
        self.assertEqual(prjDataset.GetGeoTransform(),
                         self.projectedDataset.GetGeoTransform())
        # Check projection: FIXME GetProjection() from projected dataset
        # drops AXIS["X",EAST],AXIS["Y",NORTH], from the projection
        # information compared to the match dataset.
        # self.assertEqual(prjDataset.GetProjection(),
        #                 self.projectedDataset.GetProjection())
        # Check values are correctly mapped:

        del prjDataset

    def test_reprojectDataset_same_nc_img(self):
        """Test a dataset is correctly reprojected"""
        # Write a .nc file to test
        f_nc = tempfile.NamedTemporaryFile(suffix='.nc',
                                        prefix='test_processMultipliers',
                                        delete=False)
        f_nc.close()


        # Write an .img file to test
        f_img = tempfile.NamedTemporaryFile(suffix='.img',
                                        prefix='test_processMultipliers',
                                        delete=False)
        f_img.close()

        multiplier_name = 'vmax' # what the?
        lat = np.asarray([ -23, -20, -17, -14, -11, -8, -5])
        lon = np.asarray([137, 140, 143, 146, 149, 152, 155, 158])
        dx = dy = 3
        multiplier_values = np.zeros(([lon.shape[0], lat.shape[0]]))

        multiplier_values.fill(42.5)
        #save_multiplier(multiplier_name, multiplier_values, lat,
         #               lon, f_nc.name)

        pM.createRaster(multiplier_values, lon, lat,
                        dx, dy,
                        filename=f_img.name)

        m4_max_file = f_img.name
        # pulling out a section of the processMultipliers.main
        ncobj = Dataset(f_nc.name, 'r')

        lat = ncobj.variables['lat'][:]
        lon = ncobj.variables['lon'][:]

        delta = lon[1] - lon[0]
        lon = lon - delta / 2.
        lat = lat - delta / 2.

        # Wind speed:
        wspd = ncobj.variables['vmax'][:]
        wind_raster_file = 'region_wind.tif'
        wind_prj_file = 'gust_prj.tif'

        wind_raster = pM.createRaster(wspd, lon, lat, delta, -delta,
                                   filename=wind_raster_file)
        wind_prj = pM.reprojectDataset(wind_raster, m4_max_file,
                                       wind_prj_file,
                                       match_projection=32756)

        os.remove(f_nc.name)
        os.remove(f_img.name)


if __name__ == "__main__":
    unittest.main()
