"""
Unit test suite for processMultipliers.py


"""

import sys
import os
from os.path import join as pjoin, exists
import unittest
from numpy.testing import assert_almost_equal
import numpy as np

from osgeo import osr, gdal
from osgeo.gdalconst import *

import processMultipliers as pM


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
        """Test crateRaster returns a gdal dataset"""

        result = pM.createRaster(self.array, self.x, self.y,
                                 self.dx, self.dy,
                                 filename=self.testRasterFile)
        self.assertEqual(type(result), gdal.Dataset)
        assert exists(self.testRasterFile)

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

if __name__ == "__main__":
    unittest.main()
