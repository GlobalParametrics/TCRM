"""
:mod:`processMultipliers` -- combine multipliers with wind speed data
=====================================================================

.. module:: processMultipliers
    :synopsis: Process a wind field file and combine with directional
               site-exposure multipliers to evaluate local wind
               speed.

.. moduleauthor:: Craig Arthur <craig.arthur@ga.gov.au>

Combine the regional wind speed data with the site-exposure
multipliers, taking care to select the directional multiplier that
corresponds to the direction of the maximum wind speed.

This version assumes that the site-exposure multipliers are given as a
combined value (i.e. ``Ms * Mz * Mh``), and the files are ERDAS
Imagine-format files ('*.img'). Further, the files are assumed to have
the file name ``m4_<dir>.img``, where <dir> is the direction (n, ne, e,
se, s, sw, w or nw).

Requires the Python GDAL bindings, Numpy, netCDF4 and the :mod:`files`
and :mod:`config` modules from TCRM. It assumes :mod:`Utilities` can
be found in the ``PYTHONPATH`` directory.

"""

import os
import sys
import time
import logging as log
import argparse
import traceback
from functools import wraps

from Utilities.files import flStartLog
from Utilities.config import ConfigParser
from Utilities import pathLocator

from netCDF4 import Dataset

import numpy as np
from os.path import join as pjoin, dirname, realpath, isdir, abspath, splitext

from osgeo import osr, gdal
from osgeo.gdalconst import *
from functools import reduce


gdal.UseExceptions()


def timer(f):
    """
    Basic timing functions for entire process
    """
    @wraps(f)
    def wrap(*args, **kwargs):
        t1 = time.time()
        res = f(*args, **kwargs)

        tottime = time.time() - t1
        msg = "%02d:%02d:%02d " % \
            reduce(lambda ll, b: divmod(ll[0], b) + ll[1:],
                   [(tottime,), 60, 60])

        log.info("Time for {0}: {1}".format(f.func_name, msg))
        return res

    return wrap


def createRaster(array, x, y, dx, dy, epsg=4326, filename=None, nodata=-9999):
    """
    Create an in-memory raster for processing. By default, we assume
    the input array is in geographic coordinates, using WGS84 spatial
    reference system.

    :param array: Data array to be stored.
    :type  array: :class:`numpy.ndarray`
    :param x: x-coordinate of the array values.
    :type  x: :class:`numpy.ndarray`
    :param y: y-coordinate of the array values - should be a negative
              value.
    :type  y: :class:`numpy.ndarray`
    :param float dx: Pixel size in x-direction.
    :param float dy: Pixel size in y-direction.
    :param int epsg: EPSG code of the spatial reference system
                     of the input array (default=4326, WGS84)
    :param filename: Optional path
     to store the data in.
    :type  filename: str or None

    """
    if filename:
        log.debug("Creating raster: {0}".format(filename))
    else:
        log.debug("Creating in-memory raster")
    rows, cols = array.shape
    originX, originY = x[0], y[-1]
    if filename:
        _, ext = splitext(filename)
    if filename and ext=='.tif':
        driver = gdal.GetDriverByName('GTiff')
        tempRaster = driver.Create(filename, cols, rows, 1, GDT_Float32)
    elif filename and ext=='.img':
        driver = gdal.GetDriverByName('HFA')
        tempRaster = driver.Create(filename, cols, rows, 1, GDT_Float32)
    else:
        driver = gdal.GetDriverByName('MEM')
        tempRaster = driver.Create('', cols, rows, 1, GDT_Float32)

    tempRaster.SetGeoTransform((originX, dx, 0,
                                originY, 0, dy))
    tempBand = tempRaster.GetRasterBand(1)
    tempBand.WriteArray(array[::np.sign(dy) * 1])
    tempBand.SetNoDataValue(nodata)
    tempRasterSRS = osr.SpatialReference()
    tempRasterSRS.ImportFromEPSG(epsg)
    tempRaster.SetProjection(tempRasterSRS.ExportToWkt())

    log.debug("Spatial reference system is:")
    log.debug(tempRasterSRS.ExportToWkt())
    tempBand.FlushCache()
    return tempRaster


def loadRasterFile(raster_file, fill_value=1):
    """
    Load a raster file and return the data as a :class:`numpy.ndarray`.
    No prorjection information is returned, just the actual data as an
    array.

    :param str raster_file: Path to the raster file to load.
    :param fill_value: Value to replace `nodata` values with (default=1).
    :returns: 2-d array of the data values.
    :rtype: :class:`numpy.ndarray`

    """

    log.debug("Loading raster data from {0} into array".format(raster_file))
    ds = gdal.Open(raster_file, GA_ReadOnly)
    band = ds.GetRasterBand(1)
    data = band.ReadAsArray()

    nodata = band.GetNoDataValue()

    if nodata is not None:
        np.putmask(data, data == nodata, fill_value)

    del ds
    return data


def calculateBearing(uu, vv):
    """
    Calculate the wind direction from the u (eastward) and v
    (northward) components of the wind speed.

    :param uu: :class:`numpy.ndarray` of eastward values
    :param vv: :class:`numpy.ndarray` of northward values.

    :returns: Direction of the vector, zero northwards, positive
              clockwise, in degrees.
    :rtype: :class:`numpy.ndarray`

    """
    bearing = 2 * np.pi - (np.arctan2(-vv, -uu) - np.pi / 2)
    bearing = (180. / np.pi) * np.mod(bearing, 2. * np.pi)
    return bearing


@timer
def reprojectDataset(src_file, match_filename, dst_filename,
                     resampling_method=GRA_Bilinear, match_projection=None):
    """
    Reproject a source dataset to match the projection of another
    dataset and save the projected dataset to a new file.

    :param src_filename: Filename of the source raster dataset, or an
                         open :class:`gdal.Dataset`
    :param match_filename: Filename of the dataset to match to, or an
                           open :class:`gdal.Dataset`
    :param str dst_filename: Destination filename.
    :param resampling_method: Resampling method. Default is bilinear
                              interpolation.

    """

    log.debug("Reprojecting {0}".format(repr(src_file)))
    log.debug("Match raster: {0}".format(repr(match_filename)))
    log.debug("Output raster: {0}".format(dst_filename))

    if isinstance(src_file, str):
        src = gdal.Open(src_file, GA_ReadOnly)
    else:
        src = src_file
    src_proj = src.GetProjection()
    src_geotrans = src.GetGeoTransform()

    # We want a section of source that matches this:
    if isinstance(match_filename, str):
        match_ds = gdal.Open(match_filename, GA_ReadOnly)
    else:
        match_ds = match_filename

    if match_projection:
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(match_projection)
        match_proj = srs.ExportToWkt()
    else:
        match_proj = match_ds.GetProjection()
    match_geotrans = match_ds.GetGeoTransform()
    wide = match_ds.RasterXSize
    high = match_ds.RasterYSize

    # Output / destination
    drv = gdal.GetDriverByName('GTiff')
    dst = drv.Create(dst_filename, wide, high, 1, GDT_Float32)
    dst.SetGeoTransform(match_geotrans)
    dst.SetProjection(match_proj)

    # Do the work
    gdal.ReprojectImage(src, dst, src_proj, match_proj, resampling_method)

    del dst  # Flush
    if isinstance(match_filename, str):
        del match_ds
    if isinstance(src_file, str):
        del src

    return

@timer
def processMult(result, m4_max_file, windfield_path,
                multiplier_path, track=None):

    # Use this to check values
    if track is not None:
        ncfile = track.trackfile

    gust, _, Vx, Vy, P, lon, lat = result

    wspd = gust
    uu = Vx
    vv = Vy

    bearing = calculateBearing(uu, vv)

    print "gust", gust
    print "uu",uu
    print "vv", vv
    print "lon", lon
    print "lat", lat
    print "bearing", bearing

    print "m4_max_file", m4_max_file
    print "windfield_path", windfield_path
    print "multiplier_path", multiplier_path

    delta = lon[1] - lon[0]

    print "!!!!!!!!!!!!!!!!!!!  I'm in  !!!!!!!!!!!"


    # Reproject the wind speed and bearing data:
    wind_raster_file = pjoin(windfield_path, 'region_wind.tif')
    wind_raster = createRaster(wspd, lon, lat, delta, -delta,
                               filename=wind_raster_file)
    bear_raster = createRaster(bearing, lon, lat, delta, -delta)
    uu_raster = createRaster(uu, lon, lat, delta, -delta)
    vv_raster = createRaster(vv, lon, lat, delta, -delta)

    log.info("Reprojecting regional wind data")
    wind_prj_file = pjoin(windfield_path, 'gust_prj.tif')
    bear_prj_file = pjoin(windfield_path, 'bear_prj.tif')
    uu_prj_file = pjoin(windfield_path, 'uu_prj.tif')
    vv_prj_file = pjoin(windfield_path, 'vv_prj.tif')

    wind_prj = reprojectDataset(wind_raster, m4_max_file, wind_prj_file)
    bear_prj = reprojectDataset(bear_raster, m4_max_file, bear_prj_file,
                                resampling_method=GRA_NearestNeighbour)
    uu_prj = reprojectDataset(uu_raster, m4_max_file, uu_prj_file,
                              resampling_method=GRA_NearestNeighbour)
    vv_prj = reprojectDataset(vv_raster, m4_max_file, vv_prj_file,
                              resampling_method=GRA_NearestNeighbour)

    wind_prj_ds = gdal.Open(wind_prj_file, GA_ReadOnly)
    wind_prj = wind_prj_ds.GetRasterBand(1)
    bear_prj_ds = gdal.Open(bear_prj_file, GA_ReadOnly)
    bear_prj = bear_prj_ds.GetRasterBand(1)
    uu_prj_ds = gdal.Open(uu_prj_file, GA_ReadOnly)
    uu_prj = uu_prj_ds.GetRasterBand(1)
    vv_prj_ds = gdal.Open(vv_prj_file, GA_ReadOnly)
    vv_prj = vv_prj_ds.GetRasterBand(1)
    wind_proj = wind_prj_ds.GetProjection()
    wind_geot = wind_prj_ds.GetGeoTransform()

    wind_data = wind_prj.ReadAsArray()
    bear_data = bear_prj.ReadAsArray()
    uu_data = uu_prj.ReadAsArray()
    vv_data = vv_prj.ReadAsArray()
    bearing = calculateBearing(uu_data, vv_data)

    print "wind_data", wind_data
    print "bear_data", bear_data
    # The local wind speed array:
    local = np.zeros(wind_data.shape, dtype='float32')

    indices = {
        0: {'dir': 'n', 'min': 0., 'max': 22.5},
        1: {'dir': 'ne', 'min': 22.5, 'max': 67.5},
        2: {'dir': 'e', 'min': 67.5, 'max': 112.5},
        3: {'dir': 'se', 'min': 112.5, 'max': 157.5},
        4: {'dir': 's', 'min': 157.5, 'max': 202.5},
        5: {'dir': 'sw', 'min': 202.5, 'max': 247.5},
        6: {'dir': 'w', 'min': 247.5, 'max': 292.5},
        7: {'dir': 'nw', 'min': 292.5, 'max': 337.5},
        8: {'dir': 'n', 'min': 337.5, 'max': 360.}
    }
    log.info("Processing all directions")
    for i in indices.keys():
        dn = indices[i]['dir']
        log.info("Processing {0}".format(dn))
        m4_file = pjoin(multiplier_path, 'm4_{0}.img'.format(dn.lower()))
        m4 = loadRasterFile(m4_file)
        print "*****************"
        print 'i', i
        print 'm4', m4
        idx = np.where((bear_data >= indices[i]['min']) &
                       (bear_data < indices[i]['max']))

        local[idx] = wind_data[idx] * m4[idx]
    print "local", local
    rows, cols = local.shape
    output_file = pjoin(windfield_path, 'local_wind.tif')
    log.info("Creating output file: {0}".format(output_file))
    # Save the local wind field to a raster file with the SRS of the
    # multipliers
    drv = gdal.GetDriverByName("GTiff")
    dst_ds = drv.Create(output_file, cols, rows, 1,
                        GDT_Float32, ['BIGTIFF=YES'])
    dst_ds.SetGeoTransform(wind_geot)
    dst_ds.SetProjection(wind_proj)
    dst_band = dst_ds.GetRasterBand(1)
    dst_band.WriteArray(local)

    # dst_band.FlushCache()

    del dst_ds
    log.info("Completed")


@timer
def modified_main(config_file):
    """
    Main function to combine the multipliers with the regional wind
    speed data.

    :param str configFile: Path to configuration file.

    """
    config = ConfigParser()
    config.read(config_file)
    input_path = config.get('Input', 'Path')
    try:
        gust_file = config.get('Input', 'Gust_file')
    except:
        gust_file = 'gust.interp.nc'
    windfield_path = pjoin(input_path, 'windfield')
    ncfile = pjoin(windfield_path, gust_file)
    multiplier_path = config.get('Input', 'Multipliers')

    # Load the wind data:
    log.info("Loading regional wind data from {0}".format(ncfile))
    ncobj = Dataset(ncfile, 'r')

    lat = ncobj.variables['lat'][:]
    lon = ncobj.variables['lon'][:]

    delta = lon[1] - lon[0]
    lon = lon - delta / 2.
    lat = lat - delta / 2.

    # Wind speed:
    wspd = ncobj.variables['vmax'][:]

    # Components:
    uu = ncobj.variables['ua'][:]
    vv = ncobj.variables['va'][:]

    bearing = calculateBearing(uu, vv)

    gust = wspd
    Vx = uu
    Vy = vv
    P = None

    #  WARNING, THESE COULD BE WRONG!!!
    # Plus it doesn't do anything,
    # except hightlight these var's are going in..
    lon = lon
    lat = lat
    # Need to be checked !!!

    output_file = pjoin(windfield_path, 'local_wind.tif')
    result =  gust, bearing, Vx, Vy, P, lon, lat
    # Load a multiplier file to determine the projection:
    m4_max_file = pjoin(multiplier_path, 'm4_max.img')
    log.info("Using M4 data from {0}".format(m4_max_file))

    processMult(result, m4_max_file, windfield_path,
                multiplier_path)


@timer
def main(config_file):
    """
    Main function to combine the multipliers with the regional wind
    speed data.

    :param str configFile: Path to configuration file.

    """

    config = ConfigParser()
    config.read(config_file)
    input_path = config.get('Input', 'Path')
    try:
        gust_file = config.get('Input', 'Gust_file')
    except:
        gust_file = 'gust.interp.nc'
    windfield_path = pjoin(input_path, 'windfield')
    ncfile = pjoin(windfield_path, gust_file)
    multiplier_path = config.get('Input', 'Multipliers')

    # Load the wind data:
    log.info("Loading regional wind data from {0}".format(ncfile))
    ncobj = Dataset(ncfile, 'r')

    lat = ncobj.variables['lat'][:]
    lon = ncobj.variables['lon'][:]

    delta = lon[1] - lon[0]
    lon = lon - delta / 2.
    lat = lat - delta / 2.

    # Wind speed:
    wspd = ncobj.variables['vmax'][:]

    # Components:
    uu = ncobj.variables['ua'][:]
    vv = ncobj.variables['va'][:]

    bearing = calculateBearing(uu, vv)

    # Load a multiplier file to determine the projection:
    m4_max_file = pjoin(multiplier_path, 'm4_max.img')
    log.info("Using M4 data from {0}".format(m4_max_file))

    # Reproject the wind speed and bearing data:
    wind_raster_file = pjoin(windfield_path, 'region_wind.tif')
    wind_raster = createRaster(wspd, lon, lat, delta, -delta,
                               filename=wind_raster_file)
    bear_raster = createRaster(bearing, lon, lat, delta, -delta)
    uu_raster = createRaster(uu, lon, lat, delta, -delta)
    vv_raster = createRaster(vv, lon, lat, delta, -delta)

    log.info("Reprojecting regional wind data")
    wind_prj_file = pjoin(windfield_path, 'gust_prj.tif')
    bear_prj_file = pjoin(windfield_path, 'bear_prj.tif')
    uu_prj_file = pjoin(windfield_path, 'uu_prj.tif')
    vv_prj_file = pjoin(windfield_path, 'vv_prj.tif')

    wind_prj = reprojectDataset(wind_raster, m4_max_file, wind_prj_file)
    bear_prj = reprojectDataset(bear_raster, m4_max_file, bear_prj_file,
                                resampling_method=GRA_NearestNeighbour)
    uu_prj = reprojectDataset(uu_raster, m4_max_file, uu_prj_file,
                              resampling_method=GRA_NearestNeighbour)
    vv_prj = reprojectDataset(vv_raster, m4_max_file, vv_prj_file,
                              resampling_method=GRA_NearestNeighbour)

    wind_prj_ds = gdal.Open(wind_prj_file, GA_ReadOnly)
    wind_prj = wind_prj_ds.GetRasterBand(1)
    bear_prj_ds = gdal.Open(bear_prj_file, GA_ReadOnly)
    bear_prj = bear_prj_ds.GetRasterBand(1)
    uu_prj_ds = gdal.Open(uu_prj_file, GA_ReadOnly)
    uu_prj = uu_prj_ds.GetRasterBand(1)
    vv_prj_ds = gdal.Open(vv_prj_file, GA_ReadOnly)
    vv_prj = vv_prj_ds.GetRasterBand(1)
    wind_proj = wind_prj_ds.GetProjection()
    wind_geot = wind_prj_ds.GetGeoTransform()

    wind_data = wind_prj.ReadAsArray()
    bear_data = bear_prj.ReadAsArray()
    uu_data = uu_prj.ReadAsArray()
    vv_data = vv_prj.ReadAsArray()
    bearing = calculateBearing(uu_data, vv_data)

    # The local wind speed array:
    local = np.zeros(wind_data.shape, dtype='float32')

    indices = {
        0: {'dir': 'n', 'min': 0., 'max': 22.5},
        1: {'dir': 'ne', 'min': 22.5, 'max': 67.5},
        2: {'dir': 'e', 'min': 67.5, 'max': 112.5},
        3: {'dir': 'se', 'min': 112.5, 'max': 157.5},
        4: {'dir': 's', 'min': 157.5, 'max': 202.5},
        5: {'dir': 'sw', 'min': 202.5, 'max': 247.5},
        6: {'dir': 'w', 'min': 247.5, 'max': 292.5},
        7: {'dir': 'nw', 'min': 292.5, 'max': 337.5},
        8: {'dir': 'n', 'min': 337.5, 'max': 360.}
    }
    log.info("Processing all directions")
    for i in indices.keys():
        dn = indices[i]['dir']
        log.info("Processing {0}".format(dn))
        m4_file = pjoin(multiplier_path, 'm4_{0}.img'.format(dn.lower()))
        m4 = loadRasterFile(m4_file)
        idx = np.where((bear_data >= indices[i]['min']) &
                       (bear_data < indices[i]['max']))

        local[idx] = wind_data[idx] * m4[idx]

    rows, cols = local.shape
    output_file = pjoin(windfield_path, 'local_wind.tif')
    log.info("Creating output file: {0}".format(output_file))
    # Save the local wind field to a raster file with the SRS of the
    # multipliers
    drv = gdal.GetDriverByName("GTiff")
    dst_ds = drv.Create(output_file, cols, rows, 1,
                        GDT_Float32, ['BIGTIFF=YES'])
    dst_ds.SetGeoTransform(wind_geot)
    dst_ds.SetProjection(wind_proj)
    dst_band = dst_ds.GetRasterBand(1)
    dst_band.WriteArray(local)

    # dst_band.FlushCache()

    del dst_ds
    log.info("Completed")


def startup():
    """
    Parse command line arguments and call the :func:`main` function.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file',
                        help='Path to configuration file')
    parser.add_argument('-v', '--verbose', help='Verbose output',
                        action='store_true')
    parser.add_argument('-d', '--debug', help='Allow pdb traces',
                        action='store_true')
    args = parser.parse_args()

    configFile = args.config_file
    config = ConfigParser()
    config.read(configFile)

    rootdir = pathLocator.getRootDirectory()
    os.chdir(rootdir)

    logfile = config.get('Logging', 'LogFile')
    logdir = dirname(realpath(logfile))

    # If log file directory does not exist, create it
    if not isdir(logdir):
        try:
            os.makedirs(logdir)
        except OSError:
            logfile = pjoin(os.getcwd(), 'processMultipliers.log')

    logLevel = config.get('Logging', 'LogLevel')
    verbose = config.getboolean('Logging', 'Verbose')
    datestamp = config.getboolean('Logging', 'Datestamp')
    debug = False

    if args.verbose:
        verbose = True

    if args.debug:
        debug = True

    flStartLog(logfile, logLevel, verbose, datestamp)

    if debug:
        main(configFile)
    else:
        try:
            modified_main(configFile)
        except Exception:  # pylint: disable=W0703
            # Catch any exceptions that occur and log them (nicely):
            tblines = traceback.format_exc().splitlines()
            for line in tblines:
                log.critical(line.lstrip())

if __name__ == "__main__":
    startup()
