import os
import math

import numpy as np

import osgeo.gdal

#import shapely


def find_point_location_in_raster(origin, cellsize, point_xy):

    """
    Find the location of a point in a raster

    Parameters
    ----------
    cellsize : list
        [x,y] list of pixel size of raster
    x : array
        x-coordinates point
    y : array
        y-coordinates point

    Returns
    -------
    xyo : array
        [x,y] array with x and y location of point origin in raster
    xysize : array
        [x,y] array of extent of point in raster units
    """

    # find min, max coordinates of point
    xy = np.array([old_div((point_xy[0] - origin[0]), cellsize[0]),
                   old_div((point_xy[1] - origin[1]), cellsize[1])])

    return xy


def find_elevation_and_relief(xy, dem, dimensions, origin, cellsize,
                              search_radius=3000.0, verbose=True):

    """

    :param xy:
    :param dem:
    :param origin:
    :param cellsize:
    :param search_radius:
    :return:
    """

    elevation_dem = np.zeros(xy.shape[1])
    relief = np.zeros(xy.shape[1])

    xyp = find_point_location_in_raster(origin, cellsize, xy)

    xyp = (np.round(xyp)).astype(int)

    search_radius = 3000.0
    srg = int(search_radius / cellsize[0])

    for i, xypi in enumerate(xyp.T):
        if (xypi[0] > 0 and xypi[0] < dimensions[0]
                and xypi[1] > 0 and xypi[1] < dimensions[1]):

            if dem[xypi[1], xypi[0]] == nodata:
                if verbose is True:
                    print('pt %i no data', i)
                elevation_dem[i] = np.nan
                relief[i] = np.nan
            else:
                # record elevation
                elevation_dem[i] = dem[xypi[1], xypi[0]]

                # and record relief
                xcs = [xypi[1] - srg,
                       xypi[1] + srg]
                ycs = [xypi[0] - srg,
                       xypi[0] + srg]

                for xc in xcs:
                    if xc < 0:
                        xc = 0
                    elif xc >= dimensions[0] - 1:
                        xc = dimensions[0] - 1

                for yc in ycs:
                    if yc > 0:
                        yc = 0
                    elif yc <= dimensions[1]:
                        yc = dimensions[1]

                dem_radius = dem[ycs[0]:ycs[1], xcs[0]:xcs[1]]
                ind = dem_radius != nodata
                if ind.sum() > 0:
                    relief[i] = dem_radius[ind].max() - dem_radius[ind].min()
                    if verbose is True:
                        print('ok', i, relief[i])

        else:
            if verbose is True:
                print('pt %i, outside raster' % i)
            elevation_dem[i] = np.nan
            relief[i] = np.nan

    return elevation_dem, relief
    
    
def read_raster_file(filename, verbose=False, band_number=1):
    """
    Read gdal-compatible raster file and convert to numpy array
    
    Parameters
    ----------
    filename : string
        filename of gdal compatible raster file
    verbose : bool, optional
        verbose output
    
    Returns
    -------
    raster_array : array
        raster data
    dimensions : list
        x and y size of raster
    origin : list 
        coordinates of (0,0) point of raster
    cellsize : list
        cellsize of raster
    nodata : float
        nodata value
    projection : osgeo.osr.SpatialReference class

    """

    if os.path.isfile(filename):
        raster = osgeo.gdal.Open(filename, osgeo.gdal.GA_ReadOnly)
    else:
        print('error, could not open file %s' % filename)
        return None, None, None, None, None, None

    if verbose is True:
        print('\tnumber of raster bands:', raster.RasterCount)

    inband = raster.GetRasterBand(band_number)
    geotransform = raster.GetGeoTransform()
    dimensions = [inband.XSize, inband.YSize]
    nodata = inband.GetNoDataValue()
    origin = [geotransform[0], geotransform[3]]
    cellsize = [geotransform[1], geotransform[5]]
    projection = osgeo.osr.SpatialReference()
    projection.ImportFromWkt(raster.GetProjectionRef())

    if verbose is True:
        print('\torigin = (', geotransform[0], ',', geotransform[3], ')')
        print('\tpixel Size = (', geotransform[1], ',', geotransform[5], ')')
        print('\tdimensions: x= %s, y= %s' % (inband.XSize, inband.YSize))
        print('\tstart reading raster file')

    raster_array = raster.ReadAsArray()

    if verbose is True:
        print('\tfinished reading raster file')
        print('min,max data values = %0.1f - %0.1f' \
              % (raster_array.min(), raster_array.max()))

    return raster_array, dimensions, origin, cellsize, nodata, projection
    
    
def get_raster_coordinates(dimensions, cellsize, origin):

    """
    
    Parameters
    ----------
    dimensions : list
        [nx, ny] size of input raster
    cellsize : list
        [x,y] cell size of raster
    origin : list
        [x,y] coordinates of origin of raster
    
    Returns
    -------
    raster_x : array
        2D array of x-coordinates raster
    raster_y : array
        2D array of y-coordinates raster
    
    """

    # create raster with elevation grid coordinates

    xcoords = (np.arange(dimensions[0])) * cellsize[0] + origin[0] + cellsize[0] * 0.5
    ycoords = (np.arange(dimensions[1])) * cellsize[1] + origin[1] + cellsize[1] * 0.5
    raster_x, raster_y = np.meshgrid(xcoords, ycoords)

    return raster_x, raster_y
    
    
def find_distance_to_front(pts, thrust_front, angle):
    
    """
    
    """

    # find distances thrust front and datapoints
    distances = np.array([pt.distance(thrust_front) for pt in pts.geoms])

    # get point at thrust front nearest to datapoint
    project_distances = [thrust_front.project(pt) for pt in pts.geoms]
    int_pts = [thrust_front.interpolate(d) for d in project_distances]
    
    # get distances to simplified version of thrust front, with only the first and last pt retained:
    
    #thrust_front_pts_all = [np.array(t) for t in thrust_front]
    #thrust_front_pts = np.concatenate(thrust_front_pts_all, axis=0)
    #thrust_front_simple = shapely.geometry.LineString([thrust_front_pts[0], thrust_front_pts[-1]])
    #project_distances_simplified = [thrust_front_simple.project(pt) for pt in pts.geoms]

    # calculate angle
    dxs = [int_pt.x - pt.x for int_pt, pt in zip(int_pts, pts.geoms)]
    dys = [int_pt.y - pt.y for int_pt, pt in zip(int_pts, pts.geoms)]

    # check if samples are in front of the thrust or behind
    # and make the distance for samples behind the front negative
    rads = [math.atan2(dy, dx) for dx, dy in zip(dxs, dys)]

    degrees = [math.degrees(rad) for rad in rads]

    front = np.array([(degree > angle - 180) & (degree < angle)
                      for degree in degrees])

    distance_thrust_front_corr = distances
    distance_thrust_front_corr[front == True] = -distances[front == True]

    return (np.array(distance_thrust_front_corr),
            np.array(distances),
            np.array(project_distances))
