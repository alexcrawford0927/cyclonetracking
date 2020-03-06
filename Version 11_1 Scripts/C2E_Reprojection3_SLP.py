'''
Author: Alex Crawford
Date Created: 15 Jan 2015
Date Modified: 10 Mar 2019; 22 Aug 2019 -- Update for Python 3

Purpose: Reads in netcdf files, converts to geotiffs, and reprojects to the
NSIDC EASE2 Grid North (EPSG: 3793).
'''    

'''********************
Import Modules
********************'''
print("Loading modules.")
import os
import numpy as np
from osgeo import gdal, gdalconst, gdalnumeric
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
import CycloneModule_11_1 as md

ra = "ERAI"
var = "SLP"
var1 = "SLP"
path = "/Volumes/Ferdinand"
inpath = path+"/"+ra+"_nc/"+var
outpath = path+"/"+ra+"_SLP_100km" # path+"/"+ra+"/"+var1+"/"+var+"_EASE100km/Value"
suppath = path+"/Projections"

'''********************
Define Variables
********************'''
print("Defining variables")

# File Variables:
outputtype = gdal.GDT_Float64
ncext = ".nc"
ext = ".tif"

ncvar = "msl"
nctvar = "time"

longsN = ra+"_Longs"+ext
latsN = ra+"_Lats"+ext

prjOut = path+"/ArcticCyclone/code/EASE2Projection.prj"
prjIn = path+"/ArcticCyclone/code/WGS1984.prj"

# Time Variables
ymin, ymax = 2019, 2019
mmin, mmax = 2, 5
dmin, dmax = 1,31 

months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
mons = ["01","02","03","04","05","06","07","08","09","10","11","12"]
days = ["01","02","03","04","05","06","07","08","09","10","11","12","13",\
    "14","15","16","17","18","19","20","21","22","23","24","25","26","27",\
    "28","29","30","31"]
hours = ["0000","0100","0200","0300","0400","0500","0600","0700","0800",\
    "0900","1000","1100","1200","1300","1400","1500","1600","1700","1800",\
    "1900","2000","2100","2200","2300"]

dpm = [31,28,31,30,31,30,31,31,30,31,30,31] # days per month (non leap year)

secs = 86400.0 # Number of seconds per day

timestep = 6 # in hours

startdate = [1900,1,1] # The starting date for the reanalysis time steps

# Projection info
projP = '''PROJCS["EASE",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84", 
6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],
PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],AUTHORITY["EPSG","4326"]],
PROJECTION["Lambert_Azimuthal_Equal_Area"],PARAMETER["latitude_of_center",90],
PARAMETER["longitude_of_center",0],PARAMETER["false_easting",0],
PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]]]'''

# Inputs for reprojection
bb = [-84.614,-45,-84.614,135] # in degrees
ulpnt = (-9039270.143837992,9039270.143837988) # in meters (for a 183 by 183 grid)
xsize, ysize = 100000, -100000 # in meters
nx, ny = 181, 181 # number of grid cells
lon_0 = 0 # Central Meridian (which longitude is at the 6 o'clock position)
lat_0 = 90 # Reference Latitude (center of projection)

dtype = gdal.GDT_Float32

'''*******************************************
Main Analysis
*******************************************'''
print("Main Analysis")

# Obtain list of nc files:
fileList = os.listdir(inpath)
fileList = [f for f in fileList if f.endswith(ncext)]

# Identify the time steps:
ref_netcdf = Dataset(inpath+"/"+fileList[-1])

# Create latitude and longitude arrays:
longs = ref_netcdf.variables['longitude'][:]
lats = ref_netcdf.variables['latitude'][:]

# Close reference netcdf:
ref_netcdf.close()

print("Step 2. Set up dates of analysis")
years = range(ymin,ymax+1)
mos = range(mmin,mmax+1)
hrs = [h*timestep for h in range(int(24/timestep))] 

ly = md.leapyearBoolean(years) # annual boolean for leap year or not leap year

# Start the reprojection loop
print("Step 3. Load, Reproject, and Save")
for y in years:
    Y = str(y)
    
    # Set workspace
    try:
        os.chdir(outpath+"/"+Y)
    except:
        os.mkdir(outpath+"/"+Y)
        os.chdir(outpath+"/"+Y)
        
    for m in mos:
        M = mons[m-1]
        MM = months[m-1]
            
        ncList = [f for f in fileList if Y+M in f]
    
        if len(ncList) > 1:
            print("Multiple files with the date " + Y + " -- skipping.")
            continue
        if len(ncList) == 0:
            print("No files with the date " + Y + " -- skipping.")
        else:
            nc = Dataset(inpath+"/"+ncList[0])
            tlist = nc.variables[nctvar][:]
            
            # Set workspace
            try:
                os.chdir(outpath+"/"+Y+"/"+MM)
            except:
                os.mkdir(outpath+"/"+Y+"/"+MM)
                os.chdir(outpath+"/"+Y+"/"+MM)
            
            # Restrict days to those that exist:
            if m == 2 and ly[y-ymin] == 1 and dmax > dpm[m-1]:
                dmax1 = 29
            elif dmax > dpm[m-1]:
                dmax1 = dpm[m-1]
            else:
                dmax1 = dmax
                
            # For days that DO exist:
            for d in range(dmin,dmax1+1):
                D = days[d-1]
                timeD = md.daysBetweenDates(startdate,[y,m,d])*24
                
                print(" " + Y + " " + M + " " + D)
                
                for h in hrs:
                    # Establish Time
                    H = hours[h]
                    timeH = timeD + h
                    
                    # Read from netcdf array
                    inArr = nc.variables[ncvar][np.where(tlist == timeH)[0][0],:,:]
                    
                    # Set up the map for reprojecting
                    mp = Basemap(projection='laea',lat_0=lat_0,lon_0=lon_0,\
                        llcrnrlat=bb[0], llcrnrlon=bb[1],urcrnrlat=bb[2], urcrnrlon=bb[3],resolution='c')
                
                    # Transform data
                    outArr, xs, ys = mp.transform_scalar(np.flipud(inArr),longs,np.flipud(lats),nx,ny,returnxy=True)
                    
                    # Write to file
                    outName = outpath+"/"+Y+"/"+MM+"/"+var+"_"+Y+M+D+"_"+H+ext
                    driver = gdal.GetDriverByName('GTiff')
                    outFile = driver.Create(outName,outArr.shape[1],outArr.shape[0],1,dtype) # Create file
                    outFile.GetRasterBand(1).WriteArray(np.flipud(outArr),0,0) # Write array to file
                    outFile.GetRasterBand(1).ComputeStatistics(False) # Compute stats for display purposes
                    outFile.SetGeoTransform((ulpnt[0],xsize,0,ulpnt[1],0,ysize)) # Set geotransform (those six needed values)
                    outFile.SetProjection(projP)  # Set projection
                    outFile = None

print("Complete.")