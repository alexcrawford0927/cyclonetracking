'''
Author: Alex Crawford
Date Created: 10 Mar 2019
Date Modified: 22 Aug 2019 -- Update for Python 3
               01 Apr 2020 -- Switched output to netCDF instead of GeoTIFF;
                               no longer dependent on gdal module
               19 Oct 2020 -- pulled the map creation out of the for loop

Purpose: Reads in netcdf files & reprojects to the NSIDC EASE2 Grid North.
'''    

'''********************
Import Modules
********************'''
print("Loading modules.")
import os
import numpy as np
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
import CycloneModule_11_1 as md

'''********************
Define Variables
********************'''
print("Defining variables")

# File Variables:
ra = "ERA5"
var = "SLP"

ncvar = "msl"
nctvar = "time"
ncext = '.nc'

# Time Variables
ymin, ymax = 2002, 2019
mmin, mmax = 1, 12
dmin, dmax = 1, 31 

mons = ["01","02","03","04","05","06","07","08","09","10","11","12"]
dpm = [31,28,31,30,31,30,31,31,30,31,30,31] # days per month (non leap year)
timestep = 1 # in hours
startdate = [1900,1,1] # The starting date for the reanalysis time steps

# Inputs for reprojection
bb = [-89.99,-45,-89.99,135] # in degrees [ll lat, ll lon, ur lat, ur lon]
xsize, ysize = 25000, -25000 # in meters
nx, ny = 720, 720 # number of grid cells
lon_0 = 0 # Central Meridian (which longitude is at the 6 o'clock position)
lat_0 = 90 # Reference Latitude (center of projection)

# Path Variables
path = "/Volumes/Cressida"
inpath = path+"/"+ra+"/"+var
outpath = path+"/"+ra+"/"+var+"_EASE2_N0_"+str(int(xsize/1000))+"km"
suppath = path+"/Projections"

'''*******************************************
Main Analysis
*******************************************'''
print("Main Analysis")

# Obtain list of nc files:
os.chdir(outpath)
fileList = os.listdir(inpath)
fileList = [f for f in fileList if f.endswith(ncext)]

# Identify the time steps:
ref_netcdf = Dataset(inpath+"/"+fileList[-1])

# Create latitude and longitude arrays:
lons = ref_netcdf.variables['longitude'][:]
lats = ref_netcdf.variables['latitude'][:]

# Close reference netcdf:
ref_netcdf.close()

# Set up the map for reprojecting
mp = Basemap(projection='laea',lat_0=lat_0,lon_0=lon_0,\
    llcrnrlat=bb[0], llcrnrlon=bb[1],urcrnrlat=bb[2], urcrnrlon=bb[3],resolution='c')

print("Step 2. Set up dates of analysis")
years = range(ymin,ymax+1)
mos = range(mmin,mmax+1)
hrs = [h*timestep for h in range(int(24/timestep))] 

ly = md.leapyearBoolean(years) # annual boolean for leap year or not leap year

# Start the reprojection loop
print("Step 3. Load, Reproject, and Save")
for y in years:
    Y = str(y)
        
    for m in mos:
        M = mons[m-1]
        
        mlist, hlist = [], []
        ncList = [f for f in fileList if Y+M in f]
    
        if len(ncList) > 1:
            print("Multiple files with the date " + Y + " -- skipping.")
            continue
        if len(ncList) == 0:
            print("No files with the date " + Y + " -- skipping.")
        else:
            nc = Dataset(inpath+"/"+ncList[0])
            tlist = nc.variables[nctvar][:]
            
            # Restrict days to those that exist:
            if m == 2 and ly[y-ymin] == 1 and dmax > dpm[m-1]:
                dmax1 = 29
            elif dmax > dpm[m-1]:
                dmax1 = dpm[m-1]
            else:
                dmax1 = dmax
                
            # For days that DO exist:
            for d in range(dmin,dmax1+1):
                timeD = md.daysBetweenDates(startdate,[y,m,d])*24
                
                print(" " + Y + " " + M + " " + str(d))
                
                for h in hrs:
                    # Establish Time
                    timeH = timeD + h
                    
                    # Read from netcdf array
                    inArr = nc.variables[ncvar][np.where(tlist == timeH)[0][0],:,:]
                    
                    # Transform data
                    outArr, xs, ys = mp.transform_scalar(np.flipud(inArr),lons,np.flipud(lats),nx,ny,returnxy=True)
                    
                    # Add to list
                    mlist.append(outArr)
                    hlist.append(timeH)

        # Write monthly data to netcdf file
        ncf = Dataset(ra+"_EASE2_N0_"+str(int(xsize/1000))+"km_"+var+"_Hourly_"+Y+M+".nc", 'w')
        ncf.description = 'Mean sea-level pressure from ERA5. Projection specifications\
        for the EASE2 projection (Lambert Azimuthal Equal Area;\
        lat-origin = 90°N, lon-origin=0°, # cols = ' + str(nx) + ',\
        # rows = ' + str(ny) + ', dx = ' + str(xsize) + ', dy = ' + str(ysize) + ', units = meters'
        ncf.history = 'Created ' + time.ctime(time.time())
        ncf.source = 'netCDF4 python module'
        
        ncf.createDimension('time', len(mlist))
        ncf.createDimension('x', nx)
        ncf.createDimension('y', ny)
        ncft = ncf.createVariable('time', np.int, ('time',))
        ncfx = ncf.createVariable('x', np.float64, ('x',))
        ncfy = ncf.createVariable('y', np.float64, ('y',))
        ncfArr = ncf.createVariable(ncvar, np.float64, ('time','y','x'))
        
        ncft.units = nc.variables[nctvar].units
        ncfx.units = 'm'
        ncfy.units = 'm'
        ncfArr.units = 'Pa'
        
        # For x and y, note that the upper left point is the edge of the grid cell, but
        ## for this we really want the center of the grid cell, hence dividing by 2.
        ncft[:] = np.array(hlist)
        ncfx[:] = np.arange(-xsize*(nx-1)/2, xsize*(nx-1)/2+xsize, xsize)
        ncfy[:] = np.arange(-ysize*(ny-1)/2, ysize*(ny-1)/2+ysize, ysize)
        ncfArr[:] = np.array(mlist)
        
        ncf.close()

print("Complete.")
