'''
Author: Alex Crawford
Date Created: 10 Mar 2019
Date Modified: 22 Aug 2019 -- Update for Python 3
               01 Apr 2020 -- Switched output to netCDF instead of GeoTIFF;
                               no longer dependent on gdal module (start V5)
               19 Oct 2020 -- pulled the map creation out of the for loop
               06 Oct 2021 -- added a wrap-around for inputs that prevents
                              empty cells from forming along either 180° or
                              360° longitude (start V6)
               15 Nov 2022 -- replaced "np.int" with "int"
               06 Jan 2025 -- updated to always assume monthly files
                           -- implements "periodic" parameter in xesmf
                           -- adds compatibility for ECMWF changing time dimension name and units (very grumpy about this)

Purpose: Reads in netcdf files & reprojects to the NSIDC EASE2 Grid North.
'''

'''********************
Import Modules
********************'''
print("Loading modules.")
import os
import numpy as np
from netCDF4 import Dataset
import xesmf as xe
import xarray as xr
import CycloneModule_13_2 as md

'''********************
Define Variables
********************'''
print("Defining variables")

# File Variables:
ra = "ERA5"
var = "SLP"

ncvar = "msl"
nctvar = "time"
altnctvar = 'valid_time'
ncext = '.nc'

# Time Variables
ymin, ymax = 2022, 2024
mmin, mmax = 1, 12
startdate = [1900,1,1] # The starting date for the reanalysis time steps

# Inputs for reprojection
xsize, ysize = 50000, -50000 #25000, -25000 # 100000, -100000 #  in meters
nx, ny = 360, 360 #720, 720 # 180, 180 #  number of grid cells; use 180 by 180 for 100 km grid

# Path Variables
path1 = "/media/alex/Datapool" # '/Volumes/Cressida' # 
path2 =  "/media/alex/Datapool" # "/Volumes/Cressida" # 

inpath = path1+"/"+ra+"/"+var #
outpath = path2+"/"+ra+"/"+var+"_EASE2_N0_"+str(int(xsize/1000))+"km" #
prjpath = path2+'/Projections/EASE2_N0_'+str(int(xsize/1000))+'km_Projection_uv.nc'
regridderpath = path2+"/Projections/Regridders/ERA5_NH_to_EASE2_N0_"+str(int(xsize/1000))+"km.nc" 

'''*******************************************
Main Analysis
*******************************************'''
print("Step 1. Set up regridder")
# Obtain list of nc files:
os.chdir(outpath)
fileList = os.listdir(inpath)
fileList = [f for f in fileList if (f.endswith(ncext) & f.startswith(ra))]

# Create latitude and longitude arrays for output
outprjnc = Dataset(prjpath)
outlat = outprjnc['lat'][:].data
outlon = outprjnc['lon'][:].data

# Load example netcdf file for input data 
ref_netcdf = Dataset(inpath+"/"+fileList[-1])

# Define Grids as Dictionaries
grid_in = {'lon': ref_netcdf['longitude'][:], 
           'lat': ref_netcdf['latitude'][:]}
grid_out = {'lon': outlon, 'lat': outlat}
ref_netcdf.close()

# Create Regridder
try:
    regridder = xe.Regridder(grid_in, grid_out, 'bilinear', periodic=True, ignore_degenerate=True, weights=regridderpath)
except:
    regridder = xe.Regridder(grid_in, grid_out, 'bilinear', periodic=True, ignore_degenerate=True)
    regridder.to_netcdf(regridderpath)

print("Step 2. Set up dates of analysis")
years = range(ymin,ymax+1)
mos = range(mmin,mmax+1)

# Start the reprojection loop
print("Step 3. Load, Reproject, and Save")
for y in years:
    Y = str(y)

    for m in mos:
        M = md.dd[m-1]
        
        # Check to make sure there is one and only one valid input file
        ncList = [f for f in fileList if Y+M in f]

        if len(ncList) > 1:
            print("Multiple files with the date " + Y+M + " -- skipping.")
            continue
        if len(ncList) == 0:
            print("No files with the date " + Y+M + " -- skipping.")
        else:
            print(" Processing " + Y + " " + M )
            nc = xr.open_dataset(inpath+"/"+ncList[0])
            
            if nctvar not in list(nc.variables):
                nc = nc.rename_dims({altnctvar:nctvar}).rename_vars({altnctvar:nctvar})
            
            tlist = nc.variables[nctvar][:].data
            
            # Create time list guaranteed to be hours since the reference time (redundant for some data)
            harr = md.daysBetweenDates(startdate, [y,m,1])*24 + np.arange(len(tlist)) * (md.daysBetweenDates([y,m,1],md.timeAdd([y,m,1],[0,1,0]))*24) / len(tlist)

            # Transform data
            outArr = regridder(nc.variables[ncvar][:].data)
            outArr[:,outlat < 0] = np.nan # Limits to Northern Hemisphere

            # Write monthly data to netcdf file
            ncf = Dataset(ra+"_EASE2_N0_"+str(int(xsize/1000))+"km_"+var+"_Hourly_"+Y+M+".nc", 'w')
            ncf.description = 'Mean sea-level pressure from ERA5. Projection specifications\
            for the EASE2 projection (Lambert Azimuthal Equal Area;\
            lat-origin = 90°N, lon-origin=0°, # cols = ' + str(nx) + ',\
            # rows = ' + str(ny) + ', dx = ' + str(xsize) + ', dy = ' + str(ysize) + ', units = meters'
            ncf.source = 'netCDF4 python module'
    
            ncf.createDimension('time', len(tlist))
            ncf.createDimension('x', nx)
            ncf.createDimension('y', ny)
            ncft = ncf.createVariable('time', int, ('time',))
            ncfx = ncf.createVariable('x', np.int32, ('x',))
            ncfy = ncf.createVariable('y', np.int32, ('y',))
            ncfArr = ncf.createVariable(ncvar, np.float32, ('time','y','x'))
    
            try:
                ncft.units = nc.variables[nctvar].units
            except:
                ncft.units = 'hours since 1900-01-01 00:00:00.0'
    
            ncfx.units = 'm'
            ncfy.units = 'm'
            ncfArr.units = 'Pa'
    
            # For x and y, note that the upper left point is the edge of the grid cell, but
            ## for this we really want the center of the grid cell, hence dividing by 2.
            ncft[:] = harr
            ncfx[:] = np.arange(-xsize*(nx-1)/2, xsize*(nx-1)/2+xsize, xsize)
            ncfy[:] = np.arange(-ysize*(ny-1)/2, ysize*(ny-1)/2+ysize, ysize)
            ncfArr[:] = outArr
    
            ncf.close()

print("Complete.")
