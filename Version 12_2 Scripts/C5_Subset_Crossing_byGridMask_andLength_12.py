'''
Author: Alex Crawford
Date Created: 28 Jul 2015
Date Modified: 12 Jun 2019 --> Modified for Python 3
                18 May 2020 --> Modified for using netcdf files instead of geotiffs
Purpose: Identify tracks that spend any point of their lifetime within a 
bounding box defined by a list of (long,lat) ordered pairs in a csv file.

Inputs: User must define the...
    Type of Tracks (typ) -- Cyclone Centers ("Cyclone") or System Centers ("System")
    Bounding Box Number (bboxnum) -- An ID for organizing directories
    Bounding Box Mask (bboxName) -- pathway for the mask to be used
    Versions of Module and Algorithm Run (e.g. 7.8, 9.5)
    Dates of interest
'''

'''********************
Import Modules
********************'''
# Import clock:
from time import perf_counter as clock
# Start script stopwatch. The clock starts running when time is imported
start = clock()

import pandas as pd
import numpy as np
import CycloneModule_12_2 as md
import os
import netCDF4 as nc

'''*******************************************
Set up Environment
*******************************************'''
path = "/Volumes/Cressida"
inpath = path+"/CycloneTracking/tracking12_4E5P"
outpath = inpath

'''*******************************************
Define Variables
*******************************************'''
# File Variables
typ = "System" # Cyclone, System, or Active

bboxName = path+"/Projections/EASE2_N0_5=100km_Projection.nc"
ncvar = 'z'
bboxmin = -100000 # Set to None if using values instead
bboxmax = 500 # Set to None if using values instead
values = [1] # If using a min and max, set to 1, otherwise, these are the
## acccepted values within the mask raster (e.g. the regions of interest)

bboxnum = "15"
bboxmain = "" # The main bbox your subsetting from; usually "" for "all cyclones", otherwise BBox##

# Time Variables 
starttime = [1979,1,1,0,0,0] # Format: [Y,M,D,H,M,S]
endtime = [1979,2,1,0,0,0] # stop BEFORE this time (exclusive)
monthstep = [0,1,0,0,0,0] # A Time step that increases by 1 mont [Y,M,D,H,M,S]

dateref = [1900,1,1,0,0,0] # [Y,M,D,H,M,S]

months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
mons = ["01","02","03","04","05","06","07","08","09","10","11","12"]

minlifespan = 1 # in days
mintracklength = 1000 # in km for version 11, in gridcells for version 10

'''*******************************************
Main Analysis
*******************************************'''
print("Main Analysis")

# Load Mask
bboxnc = nc.Dataset(bboxName)
bbox = bboxnc[ncvar][:].data

if bboxmin == None:
    mask0 = bbox
else:
    mask0 = np.where((bbox >= bboxmin) & (bbox <= bboxmax), 1, 0)

mask = np.isin(mask0,values).reshape(mask0.shape)

# Set up output paths
try:
    os.chdir(inpath+"/BBox"+bboxnum)
except:
    os.mkdir(inpath+"/BBox"+bboxnum)
    os.chdir(inpath+"/BBox"+bboxnum)
    
try:
    os.chdir(inpath+"/BBox"+bboxnum+"/"+typ+"Tracks")
except:
    os.mkdir(inpath+"/BBox"+bboxnum+"/"+typ+"Tracks")
    os.chdir(inpath+"/BBox"+bboxnum+"/"+typ+"Tracks")

# Main Loop
mt = starttime
while mt != endtime:
    # Extract date
    Y = str(mt[0])
    MM = months[mt[1]-1]
    M = mons[mt[1]-1]
    print(" " + Y + " - " + MM)
    
    # Load Tracks
    cs = pd.read_pickle(inpath+"/"+bboxmain+"/"+typ+"Tracks/"+Y+"/"+bboxmain+typ.lower()+"tracks"+Y+M+".pkl")
    cs = [tr for tr in cs if ((tr.lifespan() > minlifespan) and (tr.trackLength() >= mintracklength))]
    
    trs = []
    for tr in cs: # For each track
        # Collect lats and longs
        xs = list(tr.data.x)
        ys = list(tr.data.y)
        
        # Prep while loop
        test = 0
        i = 0
        while test == 0 and i < len(xs):
            # If at any point the cyclone enters the bbox, keep it
            if mask[int(ys[i]),int(xs[i])] == 1:
                trs.append(tr)
                test = 1
            else:
                i = i+1
    
    # Save Tracks
    try:
        os.chdir(inpath+"/BBox"+bboxnum+"/"+typ+"Tracks/"+Y)
    except:
        os.mkdir(inpath+"/BBox"+bboxnum+"/"+typ+"Tracks/"+Y)
        os.chdir(inpath+"/BBox"+bboxnum+"/"+typ+"Tracks/"+Y)
    
    pd.to_pickle(trs,"BBox"+bboxnum+typ.lower()+"tracks"+Y+M+".pkl")
    
    # Increment Month
    mt = md.timeAdd(mt,monthstep)

# Print elapsed time
print('Elapsed time:',round(clock()-start,2),'seconds')
print("Complete")