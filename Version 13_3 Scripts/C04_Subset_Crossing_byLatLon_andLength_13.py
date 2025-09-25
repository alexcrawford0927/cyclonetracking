'''
Author: Alex Crawford
Date Created: 28 Jul 2015
Date Modified: 12 Jun 2019 --> Modified for Python 3
                18 May 2020 --> Modified for using netcdf files instead of geotiffs
                23 Jan 2023 --> Adapted to version 13
Purpose: Identify tracks that spend any point of their lifetime within a
bounding box defined by a list of (long,lat) ordered pairs in a csv file.

Inputs: User must define the...
    Type of Tracks (typ) -- Cyclone Centers ("Cyclone") or System Centers ("System")
    Bounding Box Number (bboxnum) -- An ID for organizing directories
    Bounding Box Mask (bboxName) -- pathway for the mask to be used
    Versions of Module and Algorithm Run (e.g. 7.8, 9.5)
    Spatial Resolution
    Dates of interest
    Minimum track length and lifespan
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
import CycloneModule_13_2 as md
import os
import netCDF4 as nc

def maxDistFromGenPnt(data):
    return  np.max([md.haversine(data.lat[0],data.lat[i],data.lon[0],data.lon[i]) for i in range(len(data.lon))]) / 1000

'''*******************************************
Set up Environment
*******************************************'''
path = "/Volumes/Cressida"
inpath = path+"/CycloneTracking/tracking13testP"
outpath = inpath

'''*******************************************
Define Variables
*******************************************'''
# File Variables
typ = "System" # Cyclone, System, or Active

bboxName = path+"/Projections/EASE2_N0_25km_Projection_uv.nc"
bbox = [65,85,120,-90] # minlat, maxlat, minlon, maxlon
bboxnum = "40"
bboxmain = "" # The main bbox your subsetting from; usually "" for "all cyclones", otherwise BBox##

# Time Variables
starttime = [1979,1,1,0,0,0] # Format: [Y,M,D,H,M,S]
endtime = [1979,2,1,0,0,0] # stop BEFORE this time (exclusive)
monthstep = [0,1,0,0,0,0] # A Time step that increases by 1 mont [Y,M,D,H,M,S]

dateref = [1900,1,1,0,0,0] # [Y,M,D,H,M,S]

minlifespan = 1 # in days
mintracklength = 1000 # in km
mindisplacement = 500 # in km (at least one observation of this cyclone must be at least this far away from its genesis point)

'''*******************************************
Main Analysis
*******************************************'''
print("Main Analysis")

# Load Mask
bboxnc = nc.Dataset(bboxName)
lats = bboxnc['lat'][:].data
lons = bboxnc['lon'][:].data

if bbox[-2] > bbox[-1]:
    mask = ((lats >= bbox[0]) & (lats <= bbox[1]) & ( (lons >= bbox[2]) | (lons <= bbox[3])) )
else:
    mask = ((lats >= bbox[0]) & (lats <= bbox[1]) & (lons >= bbox[2]) & (lons <= bbox[3]))

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
    MM = md.mmm[mt[1]-1]
    M = md.dd[mt[1]-1]
    print(" " + Y + " - " + MM)

    # Load Tracks
    cs = pd.read_pickle(inpath+"/"+bboxmain+"/"+typ+"Tracks/"+Y+"/"+bboxmain+typ.lower()+"tracks"+Y+M+".pkl")
    cs = [tr for tr in cs if ((tr.lifespan() >= minlifespan) and (tr.trackLength() >= mintracklength)) and (maxDistFromGenPnt(tr.data) >= mindisplacement)]

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
