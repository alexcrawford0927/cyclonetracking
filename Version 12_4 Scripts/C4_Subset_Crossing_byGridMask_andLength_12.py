'''
Author: Alex Crawford
Date Created: 28 Jul 2015
Date Modified: 12 Jun 2019 --> Modified for Python 3
                18 May 2020 --> Modified for using netcdf files instead of geotiffs
                19 Jan 2021 --> Added pickles as acceptable file input for masks
                11 Jun 2021 --> Added option for minimum displacement
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
import CycloneModule_12_4 as md
import os
import netCDF4 as nc
from scipy import interpolate
# import pickle5

def maxDistFromGenPnt(data):        
    v = np.max([md.haversine(data.lat[0],data.lat[i],data.long[0],data.long[i]) for i in range(len(data.long))])
    return v/1000

'''*******************************************
Set up Environment
*******************************************'''
path = "/Volumes/Cressida"
inpath = path+"/CycloneTracking/tracking12_4E5P"
outpath = inpath

'''*******************************************
Define Variables
*******************************************'''
# spres = pickle5.load(open(inpath+"/cycloneparams.pkl",'rb'))['spres']
spres =  pd.read_pickle(inpath+"/cycloneparams.pkl")['spres'] # 100 #

# File Variables
typ = "System" # Cyclone, System, or Active

bboxName = path+"/Projections/EASE2_N0_"+str(spres)+"km_Projection.nc"  # "/Volumes/Cressida/Projections/EASE2_N0_100km_SeaIceRegions.pkl" # path+"/Projections/EASE2_N0_100km_GenesisRegions.pkl" #
ncvar = 'z' #None #  Set to None unless file is a netcdf
bboxmin = -1*np.inf #None #  Set to None if using values instead
bboxmax = 500 #None #  Set to None if using values instead
values = [1] #[10,11,12,13,15] #  If using a min and max, set to 1, otherwise, these are the
## acccepted values within the mask raster (e.g., the regions of interest)

bboxnum = "10"
bboxmain = "" # The main bbox your subsetting from; usually "" for "all cyclones", otherwise BBox##

# Time Variables 
starttime = [1979,1,1,0,0,0] # Format: [Y,M,D,H,M,S]
endtime = [2021,1,1,0,0,0] # stop BEFORE this time (exclusive)
monthstep = [0,1,0,0,0,0] # A Time step that increases by 1 mont [Y,M,D,H,M,S]

dateref = [1900,1,1,0,0,0] # [Y,M,D,H,M,S]

months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
mons = ["01","02","03","04","05","06","07","08","09","10","11","12"]

minlifespan = 1 # in days
mintracklength = 1000 # in km for version 11 or later, in gridcells for version 10 or earlier
mindisplacement = 0 # in km

'''*******************************************
Main Analysis
*******************************************'''
print("Main Analysis")

# Load Mask
if bboxName.endswith(".nc"):
    bboxnc = nc.Dataset(bboxName)
    bbox = bboxnc[ncvar][:].data
elif bboxName.endswith(".pkl"):
    bbox = pd.read_pickle(bboxName)

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
    # cs = pickle5.load(open(inpath+"/"+bboxmain+"/"+typ+"Tracks/"+Y+"/"+bboxmain+typ.lower()+"tracks"+Y+M+".pkl",'rb'))
    cs = [tr for tr in cs if ((tr.lifespan() >= minlifespan) and (tr.trackLength() >= mintracklength)) and (maxDistFromGenPnt(tr.data) >= mindisplacement)]
    
    trs = []
    for tr in cs: # For each track
        # Extract time and location
        xs = np.array(tr.data.x)
        ys = np.array(tr.data.y)
        hours = np.array(tr.data.time*24)
        
        # Interpolate to hourly
        f = interpolate.interp1d(hours,xs)
        xs2 = f(np.arange(hours[0],hours[-1])).astype(int)
        f = interpolate.interp1d(hours,ys)
        ys2 = f(np.arange(hours[0],hours[-1])).astype(int)
        
        # Test if at least one point is within the mask
        if mask[ys2,xs2].sum() > 0:
            trs.append(tr)
    
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
