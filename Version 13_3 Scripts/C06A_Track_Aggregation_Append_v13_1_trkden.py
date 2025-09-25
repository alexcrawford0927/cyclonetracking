'''
Author: Alex Crawford
Date Created: 10 Mar 2015
Date Modified: 18 Apr 2016; 10 Jul 2019 (update for Python 3);
                10 Sep 2020 (switch from geotiff to netcdf), switch to uniform_filter from scipy.ndimage
                30 Sep 2020 (switch back to slower custom smoother because of what scipy does to NaNs)
                18 Feb 2021 (edited seasonal caluclations to work directly from months, not monthly climatology,
                             allowing for cross-annual averaging)
                09 Sep 2021: If a pre-existing file exists, this script will append new results
                            instead of overwriting for all years. Climatologies no longer in this script.
                01 Nov 2021: Added the possibility of appending prior years as will as subsequent years.
                23 Jan 2023: Adapted to version 13
                06 Feb 2025: Changed to eliminate all NaNs -- if no cyclone tracks are found, the trakc density is now 0
                    --> Is is likely a reversion to previous behavior because the NaNs were being introduced at the monthly aggregation step, not the initial assignment
Purpose: Calculate aggergate track density (Eulerian-Lagrangian hybrid) for either
cyclone tracks or system tracks.

User inputs:
    Path Variables, including the reanalysis (ERA, MERRA, CFSR)
    Track Type (typ): Cyclone or System
    Bounding Box ID (bboxnum): 2-digit character string
    Time Variables: when to start, end, the time step of the data
    Aggregation Parameters (minls, mintl, kSizekm)

Note: Units for track density are tracks/month/gridcell
'''

'''********************
Import Modules
********************'''
# Import clock:
from time import perf_counter as clock
start = clock()
import warnings
warnings.filterwarnings("ignore")

print("Loading modules.")
import os
import pandas as pd
from scipy import ndimage
from scipy import interpolate
import numpy as np
import netCDF4 as nc
# import pickle5
import CycloneModule_13_3 as md

'''*******************************************
Set up Environment
*******************************************'''
print("Setting up environment.")
subset = "" # use "" if performing on all cyclones; or BBox##
typ = "System"
ver = "13testP"

path = "/Volumes/Cressida" # "/media/alex/Datapool" # 
inpath = path+"/CycloneTracking/tracking"+ver
outpath = inpath+"/"+subset
suppath = path+"/Projections"

'''*******************************************
Define Variables
*******************************************'''
print("Defining variables")
# Time Variables
starttime = [1979,1,1,0,0,0] # Format: [Y,M,D,H,M,S]
endtime = [1979,2,1,0,0,0] # stop BEFORE this time (exclusive)
monthstep = [0,1,0,0,0,0] # A Time step that increases by 1 month [Y,M,D,H,M,S]

seasons = np.array([1,2,3,4,5,6,7,8,9,10,11,12]) # Ending month for three-month seasons (e.g., 2 = DJF)

# Aggregation Parameters
minls = 1 # minimum lifespan (in days) for a track to be considered
mintl = 1000 # minimum track length (in km for version ≥ 11.1; grid cells for version ≤ 10.10) for a track to be considered
kSizekm = 400 # Full kernel size (in km) for spatial averaging measured between grid cell centers.
    ## For a 100 km spatial resolution, 400 is a 4 by 4 kernel; i.e., kSize = (kSizekm/spres)

# Variables
vName = "trkden"
vunits = 'count'

'''*******************************************
Main Analysis
*******************************************'''
print("Main Analysis")
# Ensure that folders exist to store outputs
try:
   os.chdir(outpath+"/Aggregation"+typ)
except:
    os.mkdir(outpath+"/Aggregation"+typ)
    os.chdir(outpath+"/Aggregation"+typ)
try:
    os.chdir(outpath+"/Aggregation"+typ+"/"+str(kSizekm)+"km")
except:
    os.mkdir(outpath+"/Aggregation"+typ+"/"+str(kSizekm)+"km")
    os.chdir(outpath+"/Aggregation"+typ+"/"+str(kSizekm)+"km")
priorfiles = os.listdir()

print("Step 1. Load Files and References")
# Read in attributes of reference files
params = pd.read_pickle(inpath+"/cycloneparams.pkl")
# params = pickle5.load(open(inpath+"/cycloneparams.pkl",'rb'))

try:
    spres = params['spres']
except:
    spres = 100
    
if int(ver.split('_')[0]) < 14:
    prjpath = path+"/Projections/EASE2_N0_"+str(spres)+"km_Projection_uv.nc"
else:
    prjpath = path+"/Projections/EASE2_N0_"+str(spres)+"km_Projection.nc"

proj = nc.Dataset(prjpath)
lats = proj['lat'][:]

kSize = int(kSizekm/spres) # This needs to be the full width ('diameter'), not the half width ('radius') for ndimage filters
kernel = md.circleKernel(int(kSize),masked_value=0)

print("Step 2. Aggregation requested for " + str(starttime[0]) + "-" + str(endtime[0]-1))
name = ver+"_AggregationFields_Monthly_"+vName+".nc"
if name in priorfiles:
    prior = nc.Dataset(name)
    nextyear = int(np.ceil(prior['time'][:].max()))
    firstyear = int(np.floor(prior['time'][:].min()))
    if starttime[0] < firstyear: # If the desired time range starts before the prior years...
        if endtime[0] >= firstyear:
            startyear, endyear = starttime[0], firstyear
            print("Years " + str(firstyear) + "-"+str(nextyear-1) + " were already aggregated.\nAggregating for " + str(startyear) + "-" + str(endyear-1) + ".")
        else:
            raise Exception("There is a gap between the ending year requested ("+str(endtime[0]-1)+") and the first year already aggregated ("+str(firstyear)+"). Either increase the ending year or choose a different destination folder.")
    elif endtime[0] > nextyear: # If the desired range ends after the prior years...
        if starttime[0] <= nextyear:
            startyear, endyear = nextyear, endtime[0]
            print("Years " + str(firstyear) + "-"+str(nextyear-1) + " were already aggregated.\nAggregating for " + str(startyear) + "-" + str(endyear-1) + ".")
        else:
            raise Exception("There is a gap between the last year already aggregated ("+str(nextyear-1)+") and the starting year requested ("+str(starttime[0])+"). Either decrease the starting year or choose a different destination folder.")
    else:
        raise Exception("All requested years are already aggregated.")
else:
    startyear, endyear, firstyear, nextyear = starttime[0], endtime[0], starttime[0], endtime[0]

# Start at the earliest necessary time for ALL variables of interest
newstarttime = [startyear,1,1,0,0,0]
newendtime = [endyear,1,1,0,0,0]

vlists = []

mt = newstarttime
while mt != newendtime:
    # Extract date
    Y = str(mt[0])
    MM = md.mmm[mt[1]-1]
    M = md.dd[mt[1]-1]
    if MM == "Jan":
        print(" " + Y)

    ### LOAD TRACKS ###
    # Load Cyclone/System Tracks
    # cs = pickle5.load(open(inpath+"/"+bboxnum+"/"+typ+"Tracks/"+Y+"/"+bboxnum+typ.lower()+"tracks"+Y+M+".pkl",'rb'))
    cs = pd.read_pickle(inpath+"/"+subset+"/"+typ+"Tracks/"+Y+"/"+subset+typ.lower()+"tracks"+Y+M+".pkl")

    ### LIMIT TRACKS & IDS ###
    # Limit to tracks that satisfy minimum lifespan and track length
    trs = [c for c in cs if ((c.lifespan() > minls) and (c.trackLength() >= mintl))]

    ### CALCULATE FIELDS ###
    trk_field = np.zeros_like(lats)
    for tr in trs:
        # Extract time and location
        xs = np.array(tr.data.x)
        ys = np.array(tr.data.y)
        hours = np.array(tr.data.time*24)

        # Interpolate to hourly
        f = interpolate.interp1d(hours,xs)
        xs2 = f(np.arange(hours[0],hours[-1])).astype(int)
        f = interpolate.interp1d(hours,ys)
        ys2 = f(np.arange(hours[0],hours[-1])).astype(int)

        # Zip together ys and xs and find unique values
        yxs2 = np.transpose(np.vstack( (ys2,xs2) ))
        yxs3 = np.unique(yxs2,axis=0)

        # Record Existance of Track at each unique point
        for i in range(yxs3.shape[0]):
            x = yxs3[i,1]
            y = yxs3[i,0]

            trk_field[y,x] += 1

    ### SMOOTH FIELDS ###
    # varFieldsm = np.array( ndimage.generic_filter( trk_field, np.nanmean, footprint=kernel, mode='nearest' ) )
    varFieldsm = ndimage.uniform_filter(trk_field,kSize,mode="nearest") # --> This cannot handle NaNs
    vlists.append(varFieldsm) # append to list

    # Increment Month
    mt = md.timeAdd(mt,monthstep,lys=1)

### SAVE FILE ###
print("Step 3. Write to NetCDF")
mnc = nc.Dataset(ver+"_AggregationFields_Monthly_"+vName+"_NEW.nc",'w')
mnc.createDimension('y', lats.shape[0])
mnc.createDimension('x', lats.shape[1])
mnc.createDimension('time', (max(nextyear,newendtime[0])-min(firstyear,newstarttime[0]))*12)
mnc.description = 'Aggregation of cyclone track characteristics on monthly time scale.'

ncy = mnc.createVariable('y', np.float32, ('y',))
ncx = mnc.createVariable('x', np.float32, ('x',))
ncy.units, ncx.units = 'm', 'm'
ncy[:] = np.arange(proj['lat'].shape[0]*spres*1000/-2 + (spres*1000/2),proj['lat'].shape[0]*spres*1000/2, spres*1000)
ncx[:] = np.arange(proj['lat'].shape[1]*spres*1000/-2 + (spres*1000/2),proj['lat'].shape[1]*spres*1000/2, spres*1000)

# Add times, lats, and lons
nctime = mnc.createVariable('time', np.float32, ('time',))
nctime.units = 'years'
nctime[:] = np.arange(min(firstyear,newstarttime[0]),max(nextyear,newendtime[0]),1/12)

nclon = mnc.createVariable('lon', np.float32, ('y','x'))
nclon.units = 'degrees'
nclon[:] = proj['lon'][:]

nclat = mnc.createVariable('lat', np.float32, ('y','x'))
nclat.units = 'degrees'
nclat[:] = proj['lat'][:]

ncvar = mnc.createVariable(vName, np.float64, ('time','y','x'))
ncvar.units = vunits + ' -- Smoothing:' + str(kSizekm) + ' km'
vout = np.array(vlists)

name = ver+"_AggregationFields_Monthly_"+vName+".nc"
if name in priorfiles: # Append data if prior data existed...
    if vout.shape[0] > 0: # ...and there is new data to be added
        prior = nc.Dataset(name)

        if starttime[0] < firstyear:
            ncvar[:] = np.concatenate( ( vout , prior[vName][:].data ) )
        else:
            ncvar[:] = np.concatenate( ( prior[vName][:].data ,vout ) )


        if (vout.shape[0] > 0) & (prior[vName].shape != vout.shape): # ...and there is new data to be added
            prior = nc.Dataset(name)

            if (startyear <= firstyear) and (endyear >= nextyear): # If the new data starts before and ends after prior data
                ncvar[:] = vout

            elif (startyear > firstyear) and (endyear < nextyear): # If the new data starts after and ends before prior data
                ncvar[:] = np.concatenate( ( prior[vName][prior['time'][:].data < newstarttime[0],:,:].data , vout , prior[vName][prior['time'][:].data >= newendtime[0],:,:].data ) )

            elif (endyear <= firstyear): # If the new data starts and ends before the prior data
                ncvar[:] = np.concatenate( ( vout , prior[vName][prior['time'][:].data >= newendtime[0],:,:].data ) )

            elif (endyear >= nextyear): # If the new data starts and ends after the prior data
                ncvar[:] = np.concatenate( ( prior[vName][prior['time'][:].data < newstarttime[0],:,:].data , vout ) )

            else:
                mnc.close()
                raise Exception('''Times are misaligned.\n
                                Requested Year Range: ''' + str(starttime[0]) + "-" + str(endtime[0]-1) + '''.
                                Processed Year Range: ''' + str(newstarttime[0]) + "-" + str(newendtime[0]-1) + '''.
                                New Data Year Range: ''' + str(startyear) + '-' + str(endyear-1)+'.')

        mnc.close()

        os.remove(name) # Remove old file
        os.rename(ver+"_AggregationFields_Monthly_"+vName+"_NEW.nc", name) # rename new file to standard name

else: # Create new data if no prior data existed
    ncvar[:] = vout
    mnc.close()
    os.rename(ver+"_AggregationFields_Monthly_"+vName+"_NEW.nc", name) # rename new file to standard name

if (nextyear < endtime[0]) & (firstyear > starttime[0]):
    print("Completed aggregating " + str(startyear) + "-" + str(endyear-1)+".\nRe-run this script to aggregate " + str(nextyear) + "-" + str(endtime[0]-1) + ".")
else:
    print("Completed aggregating " + str(startyear) + "-" + str(endyear-1)+".")
