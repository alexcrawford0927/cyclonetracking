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
Purpose: Calculate aggergate track density (Eulerian and Lagrangian) for either
cyclone tracks or system tracks - can only be run AFTER the events aggregator.

User inputs:
    Path Variables, including the reanalysis (ERA, MERRA, CFSR)
    Track Type (typ): Cyclone or System
    Bounding Box ID (bboxnum): 2-digit character string
    Time Variables: when to start, end, the time step of the data
    Aggregation Parameters (minls, mintl, kSizekm)

Other notes:
    Units for track density are tracks/month/gridcell
    Units for event counts are raw counts (#/month/gridcell)
    Units for counts relative to cyclone obs are ratios (%/gridcell/100)
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
import CycloneModule_12_4 as md

'''*******************************************
Set up Environment
*******************************************'''
print("Setting up environment.")
bboxnum = "BBox10" # use "" if performing on all cyclones; or BBox##
typ = "System"
ver = "12_9E5R"

path = "/Volumes/Cressida"
inpath = path+"/CycloneTracking/tracking"+ver
outpath = inpath+"/"+bboxnum
suppath = path+"/Projections"

'''*******************************************
Define Variables
*******************************************'''
print("Defining variables")
# Time Variables 
starttime = [1979,1,1,0,0,0] # Format: [Y,M,D,H,M,S]
endtime = [2021,1,1,0,0,0] # stop BEFORE this time (exclusive)
monthstep = [0,1,0,0,0,0] # A Time step that increases by 1 month [Y,M,D,H,M,S]

seasons = np.array([1,2,3,4,5,6,7,8,9,10,11,12]) # Ending month for three-month seasons (e.g., 2 = DJF)
months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
mons = ["01","02","03","04","05","06","07","08","09","10","11","12"]
dpm = [31,28,31,30,31,30,31,31,30,31,30,31]

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
    os.chdir(outpath+"/Aggregation"+typ+"/"+str(kSizekm)+"km")
except:
    os.mkdir(outpath+"/Aggregation"+typ+"/"+str(kSizekm)+"km")
    os.chdir(outpath+"/Aggregation"+typ+"/"+str(kSizekm)+"km")
priorfiles = os.listdir()

print("Step 1. Load Files and References")
# Read in attributes of reference files
params = pd.read_pickle(inpath+"/cycloneparams.pkl")
# params = pickle5.load(open(inpath+"/cycloneparams.pkl",'rb'))
timestep = params['timestep']
try:
    spres = params['spres']
except:
    spres = 100

proj = nc.Dataset(suppath+"/EASE2_N0_"+str(spres)+"km_Projection.nc")
lats = proj['lat'][:]

kSize = int(kSizekm/spres) # This needs to be the full width ('diameter'), not the half width ('radius') for ndimage filters

print("Step 2. Aggregate!")
name = ver+"_AggregationFields_Monthly_"+vName+".nc"
if name in priorfiles:
    prior = nc.Dataset(name)
    startyear = int(np.ceil(prior['time'][:].max()))
else:
    startyear = starttime[0]
    
# Start at the earliest necessary time for ALL variables of interest
newstarttime = [startyear,1,1,0,0,0]

vlists = []

mt = newstarttime
while mt != endtime:
    # Extract date
    Y = str(mt[0])
    MM = months[mt[1]-1]
    M = mons[mt[1]-1]
    print(" " + Y + " - " + MM)
        
    ### LOAD TRACKS ###
    # Load Cyclone/System Tracks
    # cs = pickle5.load(open(inpath+"/"+bboxnum+"/"+typ+"Tracks/"+Y+"/"+bboxnum+typ.lower()+"tracks"+Y+M+".pkl",'rb'))
    cs = pd.read_pickle(inpath+"/"+bboxnum+"/"+typ+"Tracks/"+Y+"/"+bboxnum+typ.lower()+"tracks"+Y+M+".pkl")
    
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
    varFieldsm = ndimage.uniform_filter(trk_field,kSize,mode="nearest") # --> This cannot handle NaNs
    vlists.append(varFieldsm) # append to list
    
    # Increment Month
    mt = md.timeAdd(mt,monthstep,lys=1)

### SAVE FILE ###
print("Step 3. Write to NetCDF")
mnc = nc.Dataset(ver+"_AggregationFields_Monthly_"+vName+"_NEW.nc",'w')
mnc.createDimension('y', lats.shape[0])
mnc.createDimension('x', lats.shape[1])
mnc.createDimension('time', (endtime[0]-starttime[0])*12)
mnc.description = 'Aggregation of cyclone track characteristics on monthly time scale.'

ncy = mnc.createVariable('y', np.float32, ('y',))
ncx = mnc.createVariable('x', np.float32, ('x',))
ncy.units, ncx.units = 'm', 'm'
ncy[:] = np.arange(proj['lat'].shape[0]*spres*1000/-2 + (spres*1000/2),proj['lat'].shape[0]*spres*1000/2, spres*1000)
ncx[:] = np.arange(proj['lat'].shape[1]*spres*1000/-2 + (spres*1000/2),proj['lat'].shape[1]*spres*1000/2, spres*1000)
    
# Add times, lats, and lons
nctime = mnc.createVariable('time', np.float32, ('time',))
nctime.units = 'years'
nctime[:] = np.arange(starttime[0],endtime[0],1/12)

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
        ncvar[:] = np.concatenate( ( prior[vName][prior['time'][:].data <= newstarttime[0],:,:].data , np.where(vout == 0,np.nan,vout) ) )
        prior.close(), mnc.close()
        os.remove(name) # Remove old file
        os.rename(ver+"_AggregationFields_Monthly_"+vName+".nc", name) # rename new file to standard name

else: # Create new data if no prior data existed
    ncvar[:] = np.where(vout == 0,np.nan,vout)
    mnc.close()
    os.rename(ver+"_AggregationFields_Monthly_"+vName+".nc", name) # rename new file to standard name

