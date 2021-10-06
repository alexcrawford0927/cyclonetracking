'''
Author: Alex Crawford
Date Created: 10 Mar 2015
Date Modified:  18 Apr 2016; 10 Jul 2019 (update for Python 3);
                10 Sep 2020 (switch from geotiff to netcdf), switch to uniform_filter from scipy.ndimage
                30 Sep 2020 (switch back to slower custom smoother because of what scipy does to NaNs)
                18 Feb 2021 (edited seasonal caluclations to work directly from months, not monthly climatology,
                             allowing for cross-annual averaging)
                13 Sep 2021: If a pre-existing file exists, this script will append new results 
                            instead of overwriting for all years. Climatologies no longer in this script.
Purpose: Calculate aggergate statistics (Eulerian and Lagrangian) for either
cyclone tracks or system tracks.

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
import numpy as np
import netCDF4 as nc
import CycloneModule_12_4 as md
# import pickle5

'''*******************************************
Set up Environment
*******************************************'''
print("Setting up environment.")
bboxnum = "BBox10" # use "" if performing on all cyclones; or BBox##
typ = "System"
ver = "12_4E5P"

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

dateref = [1900,1,1,0,0,0] # [Y,M,D,H,M,S]

seasons = np.array([1,2,3,4,5,6,7,8,9,10,11,12]) # Ending month for three-month seasons (e.g., 2 = DJF)
months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
mons = ["01","02","03","04","05","06","07","08","09","10","11","12"]
dpm = [31,28,31,30,31,30,31,31,30,31,30,31]

# Aggregation Parameters
minls = 1 # minimum lifespan (in days) for a track to be considered
mintl = 1 # minimum track length (in km for version ≥ 11.1; grid cells for version ≤ 10.10) for a track to be considered
kSizekm = 800 # Full kernel size (in km) for spatial averaging measured between grid cell centers.
    ## For a 100 km spatial resolution, 400 is a 4 by 4 kernel; i.e., kSize = (kSizekm/spres)

# Variables
vNames = ["countA","gen","lys","spl","mrg"]
varsi = range(1,len(vNames)) # range(0,1) #
vunits = ['ratio','count','count','count','count']
agg = [-1,-1,-1,-1,-1]

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
spres = params['spres']

proj = nc.Dataset(suppath+"/EASE2_N0_"+str(spres)+"km_Projection.nc")
lats = proj['lat'][:]

kSize = int(kSizekm/spres) # This needs to be the full width ('diameter'), not the half width ('radius') for ndimage filters

print("Step 2. Check for Prior Data and update years of new analysis.")
startyears = [starttime[0] for i in vNames]
for v in varsi:
    name = ver+"_AggregationFields_Monthly_"+vNames[v]+".nc"
    if name in priorfiles:
        prior = nc.Dataset(name)
        startyears[v] = int(np.ceil(prior['time'][:].max()))

# Start at the earliest necessary time for ALL variables of interest
newstarttime = [np.min(np.array(startyears)[varsi]),1,1,0,0,0]

print("Step 2. Aggregate!")
vlists = [ [] for v in vNames]

mt = newstarttime
while mt != endtime:
    # Extract date
    Y = str(mt[0])
    MM = months[mt[1]-1]
    M = mons[mt[1]-1]
    print(" " + Y + " - " + MM)
    
    mtdays = md.daysBetweenDates(dateref,mt,lys=1) # Convert date to days since [1900,1,1,0,0,0]
    mt0 = md.timeAdd(mt,[-i for i in monthstep],lys=1) # Identify time for the previous month
    
    # Define number of valid times for making %s from counting stats
    if MM == "Feb" and md.leapyearBoolean(mt)[0] == 1:
        n = 29*(24/timestep[3])
    else:
        n = dpm[mt[1]-1]*(24/timestep[3])
    
    ### LOAD TRACKS ###
    # Load Cyclone/System Tracks
    cs = pd.read_pickle(inpath+"/"+bboxnum+"/"+typ+"Tracks/"+Y+"/"+bboxnum+typ.lower()+"tracks"+Y+M+".pkl")
    # cs = pickle5.load(open(inpath+"/"+bboxnum+"/"+typ+"Tracks/"+Y+"/"+bboxnum+typ.lower()+"tracks"+Y+M+".pkl",'rb'))
    try:
        cs0 = pd.read_pickle(inpath+"/"+bboxnum+"/"+typ+"Tracks/"+str(mt0[0])+"/"+bboxnum+typ.lower()+"tracks"+str(mt0[0])+mons[mt0[1]-1]+".pkl")
        # cs0 = pickle5.load(open(inpath+"/"+bboxnum+"/"+typ+"Tracks/"+str(mt0[0])+"/"+bboxnum+typ.lower()+"tracks"+str(mt0[0])+mons[mt0[1]-1]+".pkl",'rb'))
    except:
        cs0 = []
    # Load Active tracks
    ct2 = pd.read_pickle(inpath+"/ActiveTracks/"+Y+"/activetracks"+Y+M+".pkl")
    # ct2 = pickle5.load(open(inpath+"/ActiveTracks/"+Y+"/activetracks"+Y+M+".pkl",'rb'))
    if typ == "Cyclone":
        cs2 = ct2
    else:
        try: # Convert active tracks to systems as well
            cs2 = md.cTrack2sTrack(ct2,[],dateref,1)[0]
        except:
            cs2 = []
    
    ### LIMIT TRACKS & IDS ###
    # Limit to tracks that satisfy minimum lifespan and track length
    trs = [c for c in cs if ((c.lifespan() > minls) and (c.trackLength() >= mintl))]
    trs2 = [c for c in cs2 if ((c.lifespan() > minls) and (c.trackLength() >= mintl))]
    trs0 = [c for c in cs0 if ((c.lifespan() > minls) and (c.trackLength() >= mintl))]
    
    ### CALCULATE FIELDS ###
    fields0 = [np.nan]
    fields1 = md.aggregateEvents([trs,trs0,trs2],typ,mtdays,lats.shape)

    fields = fields0 + fields1
    
    ### SMOOTH FIELDS ###
    for v in varsi:
        # varFieldsm = md.smoothField(fields[v],kSize) # Smooth
        varFieldsm = ndimage.uniform_filter(fields[v],kSize,mode="nearest") # --> This cannot handle NaNs
        vlists[v].append(varFieldsm) # append to list
    
    # Increment Month
    mt = md.timeAdd(mt,monthstep,lys=1)

### SAVE FILE ###
print("Step 3. Write to NetCDF")
for v in varsi:
    mnc = nc.Dataset(ver+"_AggregationFields_Monthly_"+vNames[v]+"_NEW.nc",'w')
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
    
    vout = np.array(vlists[v])
    ncvar = mnc.createVariable(vNames[v], np.float64, ('time','y','x'))
    ncvar.units = vunits[v] + ' -- Smoothing:' + str(kSizekm) + ' km'
    
    name = ver+"_AggregationFields_Monthly_"+vNames[v]+".nc"
    if name in priorfiles: # Append data if prior data existed...
        if vout.shape[0] > 0: # ...and there is new data to be added
            prior = nc.Dataset(name)
            ncvar[:] = np.concatenate( ( prior[vNames[v]][prior['time'][:].data <= newstarttime[0],:,:].data , np.where(vout == 0,np.nan,vout) ) )
            prior.close(), mnc.close()
            os.remove(name) # Remove old file
            os.rename(ver+"_AggregationFields_Monthly_"+vNames[v]+"_NEW.nc", name) # rename new file to standard name

    else: # Create new data if no prior data existed
        ncvar[:] = np.where(vout == 0,np.nan,vout)
        mnc.close()
        os.rename(ver+"_AggregationFields_Monthly_"+vNames[v]+"_NEW.nc", name) # rename new file to standard name

   
