'''
Author: Alex Crawford
Date Created: 18 Feb 2021
Date Modified: 13 Sep 2021: If a pre-existing file exists, this script will append new results
            instead of overwriting for all years. Climatologies no longer in this script.
            01 Nov 2021: Added the possibility of appending prior years as will as subsequent years.
            23 Jan 2023: Adapted to version 13

User inputs:
    Path Variables, including the reanalysis (ERA, MERRA, CFSR)
    Track Type (typ): Cyclone or System
    Bounding Box ID (bboxnum): 2-digit character string
    Time Variables: when to start, end, the time step of the data

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
# import pickle5
import CycloneModule_13_2 as md

'''*******************************************
Set up Environment
*******************************************'''
print("Setting up environment.")
bboxnum = "BBox27" # use "" if performing on all cyclones; or BBox##
typ = "System"
ver = "13_2R"

path = "/Volumes/Cressida"
inpath = path+"/CycloneTracking/tracking"+ver
outpath = inpath+"/"+bboxnum
suppath = path+"/Projections"

'''*******************************************
Define Variables
*******************************************'''
print("Defining variables")
# Time Variables
starttime = [1950,1,1,0,0,0] # Format: [Y,M,D,H,M,S]
endtime = [1951,1,1,0,0,0] # stop BEFORE this time (exclusive)
monthstep = [0,1,0,0,0,0] # A Time step that increases by 1 month [Y,M,D,H,M,S]

dateref = [1900,1,1,0,0,0] # [Y,M,D,H,M,S]

seasons = np.array([1,2,3,4,5,6,7,8,9,10,11,12]) # Ending month for three-month seasons (e.g., 2 = DJF)
months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
mons = ["01","02","03","04","05","06","07","08","09","10","11","12"]
dpm = [31,28,31,30,31,30,31,31,30,31,30,31]

# Aggregation Parameters
minls = 1 # minimum lifespan (in days) for a track to be considered
mintl = 1000 # minimum track length (in km for version ≥ 11.1; grid cells for version ≤ 10.10) for a track to be considered
kSizekm = 800 # Full kernel size (in km) for spatial averaging measured between grid cell centers.
    ## For a 100 km spatial resolution, 400 is a 4 by 4 kernel; i.e., kSize = (kSizekm/spres)

# Variables (Note that countU is mandatory)
varsi = [0] + [1,2,3] + [4] + [5,6] #
vNames = ["countU"] + ["DpDt","u","v"] + ['uv'] + ['vratio','mci']
multiplier = [1] + [0.01,1,1] + [1,1,1] + [1,1]
vunits = ['percent'] + ['hPa/day','km/h','km/h'] + ['km/h'] + ['ratio of |v| to |uv|', 'ratio of v^2 to (v^2 + u^2)']

'''*******************************************
Main Analysis
*******************************************'''
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

print("Step 2. Aggregation requested for " + str(starttime[0]) + "-" + str(endtime[0]-1))
startyears, endyears = [starttime[0] for i in vNames], [endtime[0] for i in vNames]
firstyears, nextyears = [starttime[0] for i in vNames], [endtime[0] for i in vNames]
for v in varsi:
    name = ver+"_AggregationFields_Monthly_"+vNames[v]+".nc"
    if name in priorfiles:
        prior = nc.Dataset(name)

        nextyears[v] = int(np.ceil(prior['time'][:].max()))
        firstyears[v] = int(np.floor(prior['time'][:].min()))

        if starttime[0] < firstyears[v]: # If the desired time range starts before the prior years...
            if endtime[0] >= firstyears[v]:
                startyears[v], endyears[v] = starttime[0], firstyears[v]
            else:
                raise Exception("There is a gap between the ending year requested ("+str(endtime[0]-1)+") and the first year already aggregated ("+str(firstyears[v])+"). Either increase the ending year or choose a different destination folder.")
        elif endtime[0] > nextyears[v]: # If the desired range ends after the prior years...
            if starttime[0] <= nextyears[v]:
                startyears[v], endyears[v] = nextyears[v], endtime[0]
            else:
                raise Exception("There is a gap between the last year already aggregated ("+str(nextyears[v]-1)+") and the starting year requested ("+str(starttime[0])+"). Either decrease the starting year or choose a different destination folder.")
        else:
            raise Exception("All requested years are already aggregated.")
    else:
        startyears[0], endyears[0] = starttime[0], endtime[0]

# Start at the earliest necessary time for ALL variables of interest
newstarttime = [np.min(np.array(startyears)[varsi]),1,1,0,0,0]
newendtime = [np.max(np.array(endyears)[varsi]),1,1,0,0,0]

print("Some years may have already been aggregated.\nAggregating for " + str(newstarttime[0]) + "-" + str(newendtime[0]-1) + ".")

vlists = [ [] for v in vNames]

mt = newstarttime
while mt != newendtime:
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
    # cs = pickle5.load(open(inpath+"/"+bboxnum+"/"+typ+"Tracks/"+Y+"/"+bboxnum+typ.lower()+"tracks"+Y+M+".pkl",'rb'))
    cs = pd.read_pickle(inpath+"/"+bboxnum+"/"+typ+"Tracks/"+Y+"/"+bboxnum+typ.lower()+"tracks"+Y+M+".pkl")

    ### LIMIT TRACKS & IDS ###
    # Limit to tracks that satisfy minimum lifespan and track length
    trs = [c for c in cs if ((c.lifespan() > minls) and (c.trackLength() >= mintl))]

    ### CALCULATE FIELDS ###
    # Create empty fields
    fields = [np.zeros(lats.shape) for i in range(len(vNames))]

    for tr in trs:
        uvab = np.array(tr.data['uv'])

        # V Ratio & MCI
        vratio = np.zeros_like(uvab)*np.nan
        vratio[uvab > 0] = np.abs( np.array(tr.data['v'])[uvab > 0] / uvab[uvab > 0] )
        tr.data['vratio'] = vratio

        mci = np.zeros_like(uvab)*np.nan
        mci[uvab > 0] = np.square( np.array(tr.data['v'])[uvab > 0] / uvab[uvab > 0] )
        tr.data['mci'] = mci

        # Subset
        trdata = tr.data[np.isfinite(list(tr.data.u))][:-1]

        for i in trdata.index:
            x = int(trdata.x[i])
            y = int(trdata.y[i])

            fields[0][y,x] += 1 # Add one to the count
            for vi in varsi[1:]: # Add table value for intensity measures
                fields[vi][y,x] += float(trdata[vNames[vi]][i])

    # Append to main list
    field0sm = np.array( ndimage.generic_filter( fields[0], np.nansum, kSize, mode='nearest' ) )
    vlists[0].append( field0sm/n*100 ) # convert count to a %
    for vi in varsi[1:]:
        fieldsm = np.array( ndimage.generic_filter( fields[vi], np.nansum, kSize, mode='nearest' ) ) / field0sm
        vlists[vi].append(fieldsm*multiplier[vi]) # append to list

    # Increment Month
    mt = md.timeAdd(mt,monthstep,lys=1)

### SAVE FILE ###
print("Step 3. Write to NetCDF")
for v in varsi:
    print(vNames[v])
    mnc = nc.Dataset(ver+"_AggregationFields_Monthly_"+vNames[v]+"_NEW.nc",'w')
    mnc.createDimension('y', lats.shape[0])
    mnc.createDimension('x', lats.shape[1])
    mnc.createDimension('time', (max(nextyears[v],newendtime[0])-min(firstyears[v],newstarttime[0]))*12)
    mnc.description = 'Aggregation of cyclone track ' + vNames[v] + ' on monthly time scale.'

    ncy = mnc.createVariable('y', np.float32, ('y',))
    ncx = mnc.createVariable('x', np.float32, ('x',))
    ncy.units, ncx.units = 'm', 'm'
    ncy[:] = np.arange(proj['lat'].shape[0]*spres*1000/-2 + (spres*1000/2),proj['lat'].shape[0]*spres*1000/2, spres*1000)
    ncx[:] = np.arange(proj['lat'].shape[1]*spres*1000/-2 + (spres*1000/2),proj['lat'].shape[1]*spres*1000/2, spres*1000)

    # Add times, lats, and lons
    nctime = mnc.createVariable('time', np.float32, ('time',))
    nctime.units = 'years'
    nctime[:] = np.arange(min(firstyears[v],newstarttime[0]),max(nextyears[v],newendtime[0]),1/12)

    nclon = mnc.createVariable('lon', np.float32, ('y','x'))
    nclon.units = 'degrees'
    nclon[:] = proj['lon'][:]

    nclat = mnc.createVariable('lat', np.float32, ('y','x'))
    nclat.units = 'degrees'
    nclat[:] = proj['lat'][:]

    vout = np.array(vlists[v])
    vout = np.where(vout == 0,np.nan,vout)
    ncvar = mnc.createVariable(vNames[v], np.float64, ('time','y','x'))
    ncvar.units = vunits[v] + ' -- Smoothing:' + str(kSizekm) + ' km'

    name = ver+"_AggregationFields_Monthly_"+vNames[v]+".nc"
    if name in priorfiles: # Append data if prior data existed...
        if vout.shape[0] > 0: # ...and there is new data to be added
            prior = nc.Dataset(name)

            if (newstarttime[0] <= firstyears[v]) and (newendtime[0] >= nextyears[v]): # If the new data starts before and ends after prior data
                ncvar[:] = vout

            elif (newstarttime[0] > firstyears[v]) and (newendtime[0] < nextyears[v]): # If the new data starts after and ends before prior data
                ncvar[:] = np.concatenate( ( prior[vNames[v]][prior['time'][:].data < newstarttime[0],:,:].data , vout , prior[vNames[v]][prior['time'][:].data >= newendtime[0],:,:].data ) )

            elif (newendtime[0] <= firstyears[v]): # If the new data starts and ends before the prior data
                ncvar[:] = np.concatenate( ( vout , prior[vNames[v]][prior['time'][:].data >= newendtime[0],:,:].data ) )

            elif (newstarttime[0] >= nextyears[v]): # If the new data starts and ends after the prior data
                ncvar[:] = np.concatenate( ( prior[vNames[v]][prior['time'][:].data < newstarttime[0],:,:].data , vout ) )

            else:
                mnc.close()
                raise Exception("Times are misaligned. Requested Time Range: " + str(starttime) + "-" + str(endtime) + ". Processed Time Range: " + str(newstarttime) + "-" + str(newendtime) + ".")

            prior.close(), mnc.close()
            os.remove(name) # Remove old file
            os.rename(ver+"_AggregationFields_Monthly_"+vNames[v]+"_NEW.nc", name) # rename new file to standard name

    else: # Create new data if no prior data existed
        ncvar[:] = vout
        mnc.close()
        os.rename(ver+"_AggregationFields_Monthly_"+vNames[v]+"_NEW.nc", name) # rename new file to standard name

if (newendtime[0] < endtime[0]) & (max(nextyears) < endtime[0]):
    print("Completed aggregating " + str(newstarttime[0]) + "-" + str(newendtime[0]-1)+".\nRe-run this script to aggregate any time after " + str(max(nextyears[v],newendtime[0])-1) + ".")
else:
    print("Completed aggregating " + str(newstarttime[0]) + "-" + str(newendtime[0]-1)+".")
