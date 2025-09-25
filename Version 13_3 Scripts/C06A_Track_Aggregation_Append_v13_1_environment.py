'''
Author: Alex Crawford
Date Created: 29 Aug 2023
Date Modified:  

Purpose: Calculate aggergate statistics (Eulerian and Lagrangian) for environmental 
variables that have been linked to either cyclone tracks or system tracks.

'''

'''********************
Import Modules
********************'''
# Import clock:
import CycloneModule_13_2 as md
import xarray as xr
import netCDF4 as nc
import numpy as np
from scipy import ndimage
import pandas as pd
import os
import warnings
from time import perf_counter as clock
start = clock()
warnings.filterwarnings("ignore")

print("Loading modules.")

'''*******************************************
Set up Environment
*******************************************'''
print("Setting up environment.")
bboxnum = "BBox10"  # use "" if performing on all cyclones; or BBox##
typ = "System"
ver = "13testP"

path = '/media/alex/Datapool'  # /Volumes/Cressida"
inpath = path+"/CycloneTracking/tracking"+ver
outpath = inpath+"/"+bboxnum
suppath = path+"/Projections"

'''*******************************************
Define Variables
*******************************************'''
print("Defining variables")
# Time Variables
starttime = [1978, 11, 1, 0, 0, 0]  # Format: [Y,M,D,H,M,S]
endtime = [2025, 1, 1, 0, 0, 0]  # stop BEFORE this time (exclusive)
# A Time step that increases by 1 month [Y,M,D,H,M,S]
monthstep = [0, 1, 0, 0, 0, 0]

dateref = [1900, 1, 1, 0, 0, 0]  # [Y,M,D,H,M,S]

# Inputs Parameters
insize = 1000  # radius (km) used for averaging the environmental variables
envname1 = 'SpatialAvgEnv'  # name in title of files
dataset = 'ERA5'
minlat2 = 15  # Trim any location that is south of this latitude


# Variables
# range(0,len(vNames)) # The 'n' index MUST be included and MUST be last!!!!
varsi = [-10,-9,-8,-7,-6,-5,-3, -2, -1]
vNames = ['age', 'uvAvg', 'uvMax', 'tcw', 'tcw_Adv', 'tp',
          'e', 'viwvd', 'siconc', 'peff', 'ttr', 'tnetrad', 'n']
vunits = ['ratio', 'm/s', 'm/s', 'mm', 'mm', 'mm', 'mm',
          'mm', 'ratio', 'ratio', 'W/m2', 'W/m2', 'count']

'''*******************************************
Main Analysis
*******************************************'''
print("Main Analysis")
# Ensure that folders exist to store outputs
try:
    os.chdir(outpath+"/Aggregation"+typ+"/"+str(insize*2)+"km")
except:
    os.mkdir(outpath+"/Aggregation"+typ+"/"+str(insize*2)+"km")
    os.chdir(outpath+"/Aggregation"+typ+"/"+str(insize*2)+"km")
priorfiles = os.listdir()

print("Step 1. Load Files and References")
# Read in attributes of reference files
params = pd.read_pickle(inpath+"/cycloneparams.pkl")
# params = pickle5.load(open(inpath+"/cycloneparams.pkl",'rb'))
timestep = params['timestep']
spres = params['spres']

if int(ver.split("_")[0]) < 14:
    proj = xr.open_dataset(suppath+"/EASE2_N0_" +str(spres)+"km_Projection_uv.nc")
else:
    proj = xr.open_dataset(suppath+"/EASE2_N0_"+str(spres)+"km_Projection.nc")
lats = proj['lat'][:]

print("Step 2. Aggregation requested for " +
      str(starttime[0]) + "-" + str(endtime[0]-1))
startyears, endyears = [starttime[0]+(starttime[1]-1)/12 for i in vNames], [
    endtime[0]+(endtime[1]-1)/12 for i in vNames]
firstyears, nextyears = [starttime[0]+(starttime[1]-1)/12 for i in vNames], [
    endtime[0]+(endtime[1]-1)/12 for i in vNames]

for v in varsi:
    name = ver+"_AggregationFields_Monthly_"+vNames[v]+".nc"
    if name in priorfiles:
        prior = xr.open_dataset(name)

        nextyears[v] = prior['time'][:].data.max()
        firstyears[v] = prior['time'][:].data.min()

        # If the desired time range starts before the prior years...
        if np.round(starttime[0]+(starttime[1]-1)/12, 2) < np.round(firstyears[v], 2):
            if endtime[0]+(endtime[1]-1)/12 >= firstyears[v]:
                endyears[v] = firstyears[v]
            else:
                raise Exception("There is a gap between the ending year requested ("+str(endtime[0]-1)+") and the first year already aggregated ("+str(
                    firstyears[v])+"). Either increase the ending year or choose a different destination folder.")
        # If the desired range ends after the prior years...
        elif np.round(endtime[0]+(endtime[1]-1)/12, 2) > np.round(nextyears[v], 2):
            if starttime[0]+(starttime[1]-1)/12 <= nextyears[v]:
                startyears[v] = np.round((nextyears[v]+1/12)*12)/12
            else:
                raise Exception("There is a gap between the last year already aggregated ("+str(nextyears[v]-1)+") and the starting year requested ("+str(
                    starttime[0])+"). Either decrease the starting year or choose a different destination folder.")
        else:
            raise Exception("All requested years are already aggregated.")
    else:
        startyears[v], endyears[v] = starttime[0] + \
            (starttime[1]-1)/12, endtime[0]+(endtime[1]-1)/12

# Start at the earliest necessary time for ALL variables of interest
newstartyear = np.min(np.array(startyears)[varsi])
newendyear = np.max(np.array(endyears)[varsi])

newstarttime = [int(np.floor(newstartyear)), int(
    np.round((newstartyear % 1)*12+1)), 1, 0, 0, 0]
newendtime = [int(np.floor(newendyear)), int(
    np.round((newendyear % 1)*12+1)), 1, 0, 0, 0]

print("Some years may have already been aggregated.\nAggregating for " +
      str(newstarttime) + " to " + str(newendtime) + ".")

vlists = [[] for v in vNames]

# Define kernel
kernel = md.circleKernel(int(insize/spres), masked_value=0)

# Start Loop
mt = newstarttime
while mt != newendtime:
    # Extract date
    Y = str(mt[0])
    MM = md.mmm[mt[1]-1]
    M = md.dd[mt[1]-1]
    print(" " + Y + " - " + MM)

    # Convert date to days since [1900,1,1,0,0,0]
    mtdays = md.daysBetweenDates(dateref, mt, lys=1)
    # Identify time for the previous month
    mt0 = md.timeAdd(mt, [-i for i in monthstep], lys=1)

    # Define number of valid times for making %s from counting stats
    if MM == "Feb" and md.leapyearBoolean(mt)[0] == 1:
        n = 29*(24/timestep[3])
    else:
        n = md.dpm[mt[1]-1]*(24/timestep[3])

    ### LOAD DATA ###
    # Load environmental data
    pdf = pd.read_csv(inpath+"/"+bboxnum+"/Aggregation"+typ+"/"+envname1 +
                      "_"+str(insize)+"km/"+envname1+str(insize)+"km_"+Y+M+".csv")
    pdf['peff'] = pdf['tp'] / pdf['tcw']

    ### CALCULATE FIELDS ###
    # Create empty fields
    fields = [np.zeros(lats.shape) for i in range(len(vNames))]
    narr = np.zeros(lats.shape)

    # Add to fields
    for row in range(pdf.shape[0]):
        # Find locations
        y, x = int(pdf.loc[row].y), int(pdf.loc[row].x)
        ky, kx = (np.array(kernel.shape)/2).astype(int)

        for v in varsi[:-1]:
            # Add data
            fields[v][(y-ky):(y+ky+1), (x-kx):(x+kx+1)] += kernel * \
                pdf.loc[row][vNames[v]]

        # Add n
        narr[(y-ky):(y+ky+1), (x-kx):(x+kx+1)] += kernel

    # Aggregate all rows
    for v in varsi[:-1]:
        vlists[v].append(fields[v] / narr)

    vlists[-1].append(narr)

    # Increment Month
    mt = md.timeAdd(mt, monthstep, lys=1)

### SAVE FILE ###
print("Step 3. Write to NetCDF")
for v in varsi:
    print(vNames[v])
    mnc = nc.Dataset(ver+"_AggregationFields_Monthly_" +
                     vNames[v]+"_NEW.nc", 'w')
    mnc.createDimension('y', lats.shape[0])
    mnc.createDimension('x', lats.shape[1])
    mnc.createDimension('time', int(
        np.round(12*(max(nextyears[v], newendyear)-min(firstyears[v], newstartyear)))))
    mnc.description = 'Aggregation of cyclone track ' + \
        vNames[v] + ' on monthly time scale.'

    ncy = mnc.createVariable('y', np.float32, ('y',))
    ncx = mnc.createVariable('x', np.float32, ('x',))
    ncy.units, ncx.units = 'm', 'm'
    ncy[:] = np.arange(proj['lat'].shape[0]*spres*1000/-2 +
                       (spres*1000/2), proj['lat'].shape[0]*spres*1000/2, spres*1000)
    ncx[:] = np.arange(proj['lat'].shape[1]*spres*1000/-2 +
                       (spres*1000/2), proj['lat'].shape[1]*spres*1000/2, spres*1000)

    # Add times, lats, and lons
    nctime = mnc.createVariable('time', np.float32, ('time',))
    nctime.units = 'years'
    nctime[:] = min(firstyears[v], newstartyear) + \
        np.arange(mnc['time'].shape[0])/12

    nclon = mnc.createVariable('lon', np.float32, ('y', 'x'))
    nclon.units = 'degrees'
    nclon[:] = proj['lon'][:]

    nclat = mnc.createVariable('lat', np.float32, ('y', 'x'))
    nclat.units = 'degrees'
    nclat[:] = proj['lat'][:]

    vout = np.array(vlists[v])
    vout = np.where(vout == 0, np.nan, vout)
    ncvar = mnc.createVariable(vNames[v], np.float64, ('time', 'y', 'x'))
    ncvar.units = vunits[v] + ' -- Smoothing:' + str(insize*2) + ' km'

    name = ver+"_AggregationFields_Monthly_"+vNames[v]+".nc"
    if name in priorfiles:  # Append data if prior data existed...
        if (vout.shape[0] > 0) & (prior[vNames[v]].shape != vout.shape):  # ...and there is new data to be added
            prior = nc.Dataset(name)

            # If the new data starts before and ends after prior data
            if (startyears[v] <= firstyears[v]) and (endyears[v] >= nextyears[v]):
                ncvar[:] = vout

            # If the new data starts after and ends before prior data
            elif (startyears[v] > firstyears[v]) and (endyears[v] < nextyears[v]):
                ncvar[:] = np.concatenate((prior[vNames[v]][prior['time'][:].data < newstartyear, :, :].data,
                                          vout, prior[vNames[v]][prior['time'][:].data >= newendyear, :, :].data))

            # If the new data starts and ends before the prior data
            elif (endyears[v] <= firstyears[v]):
                ncvar[:] = np.concatenate(
                    (vout, prior[vNames[v]][prior['time'][:].data >= newendyear, :, :].data))

            # If the new data starts and ends after the prior data
            elif (endyears[v] >= nextyears[v]):
                ncvar[:] = np.concatenate(
                    (prior[vNames[v]][prior['time'][:].data < newstartyear, :, :].data, vout))

            else:
                mnc.close()
                raise Exception('''Times are misaligned.\n
                                Requested Year Range: ''' + str(starttime[0]) + "-" + str(endtime[0]-1) + '''.
                                Processed Year Range: ''' + str(newstarttime[0]) + "-" + str(newendtime[0]-1) + '''.
                                New Data Year Range: ''' + str(startyears[v]) + '-' + str(endyears[v]-1)+'.')

            prior.close(), mnc.close()
            os.remove(name)  # Remove old file
            # rename new file to standard name
            os.rename(ver+"_AggregationFields_Monthly_" +
                      vNames[v]+"_NEW.nc", name)

    else:  # Create new data if no prior data existed
        ncvar[:] = vout
        mnc.close()
        # rename new file to standard name
        os.rename(ver+"_AggregationFields_Monthly_"+vNames[v]+"_NEW.nc", name)

if (newendyear < endtime[0]+(endtime[1]-1)/12) & (max(nextyears) < endtime[0]+(endtime[1]-1)/12):
    print("Completed aggregating " + str(newstarttime) + "-" + str(newendtime) +
          ".\nRe-run this script to aggregate any time after " + str(max(nextyears[v], newendtime[0])-1) + ".")
else:
    print("Completed aggregating " +
          str(newstarttime) + "-" + str(newendtime)+".")
