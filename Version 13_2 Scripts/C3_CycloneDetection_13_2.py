'''
Author: Alex Crawford
Date Created: 20 Jan 2015
Date Modified: 10 Sep 2020 -> Branch from 12_1 --> kernel size is now based on km instead of cells
                16 Dec 2020 --> updated comments
                13 Jan 2021 --> changed  when masking for elevation happens -- after minima are detected, not before
                02 Mar 2022 --> transferred some constants to the module file to save space
                            --> changed the output for cyclone fields from a
                            single field per pickled file to to a list of fields for an
                            entire month in each pickled file
                14 Nov 2022 --> Added an if statement to make this more functional for the Southern Hemisphere
                23 Jan 2023 --> Branch from 12_4 --> added kSizekm to cycloneparams.pkl and switched from "surf" to "field"

Purpose: Given a series of sea level pressure fields in netcdf files, this
    script performs several steps:
    1) Identify closed low pressure centers at each time step
    2) Store information to characterize these centers at each time step
    3) Identify multi-center and single-center cyclone systems
  Steps in the tracking part:
    4) Associate each cyclone center with a corresponding center in the
        previous time step (when applicable)
    5) Combine the timeseries of cyclone center charcteristics into a data frame for each track
    6) Record cyclone life cycle events (genesis, lysis, splits, merges, secondary genesis)

User Inputs: paths for inputs, desired projection info, various detection/tracking parameters
'''

'''********************
Import Modules
********************'''
# Import clock:
import time
# Start script stopwatch. The clock starts running when time is imported
start = time.perf_counter()

print("Loading modules")
import os
import copy
import pandas as pd
import numpy as np
import netCDF4 as nc
import CycloneModule_13_2 as md
import warnings

np.seterr(all='ignore') # This mutes warnings from numpy
warnings.filterwarnings('ignore',category=DeprecationWarning)

'''*******************************************
Set up Environment
*******************************************'''
print("Setting up environment")
path = "/Volumes/Cressida"
dataset = "ERA5"
verd = "13_2" # Detection Version
vert = 'PTest' # Tracking Version
spres = 100 # Spatial resolution (in km)

inpath = path+"/"+dataset+"/SLP_EASE2_N0_"+str(spres)+"km"
outpath = path+"/CycloneTracking"
suppath = path+"/Projections/EASE2_N0_"+str(spres)+"km_Projection.nc"

'''********************
Define Variables/Parameters
********************'''
print("Defining parameters")
# File Variables
invar = "SLP"
ncvar = "msl" # 'msl' for ERA5, 'SLP' for MERRA2 & CFSR

# Time Variables
starttime = [1979,1,1,0,0,0] # Format: [Y,M,D,H,M,S]
endtime = [1979,2,1,0,0,0] # stop BEFORE this time (exclusive)
timestep = [0,0,0,6,0,0] # Time step in [Y,M,D,H,M,S]

dateref = [1900,1,1,0,0,0]  # [Y,M,D,H,M,S]

prior = 0 # 1 = a cyclone track object exists for a prior month; 0 = otherwise

# Detection Parameters #
minfield = 80000 # minimum reasonable value in field array
maxfield = 200000 # maximum reasonable value in field array

# Size of kernel (km) used to determine if a grid cell is a local minimum
##  Starting in Version 12.1, users enter a distance in km instead of the kernel
## size. This way, the kSize can adapt to different spatial resolutions. The
## equivalent of a 3 by 3 kernel with 100 km resolution would be 100
## i.e., kSize = (2*kSizekm/spres)+1
kSizekm = 200

# Maximum fraction of neighboring grid cells with no data (Not a Number) allowed
### for a grid cell to be considered during minimum detection
nanThresh = 0.4

# minimum slp gradient for identifying (and eliminating) weak minima:
d_slp = 750 # slp difference in Pa (use 0 to turn off)
d_dist = 1000000 # distance in m (units that match units of cellsize)

# maximum elevation for masking out high elevation minima
maxelev = 1500. # elevation in m (use 10000 to turn off)

# minimum latitude for masking out the Equator (default should be 5 for NH and -5 for SH)
minlat = 5

# Contour interval (Pa; determines the interval needed to identify closed
### contours,and therefore cyclone area)
contint = 200

# Multi-center cyclone (mcc) tolerance is the maximum ratio permitted between the
### number of unshared and total contours in a multi-centered cyclone. "Unshared"
### contours are only used by the primary center. "Shared" contours are used
### by both the primary and secondary centers.
mcctol = 0.5 # (use 0 to turn off mcc's; higher makes mcc's more likely)
# Multi-center cyclone (mcc) distance is the maximum distance (in m) two minima can
### lie apart and still be considered part of the same cyclone system
mccdist = 1200000

# Tracking Parameters #
# Maximum speed is the fastest that a cyclone center is allowed to travel; given
### in units of km/h. To be realistic, the number should be between 100 and 200.
### and probably over 125 (based on Rudeva et al. 2014). To turn off, set to
### np.inf. Also, note that instabilities occur at temporal resolution of 1-hr.
### Tracking at 6-hr and a maxspeed of 125 km/hr is more comprable to tracking
### at 1-hr and a maxspeed of 300 km/hr (assuming spatial resolution of 50 km).
maxspeed = 150 # constant value
# maxspeed = 150*(3*math.log(timestep[3],6)+2)/timestep[3] # One example of scaling by temporal resolution

# The reduction parameter is a scaling of cyclone speed.  When tracking, the
### algorithm uses the cyclone speed and direction from the last interval to
### estimate a "best guess" position. This parameter modifies that speed, making
### it slower. This reflects how cyclones tend to slow down as they mature. To
### turn off, set to 1.
red = 0.75

'''*******************************************
Main Analysis
*******************************************'''
print("Loading Folders & Reference Files")

##### Ensure that folders exist to store outputs #####
detpath = outpath+"/detection"+verd
trkpath = outpath+"/tracking"+verd+vert
try:
    os.chdir(detpath)
except:
    os.mkdir(detpath)
    os.chdir(detpath)
    os.mkdir("CycloneFields")
try:
    os.chdir(trkpath)
except:
    os.mkdir(trkpath)
    os.chdir(trkpath)
    os.mkdir("CycloneTracks")
    os.mkdir("ActiveTracks")
    os.mkdir("SystemTracks")

for y in range(starttime[0],endtime[0]+1):
    Y = str(y)

    # Cyclone Tracks
    try:
        os.chdir(trkpath+"/CycloneTracks/"+Y)
    except:
        os.mkdir(trkpath+"/CycloneTracks/"+Y)

    # Active Tracks
    try:
        os.chdir(trkpath+"/ActiveTracks/"+Y)
    except:
        os.mkdir(trkpath+"/ActiveTracks/"+Y)

    # System Tracks
    try:
        os.chdir(trkpath+"/SystemTracks/"+Y)
    except:
        os.mkdir(trkpath+"/SystemTracks/"+Y)

##### Read in attributes of reference files #####
projnc = nc.Dataset(suppath)

lats = projnc['lat'][:].data
lons = projnc['lon'][:].data
yDist = projnc['yDistance'][:].data
xDist = projnc['xDistance'][:].data
elev = projnc['z'][:]

# Generate mask based on latitude and elevation
if minlat >= 0:
    mask = np.where((elev > maxelev) | (lats < minlat),np.nan,0)
else:
    mask = np.where((elev > maxelev) | (lats > minlat),np.nan,0)

# Convert kernel size to grid cells
kSize = int(2*kSizekm/spres)+1

# Convert max speed to max distance
maxdist = maxspeed*1000*timestep[3]

# Save Parameters
params = dict({"path":trkpath,"timestep":timestep, "dateref":dateref, "minfield":minfield,
    "maxfield":maxfield,"kSize":kSize, "kSizekm":kSizekm,"nanThresh":nanThresh, "d_slp":d_slp, \
    "d_dist":d_dist, "maxelev":maxelev, "minlat":minlat, "contint":contint,
    "mcctol":mcctol, "mccdist":mccdist, "maxspeed":maxspeed, "red":red, "spres":spres})
pd.to_pickle(params,trkpath+"/cycloneparams.pkl")

##### The actual detection and tracking #####
print("Cyclone Detection & Tracking")
# Print elapsed time
print(' Elapsed time:',round(time.perf_counter()-start,2),'seconds -- Starting first month')

# Load netcdf for initial time
ncf = nc.Dataset(inpath+"/"+dataset+"_EASE2_N0_"+str(spres)+"km_"+invar+"_Hourly_"+str(starttime[0])+md.dd[starttime[1]-1]+".nc")
tlist = ncf['time'][:].data

# Try loading cyclone field objects for the initial month -- if none exist, make an empty list
try:
    cflist = pd.read_pickle(detpath+"/CycloneFields/CF"+str(starttime[0])+md.dd[starttime[1]-1]+".pkl")
    cftimes = np.array([cf.time for cf in cflist])
except:
    cflistnew = []

t = copy.deepcopy(starttime)
while t != endtime:
    # Extract date
    Y = str(t[0])
    MM = md.mmm[t[1]-1]
    M = md.dd[t[1]-1]
    date = Y+M+md.dd[t[2]-1]+"_"+md.hhmm[t[3]]
    days = md.daysBetweenDates(dateref,t)

    # Load field
    try: # If the cyclone field has already been calculated, no need to repeat
        cf = cflist[np.where(cftimes == days)[0]]
    except:
        field = ncf[ncvar][np.where(tlist == md.daysBetweenDates(dateref,t)*24)[0][0],:,:]
        field = np.where((field < minfield) | (field > maxfield), np.nan, field)

        # Create a cyclone field object
        cf = md.cyclonefield(days)

        # Identify cyclone centers
        cf.findCenters(field, mask, kSize, nanThresh, d_slp, d_dist, yDist, xDist, lats, lons) # Identify Cyclone Centers

        # Calculate cyclone areas (and MCCs)
        cf.findAreas(field+mask, contint, mcctol, mccdist, lats, lons, kSize) # Calculate Cyclone Areas

        # Append to lists
        cflistnew.append( cf )

    # Track Cyclones
    if t == starttime: # If this is the first time step, must initiate tracking
        if prior == 0: #If this is the first time step and there are no prior months
            ct, cf.cyclones = md.startTracks(cf.cyclones)

        else: #If this is the first time step but there is a prior month
            # Identify date/time of prior timestep
            tp = md.timeAdd(t,[-i for i in timestep])

            # Load cyclone tracks and cyclone field from prior time step
            ct = pd.read_pickle(trkpath+"/ActiveTracks/"+str(tp[0])+"/activetracks"+str(tp[0])+md.dd[tp[1]-1]+".pkl")
            cf1 = pd.read_pickle(detpath+"/CycloneFields/CF"+str(tp[0])+md.dd[tp[1]-1]+".pkl")[-1] # Note, just taking final field from prior month

            md.realignPriorTID(ct,cf1)

            # Start normal tracking
            ct, cf = md.trackCyclones(cf1,cf,ct,maxdist,red,timestep[3])

    else: #If this isn't the first time step, just keep tracking
        ct, cf = md.trackCyclones(cf1,cf,ct,maxdist,red,timestep[3])

    # Increment time step indicator
    t = md.timeAdd(t,timestep)
    cf1 = copy.deepcopy(cf)

    # Save Tracks & Fields (at the end of each month)
    if t[2] == 1 and t[3] == 0: # If the next timestep is the 0th hour of the 1st day of a month,
        print("  Exporting Tracks for " + Y + " " + MM + ' -- Elapsed Time: ' + str(round(time.perf_counter()-start,2)) + ' seconds')
        start = time.perf_counter() # Reset clock
        ct, ct_inactive = md.splitActiveTracks(ct, cf1)

        # Export inactive tracks
        pd.to_pickle(ct_inactive,trkpath+"/CycloneTracks/"+Y+"/cyclonetracks"+Y+M+".pkl")
        pd.to_pickle(ct,trkpath+"/ActiveTracks/"+Y+"/activetracks"+Y+M+".pkl")

        # Export Cyclone Fields (if new)
        try:
            pd.to_pickle(cflistnew,detpath+"/CycloneFields/CF"+Y+M+".pkl")
        except:
            cflist = []

        if t != endtime:
            # Load netcdf for next month
            ncf = nc.Dataset(inpath+"/"+dataset+"_EASE2_N0_"+str(spres)+"km_"+invar+"_Hourly_"+str(t[0])+md.dd[t[1]-1]+".nc")
            tlist = ncf['time'][:].data

            # Load Cyclone Fields (if they exist)
            try:
                cflist = pd.read_pickle(detpath+"/CycloneFields/CF"+str(t[0])+md.dd[t[1]-1]+".pkl")
                cftimes = np.array([cf.time for cf in cflist])
            except:
                cflistnew = []

print("Complete")
