'''
Author: Alex Crawford
Date Created: 20 Jan 2015
Date Modified: 21 Jun 2018 -> Added automatic folder creation for outputs
    4 Jun 2019 -> Updated for Python 3
Purpose: Given a series of sea level pressure fields in netcdf files, this 
    script performs several steps:
    1) Identify closed low pressure centers at each time step
    2) Store information to characterize these centers at each time step
    3) Calculate aggregate statistics about the low pressure systems at each 
        time step
  Steps in the tracking part:
    4) Associate each low pressure center with a corresponding center in the 
        previous time step (when applicable)
    5) Identify low pressure center tracks
    6) Charaterize low pressure center tracks

User Inputs: paths for inputs, desired projection info, limiting parameters
'''

'''********************
Import Modules
********************'''
# Import clock:
import time
# Start script stopwatch. The clock starts running when time is imported
start = time.perf_counter()

print("Loading modules.")
import os
import copy
import pandas as pd
import numpy as np
np.seterr(all='ignore')
from osgeo import gdal, gdalconst, gdalnumeric
import scipy.ndimage.measurements
import CycloneModule_11_1 as md

'''*******************************************
Set up Environment
*******************************************'''
print("Setting up environment.")
path = "/Volumes/Ferdinand/Cyclones/Algorithm Sharing/Version11_1/Test Data"
ra = "ERA"
verd = "11" # Detection Version
vert = "1E" # Tracking Version
inpath = path+"/"+ra+"_SLP_100km" # path+"/"+ra+"/SLP/SLP_EASE100km/Value" # 
#inpath2 = path+"/"+ra+"/Precip"
#inpath3 = path+"/"+ra+"/svp/svpratio_EASE"
suppath = path
outpath = path+"/Results"

'''********************
Define Variables
********************'''
print("Defining variables")
# File Variables
invar = "SLP"
outputtype = gdal.GDT_Float64
ext = ".tif"

demN = "EASE2_N0_100km_etopo1"+ext
latsN = "EASE2_N0_100km_Lats"+ext
longsN = "EASE2_N0_100km_Lons"+ext
xDistN = "EASE2_N0_100km_xDistance"+ext
yDistN = "EASE2_N0_100km_yDistance"+ext
reffile = latsN

# Time Variables 
starttime = [2016,8,1,0,0,0] # Format: [Y,M,D,H,M,S]
endtime = [2016,9,1,0,0,0] # stop BEFORE this time (exclusive)
timestep = [0,0,0,6,0,0] # Time step in [Y,M,D,H,M,S]

dateref = [1900,1,1,0,0,0]  # [Y,M,D,H,M,S]

months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
mons = ["01","02","03","04","05","06","07","08","09","10","11","12"]
days = ["01","02","03","04","05","06","07","08","09","10","11","12","13",\
    "14","15","16","17","18","19","20","21","22","23","24","25","26","27",\
    "28","29","30","31"]
hours = ["0000","0100","0200","0300","0400","0500","0600","0700","0800",\
    "0900","1000","1100","1200","1300","1400","1500","1600","1700","1800",\
    "1900","2000","2100","2200","2300"]

print("Defining parameters")
prior = 0 # 1 = a cyclone track object exists for a prior month; 0 = otherwise

# Detection Parameters
minsurf = 80000 # minimum reasonable value in surface tif
maxsurf = 200000 # maximum reasonable value in surface tif

# Number of places to search on either side of a gridcell 
### to determine if it's a local minimum
kSize = 2

# Maximum fraction of neighboring grid cells with no data (Not a Number) allowed
### for a grid cell to be considered during minimum detection 
nanThresh = 0.4

# minimum slp gradient for identifying weak minima:
d_slp = 750 # slp difference in Pa (use 0 to turn off)
d_dist = 1000000 # distance in m (units that match units of cellsize)

# maximum elevation for masking out high elevation minima
max_elev = 1500. # elevation in m (use 10000 to turn off)

# Contour interval (Pa; determines the interval needed to identify closed 
### contours,and therefore cyclone area)
contint = 200

# Minimum precipitation rate (mm) per timestep (hr)
## Used to determine contiguous precipitation areas
pMin = 1.5*timestep[3]/24 # input is: mm/day * timestep / 24

# Minimum radius for cyclone area (m) (additive with algorithm's cyclone area calculation)
rPrecip = 250000.

# Multi-center cyclone (mcc) tolerance is the maximum ratio permitted between the
### number of unshared and total contours in a multi-centered cyclone. "Unshared"
### contours are only used by the primary center. "Shared" contours are used
### by both the primary and secondary centers.
mcctol = 0.5 # (use 0 to turn off mcc's; higher makes mcc's more likely)
# Multi-center cyclone (mcc) distance is the maximum distance (in m) two minima can
### lie apart and still be considered part of the same cyclone system
mccdist = 1200000

# Tracking Parameters
# Maximum speed is the fastest that a cyclone center is allowed to travel; given
### in units of km/h. To be realistic, the number should be between 100 and 200.
### and probably over 125 (based on Rudeva et al. 2014). To turn off, set to 
### the cellsize times the number of rows or columns (whichever is larger).
maxspeed = 150

# The reduction parameter is a scaling of cyclone speed.  When tracking, the
### algorithm uses the cyclone speed and direction from the last interval to 
### estimate a "best guess" position. This parameter modifies that speed, making
### it slower. This reflects how cyclones tend to slow down as they mature. To
### turn off, set to 1.
red = 0.75

'''*******************************************
Main Analysis
*******************************************'''
print("Main Analysis")

# Ensure that folders exist to store outputs
detpath = outpath+"/detection"+verd
trkpath = outpath+"/tracking"+verd+"_"+vert

for y in range(starttime[0],endtime[0]+1):
    Y = str(y)
    
    # Cyclone Fields
    try:
        os.chdir(detpath+"/CycloneFields/"+Y)
    except:
        os.mkdir(detpath+"/CycloneFields/"+Y)
        for mm in range(12):
            os.mkdir(detpath+"/CycloneFields/"+Y+"/"+months[mm])
    
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

print(" Load Files and References")
# Read in attributes of reference files
ref = gdal.Open(suppath+"/"+latsN)

lats = gdalnumeric.LoadFile(suppath+"/"+latsN)#[1:-1,1:-1]
longs = gdalnumeric.LoadFile(suppath+"/"+longsN)#[1:-1,1:-1]
yDist = gdalnumeric.LoadFile(suppath+"/"+yDistN)
xDist = gdalnumeric.LoadFile(suppath+"/"+xDistN)
elev = gdalnumeric.LoadFile(suppath+"/"+demN)#[1:-1,1:-1]
fieldElevMask = np.where(elev > max_elev,np.nan,0)

# Set up minimum precip
#pMins = [gdalnumeric.LoadFile(inpath3+"/svpratio_"+m+"_EASE.tif") for m in mons]

t = copy.deepcopy(starttime)

# Save Parameters
params = dict({"path":trkpath,"timestep":timestep, "dateref":dateref, "Lats":latsN, "Longs":longsN, \
    "DEM":demN, "minsurf":minsurf, "maxsurf":maxsurf,"kSize":kSize, "nanThresh":nanThresh, "d_slp":d_slp, \
    "d_dist":d_dist, "max_elev":max_elev, "contint":contint, "pMin":pMin/(timestep[3]/24.), \
    "rPrecip":rPrecip, "mcctol":mcctol, "mccdist":mccdist, "maxspeed":maxspeed, "red":red})
pd.to_pickle(params,trkpath+"/cycloneparams.pkl")
del params

print(" Cyclone Detection & Tracking")
# Print elapsed time
print('Elapsed time:',round(time.perf_counter()-start,2),'seconds')

while t != endtime: 
    if t[3] == 0:    
        print(t)
    
    # Extract date
    Y = str(t[0])
    MM = months[t[1]-1]
    M = mons[t[1]-1]
    date = Y+M+days[t[2]-1]+"_"+hours[t[3]]
    
    # Load surface
    try:
        cf = pd.read_pickle(detpath+"/CycloneFields/"+Y+"/"+MM+"/CF"+date+".pkl")
    except:    
        surf = gdalnumeric.LoadFile(inpath+"/"+Y+"/"+MM+"/"+invar+"_"+date+ext)
        surf = np.where((surf < minsurf) | (surf > maxsurf), np.nan, surf)
        
        # Mask elevation if desired
        surfMask = surf+fieldElevMask
        
        # Create a cyclone field object
        cf = md.cyclonefield(md.daysBetweenDates(dateref,t))
        cf.findCenters(surf, surfMask, kSize, nanThresh, d_slp, d_dist, yDist, xDist, lats, longs) # STEP 1: Identify Cyclone Centers
        cf.findAreas(surfMask, contint, mcctol, mccdist, lats, longs, kSize) # STEP 2: Calculate Cyclone Areas
    
        pd.to_pickle(cf,detpath+"/CycloneFields/"+Y+"/"+MM+"/CF"+date+".pkl")

        # STEP 3: Cyclone-Associated Precipitation
#        if t != [1979,1,1,0,0,0]:
#            plsc = gdalnumeric.LoadFile(inpath2+"/PrecipLgScl_EASE/Value/"+Y+"/"+MM+"/PrecipLgScl_"+date+ext)[1:-1,1:-1]
#            ptot = gdalnumeric.LoadFile(inpath2+"/PrecipTotal_EASE/Value/"+Y+"/"+MM+"/PrecipTotal_"+date+ext)[1:-1,1:-1]
#            plsc = np.hstack(( np.zeros( (plsc.shape[0]+2,1) ) , np.vstack(( np.zeros( (1,plsc.shape[1]) ), plsc ,np.zeros( (1,plsc.shape[1]) ) )), np.zeros( (plsc.shape[0]+2,1) ) ))
#            ptot = np.hstack(( np.zeros( (ptot.shape[0]+2,1) ) , np.vstack(( np.zeros( (1,ptot.shape[1]) ), ptot ,np.zeros( (1,ptot.shape[1]) ) )), np.zeros( (ptot.shape[0]+2,1) ) ))
#            cap = md.findCAP2(cf,plsc,ptot,yDist,xDist,lats,longs,pMins[t[1]-1],rPrecip)
#            md.writeNumpy_gdalObj(cap,detpath+"/PrecipCAP_EASE/Value/"+Y+"/"+MM+"/PrecipCAP_"+date+ext,ref,outputtype)
#            #cf.findCAP(plsc,ptot,yDist,xDist,lats,longs,pMin,rPrecip)
    
    # STEP 4. Track Cyclones
    if t == starttime and prior == 0: #If this is the first time step and there are no prior months
        ct, cf.cyclones = md.startTracks(cf.cyclones)
    
    elif t == starttime and prior == 1: #If this is the first time step but there is a prior month
        # Identify date/time of prior timestep
        tp = md.timeAdd(t,[-i for i in timestep])
        datep = str(tp[0]) + str(mons[tp[1]-1]) + days[tp[2]-1]+"_"+hours[tp[3]]
        cffilep = "CF"+datep+".pkl"
        
        # Load cyclone tracks and cyclone field from prior time step
        ct = pd.read_pickle(trkpath+"/ActiveTracks/"+str(tp[0])+"/activetracks"+str(tp[0])+str(mons[tp[1]-1])+".pkl")
        cf1 = pd.read_pickle(trkpath+"/CycloneFields/"+str(tp[0])+"/"+str(months[tp[1]-1])+"/"+cffilep)
        md.realignPriorTID(ct,cf1)
        
        # move into normal tracking
        ct, cf = md.trackCyclones(cf1,cf,ct,maxspeed,red)
    
    else: #If this isn't the first time step, track
        ct, cf = md.trackCyclones(cf1,cf,ct,maxspeed,red)
     

    # Increment time step indicator
    t = md.timeAdd(t,timestep)
    cf1 = copy.deepcopy(cf)
    
    # Save Tracks (at the end of each month)
    if t[2] == 1 and t[3] == starttime[3]: # If the next timestep is the 0th hour of the 1st day of a month,
        print("  Exporting Tracks for " + Y + " " + MM)
        ct, ct_inactive = md.splitActiveTracks(ct, cf1)
        
        # Export inactive tracks
        pd.to_pickle(ct_inactive,trkpath+"/CycloneTracks/"+Y+"/cyclonetracks"+Y+M+".pkl")
        pd.to_pickle(ct,trkpath+"/ActiveTracks/"+Y+"/activetracks"+Y+M+".pkl")
        
        # Print elapsed time
        print('Elapsed time:',round(time.perf_counter()-start,2),'seconds')