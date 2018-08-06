'''
Author: Alex Crawford
Date Created: 20 Jan 2015
Date Modified: 12 Dec 2015
Purpose: Given a series of sea level pressure fields, this 
    script performs several steps:
    1) Identify closed low pressure centers at each time step
    2) Store information to characterize these centers at each time step
    3) Identify cyclone area and multi-center cyclones
    4) Calculate aggregate statistics about the low pressure systems at each 
        time step
  Steps in the tracking part:
    5) Associate each low pressure center with a corresponding center in the 
        previous time step (when applicable)
    6) Identify cyclone center tracks
    7) Charaterize cyclone center tracks
    8) Identify relationships amongst cyclone center tracks
'''

'''********************
Import Modules
********************'''
# Import clock:
from time import clock
# Start script stopwatch. The clock starts running when time is imported
start = clock()

print "Loading modules."
import os
import copy
import pandas as pd
import cPickle
import numpy as np
from osgeo import gdal, gdalconst, gdalnumeric
import scipy.ndimage.measurements
import CycloneModule_10_3 as md

'''*******************************************
Set up Environment
*******************************************'''
print "Setting up environment."
ra = "ERA"
path = "/VOLUMES/Horatio"
inpath = path+"/"+ra+"/SLP/SLP_EASE100000/Value" # path+"/ArcticCyclone/SLP_EASE100000/Value" #
#inpath2 = path+"/"+ra+"/Precip"
outpath = path+"/ArcticCyclone/detection10_10E"
suppath = path+"/"+ra+"/"+ra+"_Support"

'''********************
Define Variables
********************'''
print "Defining variables"
# File Variables
invar = "SLP"
outputtype = gdal.GDT_Float64

demN = "EASE2_N0_100km_etopo1.tif"
latsN = "EASE2_N0_100km_Lats.tif"
longsN = "EASE2_N0_100km_Lons.tif"

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

print "Defining parameters"
prior = 0 # 1 = a cyclone track object exists for a prior month; 0 = otherwise

# Detection Parameters
minsurf = 80000 # minimum reasonable value in surface tif
maxsurf = 200000 # maximum reasonable value in surface tif

# Number of places to search on either side of a gridcell 
### to determine if it's a local minimum
kSize = 1

# Maximum fraction of neighboring grid cells with no data (Not a Number) allowed
### for a grid cell to be considered during minimum detection 
nanThresh = 0.25 

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
mccdist = 1000000

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
print "Main Analysis"

print "Step 1. Load Files and References"
# Read in attributes of reference files
ref = gdal.Open(suppath+"/"+latsN)
refG = ref.GetGeoTransform()
cellsize = refG[1]

lats = gdalnumeric.LoadFile(suppath+"/"+latsN)
longs = gdalnumeric.LoadFile(suppath+"/"+longsN)
elev = gdalnumeric.LoadFile(suppath+"/"+demN)
elevMask = np.where(elev > max_elev,np.nan,0)

t = starttime
tcount = 0

# Save Parameters
params = dict({"path":outpath,"timestep":timestep, "dateref":dateref, "Lats":latsN, "Longs":longsN, \
    "DEM":demN, "minsurf":minsurf, "maxsurf":maxsurf,"kSize":kSize, "nanThresh":nanThresh, "d_slp":d_slp, \
    "d_dist":d_dist, "max_elev":max_elev, "contint":contint, "pMin":pMin/(timestep[3]/24.), \
    "rPrecip":rPrecip, "mcctol":mcctol, "mccdist":mccdist, "maxspeed":maxspeed, "red":red})
md.pickle(params,outpath+"/cycloneparams.pkl")
del params

print "Step 2. Cyclone Detection & Tracking"
# Print elapsed time
print 'Elapsed time:',round(clock()-start,2),'seconds'

while t != endtime: 
    if t[3] == 0:    
        print t
    
    # Extract date
    Y = str(t[0])
    MM = months[t[1]-1]
    M = mons[t[1]-1]
    date = Y+M+days[t[2]-1]+"_"+hours[t[3]]
    
    # Load surface
    surf = gdalnumeric.LoadFile(inpath+"/"+Y+"/"+MM+"/"+invar+"_"+date+".tif")
    surf = np.where((surf < minsurf) | (surf > maxsurf), np.nan, surf)
    
    # Mask elevation if desired
    surfMask = surf+elevMask
    
    # Create a cyclone field object
    cf = md.cyclonefield(md.daysBetweenDates(dateref,t))
    cf.findCenters(surf, surfMask, kSize, d_slp, d_dist, cellsize, lats, longs, nanThresh) # STEP 2.1: Identify Cyclone Centers
    cf.findAreas2(surfMask, contint, mcctol, mccdist, cellsize, kSize) # STEP 2.2: Calculate Cyclone Areas
    
    # STEP 2.3: Cyclone-Associated Precipitation
#    if t != [1979,1,1,0,0,0]:
#        plsc = gdalnumeric.LoadFile(inpath2+"/PrecipLgScl_EASE/Value/"+Y+"/"+MM+"/PrecipLgScl_"+date+".tif")
#        ptot = gdalnumeric.LoadFile(inpath2+"/PrecipTotal_EASE/Value/"+Y+"/"+MM+"/PrecipTotal_"+date+".tif")
#        plsc = np.hstack(( np.zeros( (plsc.shape[0]+2,1) ) , np.vstack(( np.zeros( (1,plsc.shape[1]) ), plsc ,np.zeros( (1,plsc.shape[1]) ) )), np.zeros( (plsc.shape[0]+2,1) ) ))
#        ptot = np.hstack(( np.zeros( (ptot.shape[0]+2,1) ) , np.vstack(( np.zeros( (1,ptot.shape[1]) ), ptot ,np.zeros( (1,ptot.shape[1]) ) )), np.zeros( (ptot.shape[0]+2,1) ) ))
#        cap = md.findCAP(cf,plsc,ptot,pMin,rPrecip,cellsize)
#        md.writeNumpy_gdalObj(cap,outpath+"/PrecipCAP_EASE/Value/"+Y+"/"+MM+"/PrecipCAP_"+date+".tif",ref,outputtype)
#        #cf.findCAP(plsc,ptot,pMin,rPrecip,cellsize)
#        #md.writeNumpy_gdalObj(cf.CAP,outpath+"/Member"+E+"/PrecipCAP_EASE/Value/"+Y+"/"+MM+"/PrecipCAP_"+date+".tif",ref,outputtype)
    
    # Track Cyclones
    if t == starttime and prior == 0: #If this is the first time step and there are no prior months
        ct, cf.cyclones = md.startTracks(cf.cyclones)
    
    elif t == starttime and prior == 1: #If this is the first time step but there is a prior month
        # Identify date/time of prior timestep
        tp = md.timeAdd(t,[-i for i in timestep])
        datep = str(tp[0]) + str(mons[tp[1]-1]) + days[tp[2]-1]+"_"+hours[tp[3]]
        cffilep = "CF"+datep+".pkl"
        
        # Load cyclone tracks and cyclone field from prior time step
        ct = md.unpickle(outpath+"/ActiveTracks/"+str(tp[0])+"/activetracks"+str(tp[0])+str(mons[tp[1]-1])+".pkl")
        cf1 = md.unpickle(outpath+"/CycloneFields/"+str(tp[0])+"/"+str(months[tp[1]-1])+"/"+cffilep)
        md.realignPriorTID(ct,cf1)
        
        # move into normal tracking
        ct, cf = md.trackCyclones(cf1,cf,ct,cellsize,maxspeed,red)
    
    else: #If this isn't the first time step, track
        ct, cf = md.trackCyclones(cf1,cf,ct,cellsize,maxspeed,red)
     
    # Save Cyclone Objects & Fields
    md.pickle(cf,outpath+"/CycloneFields/"+Y+"/"+MM+"/CF"+date+".pkl")
    #md.writeNumpy_gdalObj(cf.fieldCenters2,outpath+"/Centers2Fields/"+Y+"/"+MM+"/sysMin2_"+date+".tif",ref,outputtype)
    #md.writeNumpy_gdalObj(cf.fieldAreas,outpath+"/AreaFields/"+Y+"/"+MM+"/cycField_"+date+".tif",ref,outputtype)
    
    # Increment time step indicator
    t = md.timeAdd(t,timestep)
    tcount = tcount + 1
    cf1 = copy.deepcopy(cf)
    
    # Save Tracks (at the end of each month)
    if t[2] == 1 and t[3] == 0: # If the next timestep is the 0th hour of the 1st day of a month,
        print "  Exporting Tracks for " + Y + " " + MM
        ct, ct_inactive = md.splitActiveTracks(ct, cf1)
        
        # Export inactive tracks
        md.pickle(ct_inactive,outpath+"/CycloneTracks/"+Y+"/cyclonetracks"+Y+M+".pkl")
        md.pickle(ct,outpath+"/ActiveTracks/"+Y+"/activetracks"+Y+M+".pkl")
        
        # Print elapsed time
        print 'Elapsed time:',round(clock()-start,2),'seconds'