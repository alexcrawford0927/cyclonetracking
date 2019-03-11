'''
Author: Alex Crawford
Date Created: 11 Jan 2016
Date Modified: 11 Jan 2016
Purpose: Convert a series of center tracks to system tracks.

User inputs:
    Path Variables, including the reanalysis (ERA, MERRA, CFSR)
    Bounding Box ID (bboxnum): 2-digit character string
    Time Variables: when to start, end, the time step of the data
    Regenesis Paramter: 0 or 1, depending on whether regenesis continues tracks
'''

'''********************
Import Modules
********************'''
# Import clock:
from time import clock
# Start script stopwatch. The clock starts running when time is imported
start = clock()

print "Loading modules."
import pandas as pd
import cPickle
import numpy as np
import CycloneModule_10_3 as md

'''*******************************************
Set up Environment
*******************************************'''
print "Setting up environment."
BBox = "" # use "" if performing on all cyclones

path = "$INSERT PATH HERE$/Version10_3/Test Data"
inpath = path+"/Results"
outpath = inpath

'''*******************************************
Define Variables
*******************************************'''
print "Defining variables"
# Regenesis Paramater
rg = 1 
# 0 = regenesis starts a new system track; 
# 1 = regenesis continues preivous system track with new ptid

# Time Variables
time1 = [2016,8,1,0,0,0] # Format: [Y,M,D,H,M,S]
time2 = [2016,9,1,0,0,0] # stop BEFORE this time (exclusive)
monthstep = [0,1,0,0,0,0] # A Time step that increases by 1 mont [Y,M,D,H,M,S]

dateref = [1900,1,1,0,0,0] # [Y,M,D,H,M,S]
starttime = [2016,8,1,0,0,0] # [Y,M,D,H,M,S]... starttime for the cyclone detection script

months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
mons = ["01","02","03","04","05","06","07","08","09","10","11","12"]

'''*******************************************
Main Analysis
*******************************************'''
print "Main Analysis"

mt = time1
while mt != time2:
    # Extract date
    Y = str(mt[0])
    M = mons[mt[1]-1]
    print " " + Y + " - " + M
    
    # Load Cyclone Tracks
    ct = md.unpickle(inpath+"/CycloneTracks/"+Y+"/"+BBox+"cyclonetracks"+Y+M+".pkl")
    
    # Create System Tracks
    if mt == starttime:
        cs, cs0 = md.cTrack2sTrack(ct,[],dateref,rg)
        md.pickle(cs,inpath+"/SystemTracks/"+Y+"/"+BBox+"systemtracks"+Y+M+".pkl")
    
    else:
        # Extract date for previous month
        mt0 = md.timeAdd(mt,[-d for d in monthstep])
        Y0 = str(mt0[0])
        M0 = mons[mt0[1]-1]
        
        # Load previous month's system tracks
        cs0 = md.unpickle(inpath+"/SystemTracks/"+Y0+"/"+BBox+"systemtracks"+Y0+M0+".pkl")
        
        # Create system tracks
        cs, cs0 = md.cTrack2sTrack(ct,cs0,dateref,rg)
        md.pickle(cs,inpath+"/SystemTracks/"+Y+"/"+BBox+"systemtracks"+Y+M+".pkl")
        md.pickle(cs0,inpath+"/SystemTracks/"+Y0+"/"+BBox+"systemtracks"+Y0+M0+".pkl")
    
    # Increment Time Step
    mt = md.timeAdd(mt,monthstep)

print 'Elapsed time:',round(clock()-start,2),'seconds'
print "Complete."