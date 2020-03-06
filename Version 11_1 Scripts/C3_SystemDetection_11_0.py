'''
Author: Alex Crawford
Date Created: 11 Jan 2016
Date Modified: 8 Dec 2017, 4 Jun 2019 (Python 3), 13 Jun 2019 (warning added)
Purpose: Convert a series of center tracks to system tracks. Warning: If a) you
wish to re-run this process on some of the data and b) you are using rg = 1
(allowing regenertion), you need to re-run FROM THE REFTIME. Otherwise, it 
will break.

User inputs:
    Path Variables
    Bounding Box ID (bboxnum): 2-digit character string
    Time Variables: when to start, end, the time step of the data
    Regenesis Paramter: 0 or 1, depending on whether regenesis continues tracks
'''

'''********************
Import Modules
********************'''
# Import clock:
from time import perf_counter
# Start script stopwatch. The clock starts running when time is imported
start = perf_counter()

print("Loading modules.")
import pandas as pd
import numpy as np
import CycloneModule_11_1 as md

'''*******************************************
Set up Environment
*******************************************'''
print("Setting up environment.")
BBox = "" # use "" if performing on all cyclones

path = "/Volumes/Ferdinand/Cyclones/Algorithm Sharing/Version11_1/Test Data/Results"
inpath = path+"/tracking11_1E/"+BBox
outpath = inpath

'''*******************************************
Define Variables
*******************************************'''
print("Defining variables")
# Regenesis Paramater
rg = 1
# 0 = regenesis starts a new system track; 
# 1 = regenesis continues previous system track with new ptid

# Time Variables
starttime = [2016,8,1,0,0,0] # Format: [Y,M,D,H,M,S]
endtime = [2016,9,1,0,0,0] # stop BEFORE this time (exclusive)
reftime = [2016,8,1,0,0,0] # start of first month in a multi-month sequence
monthstep = [0,1,0,0,0,0] # A Time step that increases by 1 mont [Y,M,D,H,M,S]

dateref = [1900,1,1,0,0,0] # [Y,M,D,H,M,S]

months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
mons = ["01","02","03","04","05","06","07","08","09","10","11","12"]

'''*******************************************
Main Analysis
*******************************************'''
print("Main Analysis")

mt = starttime
while mt != endtime:
    # Extract date
    Y = str(mt[0])
    M = mons[mt[1]-1]
    print (" " + Y + " - " + M)
    
    # Load Cyclone Tracks
    ct = pd.read_pickle(inpath+"/CycloneTracks/"+Y+"/"+BBox+"cyclonetracks"+Y+M+".pkl")
    
    # Create System Tracks
    if mt == reftime:
        cs, cs0 = md.cTrack2sTrack(ct,[],dateref,rg)
        pd.to_pickle(cs,inpath+"/SystemTracks/"+Y+"/"+BBox+"systemtracks"+Y+M+".pkl")
    
    else:
        # Extract date for previous month
        mt0 = md.timeAdd(mt,[-d for d in monthstep])
        Y0 = str(mt0[0])
        M0 = mons[mt0[1]-1]
        
        # Load previous month's system tracks
        cs0 = pd.read_pickle(inpath+"/SystemTracks/"+Y0+"/"+BBox+"systemtracks"+Y0+M0+".pkl")
        
        # Create system tracks
        cs, cs0 = md.cTrack2sTrack(ct,cs0,dateref,rg)
        pd.to_pickle(cs,inpath+"/SystemTracks/"+Y+"/"+BBox+"systemtracks"+Y+M+".pkl")
        pd.to_pickle(cs0,inpath+"/SystemTracks/"+Y0+"/"+BBox+"systemtracks"+Y0+M0+".pkl")
    
    # Increment Time Step
    mt = md.timeAdd(mt,monthstep)

print('Elapsed time:',round(perf_counter()-start,2),'seconds')