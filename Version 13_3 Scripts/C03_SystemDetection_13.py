'''
Author: Alex Crawford
Date Created: 11 Jan 2016
Date Modified: 8 Dec 2017, 4 Jun 2019 (Python 3), 13 Jun 2019 (warning added)
Purpose: Convert a series of center tracks to system tracks. Warning: If a) you
wish to re-run this process on some of the data and b) you are using rg = 1
(allowing regeneration), you need to re-run from the reftime or accept that
some active storms at the re-start point will get truncated.

User inputs:
    Path Variables
    Bounding Box ID (subsetnum): 2-digit character string
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
import CycloneModule_13_3 as md

'''*******************************************
Set up Environment
*******************************************'''
print("Setting up environment.")
subset = "" # use "" if performing on all cyclones

inpath = "/Volumes/Cressida/CycloneTracking/tracking13testP/"
outpath = inpath+"/"+subset

'''*******************************************
Define Variables
*******************************************'''
print("Defining variables")
# Regenesis Paramater
rg = 1
# 0 = regenesis starts a new system track;
# 1 = regenesis continues previous system track with new ptid

# Time Variables
lyb, dpy = 1, 365 # Whether there are leap years; how many days in a a non-leap year

starttime = [1979,1,1,0,0,0] # Format: [Y,M,D,H,M,S]
endtime = [1979,2,1,0,0,0] # stop BEFORE this time (exclusive)
reftime = [1950,1,1,0,0,0]
monthstep = [0,1,0,0,0,0] # A Time step that increases by 1 month [Y,M,D,H,M,S]

dateref = [1900,1,1,0,0,0] # [Y,M,D,H,M,S]

'''*******************************************
Main Analysis
*******************************************'''
print("Main Analysis")

mt = starttime
while mt != endtime:
    # Extract date
    Y = str(mt[0])
    M = md.dd[mt[1]-1]
    print (" " + Y + " - " + M)

    # Load Cyclone Tracks
    ct = pd.read_pickle(inpath+"/"+subset+"/CycloneTracks/"+Y+"/"+subset+"cyclonetracks"+Y+M+".pkl")

    # Create System Tracks
    if mt == reftime:
        cs, cs0 = md.cTrack2sTrack(ct, [], dateref, rg, lyb, dpy)
        pd.to_pickle(cs,outpath+"/SystemTracks/"+Y+"/"+subset+"systemtracks"+Y+M+".pkl")

    else:
        # Extract date for previous month
        mt0 = md.timeAdd(mt,[-d for d in monthstep])
        Y0 = str(mt0[0])
        M0 = md.dd[mt0[1]-1]

        # Load previous month's system tracks
        cs0 = pd.read_pickle(outpath+"/SystemTracks/"+Y0+"/"+subset+"systemtracks"+Y0+M0+".pkl")

        # Create system tracks
        cs, cs0 = md.cTrack2sTrack(ct, cs0, dateref, rg, lyb, dpy)
        pd.to_pickle(cs,outpath+"/SystemTracks/"+Y+"/"+subset+"systemtracks"+Y+M+".pkl")
        pd.to_pickle(cs0,outpath+"/SystemTracks/"+Y0+"/"+subset+"systemtracks"+Y0+M0+".pkl")

    # Increment Time Step
    mt = md.timeAdd(mt,monthstep)

print('Elapsed time:',round(perf_counter()-start,2),'seconds')
