'''
Author: Alex Crawford
Date Created: 11 Mar 2015
Date Modified: 21 Jun 2018 -> Switch to iloc, loc method of subsetting
12 Jun 2019 --> Update for Python 3
19 May 2020 --> Added automatic creation of output directory
02 Jul 2020 --> Removed reliance on GDAL; using regions stored in netcdf file
Purpose: Records information that summarizes each track (e.g., length, lifespan, 
region of origin, number of merges and splits).
'''

'''********************
Import Modules
********************'''

#print "Loading modules."
import pandas as pd
import os
import numpy as np
import CycloneModule_12_2 as md

'''*******************************************
Set up Environment
*******************************************'''
BBoxNum = "BBox15" # Use "BBox##" or "" if no subset
path = "/Volumes/Cressida"
version = "12_4E5R"
inpath = path+"/CycloneTracking/tracking"+version+"/"+BBoxNum

'''*******************************************
Define Variables
*******************************************'''
# File Variables
ext = ".tif"
kind1 = "System" # System, Cyclone
kind = kind1+"Tracks" # Can be AFZ, Arctic, or other region (or no region), followed by System or Cyclone

rg = 1 # Whether regenesis of a cyclone counts as a track split (0) or track continuation (1)

V = "_N60" # An optional version name; I suggest you start with "_" or "-" to separate from years in file title

# Aggregation Parameters
minls = 0 # minimum lifespan (in  days) for a track to be considered
mintl = 0 # minimum track length (in km)
minlat = 60 # minimum latitude

# Time Variables 
starttime = [1979,1,1,0,0,0] # Format: [Y,M,D,H,M,S]
endtime = [2020,1,1,0,0,0] # stop BEFORE this time (exclusive)
monthstep = [0,1,0,0,0,0] # A Time step that increases by 1 mont [Y,M,D,H,M,S]

dateref = [1900,1,1,0,0,0] # [Y,M,D,H,M,S]

months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
mons = ["01","02","03","04","05","06","07","08","09","10","11","12"]

'''*******************************************
Main Analysis
*******************************************'''
print("Main Analysis")
# Prep empty pdf
pdf = pd.DataFrame()

mt = starttime
while mt != endtime:
    # Extract date
    Y = str(mt[0])
    MM = months[mt[1]-1]
    M = mons[mt[1]-1]
    print ("  " + Y + " - " + MM )
    
    mtdays = md.daysBetweenDates(dateref,mt) # Convert date to days since [1900,1,1,0,0,0]
    mt0 = md.timeAdd(mt,[-i for i in monthstep]) # Identify time for the previous month
    mt2 = md.timeAdd(mt,monthstep) # Identify time for next month
    
    # Load tracks
    cs = pd.read_pickle(inpath+"/"+kind+"/"+Y+"/"+BBoxNum+kind.lower()+Y+M+".pkl") # Current Month
    try:
        cs0 = pd.read_pickle(inpath+"/"+kind+"/"+str(mt0[0])+"/"+BBoxNum+kind.lower()+str(mt0[0])+mons[mt0[1]-1]+".pkl")
    except:
        cs0 = []
    try: # Next Month
        cs2 = pd.read_pickle(inpath+"/"+kind+"/"+str(mt2[0])+"/"+BBoxNum+kind.lower()+str(mt2[0])+mons[mt2[1]-1]+".pkl")
        cs2 = [c for c in cs2 if np.isnan(c.ftid) == 0] # Only care about tracks that existed in current month
    except: # For final month in series, forced to used active tracks for partial tabulation of events
        try:
            cs2 = pd.read_pickle(inpath+"/ActiveTracks/"+str(mt[0])+"/"+BBoxNum+"activetracks"+str(mt[0])+mons[mt[1]-1]+".pkl")
            cs2, cs = md.cTrack2sTrack(cs2,cs,dateref,rg)
        except:
            cs2 = []
    
    # Limit to tracks that satisfy minimum lifespan and track length
    trs = [c for c in cs if ((c.lifespan() >= minls) and (c.trackLength() >= mintl) and (np.max(c.data.lat) >= minlat))]
    trs0 = [c for c in cs0 if ((c.lifespan() >= minls) and (c.trackLength() >= mintl) and (np.max(c.data.lat) >= minlat))]
    trs2 = [c for c in cs2 if ((c.lifespan() >= minls) and (c.trackLength() >= mintl) and (np.max(c.data.lat) >= minlat))]
    
    ### EVENT FIELDS ###
    # Limit events to only those tracks that satisfy above criteria
    tids = [tr.tid for tr in trs]
    ftids = [tr.ftid for tr in trs]
    tids0 = [tr.tid for tr in trs0]
    ftids2 = [tr.ftid for tr in trs2]
    
    for tr in trs: # For each track
        # Establish event counters
        mrg1_count, mrg2_count, mrg3_count = 0, 0, 0
        spl1_count, spl2_count, spl3_count = 0, 0, 0
        rge2_count = 0
        for e in range(len(tr.events)): # Check the stats for each event    
            # Only record the event if the interacting track also satisfies the lifespan/track length criteria
            # If the event time occurs during the month of interest...
            # Check if the otid track exists in either this month or the next month:
            if tr.events.time.iloc[e] >= mtdays and ( (tr.events.otid.iloc[e] in tids) or (tr.events.otid.iloc[e] in ftids2) ):
                # And if so, record the event type
                if tr.events.event.iloc[e] == "mg":
                    if tr.events.Etype.iloc[e] == 1:
                        mrg1_count =  mrg1_count + 1
                    elif tr.events.Etype.iloc[e] == 2:
                        mrg2_count = mrg2_count + 1
                    elif tr.events.Etype.iloc[e] == 3:
                        mrg3_count = mrg3_count + 1
                elif tr.events.event.iloc[e] == "sp":
                    if tr.events.Etype.iloc[e] == 1:
                        spl1_count =  spl1_count + 1
                    elif tr.events.Etype.iloc[e] == 2:
                        spl2_count =  spl2_count + 2
                    elif tr.events.Etype.iloc[e] == 3:
                        spl3_count =  spl3_count + 3
            
            # If the event time occurs during the previous month...
            # Check if the otid track exists in either this month or the previous month:
            elif tr.events.time.iloc[e] < mtdays and ( (tr.events.otid.iloc[e] in tids0) or (tr.events.otid.iloc[e] in ftids) ):
                # And if so, record the event type
                if tr.events.event.iloc[e] == "mg":
                    if tr.events.Etype.iloc[e] == 1:
                        mrg1_count =  mrg1_count + 1
                    elif tr.events.Etype.iloc[e] == 2:
                        mrg2_count = mrg2_count + 1
                    elif tr.events.Etype.iloc[e] == 3:
                        mrg3_count = mrg3_count + 1
                elif tr.events.event.iloc[e] == "sp":
                    if tr.events.Etype.iloc[e] == 1:
                        spl1_count =  spl1_count + 1
                    elif tr.events.Etype.iloc[e] == 2:
                        spl2_count =  spl2_count + 2
                    elif tr.events.Etype.iloc[e] == 3:
                        spl3_count =  spl3_count + 3
        
        row = pd.DataFrame([{"tid":tr.tid,"year":mt[0],"month":mt[1],"avguv":tr.data.uv.mean(),\
        "maxdsqp":tr.data.DsqP.max(),"minp":tr.data.p_cent.min(),"maxdepth":tr.data.depth.max(),\
        "avgdsqp":tr.data.DsqP.mean(),"avgp":tr.data.p_cent.mean(),"avgdepth":tr.data.depth.mean(),\
        "lifespan":tr.lifespan(),"trlen":tr.trackLength(),"avgarea":tr.avgArea(),\
        "mcc":tr.mcc(),"spl1":spl1_count,"spl2":spl2_count,"spl3":spl3_count,\
        "mrg1":mrg1_count,"mrg2":mrg2_count,"mrg3":mrg3_count,"rge2":rge2_count},])
        pdf = pdf.append(row,ignore_index=True,sort=False)
    
    mt = md.timeAdd(mt,monthstep)

# Append columns for overall merge, split, non-interacting
pdf["mrg"] = np.array((pdf["mrg2"] >= 1) | (pdf["mrg3"] >= 1)).astype(int)
pdf["spl"] = np.array((pdf["spl2"] >= 1) | (pdf["spl3"] >= 1)).astype(int)
pdf["nonint"] = np.array((pdf["mrg"] == 0) & (pdf["spl"] == 0)).astype(int)

# Write to File
try:
    os.chdir(inpath+"/Aggregation"+kind1)
except:
    os.mkdir(inpath+"/Aggregation"+kind1)
    os.chdir(inpath+"/Aggregation"+kind1)
    
YY = str(starttime[0]) + "_" + str(md.timeAdd(endtime,[0,-1,0,0,0,0])[0])
pdf.to_csv(BBoxNum+"_"+kind+"Events_"+version+"_"+YY+V+".csv",index=False)
