'''
Author: Alex Crawford
Date Created: 11 Mar 2015
Date Modified: 21 Jun 2018 -> Switch to iloc, loc method of subsetting
12 Jun 2019 --> Update for Python 3
Purpose: Plot histograms of several cyclone characteristics. There are a BUNCH of
options here for subsetting and type -- system or cyclone, months of interest, 
spatial area of interest -- make sure all of the "kind" variables and the "minlat"
are satisfactory. You may also want to tweak the version and the inpath.
'''

'''********************
Import Modules
********************'''

#print "Loading modules."
import pandas as pd
import numpy as np
import CycloneModule_10_3 as md
from osgeo import gdal, gdalnumeric

'''*******************************************
Set up Environment
*******************************************'''
BBoxNum = "BBox16" # Use "" for no subset, use "BBOX##" otherwise, replacing ## with the appropriate value
path = "/Volumes/Ferdinand/Cyclones/Algorithm Sharing/Version11_1/Test Data"# 
version = "11_1E"
inpath = path+"/Results/tracking"+version+"/"+BBoxNum
suppath = path

'''*******************************************
Define Variables
*******************************************'''
# File Variables
ext = ".tif"
reffileG = "GenesisRegions"+ext
values = [0,1] # [1] for the DEM mask; [0,1] for Central/Pacific-Side Arctic Ocean in the the Genesis Regions mask

kind1 = "Cyclone" # Systems, Cyclones
kind = kind1+"Tracks" # Can be "Arctic" or other region (or no region), followed by System or Cyclone

rg = 1 # Whether regenesis of a cyclone counts as a track split (0) or track continuation (1)

# Aggregation Parameters
minls = 1 # minimum lifespan (in  days) for a track to be considered
mintl = 1 # minimum track length (in km for version 11.1, in grid cells for version 10)

# Time Variables 
starttime = [2016,8,1,0,0,0] # Format: [Y,M,D,H,M,S]
endtime = [2016,9,1,0,0,0] # stop BEFORE this time (exclusive)
monthstep = [0,1,0,0,0,0] # A Time step that increases by 1 mont [Y,M,D,H,M,S]

dateref = [1900,1,1,0,0,0] # [Y,M,D,H,M,S]

months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
mons = ["01","02","03","04","05","06","07","08","09","10","11","12"]

'''*******************************************
Main Analysis
*******************************************'''
print("Main Analysis")

# Read in reference file
refG = gdalnumeric.LoadFile(suppath+"/"+reffileG)

# Prep empty pdf
pdf = pd.DataFrame(columns=["tid","year","month","spl1","spl2","spl3","mrg1","mrg2","mrg3","rge2",\
"maxuv","maxdpdt","maxdsqp","minp","lifespan","trlen","avgarea","mcc","genRegion","lysRegion","bboxPer"])

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
    trs = [c for c in cs if ((c.lifespan() >= minls) and (c.trackLength() >= mintl))]
    trs0 = [c for c in cs0 if ((c.lifespan() >= minls) and (c.trackLength() >= mintl))]
    trs2 = [c for c in cs2 if ((c.lifespan() >= minls) and (c.trackLength() >= mintl))]
    
    ### EVENT FIELDS ###
    # Limit events to only those tracks that satisfy above criteria
    tids = [tr.tid for tr in trs]
    ftids = [tr.ftid for tr in trs]
    tids0 = [tr.tid for tr in trs0]
    ftids2 = [tr.ftid for tr in trs2]
    
    for tr in trs: # For each track
        # Identify genesis region & lysis region
        genRegion = refG[int(tr.data.y.iloc[0]),int(tr.data.x.iloc[0])]
        lysRegion = refG[int(list(tr.data.y)[-1]),int(list(tr.data.x)[-1])]
        
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
        
        # Calculate % of time that cyclone satisfies the bbox
        xs = list(tr.data.x)
        ys = list(tr.data.y)
        mask = np.in1d(refG,values).reshape(refG.shape)
        
        # Prep for loop
        total = 0
        for i in range(len(xs)):
            total = total + mask[int(ys[i]),int(xs[i])]
        
        row = pd.DataFrame([{"tid":tr.tid,"year":mt[0],"month":mt[1],"maxuv":tr.maxUV()[0],\
        "maxdpdt":tr.maxDpDt()[0],"maxdsqp":tr.maxDsqP()[0],"minp":tr.minP()[0],\
        "lifespan":tr.lifespan(),"trlen":tr.trackLength(),"avgarea":tr.avgArea(),\
        "mcc":tr.mcc(),"spl1":spl1_count,"spl2":spl2_count,"spl3":spl3_count,\
        "mrg1":mrg1_count,"mrg2":mrg2_count,"mrg3":mrg3_count,"rge2":rge2_count,\
        "genRegion":genRegion,"lysRegion":lysRegion,"bboxPer":float(total)/len(xs)},])
        pdf = pdf.append(row,ignore_index=True,sort=False)
    
    mt = md.timeAdd(mt,monthstep)

YYMM = str(starttime[0])+mons[starttime[1]-1] + "_" + str(md.timeAdd(endtime,[0,-1,0,0,0,0])[0])+mons[endtime[1]-1]
pdf.to_csv(inpath+"/Aggregation"+kind1+"/"+BBoxNum+"_"+kind+"Events_"+version+"_"+YYMM+".csv",index=False)
