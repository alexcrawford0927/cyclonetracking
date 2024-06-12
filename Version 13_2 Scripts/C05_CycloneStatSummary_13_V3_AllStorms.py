'''
Author: Alex Crawford
Date Created: 11 Mar 2015
Date Modified: 21 Jun 2018 -> Switch to iloc, loc method of subsetting
12 Jun 2019 --> Update for Python 3
19 May 2020 --> Added automatic creation of output directory
02 Jul 2020 --> Removed reliance on GDAL; using regions stored in netcdf file
21 Jan 2021 --> Added dispalcement as subset option; added genesis/lysis region check
06 Oct 2021 --> Replaced pickled regions file with a netcdf
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
import netCDF4 as nc
import CycloneModule_13_2 as md

'''*******************************************
Set up Environment
*******************************************'''
BBoxNum = "BBox27" # Use "BBox##" or "" if no subset
path = "/Volumes/Cressida"
version = "13_2R"
inpath = path+"/CycloneTracking/tracking"+version+"/"+BBoxNum
regpath = path+"/Projections/EASE2_N0_25km_GenesisRegions.nc"

'''*******************************************
Define Variables
*******************************************'''
# File Variables
ext = ".tif"
kind1 = "System" # System, Cyclone
kind = kind1+"Tracks" # Can be AFZ, Arctic, or other region (or no region), followed by System or Cyclone

rg = 1 # Whether regenesis of a cyclone counts as a track split (0) or track continuation (1)

V = "_GenReg" # An optional version name; suggested to start with "_" or "-" to separate from years in file title

# Aggregation Parameters
minls = 1 # minimum lifespan (in  days) for a track to be considered
mintl = 1000 # minimum track length (in km)
mindisp = 0
minlat = 0 # minimum latitude

# Time Variables
starttime = [1950,1,1,0,0,0] # Format: [Y,M,D,H,M,S]
endtime = [1980,1,1,0,0,0] # stop BEFORE this time (exclusive)
monthstep = [0,1,0,0,0,0] # A Time step that increases by 1 mont [Y,M,D,H,M,S]

dateref = [1900,1,1,0,0,0] # [Y,M,D,H,M,S]

months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
mons = ["01","02","03","04","05","06","07","08","09","10","11","12"]

'''*******************************************
Main Analysis
*******************************************'''
print("Main Analysis")
# Load Regions
regs = nc.Dataset(regpath)['reg'][:].data
spres = pd.read_pickle(path+"/CycloneTracking/tracking"+version+"/cycloneparams.pkl")['spres']

# Create Empty lists
sid, tid, year, month, avguv, maxdsqp, minp, maxdepth, maxpgrad, avgdsqp, avgp, avgdepth, avgpgrad = [[] for i in range(13)]
lifespan, trlen, avgarea, mcc, spl1, spl2, spl3, mrg1, mrg2, mrg3, rge2, genReg, lysReg = [[] for i in range(13)]

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
    try: # Previous Month
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
    trs = [c for c in cs if ((c.lifespan() > minls) and (c.trackLength() >= mintl) and (np.max(c.data.lat) >= minlat) and (c.maxDistFromGenPnt() >= mindisp))]
    trs0 = [c for c in cs0 if ((c.lifespan() > minls) and (c.trackLength() >= mintl) and (np.max(c.data.lat) >= minlat) and (c.maxDistFromGenPnt() >= mindisp))]
    trs2 = [c for c in cs2 if ((c.lifespan() > minls) and (c.trackLength() >= mintl) and (np.max(c.data.lat) >= minlat) and (c.maxDistFromGenPnt() >= mindisp))]

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


        # Append to lists
        try:
            sid.append(tr.sid)
        except:
            tid.append(tr.tid)

        year.append(mt[0]), month.append(mt[1]), avguv.append( tr.data.uv.mean() )
        maxdsqp.append( tr.data.DsqP.max() ), minp.append( tr.data.p_cent.min() )
        maxdepth.append( tr.data.depth.max() ), maxpgrad.append( tr.data.p_grad.max() )
        avgdsqp.append( tr.data.DsqP.mean() ), avgpgrad.append( tr.data.p_grad.mean() )
        avgp.append( tr.data.p_cent.mean() ), avgdepth.append( tr.data.depth.mean() )
        lifespan.append( tr.lifespan() ), trlen.append( tr.trackLength() )
        avgarea.append( tr.avgArea() ), mcc.append( tr.mcc() ), rge2.append( rge2_count )
        spl1.append( spl1_count ), spl2.append( spl2_count ), spl3.append( spl3_count )
        mrg1.append( mrg1_count ), mrg2.append( mrg2_count ), mrg3.append( mrg3_count )
        genReg.append( regs[list(tr.data.loc[tr.data.type > 0,'y'])[0],list(tr.data.loc[tr.data.type > 0,'x'])[0]] )
        lysReg.append( regs[list(tr.data.y)[-1],list(tr.data.x)[-1]] )

    mt = md.timeAdd(mt,monthstep)

# Construct Pandas Dataframe
if kind1 == 'System':
    pdf = pd.DataFrame({"sid":sid,"year":year,"month":month,"avguv":avguv,\
    "maxdsqp":maxdsqp,"minp":minp,"maxdepth":maxdepth,"maxpgrad":maxpgrad,\
    "avgdsqp":avgdsqp,"avgp":avgp,"avgdepth":avgdepth,"avgpgrad":avgpgrad,\
    "lifespan":lifespan,"trlen":trlen,"avgarea":avgarea,"mcc":mcc,\
    "spl1":spl1,"spl2":spl2,"spl3":spl3,"mrg1":mrg1,"mrg2":mrg2,"mrg3":mrg3,\
    "rge2":rge2,"genReg":genReg,"lysReg":lysReg})
else:
    pdf = pd.DataFrame({"tid":tid,"year":year,"month":month,"avguv":avguv,\
    "maxdsqp":maxdsqp,"minp":minp,"maxdepth":maxdepth,"maxpgrad":maxpgrad,\
    "avgdsqp":avgdsqp,"avgp":avgp,"avgdepth":avgdepth,"avgpgrad":avgpgrad,\
    "lifespan":lifespan,"trlen":trlen,"avgarea":avgarea,"mcc":mcc,\
    "spl1":spl1,"spl2":spl2,"spl3":spl3,"mrg1":mrg1,"mrg2":mrg2,"mrg3":mrg3,\
    "rge2":rge2,"genReg":genReg,"lysReg":lysReg})

# Append columns for overall merge, split, non-interacting
pdf["mrg"] = np.array((pdf["mrg2"] >= 1) | (pdf["mrg3"] >= 1)).astype(int)
pdf["spl"] = np.array((pdf["spl2"] >= 1) | (pdf["spl3"] >= 1)).astype(int)
pdf["nonint"] = np.array((pdf["mrg"] == 0) & (pdf["spl"] == 0)).astype(int)

# Convert units
pdf['avgp'] = pdf['avgp']/100 # Pa --> hPa
pdf['minp'] = pdf['minp']/100 # Pa --> hPa
pdf['avgdepth'] = pdf['avgdepth']/100 # Pa --> hPa
pdf['maxdepth'] = pdf['maxdepth']/100 # Pa --> hPa
pdf['avgpgrad'] = pdf['avgpgrad']/100*1000*1000 # Pa/m --> hPa/[1000 km]
pdf['maxpgrad'] = pdf['maxpgrad']/100*1000*1000 # Pa/m --> hPa/[1000 km]
pdf['avgdsqp'] = pdf['avgdsqp']/spres/spres*100 # Pa/gridcell^2 --> hPa/[100 km]^2 (1/100*100*100 = 100)
pdf['maxdsqp'] = pdf['maxdsqp']/spres/spres*100 # Pa/gridcell^2 --> hPa/[100 km]^2 (1/100*100*100 = 100)
pdf['avgarea'] = pdf['avgarea']*spres*spres # gridcell^2 --> km^2

# Write to File
try:
    os.chdir(inpath+"/Aggregation"+kind1)
except:
    os.mkdir(inpath+"/Aggregation"+kind1)
    os.chdir(inpath+"/Aggregation"+kind1)

YY = str(starttime[0]) + "_" + str(md.timeAdd(endtime,[0,-1,0,0,0,0])[0])
pdf.to_csv(BBoxNum+"_"+kind+"Events_"+version+"_"+YY+V+".csv",index=False)
