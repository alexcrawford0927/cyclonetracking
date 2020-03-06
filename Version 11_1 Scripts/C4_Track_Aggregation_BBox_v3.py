'''
Author: Alex Crawford
Date Created: 10 Mar 2015
Date Modified: 18 Apr 2016; 10 Jul 2019 (update for Python 3)
Purpose: Calculate aggergate statistics (Eulerian and Lagrangian) for either
cyclone tracks or system tracks.

User inputs:
    Path Variables, including the reanalysis (ERA, MERRA, CFSR)
    Track Type (typ): Cyclone or System
    Bounding Box ID (bboxnum): 2-digit character string
    Time Variables: when to start, end, the time step of the data
'''

'''********************
Import Modules
********************'''
# Import clock:
from time import perf_counter as clock
start = clock()

print("Loading modules.")
import os
import pandas as pd
import numpy as np
from osgeo import gdal, gdalconst, gdalnumeric
import CycloneModule_11_1 as md

'''*******************************************
Set up Environment
*******************************************'''
print("Setting up environment.")
bboxnum = "BBox16" # use "" if performing on all cyclones; or BBox##
typ = "System"  # "System" or "Cyclone"

path = "/Volumes/Ferdinand/Cyclones/Algorithm Sharing/Version11_1/Test Data"
inpath = path+"/Results/tracking11_1E/"+bboxnum
outpath = inpath
suppath = path

'''*******************************************
Define Variables
*******************************************'''
print("Defining variables")
# File Variables
outputtype = gdal.GDT_Float64
ext = ".tif"

reffile = "EASE2_N0_100km_Lats"+ext

# Time Variables 
starttime = [2016,8,1,0,0,0] # Format: [Y,M,D,H,M,S]
endtime = [2016,9,1,0,0,0] # stop BEFORE this time (exclusive)
timestep = [0,0,0,6,0,0] # Time step in [Y,M,D,H,M,S]
monthstep = [0,1,0,0,0,0] # A Time step that increases by 1 month [Y,M,D,H,M,S]

dateref = [1900,1,1,0,0,0] # [Y,M,D,H,M,S]

months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
mons = ["01","02","03","04","05","06","07","08","09","10","11","12"]
dpm = [31,28,31,30,31,30,31,31,30,31,30,31]
days = ["01","02","03","04","05","06","07","08","09","10","11","12","13",\
    "14","15","16","17","18","19","20","21","22","23","24","25","26","27",\
    "28","29","30","31"]
hours = ["0000","0100","0200","0300","0400","0500","0600","0700","0800",\
    "0900","1000","1100","1200","1300","1400","1500","1600","1700","1800",\
    "1900","2000","2100","2200","2300"]
    
# Aggregation Parameters
minls = 1 # minimum lifespan (in days) for a track to be considered
mintl = 1 # minimum track length (in km for version 11.1; grid cells for version 10.x) for a track to be considered
kSize = 2 # kernal size for spatial averaging

# Variables
vNames = ["countA","gen","lys","spl","mrg",\
"maxuv","maxdpdt","maxdep","minp","maxdsqp",\
"trkden","countC","countP","countU","countR","mcc",\
"uvAb","uvDi","radius","area",\
"depth","dpdr","dpdt","pcent","dsqp"]
varsi = range(1,len(vNames)) # range(0,1) #
agg = [-1,-1,-1,-1,-1,\
-1,-1,-1,-1,-1,\
-1,-1,-1,-1,-1,-1,\
13,-2,12,12,\
14,14,13,12,12]

'''*******************************************
Main Analysis
*******************************************'''
print("Main Analysis")
# Ensure that folders exist to store outputs
for y in range(starttime[0],endtime[0]+1):
    Y = str(y)
    
    # Cyclone Fields
    try:
        os.chdir(outpath+"/Aggregation"+typ+"/"+Y)
    except:
        os.mkdir(outpath+"/Aggregation"+typ+"/"+Y)
        for mm in range(12):
            os.mkdir(outpath+"/Aggregation"+typ+"/"+Y+"/"+months[mm])

print("Step 1. Load Files and References")
# Read in attributes of reference files
ref = gdal.Open(suppath+"/"+reffile,gdalconst.GA_ReadOnly)
refA = gdalnumeric.LoadFile(suppath+"/"+reffile)

mt = starttime
while mt != endtime:    
    # Extract date
    Y = str(mt[0])
    MM = months[mt[1]-1]
    M = mons[mt[1]-1]
    print(" " + Y + " - " + MM)
    
    mtdays = md.daysBetweenDates(dateref,mt,lys=1) # Convert date to days since [1900,1,1,0,0,0]
    mt0 = md.timeAdd(mt,[-i for i in monthstep],lys=1) # Identify time for the previous month
    
    # Define number of valid times for making %s from counting stats
    if MM == "Feb" and md.leapyearBoolean(mt)[0] == 1:
        n = 29*(24/timestep[3])
    else:
        n = dpm[mt[1]-1]*(24/timestep[3])
    
    ### LOAD TRACKS ###
    # Load Cyclone/System Tracks
    cs = pd.read_pickle(inpath+"/"+typ+"Tracks/"+Y+"/"+bboxnum+typ.lower()+"tracks"+Y+M+".pkl")
    try:
        cs0 = pd.read_pickle(inpath+"/"+typ+"Tracks/"+str(mt0[0])+"/"+bboxnum+typ.lower()+"tracks"+str(mt0[0])+mons[mt0[1]-1]+".pkl")
    except:
        cs0 = []
    # Load Active tracks
    ct2 = pd.read_pickle(inpath+"/ActiveTracks/"+Y+"/"+bboxnum+"activetracks"+Y+M+".pkl")
    if typ == "Cyclone":
        cs2 = ct2
    else:
        try: # Convert active tracks to systems as well
            cs2 = md.cTrack2sTrack(ct2,[],dateref,1)[0]
        except:
            cs2 = []
    
    ### LIMIT TRACKS & IDS ###
    # Limit to tracks that satisfy minimum lifespan and track length
    trs = [c for c in cs if ((c.lifespan() >= minls) and (c.trackLength() >= mintl))]
    trs2 = [c for c in cs2 if ((c.lifespan() >= minls) and (c.trackLength() >= mintl))]
    trs0 = [c for c in cs0 if ((c.lifespan() >= minls) and (c.trackLength() >= mintl))]
    
    ### CALCULATE FIELDS ###
    print("  fields")
    try:
        start = clock()
        fields0 = md.aggregateTimeStepFields(inpath,trs,mt,timestep,lys=1)
        print(clock() - start)
    except:
        fields0 = [np.nan]
    
    print("  events")
    fields1 = md.aggregateEvents([trs,trs0,trs2],typ,mtdays,refA.shape)
    print("  tracks")
    fields2 = md.aggregateTrackWiseStats(trs,mt,refA.shape)
    statsPDF = fields2[0]
    print("  points")
    try:
        fields3 = md.aggregatePointWiseStats(trs,n,refA.shape)
    except:
        fields3 = [np.zeros_like(refA), np.zeros_like(refA), np.zeros_like(refA), \
        np.zeros_like(refA), np.zeros_like(refA), np.zeros_like(refA), \
        np.zeros_like(refA)*np.nan, np.zeros_like(refA)*np.nan, np.zeros_like(refA)*np.nan, \
        np.zeros_like(refA)*np.nan, np.zeros_like(refA)*np.nan, np.zeros_like(refA)*np.nan, \
        np.zeros_like(refA)*np.nan, np.zeros_like(refA)*np.nan, np.zeros_like(refA)*np.nan]
    
    fields = fields0 + fields1 + fields2[1:] + fields3
#    fields = fields0
    
    ### SMOOTH FIELDS, SAVE FILES ###
    print("  smooth and save files")
#    varFieldsm = md.smoothField(fields3[2],kSize)
#    md.writeNumpy_gdalObj(varFieldsm,outpath+"/Aggregation"+typ+"/"+Y+"/"+MM+"/"+vNames[12]+"Field1000"+Y+M+ext,ref,outputtype) # Write File

    statsPDF.to_csv(outpath+"/Aggregation"+typ+"/"+Y+"/"+MM+"/BasicStats"+Y+M+".csv")
    
    for v in varsi:
        varFieldsm = md.smoothField(fields[v],kSize) # Smooth
        md.writeNumpy_gdalObj(varFieldsm,outpath+"/Aggregation"+typ+"/"+Y+"/"+MM+"/"+vNames[v]+"Field"+Y+M+ext,ref,outputtype) # Write File
    
    # Increment Month
    mt = md.timeAdd(mt,monthstep,lys=1)

### Delete Unnecessary Files ###
#del fields, fields0, fields1, fields2, fields3, statsPDF, trs, trs0, trs2, cs, cs0, cs2, ct2, mtdays, mt0, mt, M, MM, Y, v, c, i

#### MONTHLY CLIMATOLOGIES ###
#print("Step 3. Aggregation By Month")
#for m in range(1,12+1):
#    MM = months[m-1]
#    M = mons[m-1]
#    print(" " + MM)
#    
#    # Define starting and ending years for this month:
#    if starttime[1] <= m:
#        startyear = starttime[0]
#    else:
#        startyear = starttime[0]+1
#    
#    lasttime = md.timeAdd(endtime,[-1*d for d in timestep],lys=1)
#    if lasttime[1] >= m:
#        endyear = lasttime[0]
#    else:
#        endyear = lasttime[0]-1
#    
#    n = endyear-startyear
#    YY = str(startyear)+"_"+str(endyear)
#    
#    # Create Empty Objects
#    #statsPDF_MC = pd.DataFrame(columns=["year","month","maxUV","maxDpDt","maxDep","minP","lifespan","trlength","avgArea","MCC"])
#    fields_MC = [[] for i in range(len(vNames))]
#    
#    # Load Files
#    for y in range(startyear,endyear):
#        Y = str(y)        
#        # Load Monthly Values
#        #statsPDF_M = pd.read_csv(outpath+"/Aggregation"+typ+"/"+Y+"/"+MM+"/BasicStats"+Y+M+".csv")
#        #statsPDF_MC = statsPDF_MC.append(statsPDF_M)
#        for v in varsi:
#            field_M = gdalnumeric.LoadFile(outpath+"/Aggregation"+typ+"/"+Y+"/"+MM+"/"+vNames[v]+"Field"+Y+M+ext)
#            fields_MC[v].append(np.where(field_M == -99, np.nan, field_M))
#    
#    # Take Monthly Climatologies
#    for v in varsi:
#        if agg[v] == -1:
#            field_MC = np.apply_along_axis(np.nanmean,0,fields_MC[v])
#        elif agg[v] == -2:
#            field_MC = md.meanArraysCircular_nan(fields_MC[v],0,360)
#        else:
#            field_MC = np.apply_along_axis(np.nansum,0,np.array(fields_MC[v])*np.array(fields_MC[agg[v]])) / np.apply_along_axis(np.nansum,0,fields_MC[agg[v]])
#        
#        md.writeNumpy_gdalObj(field_MC,outpath+"/Aggregation"+typ+"/Monthly"+YY+"/"+MM+"/"+vNames[v]+"Field"+M+ext,ref,outputtype)
#
##### SEASONAL MEANS ###
#print("Step 4. Aggregate By Season")
#os.chdir(outpath+"/Aggregation"+typ+"/Monthly"+YY)
##os.mkdir("DJF")
##os.mkdir("MAM")
##os.mkdir("JJA")
##os.mkdir("SON")
#
#for v in varsi:
#    for m in [2,5,8,11]:
#        MM = months[m-1]
#        M = mons[m-1]
#        SSS = months[m-3][0]+months[m-2][0]+months[m-1][0]
#        v1 = gdalnumeric.LoadFile(outpath+"/Aggregation"+typ+"/Monthly"+YY+"/"+MM+"/"+vNames[v]+"Field"+M+ext)
#        v2 = gdalnumeric.LoadFile(outpath+"/Aggregation"+typ+"/Monthly"+YY+"/"+months[m-2]+"/"+vNames[v]+"Field"+mons[m-2]+ext)
#        v3 = gdalnumeric.LoadFile(outpath+"/Aggregation"+typ+"/Monthly"+YY+"/"+months[m-3]+"/"+vNames[v]+"Field"+mons[m-3]+ext)
#        v1 = np.where(v1 == -99, np.nan, v1) # Set NaNs
#        v2 = np.where(v2 == -99, np.nan, v2) # Set NaNs
#        v3 = np.where(v3 == -99, np.nan, v3) # Set NaNs
#        v4 = np.apply_along_axis(np.mean,0,np.array([v1,v2,v3]))
#        md.writeNumpy_gdalObj(v4,outpath+"/Aggregation"+typ+"/Monthly"+YY+"/"+SSS+"/"+vNames[v]+"Field"+SSS+ext,ref,outputtype)
#
#print('Complete')