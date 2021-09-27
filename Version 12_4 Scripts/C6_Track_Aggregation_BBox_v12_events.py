'''
Author: Alex Crawford
Date Created: 10 Mar 2015
Date Modified: 18 Apr 2016; 10 Jul 2019 (update for Python 3);
                10 Sep 2020 (switch from geotiff to netcdf), switch to uniform_filter from scipy.ndimage
                30 Sep 2020 (switch back to slower custom smoother because of what scipy does to NaNs)
                18 Feb 2021 (edited seasonal caluclations to work directly from months, not monthly climatology,
                             allowing for cross-annual averaging)
Purpose: Calculate aggergate statistics (Eulerian and Lagrangian) for either
cyclone tracks or system tracks.

User inputs:
    Path Variables, including the reanalysis (ERA, MERRA, CFSR)
    Track Type (typ): Cyclone or System
    Bounding Box ID (bboxnum): 2-digit character string
    Time Variables: when to start, end, the time step of the data

Other notes:
    Units for track density are tracks/month/gridcell
    Units for event counts are raw counts (#/month/gridcell)
    Units for counts relative to cyclone obs are ratios (%/gridcell/100)
'''

'''********************
Import Modules
********************'''
# Import clock:
from time import perf_counter as clock
start = clock()
import warnings
warnings.filterwarnings("ignore")

print("Loading modules.")
import os
import pandas as pd
from scipy import ndimage
import numpy as np
import netCDF4 as nc
import CycloneModule_12_4 as md
# import pickle5

'''*******************************************
Set up Environment
*******************************************'''
print("Setting up environment.")
bboxnum = "" # use "" if performing on all cyclones; or BBox##
typ = "System"
ver = "12_4TestTracks"

path = "/Volumes/Cressida"
inpath = path+"/CycloneTracking/tracking"+ver
outpath = inpath+"/"+bboxnum
suppath = path+"/Projections"

'''*******************************************
Define Variables
*******************************************'''
print("Defining variables")
# Time Variables 
starttime = [1979,1,1,0,0,0] # Format: [Y,M,D,H,M,S]
endtime = [2020,1,1,0,0,0] # stop BEFORE this time (exclusive)
monthstep = [0,1,0,0,0,0] # A Time step that increases by 1 month [Y,M,D,H,M,S]

ymin, ymax = 1981, 2010 # years for climatology

dateref = [1900,1,1,0,0,0] # [Y,M,D,H,M,S]

seasons = np.array([1,2,3,4,5,6,7,8,9,10,11,12]) # Ending month for three-month seasons (e.g., 2 = DJF)
months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
mons = ["01","02","03","04","05","06","07","08","09","10","11","12"]
dpm = [31,28,31,30,31,30,31,31,30,31,30,31]

# Aggregation Parameters
minls = 1 # minimum lifespan (in days) for a track to be considered
mintl = 1 # minimum track length (in km for version ≥ 11.1; grid cells for version ≤ 10.10) for a track to be considered
kSizekm = 400 # Full kernel size (in km) for spatial averaging measured between grid cell centers.
    ## For a 100 km spatial resolution, 400 is a 4 by 4 kernel; i.e., kSize = (kSizekm/spres)

# Variables
vNames = ["countA","gen","lys","spl","mrg"]
varsi = range(1,len(vNames)) # range(0,1) #
vunits = ['ratio','count','count','count','count']
agg = [-1,-1,-1,-1,-1]

'''*******************************************
Main Analysis
*******************************************'''
print("Main Analysis")
# Ensure that folders exist to store outputs
try:
    os.chdir(outpath+"/Aggregation"+typ)
except:
    os.mkdir(outpath+"/Aggregation"+typ)
    os.chdir(outpath+"/Aggregation"+typ)

print("Step 1. Load Files and References")
# Read in attributes of reference files
params = pd.read_pickle(inpath+"/cycloneparams.pkl")
# params = pickle5.load(open(inpath+"/cycloneparams.pkl",'rb'))
timestep = params['timestep']
spres = params['spres']

proj = nc.Dataset(suppath+"/EASE2_N0_"+str(spres)+"km_Projection.nc")
lats = proj['lat'][:]

kSize = int(kSizekm/spres) # This needs to be the full width ('diameter'), not the half width ('radius') for ndimage filters

YY = str(starttime[0])+'-'+str(endtime[0]-1)
YY2 = str(ymin) + "-" + str(ymax)

print("Step 2. Aggregate!")
vlists = [ [] for v in vNames]

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
    cs = pd.read_pickle(inpath+"/"+bboxnum+"/"+typ+"Tracks/"+Y+"/"+bboxnum+typ.lower()+"tracks"+Y+M+".pkl")
    # cs = pickle5.load(open(inpath+"/"+bboxnum+"/"+typ+"Tracks/"+Y+"/"+bboxnum+typ.lower()+"tracks"+Y+M+".pkl",'rb'))
    try:
        cs0 = pd.read_pickle(inpath+"/"+bboxnum+"/"+typ+"Tracks/"+str(mt0[0])+"/"+bboxnum+typ.lower()+"tracks"+str(mt0[0])+mons[mt0[1]-1]+".pkl")
        # cs0 = pickle5.load(open(inpath+"/"+bboxnum+"/"+typ+"Tracks/"+str(mt0[0])+"/"+bboxnum+typ.lower()+"tracks"+str(mt0[0])+mons[mt0[1]-1]+".pkl",'rb'))
    except:
        cs0 = []
    # Load Active tracks
    ct2 = pd.read_pickle(inpath+"/ActiveTracks/"+Y+"/activetracks"+Y+M+".pkl")
    # ct2 = pickle5.load(open(inpath+"/ActiveTracks/"+Y+"/activetracks"+Y+M+".pkl",'rb'))
    if typ == "Cyclone":
        cs2 = ct2
    else:
        try: # Convert active tracks to systems as well
            cs2 = md.cTrack2sTrack(ct2,[],dateref,1)[0]
        except:
            cs2 = []
    
    ### LIMIT TRACKS & IDS ###
    # Limit to tracks that satisfy minimum lifespan and track length
    trs = [c for c in cs if ((c.lifespan() > minls) and (c.trackLength() >= mintl))]
    trs2 = [c for c in cs2 if ((c.lifespan() > minls) and (c.trackLength() >= mintl))]
    trs0 = [c for c in cs0 if ((c.lifespan() > minls) and (c.trackLength() >= mintl))]
    
    ### CALCULATE FIELDS ###
    fields0 = [np.nan]
    fields1 = md.aggregateEvents([trs,trs0,trs2],typ,mtdays,lats.shape)

    fields = fields0 + fields1
    
    ### SMOOTH FIELDS ###
    for v in varsi:
        # varFieldsm = md.smoothField(fields[v],kSize) # Smooth
        varFieldsm = ndimage.uniform_filter(fields[v],kSize,mode="nearest") # --> This cannot handle NaNs
        vlists[v].append(varFieldsm) # append to list
    
    # Increment Month
    mt = md.timeAdd(mt,monthstep,lys=1)

### SAVE FILE ###
print("Step 3. Write to NetCDF")
for v in varsi:
    mnc = nc.Dataset(ver+"_AggregationFields_Monthly_"+vNames[v]+".nc",'w')
    mnc.createDimension('y', lats.shape[0])
    mnc.createDimension('x', lats.shape[1])
    mnc.createDimension('time', (endtime[0]-starttime[0])*12)
    mnc.description = 'Aggregation of cyclone track characteristics on monthly time scale.'
    
    ncy = mnc.createVariable('y', np.float32, ('y',))
    ncx = mnc.createVariable('x', np.float32, ('x',))
    
    # Add times, lats, and lons
    nctime = mnc.createVariable('time', np.float32, ('time',))
    nctime.units = 'years'
    nctime[:] = np.arange(starttime[0],endtime[0],1/12)
    
    nclon = mnc.createVariable('lon', np.float32, ('y','x'))
    nclon.units = 'degrees'
    nclon[:] = proj['lon'][:]
    
    nclat = mnc.createVariable('lat', np.float32, ('y','x'))
    nclat.units = 'degrees'
    nclat[:] = proj['lat'][:]

    ncvar = mnc.createVariable(vNames[v], np.float64, ('time','y','x'))
    ncvar.units = vunits[v] + ' -- Smoothing:' + str(kSizekm) + ' km'
    ncvar[:] = np.array(vlists[v])
    
    mnc.close()

#################################
##### MONTHLY CLIMATOLOGIES #####
#################################
print("Step 4. Aggregation By Month")
vlists = [ [] for v in vNames ]

for v in varsi:
    print(vNames[v])
    ncf = nc.Dataset(ver+"_AggregationFields_Monthly_"+vNames[v]+".nc",'r')
    times = ncf['time'][:]

    for m in range(1,12+1):
        tsub = np.where( ((times-((m-1)/12))%1 == 0) & (times >= ymin) & (times < ymax+1) )[0]
        
        # Take Monthly Climatologies

        field_M = ncf[vNames[v]][tsub,:,:].data
        field_MC = np.apply_along_axis(np.nanmean, 0, field_M)
        vlists[v].append(field_MC)
ncf.close()

# Write NetCDF File
mname = ver+"_AggregationFields_MonthlyClimatology_"+YY2+".nc"
try:
    mnc = nc.Dataset(mname,'r+')
except:
    mnc = nc.Dataset(mname,'w',format="NETCDF4")
    mnc.createDimension('y', lats.shape[0])
    mnc.createDimension('x', lats.shape[1])
    mnc.createDimension('time', 12)
    mnc.description = 'Climatology ('+YY2+') of aggregation of cyclone track characteristics on monthly time scale.'
    
    ncy = mnc.createVariable('y', np.float32, ('y',))
    ncx = mnc.createVariable('x', np.float32, ('x',))
    
    # Add times, lats, and lons
    nctime = mnc.createVariable('time', np.float32, ('time',))
    nctime.units = 'months'
    nctime[:] = np.arange(1,12+1,1)
    
    nclon = mnc.createVariable('lon', np.float32, ('y','x'))
    nclon.units = 'degrees'
    nclon[:] = proj['lon'][:]
    
    nclat = mnc.createVariable('lat', np.float32, ('y','x'))
    nclat.units = 'degrees'
    nclat[:] = proj['lat'][:]

for v in varsi:
    try:
        ncvar = mnc.createVariable(vNames[v], np.float64, ('time','y','x'))
        ncvar.units = vunits[v] + ' -- Smoothing:' + str(kSizekm) + ' km'
        ncvar[:] = np.array(vlists[v])
    except:
        mnc[vNames[v]][:] = np.array(vlists[v])

mnc.close()

##### SEASONAL MEANS ###
print("Step 5. Aggregate By Season")
sname = ver+"_AggregationFields_SeasonalClimatology_"+YY2+".nc"
if sname in os.listdir():
    snc = nc.Dataset(sname,'r+')
else:
    snc = nc.Dataset(sname,'w')
    snc.createDimension('y', lats.shape[0])
    snc.createDimension('x', lats.shape[1])
    snc.createDimension('time', len(seasons))
    snc.description = 'Climatology ('+YY+') of aggregation of cyclone track characteristics on seasonal time scale.'
    
    ncy = snc.createVariable('y', np.float32, ('y',))
    ncx = snc.createVariable('x', np.float32, ('x',))
    
    # Add times, lats, and lons
    nctime = snc.createVariable('time', np.int8, ('time',))
    nctime.units = 'seasonal end months'
    nctime[:] = seasons
    
    nclon = snc.createVariable('lon', np.float32, ('y','x'))
    nclon.units = 'degrees'
    nclon[:] = proj['lon'][:]
    
    nclat = snc.createVariable('lat', np.float32, ('y','x'))
    nclat.units = 'degrees'
    nclat[:] = proj['lat'][:]

for v in varsi:
    ncf = nc.Dataset(ver+"_AggregationFields_Monthly_"+vNames[v]+".nc",'r')
    times = ncf['time'][:]
    
    print("  " + vNames[v])

    varr = ncf[vNames[v]][:]
    seaslist = []
    for si in seasons:    
        tsub = np.where( ((times-((si-1)/12))%1 == 0) & (times >= ymin) & (times < ymax+1) )[0]
        seasarr = (varr[tsub,:,:] + varr[tsub-1,:,:] + varr[tsub-2,:,:]) / 3
        seaslist.append( np.apply_along_axis(np.nanmean, 0, seasarr) )
        
    try:
        ncvar = snc.createVariable(vNames[v], np.float64, ('time','y','x'))
        ncvar.units = vunits[v] + ' -- Smoothing:' + str(kSizekm) + ' km'
        ncvar[:] = np.array(seaslist)
    except:
        snc[vNames[v]][:] = np.array(seaslist)
    
ncf.close()
snc.close()
