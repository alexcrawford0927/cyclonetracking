'''
Author: Alex Crawford
Date Created: 10 Mar 2015
Date Modified: 18 Apr 2016; 10 Jul 2019 (update for Python 3);
                10 Sep 2020 (switch from geotiff to netcdf); switch to uniform_filter from scipy.ndimage
                30 Sep 2020 (switch back to slower custom smoother because of what scipy.ndimage does to NaNs)
Purpose: Calculate aggergate statistics for either cyclone tracks or system tracks.

User inputs:
    Path Variables
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
import numpy as np
import netCDF4 as nc
import CycloneModule_12_2 as md

'''*******************************************
Set up Environment
*******************************************'''
print("Setting up environment.")
bboxnum = "BBox15" # use "" if performing on all cyclones; or BBox##
typ = "System"
ver = "12_4E5B"
spres = 100

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

dateref = [1900,1,1,0,0,0] # [Y,M,D,H,M,S]

months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
mons = ["01","02","03","04","05","06","07","08","09","10","11","12"]
dpm = [31,28,31,30,31,30,31,31,30,31,30,31]

# Aggregation Parameters
minls = 1 # minimum lifespan (in days) for a track to be considered
mintl = 1 # minimum track length (in km for version ≥ 11.1; grid cells for version ≤ 10.10) for a track to be considered
kSizekm = 400 # Full kernel size (in km) for spatial averaging measured between grid cell centers.
    ## For a 100 km spatial resolution, 400 is a 5 by 5 kernel; i.e., kSize = (kSizekm/spres)/2

# Variables
vNames = ["countA","gen","lys","spl","mrg",\
"maxuv","maxdpdt","maxdep","minp","maxdsqp",\
"trkden","countC","countP","countU","countR","mcc",\
"uvAb","uvDi","radius","area",\
"depth","dpdr","dpdt","pcent","dsqp"]
varsi = range(1,len(vNames)) # range(0,1) #
vunits = ['ratio','count','count','count','count',\
         'count', 'count', 'count', 'count', 'count',\
        'count','ratio','ratio','ratio','ratio','ratio',\
        'km/hr','degrees','cell size','(cell size)^2',\
        'Pa','Pa/cell size','Pa/day','Pa','Pa/(cell size)^2']
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
try:
    os.chdir(outpath+"/Aggregation"+typ)
except:
    os.mkdir(outpath+"/Aggregation"+typ)
    os.chdir(outpath+"/Aggregation"+typ)

print("Step 1. Load Files and References")
# Read in attributes of reference files
params = pd.read_pickle(inpath+"/cycloneparams.pkl")
timestep = params['timestep']
spres = params['spres']

proj = nc.Dataset(suppath+"/EASE2_N0_"+str(spres)+"km_Projection.nc")
lats = proj['lat'][:]

# Define kernel size
kSize = int(kSizekm/spres/2)

# Define year text
YY = str(starttime[0])+'-'+str(endtime[0]-1)

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
    try:
        cs0 = pd.read_pickle(inpath+"/"+bboxnum+"/"+typ+"Tracks/"+str(mt0[0])+"/"+bboxnum+typ.lower()+"tracks"+str(mt0[0])+mons[mt0[1]-1]+".pkl")
    except:
        cs0 = []
    # Load Active tracks
    ct2 = pd.read_pickle(inpath+"/ActiveTracks/"+Y+"/activetracks"+Y+M+".pkl")
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
    fields0 = [np.nan]
    fields1 = md.aggregateEvents([trs,trs0,trs2],typ,mtdays,lats.shape)
    fields2 = md.aggregateTrackWiseStats(trs,mt,lats.shape)
    try:
        fields3 = md.aggregatePointWiseStats(trs,n,lats.shape)
    except:
        fields3 = [np.zeros_like(lats), np.zeros_like(lats), np.zeros_like(lats), \
        np.zeros_like(lats), np.zeros_like(lats), np.zeros_like(lats), \
        np.zeros_like(lats)*np.nan, np.zeros_like(lats)*np.nan, np.zeros_like(lats)*np.nan, \
        np.zeros_like(lats)*np.nan, np.zeros_like(lats)*np.nan, np.zeros_like(lats)*np.nan, \
        np.zeros_like(lats)*np.nan, np.zeros_like(lats)*np.nan, np.zeros_like(lats)*np.nan]
    
    fields = fields0 + fields1 + fields2 + fields3
    
    ### SMOOTH FIELDS ###
    print("  smooth fields")    
    for v in varsi:
        varFieldsm = md.smoothField(fields[v],kSize) # Smooth
        # varFieldsm = ndimage.uniform_filter(fields[v],kSize,mode="nearest") # --> This cannot handle NaNs
        vlists[v].append(varFieldsm) # append to list
    
    # Increment Month
    mt = md.timeAdd(mt,monthstep,lys=1)

### SAVE FILE ###
print("Step 3. Write to NetCDF")
mnc = nc.Dataset(ver+"_AggregationFields_Monthly.nc",'w',format="NETCDF4")
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

for v in varsi:
    ncvar = mnc.createVariable(vNames[v], np.float64, ('time','y','x'))
    ncvar.units = vunits[v]
    ncvar[:] = np.array(vlists[v])

mnc.close()

##################################
###### MONTHLY CLIMATOLOGIES #####
##################################
ncf = nc.Dataset(ver+"_AggregationFields_Monthly.nc",'r')
times = ncf['time'][:]

vlists = [ [] for v in vNames ]
print("Step 4. Aggregation By Month")
for m in range(1,12+1):
    print(" " + months[m-1])
    tsub = np.where((times-((m-1)/12))%1 == 0)[0]
    
    # Take Monthly Climatologies
    for v in varsi:
        field_M = ncf[vNames[v]][tsub,:,:].data
        
        if agg[v] == -1:
            field_MC = np.apply_along_axis(np.nanmean, 0, field_M)
        elif agg[v] == -2:
            field_MC = md.meanArraysCircular_nan(field_M, 0, 360)
        else:
            # field_MC = np.apply_along_axis(np.nansum, 0, field_M*ncf[vNames[agg[v]]][tsub,:,:].data) / np.apply_along_axis(np.nansum, 0, ncf[vNames[agg[v]]][tsub,:,:].data)

            field_Count = ncf[vNames[agg[v]]][tsub,:,:].data
            field_MC = np.zeros((field_M.shape[1],field_M.shape[2]))
            for row in range(field_M.shape[1]):
                for col in range(field_M.shape[2]):
                    fsub = np.where(np.isfinite(field_M[:,row,col]))
                    field_MC[row,col] = np.sum(field_M[fsub,row,col]*field_Count[fsub,row,col]) / np.sum(field_Count[fsub,row,col])
        
        vlists[v].append(field_MC)
ncf.close()

# Write NetCDF File
mnc = nc.Dataset(ver+"_AggregationFields_MonthlyClimatology_"+YY+".nc",'w',format="NETCDF4")
mnc.createDimension('y', lats.shape[0])
mnc.createDimension('x', lats.shape[1])
mnc.createDimension('time', 12)
mnc.description = 'Climatology ('+YY+') of aggregation of cyclone track characteristics on monthly time scale.'

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
    ncvar = mnc.createVariable(vNames[v], np.float64, ('time','y','x'))
    ncvar.units = vunits[v]
    ncvar[:] = np.array(vlists[v])

mnc.close()

##### SEASONAL MEANS ###
mnc = nc.Dataset(ver+"_AggregationFields_MonthlyClimatology_"+YY+".nc",'r')

print("Step 5. Aggregate By Season")
snc = nc.Dataset(ver+"_AggregationFields_SeasonalClimatology_"+YY+".nc",'w',format="NETCDF4")
snc.createDimension('y', lats.shape[0])
snc.createDimension('x', lats.shape[1])
snc.createDimension('time', 4)
snc.description = 'Climatology ('+YY+') of aggregation of cyclone track characteristics on seasonal time scale.'

ncy = snc.createVariable('y', np.float32, ('y',))
ncx = snc.createVariable('x', np.float32, ('x',))

# Add times, lats, and lons
nctime = snc.createVariable('time', np.int8, ('time',))
nctime.units = 'seasonal end months'
nctime[:] = np.array([2,5,8,11])

nclon = snc.createVariable('lon', np.float32, ('y','x'))
nclon.units = 'degrees'
nclon[:] = proj['lon'][:]

nclat = snc.createVariable('lat', np.float32, ('y','x'))
nclat.units = 'degrees'
nclat[:] = proj['lat'][:]

for v in varsi:
    varr = mnc[vNames[v]][:]
    ss1 = ( varr[-1,:,:] + varr[0,:,:] + varr[1,:,:]) / 3
    ss2 = ( varr[2,:,:] + varr[3,:,:] + varr[4,:,:]) / 3
    ss3 = ( varr[5,:,:] + varr[6,:,:] + varr[7,:,:]) / 3
    ss4 = ( varr[8,:,:] + varr[9,:,:] + varr[10,:,:]) / 3
    
    ncvar = snc.createVariable(vNames[v], np.float64, ('time','y','x'))
    ncvar.units = vunits[v]
    ncvar[:] = np.array([ss1,ss2,ss3,ss4])
    
mnc.close()
snc.close()
