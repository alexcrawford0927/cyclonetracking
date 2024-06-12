'''
Author: Alex Crawford
Date Created: 13 Sep 2021
Date Modified: 23 Jan 2023
Purpose: Calculate Climatologies From Aggregated Cyclone Characteristics
'''

'''********************
Import Modules
********************'''
import warnings
warnings.filterwarnings("ignore")

print("Loading modules.")
import os
import pandas as pd
import numpy as np
import netCDF4 as nc
# import pickle5
import CycloneModule_13_2 as md

'''*******************************************
Set up Environment
*******************************************'''
print("Setting up environment.")
bboxnum = "BBox27" # use "" if performing on all cyclones; or BBox##
typ = "System"
ver = "13_2"
kSizekm = 800
subset2 = '' # '_DeepeningDsqP' + ''

path = "/Volumes/Cressida"
inpath = path+"/CycloneTracking/tracking"+ver
outpath = inpath+"/"+bboxnum
suppath = path+"/Projections"

'''*******************************************
Define Variables
*******************************************'''
print("Defining variables")

ymin, ymax = 1980, 2019 # years for climatology

dateref = [1900,1,1,0,0,0] # [Y,M,D,H,M,S]
seasons = np.array([1,2,3,4,5,6,7,8,9,10,11,12]) # Ending month for three-month seasons (e.g., 2 = DJF)

# Variables (note that counts are mandatory if using them as the weight for non-count variables)
varsi = [1,-1] # [0,2,3,4] #[7,9,10,11,15,16,17,18,19] # [10,11] #
# varsi = [0,1] + [2,3,4] + [5,6,7] + [8,9] + \
#         [10,11,12,13] + [14] + [15,16,17,18,19]
vNames = ["countU","countP"] + ["DpDt","u","v"] + ['uab','vab','uvab'] + ['vratio','mci'] + \
        ["gen","lys","spl","mrg"] + ['trkden'] + ["p_cent","depth","DsqP","p_grad","radius"]
vunits = ['percent','percent'] + ['hPa/day','km/h','km/h'] + ['km/h','km/h','km/h'] + ['ratio of |v| to |uv|', 'ratio of v^2 to (v^2 + u^2)'] + \
        ['#/month','#/month','#/month','#/month'] + ['#/month'] + ['hPa','hPa','hPa/[100 km]^2','hPa/[1000 km]','km']
vcount = ['none','none'] + ['countU','countU','countU'] + ['countU','countU','countU'] + ['countU','countU'] + \
        ['none','none','none','none'] +['none'] + ['countP','countP','countP','countP','countP']

'''*******************************************
Main Analysis
*******************************************'''
print("Main Analysis")
# Ensure that folders exist to store outputs
try:
    os.chdir(outpath+"/Aggregation"+typ+subset2+"/"+str(kSizekm)+"km")
except:
    os.mkdir(outpath+"/Aggregation"+typ+subset2+"/"+str(kSizekm)+"km")
    os.chdir(outpath+"/Aggregation"+typ+subset2+"/"+str(kSizekm)+"km")

# Read in attributes of reference files
params = pd.read_pickle(inpath+"/cycloneparams.pkl")
# params = pickle5.load(open(inpath+"/cycloneparams.pkl",'rb'))

try:
    spres = params['spres']
except:
    spres = 100

proj = nc.Dataset(suppath+"/EASE2_N0_"+str(spres)+"km_Projection.nc")
size = proj['lat'].shape

YY2 = str(ymin) + "-" + str(ymax)

#################################
##### MONTHLY CLIMATOLOGIES #####
#################################
print("Step 4. Aggregation By Month")
# Write NetCDF File
mname = ver+"_AggregationFields_MonthlyClimatology_"+YY2+".nc"
try:
    mnc = nc.Dataset(mname,'r+')
except:
    mnc = nc.Dataset(mname,'w',format="NETCDF4")
    mnc.createDimension('y', size[0])
    mnc.createDimension('x', size[1])
    mnc.createDimension('time', 12)
    mnc.description = 'Climatology ('+YY2+') of aggregation of cyclone track characteristics on monthly time scale.'

    ncy = mnc.createVariable('y', np.float32, ('y',))
    ncx = mnc.createVariable('x', np.float32, ('x',))
    ncy.units, ncx.units = 'm', 'm'
    ncy[:] = np.arange(proj['lat'].shape[0]*spres*1000/-2 + (spres*1000/2),proj['lat'].shape[0]*spres*1000/2, spres*1000)
    ncx[:] = np.arange(proj['lat'].shape[1]*spres*1000/-2 + (spres*1000/2),proj['lat'].shape[1]*spres*1000/2, spres*1000)

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
    print("  " + vNames[v])
    ncf = nc.Dataset(ver+"_AggregationFields_Monthly_"+vNames[v]+".nc",'r')
    times = ncf['time'][:]

    mlist = []
    for m in range(1,12+1):
        tsub = np.where( ((times-((m-1)/12))%1 == 0) & (times >= ymin) & (times < ymax+1) )[0]

        # Load Primary Monthly Data
        field_M = ncf[vNames[v]][tsub,:,:].data
        field_M[np.isnan(field_M)] = 0

        if vcount[v] == 'none': # If this is count data...
            field_MC = field_M.mean(axis=0)

        else: # If this is not count data...
            # Load Count Data for Weighting
            ncf0 = nc.Dataset(ver+"_AggregationFields_Monthly_"+vcount[v]+".nc",'r')
            field0_M = ncf0[vcount[v]][tsub,:,:].data
            field0_M[np.isnan(field0_M)] = 0

            # Calculate the weighted average
            field_MC = (field_M*field0_M).sum(axis=0) / field0_M.sum(axis=0)
            ncf0.close()

        mlist.append(field_MC)

    try:
        ncvar = mnc.createVariable(vNames[v], np.float64, ('time','y','x'))
        ncvar.units = ncf[vNames[v]].units
        ncvar[:] = np.array(mlist)
    except:
        mnc[vNames[v]][:] = np.array(mlist)

mnc.close()
ncf.close()

##### SEASONAL MEANS ###
print("Step 5. Aggregate By Season")
sname = ver+"_AggregationFields_SeasonalClimatology_"+YY2+".nc"
if sname in os.listdir():
    snc = nc.Dataset(sname,'r+')
else:
    snc = nc.Dataset(sname,'w')
    snc.createDimension('y', size[0])
    snc.createDimension('x', size[1])
    snc.createDimension('time', len(seasons))
    snc.description = 'Climatology ('+YY2+') of aggregation of cyclone track characteristics on seasonal time scale.'

    ncy = snc.createVariable('y', np.float32, ('y',))
    ncx = snc.createVariable('x', np.float32, ('x',))
    ncy.units, ncx.units = 'm', 'm'
    ncy[:] = np.arange(proj['lat'].shape[0]*spres*1000/-2 + (spres*1000/2),proj['lat'].shape[0]*spres*1000/2, spres*1000)
    ncx[:] = np.arange(proj['lat'].shape[1]*spres*1000/-2 + (spres*1000/2),proj['lat'].shape[1]*spres*1000/2, spres*1000)

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

    # Load Primary Monthly Data
    mlist = []
    for m in seasons:
        tsub = np.where( ((times-((m-1)/12))%1 == 0) & (times >= ymin) & (times < ymax+1) )[0]
        sitsub = np.concatenate( (tsub, tsub-1, tsub-2) )

        field_M = ncf[vNames[v]][sitsub,:,:].data
        field_M[np.isnan(field_M)] = 0

        if vcount[v] == 'none': # If this is count data...
            field_MC = field_M.mean(axis=0)

        else: # If this is not count data...
            # Load Count Data for Weighting
            ncf0 = nc.Dataset(ver+"_AggregationFields_Monthly_"+vcount[v]+".nc",'r')
            field0_M = ncf0[vcount[v]][sitsub,:,:].data
            field0_M[np.isnan(field0_M)] = 0

            # Calculate the weighted average
            field_MC = (field_M*field0_M).sum(axis=0) / field0_M.sum(axis=0)
            ncf0.close()

        mlist.append(field_MC)

    try:
        ncvar = snc.createVariable(vNames[v], np.float64, ('time','y','x'))
        ncvar.units = ncf[vNames[v]].units
        ncvar[:] = np.array(mlist)
    except:
        snc[vNames[v]][:] = np.array(mlist)

snc.close()
ncf.close()
