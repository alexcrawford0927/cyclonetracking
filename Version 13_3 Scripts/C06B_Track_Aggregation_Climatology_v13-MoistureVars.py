'''
Author: Alex Crawford
Date Created: 13 Sep 2021
Date Modified: 23 Jan 2023
                05 Mar 2025 --> Make count variables seasonal sums instead of seasonal averages of monthly sums 
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
import xarray as xr
# import pickle5
import CycloneModule_13_3 as md

'''*******************************************
Set up Environment
*******************************************'''
print("Setting up environment")
bboxnum = "" # use "" if performing on all cyclones; or BBox##
typ = "System"
ver = "13testP"
kSizekm = 800 # Radius of smoothing (folder is 2*this value)
subset2 = '' # '_DeepeningDsqP' + ''

path = "/media/alex/Datapool" # "/Volumes/Cressida"
inpath = path+"/CycloneTracking/tracking"+ver
outpath = inpath+"/"+bboxnum

'''*******************************************
Define Variables
*******************************************'''
print("Defining variables")

ymin, ymax = 1981, 2010 # years for climatology

dateref = [1900,1,1,0,0,0] # [Y,M,D,H,M,S]
seasons = np.array([1,2,3,4,5,6,7,8,9,10,11,12]) # Ending month for three-month seasons (e.g., 2 = DJF)

# Variables (note that counts are mandatory if using them as the weight for non-count variables)
vNames =  ['tp','e','viwvd'] #

'''*******************************************
Main Analysis
*******************************************'''
print("Main Analysis")
# Ensure that folders exist to store outputs
try:
    os.chdir(outpath+"/Aggregation"+typ+subset2+"/"+str(kSizekm*2)+"km")
except:
    os.mkdir(outpath+"/Aggregation"+typ+subset2+"/"+str(kSizekm*2)+"km")
    os.chdir(outpath+"/Aggregation"+typ+subset2+"/"+str(kSizekm*2)+"km")

# Read in attributes of reference files
params = pd.read_pickle(inpath+"/cycloneparams.pkl")
# params = pickle5.load(open(inpath+"/cycloneparams.pkl",'rb'))

try:
    spres = params['spres']
except:
    spres = 100
    
if int(ver.split('_')[0]) < 14:
    prjpath = path+"/Projections/EASE2_N0_"+str(spres)+"km_Projection_uv.nc"
else:
    prjpath = path+"/Projections/EASE2_N0_"+str(spres)+"km_Projection.nc"

proj = xr.open_dataset(prjpath)
size = proj['lat'].shape

YY2 = str(ymin) + "-" + str(ymax)

##################################
##### SEASONAL CLIMATOLOGIES #####
##################################
print("Aggregation By Season")
sname = ver+"_AggregationFields_SeasonalClimatology_"+YY2+".nc"
if sname in md.listdir(outpath+"/Aggregation"+typ+subset2+"/"+str(kSizekm*2)+"km"):
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

for v in range(len(vNames)):
    ncf = xr.open_dataset(ver+"_AggregationFields_Monthly_"+vNames[v]+".nc", decode_times=False)
    
    # Subset to Climatology Years (plus two months)
    times = ncf['time'][:]
    tsub = np.where( (times >= ymin-2/12) & (times < ymax+1) )[0]
    ncf = ncf.isel(time=slice(tsub[0],tsub[-1]+1))

    # Monthly --> Seasonal
    ncfseas = ncf.rolling(time=3,center=False).sum()
    ncfseas = ncfseas.isel(time=slice(2,None))
    
    # Add Ratios
    if vNames[v] in ['e','viwvd']:
        vNames2 = [vNames[v]+'-pos',vNames[v]+'-neg']
    else:
        vNames2 = [vNames[v]]
    
    for vv in range(len(vNames2)):
        ratio = ncfseas[vNames2[vv]+'-cyc'].data / ncfseas[vNames2[vv]+'-all'].data
        ratio[np.where(np.isposinf(ratio))] = 1
        ratio[np.where(np.isneginf(ratio))] = np.nan
        ncfseas[vNames2[vv]+'-cycratio'] = (('time','y','x'),ratio)
    
    # Seasonal Climatology
    endmonths = (np.round((ncfseas['time'].data - np.floor(ncfseas['time'].data) ) *12,0)+1).astype(int)
    ncfseas = ncfseas.assign_coords({'endmonth':('time', endmonths)})
    ncfseas = ncfseas.swap_dims({'time':'endmonth'})
    seasclim = ncfseas.groupby('endmonth').mean()
    
    for vv in range(len(vNames2)):
        try:
            snc.createVariable(vNames2[vv]+'-cyc', np.float64, ('time','y','x'))
            snc[vNames2[vv]+'-cyc'].units = ncf[vNames2[vv]+'-cyc'].units
            snc[vNames2[vv]+'-cyc'][:] = seasclim[vNames2[vv]+'-cyc'].data
            
            snc.createVariable(vNames2[vv]+'-all', np.float64, ('time','y','x'))
            snc[vNames2[vv]+'-all'].units = ncf[vNames2[vv]+'-all'].units
            snc[vNames2[vv]+'-all'][:] = seasclim[vNames2[vv]+'-all'].data

            snc.createVariable(vNames2[vv]+'-cycratio', np.float64, ('time','y','x'))
            snc[vNames2[vv]+'-cycratio'].units = 'ratio of ' + ncf[vNames2[vv]+'-cyc'].units + ' during cyclone hours to that during all hours'
            snc[vNames2[vv]+'-cycratio'][:] = seasclim[vNames2[vv]+'-cycratio'].data
            
        except:
            snc[vNames2[vv]+'-cyc'][:] = seasclim[vNames2[vv]+'-cyc'].data
            snc[vNames2[vv]+'-all'][:] = seasclim[vNames2[vv]+'-all'].data
            snc[vNames2[vv]+'-cycratio'][:] = seasclim[vNames2[vv]+'-cycratio'].data


snc.close()
ncf.close()
