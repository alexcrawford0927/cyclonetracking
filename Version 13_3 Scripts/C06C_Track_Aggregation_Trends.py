'''
Author: Alex Crawford
Date Created: 24 Jan 2017
Date Modified: 13 May 2020 - Updated for Python 3
               13 Nov 2020 - Updated for netcdf instead of gdal
               13 Oct 2021 - Added tracking of the smoothing size and minimum number of years to outputs
Purpose: Calculate trends for aggergated cyclone fields (from C4 script) on 
monthly and seasonal scales.

User inputs:
    Path Variables
    Track Type (typ): Cyclone or System
    Bounding Box ID (bboxnum): 2-digit character string
    Time Variables: when to start, end, the time step of the data
'''

'''********************
Import Modules
********************'''
print("Loading modules.")
import os
import numpy as np
import netCDF4 as nc
import xarray as xr
import CycloneModule_13_3 as md
import scipy.stats as stats
from scipy import ndimage

'''*******************************************
Set up Environment
*******************************************'''
print("Setting up environment.")
bboxnum = "" # use "" if performing on all cyclones; or BBox##
typ = "System"
newspres = '' # Leave empty unless using a reduced-resolution version, e.g., "_100km", "_50km", etc.
ver = "13testP"
rad = 800 # Radius of smoothing (folder is 2*this value)

path = "/media/alex/Datapool" #"/Volumes/Cressida"
inpath = path+"/CycloneTracking/tracking"+ver+"/"+bboxnum+"/Aggregation"+typ+"/"+str(rad*2)+"km"

'''*******************************************
Define Variables
*******************************************'''
print("Defining variables")
# File Variables
minN = 10

# Time Variables 
mos = range(1,12+1)
ymin, ymax = 1981 , 2010

# Variables

vNames = ['trkden','gen','lys'] + ["countU"] + ["DpDt","u","v"] + ['uab','vab','uvab','vratio','mci'] + ['countP'] + ['depth','p_cent','DsqP','radius']
aggtype = ['sum','sum','sum'] + ['sum'] + ['mean','mean','mean'] + ['mean','mean','mean','mean','mean'] + ['sum'] + ['mean','mean','mean','mean']

'''*******************************************
Main Analysis
*******************************************'''
print("Main Analysis")   
YY = "_"+str(ymin) + "-" + str(ymax)+"_"

## MONTHLY TRENDS ###
print("Step 1. Trends By Month")
vlist, vunits = [[] for i in range(len(vNames))], [[] for i in range(len(vNames))]

# Perform trend analysis for other variables
for v in range(len(vNames)):
    print("--Variable: "+vNames[v])
    
    ncf = xr.open_dataset(inpath+"/"+ver+"_AggregationFields_Monthly"+newspres+"_"+vNames[v]+".nc", decode_times=False)
    times = ncf['time'][:]

    field = ncf[vNames[v]][:].data
    
    # Trend Analysis
    blist, alist, rlist, plist, elist = [], [], [], [], []
    for m in mos:
        tsub = np.where( (np.round((times-((m-1)/12))%1,3) == 0) & (times >= ymin) & (times < ymax+1) )[0]
    
        varr = field[tsub,:,:]
        
        # Perform a linear regression
        b, a, r2, pvalue, stderr, stderra = md.lm(np.arange(ymin,ymax+1),varr,minN)
        
        blist.append(b), alist.append(a), rlist.append(r2), plist.append(pvalue), elist.append(stderr)
    
    vlist[v] = [blist,alist,rlist,plist,elist]
    vunits[v] = ncf[vNames[v]].units

## Write to File ##
mname = ver+"_Trend"+YY+"Fields_Monthly"+newspres+"_MinYrs"+str(minN)+".nc"
if mname in os.listdir(inpath): 
    mnc = nc.Dataset(inpath+"/"+mname,'r+')
else:
    mnc = nc.Dataset(inpath+"/"+mname,'w')
    mnc.createDimension('y', ncf['y'].shape[0])
    mnc.createDimension('x', ncf['x'].shape[0])
    mnc.createDimension('time', 12)
    mnc.description = 'Trend ('+YY+') of aggregation of cyclone track characteristics on monthly time scale (minimum number of years required: ' + str(minN)+').'
    
    ncy = mnc.createVariable('y', np.float32, ('y',))
    ncx = mnc.createVariable('x', np.float32, ('x',))
    ncy.units, ncx.units = 'm', 'm'
    ncy[:] = ncf['y'][:].data
    ncx[:] = ncf['x'][:].data
    
    # Add times, lats, and lons
    nctime = mnc.createVariable('time', np.float32, ('time',))
    nctime.units = 'months'
    nctime[:] = np.arange(1,12+1,1)
    
    nclon = mnc.createVariable('lon', np.float32, ('y','x'))
    nclon.units = 'degrees'
    nclon[:] = ncf['lon'][:].data
    
    nclat = mnc.createVariable('lat', np.float32, ('y','x'))
    nclat.units = 'degrees'
    nclat[:] = ncf['lat'][:].data

ncf.close()

for v in range(len(vNames)):
    # Trend
    try:
        ncvar = mnc.createVariable(vNames[v]+"_trend", np.float64, ('time','y','x'))
        ncvar.units = vunits[v]+"/yr"
        ncvar[:] = np.array(vlist[v][0])
    except:
        mnc[vNames[v]+"_trend"][:] = np.array(vlist[v][0])
    
    # Y-Intercept
    try:
        ncvar = mnc.createVariable(vNames[v]+"_yint", np.float64, ('time','y','x'))
        ncvar.units = vunits[v]
        ncvar[:] = np.array(vlist[v][1])
    except:
        mnc[vNames[v]+"_yint"][:] = np.array(vlist[v][1])

    # R^2
    try:
        ncvar = mnc.createVariable(vNames[v]+"_r2", np.float64, ('time','y','x'))
        ncvar.units = "ratio"
        ncvar[:] = np.array(vlist[v][2])
    except:
        mnc[vNames[v]+"_r2"][:] = np.array(vlist[v][2])

    # P-Value
    try:
        ncvar = mnc.createVariable(vNames[v]+"_pvalue", np.float64, ('time','y','x'))
        ncvar.units = "ratio"
        ncvar[:] = np.array(vlist[v][3])
    except:
        mnc[vNames[v]+"_pvalue"][:] = np.array(vlist[v][3])

    # Std Err
    try:
        ncvar = mnc.createVariable(vNames[v]+"_sdtrend", np.float64, ('time','y','x'))
        ncvar.units = vunits[v]+"/yr"
        ncvar[:] = np.array(vlist[v][4])
    except:
        mnc[vNames[v]+"_sdtrend"][:] = np.array(vlist[v][4])

mnc.close()

### SEASONAL TRENDS ###

print("Step 2. Trends By 3-Month Season")
vlist, vunits = [[] for i in range(len(vNames))], [[] for i in range(len(vNames))]

for v in range(len(vNames)):
    print("--Variable: "+vNames[v])
    ncf = xr.open_dataset(inpath+"/"+ver+"_AggregationFields_Monthly"+newspres+"_"+vNames[v]+".nc", decode_times=False)
    times = ncf['time'][:]
    
    vunits[v] = ncf[vNames[v]].units
    
    blist, alist, rlist, plist, elist = [], [], [], [], []
    for m in mos: # by ending month
        tsub = np.where( (np.round((times-((m-1)/12))%1,3) == 0) & (times >= ymin) & (times < ymax+1) )[0]
    
        varr3 = ncf[vNames[v]][tsub,:,:].data
        varr2 = ncf[vNames[v]][tsub-1,:,:].data
        varr1 = ncf[vNames[v]][tsub-2,:,:].data
        
        # Aggregate month to season
        if aggtype[v] == 'sum':
            svarr = varr1 + varr2 + varr3
        else:
            varr = np.array([varr1,varr2,varr3])
            narr = np.isfinite( varr )
            varr[narr == 0] = 0
            
            svarr = varr.sum(axis=0) / narr.sum(axis=0)

        # Perform a linear regression
        b, a, r2, pvalue, stderr, stderra = md.lm(np.arange(ymin,ymax+1,1),svarr,minN)
        
        blist.append(b), alist.append(a), rlist.append(r2), plist.append(pvalue), elist.append(stderr)
    
    vlist[v] = [blist,alist,rlist,plist,elist]

## Write to File ##
sname = ver+"_Trend"+YY+"Fields_Seasonal"+newspres+"_MinYrs"+str(minN)+".nc"
if sname in os.listdir(inpath):
    snc = nc.Dataset(inpath+"/"+sname,'r+')

else:
    snc = nc.Dataset(inpath+"/"+sname,'w')
    snc.createDimension('y', ncf['y'].shape[0])
    snc.createDimension('x', ncf['x'].shape[0])
    snc.createDimension('time', 12)
    snc.description = 'Trend ('+YY+') of aggregation of cyclone track characteristics on seasonal time scale (Notes: minimum number of years required: ' + str(minN) + '; count data is summed for each season).'
    
    ncy = snc.createVariable('y', np.float32, ('y',))
    ncx = snc.createVariable('x', np.float32, ('x',))
    ncy.units, ncx.units = 'm', 'm'
    ncy[:] = ncf['y'][:].data
    ncx[:] = ncf['x'][:].data
    
    # Add times, lats, and lons
    nctime = snc.createVariable('time', np.float32, ('time',))
    nctime.units = 'ending months'
    nctime[:] = np.arange(1,12+1,1)
    
    nclon = snc.createVariable('lon', np.float32, ('y','x'))
    nclon.units = 'degrees'
    nclon[:] = ncf['lon'][:].data
    
    nclat = snc.createVariable('lat', np.float32, ('y','x'))
    nclat.units = 'degrees'
    nclat[:] = ncf['lat'][:].data

ncf.close()

for v in range(len(vNames)):
    # Trend
    try:
        ncvar = snc.createVariable(vNames[v]+"_trend", np.float64, ('time','y','x'))
        ncvar.units = vunits[v]+"/yr"
        ncvar[:] = np.array(vlist[v][0])
    except:
        snc[vNames[v]+"_trend"][:] = np.array(vlist[v][0])
    
    # Y-Intercept
    try:
        ncvar = snc.createVariable(vNames[v]+"_yint", np.float64, ('time','y','x'))
        ncvar.units = vunits[v]
        ncvar[:] = np.array(vlist[v][1])
    except:
        snc[vNames[v]+"_yint"][:] = np.array(vlist[v][1])

    # R^2
    try:
        ncvar = snc.createVariable(vNames[v]+"_r2", np.float64, ('time','y','x'))
        ncvar.units = "ratio"
        ncvar[:] = np.array(vlist[v][2])
    except:
        snc[vNames[v]+"_r2"][:] = np.array(vlist[v][2])

    # P-Value
    try:
        ncvar = snc.createVariable(vNames[v]+"_pvalue", np.float64, ('time','y','x'))
        ncvar.units = "ratio"
        ncvar[:] = np.array(vlist[v][3])
    except:
        snc[vNames[v]+"_pvalue"][:] = np.array(vlist[v][3])

    # Std Err
    try:
        ncvar = snc.createVariable(vNames[v]+"_sdtrend", np.float64, ('time','y','x'))
        ncvar.units = vunits[v]+"/yr"
        ncvar[:] = np.array(vlist[v][4])
    except:
        snc[vNames[v]+"_sdtrend"][:] = np.array(vlist[v][4])

snc.close()

print('Complete')