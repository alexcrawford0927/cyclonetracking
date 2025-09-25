#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Date Created: 2023 May 26
Date Modified: 2025 Feb 13
Author: Alex Crawford

Purpose: Calculate monthly values for an ERA5 variable when a cyclone is present
(i.e., within radius "rad" from a given gridcell).
"""

'''**********
Load Modules
**********'''
import xarray as xr
import xesmf as xe
import CycloneModule_13_3 as md
import pandas as pd
import numpy as np
from scipy import interpolate
import warnings
import os
# warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

'''**********
Define Variables
**********'''
starttime = [1978,11,1,0,0,0] # inclusive
endtime = [2025,1,1,0,0,0] # exclusive
filetimestep = [0,1,0,0,0,0] # time step between cyclone files
inittime = [1940,1,1,0,0,0] # Initiation time for cyclone tracking dataset

dataset = 'ERA5'
cycver = '13_2E5R'
subset = 'BBox10'
typ = 'System'

vi = 1
rad = 800 # radius for constant kernel size (units: km)
minlat = 20 # Cyclone must reach this latitude at least once
minlat2 = 15 # Trim any location that is south of this latitude

path = '/media/alex/Datapool'
apath = path+'/'+dataset #'/Volumes/Telemachus/'+dataset #
apath2 =path+'/'+dataset #'/Volumes/Prospero/'+dataset #
ppath = path+'/Projections' # '/Volumes/Cressida/Projections'
cpath = path+'/CycloneTracking/tracking'+cycver
# cpath = '/Volumes/Cressida/CycloneTracking/tracking'+cycver+'/'+subset+'/'+typ+'Tracks'
outpath = path+"/CycloneTracking/tracking"+cycver+"/"+subset+"/Aggregation"+typ+"/"+str(2*rad)+'km'
# outpath = "/Volumes/Cressida/CycloneTracking/tracking"+cycver+"/"+subset+"/Aggregation"+typ+"/MoistureBudget_"+str(rad)

# Use the _uv suffix if cycver < 14
if int(cycver.split('_')[0]) < 14:
    regridname = 'ERA5_NH_to_EASE_NH_25km_uv.nc'
    regridnnname = 'ERA5_NH_to_EASE_NH_25km_uv_nearestneighbor.nc'
    outprjname = 'EASE2_N0_25km_Projection_uv.nc'
    xvarname, yvarname = 'u', 'v'
    
else:
    regridname = 'ERA5_NH_to_EASE_NH_25km.nc'
    regridnnname = 'ERA5_NH_to_EASE_NH_25km_nearestneighbor.nc'
    outprjname = 'EASE2_N0_25km_Projection.nc'
    xvarname, yvarname = 'x', 'y'

inprjname = 'ERA5_NH_Projection.nc'
landseamaskname = apath2+'/Invariant/'+dataset+'_LandSeaMask.nc'

# Variable Names
filevars = ['Evaporation','MoistureFluxDiv','Radiation_TOA','Radiation_TOA']
filetres = ['Hourly','Hourly','Hourly']
ncvars = ['e','viwvd','ttr','tsr']
# multiplier = [1000,3600,1,1] # to make the units mm (i.e., kg/m^2) of water -- and summed over an hour for fluxes
multiplier = [1,1,1,1]
longnames = ['Sum of Evaporation','Sum of Vertical Integral of Moisture Flux Divergence','Average TOA Net Longwave Radiation','Average TOA Net Shortwave Radiation']
units = ['total mm of liquid water equivalent','total mm of liquid water equivalent','total Joules per sq. m','total Joules per sq. m']

'''**********
Main Analysis
**********'''
print("Setting up regridders")
# Set up a regridder (load or create as necessary)
outprj = xr.open_dataset(ppath+"/"+outprjname)
inprj = xr.open_dataset(ppath+"/"+inprjname)
try:
    regridder = xe.Regridder(inprj, outprj, 'bilinear', periodic=True, weights=xr.open_dataset(ppath+"/Regridders/"+regridname))

except:
    regridder = xe.Regridder(inprj, outprj, 'bilinear', periodic=True)
    regridder.to_netcdf(ppath+"/"+regridname)

# Define kernel
spres = pd.read_pickle(cpath+"/cycloneparams.pkl")['spres']
dateref = pd.read_pickle(cpath+"/cycloneparams.pkl")['dateref']
kernel = md.circleKernel(int(rad/spres),masked_value=0)

# Prep Outputs
monthlylists = [[] for i in range(4)] # pos-cyc, neg-cyc, pos-all, neg-all
monthlytimelist = []

mt = starttime+[]
mt2 = md.timeAdd(mt,filetimestep)
xdlist = []
while mt != endtime:
    Y, M = str(mt[0]), md.dd[mt[1]-1]
    Y2, M2 = str(mt2[0]), md.dd[mt2[1]-1]
    print('Loading ' +str(mt))

    # Load fields for the given month -- and reproject
    try:
        ds = xr.open_dataset(apath+"/"+filevars[vi]+"/"+"_".join([dataset,filevars[vi],filetres[vi],str(mt[0])+md.dd[mt[1]-1]+'.nc']))
    except:
        ds = xr.open_dataset(apath2+"/"+filevars[vi]+"/"+"_".join([dataset,filevars[vi],filetres[vi],str(mt[0])+md.dd[mt[1]-1]+'.nc']))
    ds = regridder(ds)
    
    # Separate Positive and Negative Values
    negall = np.where(ds[ncvars[vi]].data <= 0, ds[ncvars[vi]].data, 0)
    posall = np.where(ds[ncvars[vi]].data >= 0, ds[ncvars[vi]].data, 0)

    dates = np.array((ds.time.dt.year,ds.time.dt.month,ds.time.dt.day,ds.time.dt.hour)).transpose()
    times = [md.daysBetweenDates(dateref,list(date)+[0,0]) for date in dates]

    # Load cyclone for current month
    cts  = pd.read_pickle(cpath+'/'+subset+'/'+typ+'Tracks'+"/"+Y+"/"+subset+typ.lower()+"tracks"+Y+M+".pkl")
    cts = [ct for ct in cts if (ct.data.lat.max() >= minlat)]

    # Load cyclone for next month
    try:
        cts2 = pd.read_pickle(cpath+'/'+subset+'/'+typ+'Tracks'+"/"+Y2+"/"+subset+typ.lower()+"tracks"+Y2+M2+".pkl")
        cts2 = [ct for ct in cts2 if (ct.data.lat.max() >= minlat)]
        cts = cts + cts2
    except:
        cts = cts + []
        
    # Initiate empty array
    outarrs = [np.zeros_like(ds[ncvars[vi]]) for i in range(2)]

    print('Processing ' +str(mt))
    # Create dataframe for cyclone that includes relevant values for all fields
    for ct in cts:
        ### Trim Cyclone based on minlat2 ###
        ct.data = ct.data.loc[(ct.data.lat >= minlat2) & (ct.data.time >= times[0]) & (ct.data.time <= times[-1])]

        ### Identify locations -- interpolate if needed ###
        xs = np.array(ct.data.x)
        ys = np.array(ct.data.y)
        hours = np.array(ct.data.time*24)
        
        if len(hours) > 0:
            # Interpolate to hourly if needed
            if filetres[vi] == 'Hourly':
                f = interpolate.interp1d(hours,xs)
                xs = f(np.arange(hours[0],hours[-1])).astype(int)
                f = interpolate.interp1d(hours,ys)
                ys = f(np.arange(hours[0],hours[-1])).astype(int)
                hours = np.arange(hours[0],hours[-1])
    
            ### Calculate Average or Sum Within Cyclone Area ###
            for h in range(len(hours))[:]:
                # Identitfy location indices
                y, x = ys[h], xs[h] # spatial indices
                t = np.where(times == hours[h]/24)[0][0]
    
                # Mask Creation
                ky, kx = (np.array(kernel.shape)/2).astype(int)
    
                # Sum for pos-cyc & neg-cyc
                outarrs[0][t,(y-ky):(y+ky+1),(x-kx):(x+kx+1)] = kernel*posall[t,(y-ky):(y+ky+1),(x-kx):(x+kx+1)]*multiplier[vi]
                outarrs[1][t,(y-ky):(y+ky+1),(x-kx):(x+kx+1)] = kernel*negall[t,(y-ky):(y+ky+1),(x-kx):(x+kx+1)]*multiplier[vi]
        
    # Calculate Monthly Values
    monthlylists[0].append( outarrs[0].sum(axis=0) )
    monthlylists[1].append( outarrs[1].sum(axis=0) )
    monthlylists[2].append( posall.sum(axis=0) )
    monthlylists[3].append( negall.sum(axis=0) )

    # Include count variable and time variable
    monthlytimelist.append(float(ds.time[0].dt.year + (ds.time[0].dt.month-1)/12))
    
    mt = mt2+[]
    mt2 = md.timeAdd(mt2,filetimestep)
    
### Write to File ###


## If this variable already exists and we're just replacing part of it ##
if cycver+"_AggregationFields_Monthly_"+ncvars[vi]+".nc" in md.listdir(outpath):
    # outds = xr.open_dataset(outpath+"/"+cycver+"_AggregationFields_Monthly_"+ncvars[vi][i]+".nc")

    # outarr = outds[ncvars[vi]].data
    # outarr[np.in1d(outds['time'], monthlytimelist),:,:] = monthlylists[i]
    # outds[ncvars[vi][i]] = (['time','y','x'], outarr)
    # outds[ncvars[vi][i]].attrs = {'long_name':longnames[vi][i], 'units':units[vi][i]}
    
    # outds.to_netcdf(outpath+"/"+cycver+"_AggregationFields_Monthly_"+ncvars[vi][i]+"_NEW.nc")
    # outds.close()
    # os.remove(outpath+"/"+cycver+"_AggregationFields_Monthly_"+ncvars[vi][i]+".nc")
    # os.rename(outpath+"/"+cycver+"_AggregationFields_Monthly_"+ncvars[vi][i]+"_NEW.nc",outpath+"/"+cycver+"_AggregationFields_Monthly_"+ncvars[vi][i]+".nc")

    print("WARNING THIS PART INCOMPLETE --- file not saved yet")
## If this is a new variable ##
else:
    outds = xr.Dataset({'lat':(['y','x'],outprj.lat.values),
                        'lon':(['y','x'],outprj.lon.values)},
        coords={'time':np.array(monthlytimelist),
                        'y':outprj[yvarname].values,
                        'x':outprj[xvarname].values})
    
    # Cyc precip
    outds[ncvars[vi]+"-pos-cyc"] = (['time','y','x'], np.array(monthlylists[0]))
    outds[ncvars[vi]+"-pos-cyc"].attrs = {'long_name':longnames[vi]+ " within " + str(rad) + " km of a cyclone center when positive", 'units':units[vi]}

    outds[ncvars[vi]+"-neg-cyc"] = (['time','y','x'], np.array(monthlylists[1]))
    outds[ncvars[vi]+"-neg-cyc"].attrs = {'long_name':longnames[vi]+ " within " + str(rad) + " km of a cyclone center when negative", 'units':units[vi]}
            
    # Total precip
    outds[ncvars[vi]+"-pos-all"] = (['time','y','x'], np.array(monthlylists[2]))
    outds[ncvars[vi]+"-pos-all"].attrs = {'long_name':longnames[vi] + " when positive", 'units':units[vi]}        

    outds[ncvars[vi]+"-neg-all"] = (['time','y','x'], np.array(monthlylists[3]))
    outds[ncvars[vi]+"-neg-all"].attrs = {'long_name':longnames[vi] + " when negative", 'units':units[vi]}        
            
    outds.to_netcdf(outpath+"/"+cycver+"_AggregationFields_Monthly_"+ncvars[vi]+".nc")
    outds.close()
    
print("Complete")