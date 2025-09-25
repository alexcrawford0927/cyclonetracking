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
rad = 1200 # radius for constant kernel size (units: km)
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
# filevars = ['TCW','TCWV','CloudWater','Precipitation','Evaporation','MoistureFluxDiv','SIC']
# filetres = ['3h','3h','3h','Hourly','Hourly','Hourly','3h']
# ncvars = [['tcw'],['tcwv'],['tclw','tciw'],['tp'],['e'],['vivwd'],['siconc']]
# diffvari = [0,1,2]
# multiplier = [[1],[1],[1,1],[1000],[1000],[3600],[100]] # to make the units mm (i.e., kg/m^2) of water -- and summed over an hour for fluxes

filevars = ['TCW','Precipitation','Evaporation','MoistureFluxDiv','SIC','Radiation_TOA']
filetres = ['3h','Hourly','Hourly','Hourly','3h','Hourly']
ncvars = [['tcw'],['tp'],['e'],['viwvd'],['siconc'],['ttr','tsr']]
# multiplier = [[1],[1000],[1000],[3600],[100],[1/3600,1/3600]] # to make the units mm (i.e., kg/m^2) of water -- and summed over an hour for fluxes
multiplier = [[1],[1],[1],[1],[1],[1,1]]
longnames = [['Average Total Column Water'],['Sum of Total Precipitation'],['Sum of Evaporation'],['Sum of Vertical Integral of Moisture Flux Divergence'],['Average Sea Ice Concentration'],['Average TOA Net Longwave Radiation','Average TOA Net Shortwave Radiation']]
units = [['average mm of liquid water equivalent'], ['total mm of liquid water equivalent'], ['total mm of liquid water equivalent'], ['total mm of liquid water equivalent'],['average percent'],['average W per sq. m','average W per sq. m']]
minvals = [0,0.0001,-np.inf,-np.inf,0,-np.inf]
abvlab = ['0','0.1mm','NA','NA','0','NA']

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

for vi in [1]:
    # Prep Outputs
    monthlylists = [[] for i in range(1+len(ncvars[vi]))]
    monthlybaselists = [[] for i in range(1+len(ncvars[vi]))]
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
        ds[ncvars[vi][0]][:] = np.where(ds[ncvars[vi][0]][:] > minvals[vi], ds[ncvars[vi][0]][:], 0 )
    
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
        outarrs = [np.zeros_like(ds[ncvars[vi][i]]) for i in range(len(ncvars[vi]))]
        narr = np.zeros_like(ds[ncvars[vi][0]])
    
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
        
                    # Sum for the other fields
                    for i in range(len(ncvars[vi])):
                        outarrs[i][t,(y-ky):(y+ky+1),(x-kx):(x+kx+1)] = kernel*ds[ncvars[vi][i]].data[t,(y-ky):(y+ky+1),(x-kx):(x+kx+1)]*multiplier[vi][i]
            
        # Calculate Monthly Values
        for i in range(len(ncvars[vi])):
            monthlylists[i].append( outarrs[i].sum(axis=0) )
            monthlybaselists[i].append( ds[ncvars[vi][i]].data.sum(axis=0) )
    
        # Include count variable and time variable
        monthlytimelist.append(float(ds.time[0].dt.year + (ds.time[0].dt.month-1)/12))
        
        mt = mt2+[]
        mt2 = md.timeAdd(mt2,filetimestep)
        
    ### Write to File ###
    for i in range(len(ncvars[vi])):
        
        ## If this variable already exists and we're just replacing part of it ##
        if cycver+"_AggregationFields_Monthly_"+ncvars[vi][i]+".nc" in md.listdir(outpath):
            outds = xr.open_dataset(outpath+"/"+cycver+"_AggregationFields_Monthly_"+ncvars[vi][i]+".nc")
    
            # Cyc precip
            outds[ncvars[vi][i]+"abv"+abvlab[vi]+"-cyc"] = (['time','y','x'], np.array(monthlylists[i]))
            outds[ncvars[vi][i]+"abv"+abvlab[vi]+"-cyc"].attrs = {'long_name':longnames[vi][i]+ " within " + str(rad) + " km of a cyclone center", 'units':units[vi][i]}
            
            # Total precip
            outds[ncvars[vi][i]+"abv"+abvlab[vi]+"-all"] = (['time','y','x'], np.array(monthlybaselists[i]))
            outds[ncvars[vi][i]+"abv"+abvlab[vi]+"-all"].attrs = {'long_name':longnames[vi][i], 'units':units[vi][i]}        
            
            # Overwrite        
            outds.to_netcdf(outpath+"/"+cycver+"_AggregationFields_Monthly_"+ncvars[vi][i]+"_NEW.nc")
            outds.close()
            os.remove(outpath+"/"+cycver+"_AggregationFields_Monthly_"+ncvars[vi][i]+".nc")
            os.rename(outpath+"/"+cycver+"_AggregationFields_Monthly_"+ncvars[vi][i]+"_NEW.nc",outpath+"/"+cycver+"_AggregationFields_Monthly_"+ncvars[vi][i]+".nc")
    
        ## If this is a new variable ##
        else:
            outds = xr.Dataset({'lat':(['y','x'],outprj.lat.values),
                                'lon':(['y','x'],outprj.lon.values)},
                coords={'time':np.array(monthlytimelist),
                                'y':outprj[yvarname].values,
                                'x':outprj[xvarname].values})
            
            # Cyc precip
            outds[ncvars[vi][i]+"abv"+abvlab[vi]+"-cyc"] = (['time','y','x'], np.array(monthlylists[i]))
            outds[ncvars[vi][i]+"abv"+abvlab[vi]+"-cyc"].attrs = {'long_name':longnames[vi][i]+ " within " + str(rad) + " km of a cyclone center", 'units':units[vi][i]}
            
            # Total precip
            outds[ncvars[vi][i]+"abv"+abvlab[vi]+"-all"] = (['time','y','x'], np.array(monthlybaselists[i]))
            outds[ncvars[vi][i]+"abv"+abvlab[vi]+"-all"].attrs = {'long_name':longnames[vi][i], 'units':units[vi][i]}        
            
            outds.to_netcdf(outpath+"/"+cycver+"_AggregationFields_Monthly_"+ncvars[vi][i]+".nc")
            outds.close()
    
    print("Complete")