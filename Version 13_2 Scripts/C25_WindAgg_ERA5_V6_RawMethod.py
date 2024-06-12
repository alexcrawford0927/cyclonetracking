#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Date Created: 2023 May 26
Date Modified: 2023 Aug 30

Purpose: Aggregates average and maximum wind speed within a specified distance
of a cyclone center

New in V6: Added an if statement for using different EASE2 grids depending on 
cyclone data version (2023 Aug 30)
"""

'''**********
Load Modules
**********'''
import xarray as xr
import xesmf as xe
import CycloneModule_13_2 as md
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

'''**********
Define Variables
**********'''
starttime = [1978,11,1,0,0,0] # inclusive
endtime = [2022 ,1,1,0,0,0] # exclusive
filetimestep = [0,1,0,0,0,0] # time step between cyclone files
inittime = [1940,1,1,0,0,0] # Initiation time for cyclone tracking dataset
dateref = [1900,1,1,0,0,0] # Reference date for input data

dataset = 'ERA5'
cycver = '13_2E5R'
subset = 'BBox10'
typ = 'System'

V = 'V6'
rad = 800 # radius for constant kernel size (units: km)
minlat = 20 # Cyclone must reach this latitude at least once
minlat2 = 15 # Trim any location that is south of this latitude
level = 850 # Pressure level in hPa

# Variable Names
filevar = 'HorizontalWind_LowLevel'
filetres = '3h'
ncvars = ['uv']

multiplier = [1]

# Path Names
apath = '/Volumes/Telemachus/'+dataset # '/Users/acrawfora/Documents/'+dataset #
apath2 ='/Volumes/Prospero/'+dataset #
cpath = '/Volumes/Cressida/CycloneTracking/tracking'+cycver+'/'+subset+'/'+typ+'Tracks'
ppath = '/Volumes/Cressida/Projections'
outpath = "/Volumes/Cressida/CycloneTracking/tracking"+cycver+"/"+subset+"/Aggregation"+typ+"/"+filevar.split('_')[0]+str(level)+"_"+str(rad)

# Use the _uv suffix if cycver < 14
if int(cycver.split('_')[0]) < 14:
    regridname = 'ERA5_NH_to_EASE_NH_25km_uv.nc'
    outprjname = 'EASE2_N0_25km_Projection_uv.nc'
else:
    regridname = 'ERA5_NH_to_EASE_NH_25km.nc'
    outprjname = 'EASE2_N0_25km_Projection.nc'
inprjname = 'ERA5_NH_Projection.nc'

'''**********
Main Analysis
**********'''
print("Setting up regridders")
# Set up a regridder (load or create as necessary)
outprj = xr.open_dataset(ppath+"/"+outprjname)
inprj = xr.open_dataset(ppath+"/"+inprjname)
try:
    regridder = xe.Regridder(inprj, outprj, 'bilinear', weights=xr.open_dataset(ppath+"/"+regridname))

except:
    regridder = xe.Regridder(inprj, outprj, 'bilinear')
    regridder.to_netcdf(ppath+"/"+regridname)

# Define kernel
spres = pd.read_pickle('/Volumes/Cressida/CycloneTracking/tracking'+cycver+"/cycloneparams.pkl")['spres']
kernel = md.circleKernel(int(rad/spres))
nanmask = outprj['z'].data*np.nan

# Check if there is a prior month, and if so, load and reproject those fields
if starttime != inittime:
    print("Loading prior month's data for reference")
    mt0 = md.timeAdd(starttime,[-1*f for f in filetimestep])
    try:
        ds0 = xr.open_dataset(apath+"/"+filevar+"/"+"_".join([dataset,filevar,filetres,str(mt0[0])+md.dd[mt0[1]-1]+'.nc']))
    except:
        ds0 = xr.open_dataset(apath2+"/"+filevar+"/"+"_".join([dataset,filevar,filetres,str(mt0[0])+md.dd[mt0[1]-1]+'.nc']))
    ds0 = ds0.rename({'u':'zonal','v':'meridional'})
    ds0 = regridder(ds0.sel({'level':level}))
    ds0['uv'] = np.sqrt( np.square(ds0['zonal']) + np.square(ds0['meridional']) )

mt = starttime+[]
while mt != endtime:
    Y, M = str(mt[0]), md.dd[mt[1]-1]
    print('Loading ' +str(mt))

    # Load fields for the given month -- and reproject
    try:
        ds = xr.open_dataset(apath+"/"+filevar+"/"+"_".join([dataset,filevar,filetres,str(mt[0])+md.dd[mt[1]-1]+'.nc']))
    except:
        ds = xr.open_dataset(apath2+"/"+filevar+"/"+"_".join([dataset,filevar,filetres,str(mt[0])+md.dd[mt[1]-1]+'.nc']))
    
    ds = ds.rename({'u':'zonal','v':'meridional'})
    ds = regridder(ds.sel({'level':level}))
    ds['uv'] = np.sqrt( np.square(ds['zonal']) + np.square(ds['meridional']) )

    # Load cyclone for current month
    cts  = pd.read_pickle(cpath+"/"+Y+"/"+subset+typ.lower()+"tracks"+Y+M+".pkl")
    cts = [ct for ct in cts if ct.data.lat.max() >= minlat]

    print('Processing ' +str(mt))

    # Create dataframe for cyclone that includes relevant values for all fields
    pdf = pd.DataFrame() # Initiate output dataframe
    for ct in cts:
        data2 = pd.DataFrame() # Initiate intermediate dataframe

        ### Trim Cyclone based on minlat2 ###
        ct.data = ct.data.loc[ct.data.lat >= minlat2]

        ### Define Area of Cyclone At Each Time Step ###
        tlist = list(ct.data.time)
        datelist = [md.timeAdd(dateref,[0,0,tlist[ii],0,0,0]) for ii in range(len(tlist))]

        ### Calculate Average or Sum Within Cyclone Area ###
        avglists = [[] for ii in range(len(ncvars))]
        maxlists = [[] for ii in range(len(ncvars))]
        for i in range(len(datelist))[:]:
            # Identitfy location indices
            y, x = int(ct.data.iloc[i].y),int(ct.data.iloc[i].x) # spatial indices

            # Mask Creation
            ky, kx = (np.array(kernel.shape)/2).astype(int)
            mask = nanmask+0
            mask[(y-ky):(y+ky+1),(x-kx):(x+kx+1)] = kernel

            # For hourly fields, interpolate to improve precision of location
            y1 = int(np.round( 0.33*ct.data.iloc[i-1].y +  0.67*ct.data.iloc[i].y ))
            x1 = int(np.round( 0.33*ct.data.iloc[i-1].x +  0.67*ct.data.iloc[i].x ))
            y2 = int(np.round( 0.67*ct.data.iloc[i-1].y +  0.33*ct.data.iloc[i].y ))
            x2 = int(np.round( 0.67*ct.data.iloc[i-1].x +  0.33*ct.data.iloc[i].x ))
            y3, x3 = int(ct.data.iloc[i-1].y), int(ct.data.iloc[i-1].x)

            # Mask creation for hourly fields
            mask1, mask2, mask3 = nanmask+0, nanmask+0, nanmask+0
            mask1[(y1-ky):(y1+ky+1),(x1-kx):(x1+kx+1)] = kernel
            mask2[(y2-ky):(y2+ky+1),(x2-kx):(x2+kx+1)] = kernel
            mask3[(y3-ky):(y3+ky+1),(x3-kx):(x3+kx+1)] = kernel

            # Aggregate time-varying fields
            date = datelist[i] # Calendar date for current synoptic time
            if mt[:2] == date[:2]: # If the cyclone time is in the current month
                # Identitfy time indices
                k3 = int(md.daysBetweenDates(mt, date)*8) # Index for 3h data
                k1 = int(k3*3) # for hourly data

                # Aggregate Integrated and Instantaneous Fields
                if filetres == 'Hourly':
                    for n in range(len(ncvars)):
                        avglists[n].append( (np.nanmean( ds[ncvars[n]].data[k1,:,:]*mask ) + np.nanmean( ds[ncvars[n]].data[k1-1,:,:]*mask1 ) + np.nanmean( ds[ncvars[n]].data[k1-2,:,:]*mask2 )) * multiplier[n] )
                        maxlists[n].append( np.nanmax( ds[ncvars[n]].data[(k1-2):(k1+1),:,:]*mask ) * multiplier[n] )
                else:
                    for n in range(len(ncvars)):
                        avglists[n].append( np.nanmean( ds[ncvars[n]].data[k3,:,:]*mask ) * multiplier[n] )
                        maxlists[n].append( np.nanmax( ds[ncvars[n]].data[k3,:,:]*mask ) * multiplier[n] )

            else: # If the cyclone time is in the prior month
                # Identitfy time indices
                k3 = int(md.daysBetweenDates(mt0, date)*8) # Index for 3h data
                k1 = int(k3*3) # for hourly data

                # Aggregate Integrated and Instantaneous Fields
                if filetres == 'Hourly':
                    for n in range(len(ncvars)):
                        avglists[n].append( (np.nanmean( ds0[ncvars[n]].data[k1,:,:]*mask ) + np.nanmean( ds0[ncvars[n]].data[k1-1,:,:]*mask1 ) + np.nanmean( ds0[ncvars[n]].data[k1-2,:,:]*mask2 )) * multiplier[n] )
                        maxlists[n].append( np.nanmax( ds0[ncvars[n]].data[(k1-2):(k1+1),:,:]*mask ) * multiplier[n] )
                else:
                    for n in range(len(ncvars)):
                        avglists[n].append( np.nanmean( ds0[ncvars[n]].data[k3,:,:]*mask ) * multiplier[n] )
                        maxlists[n].append( np.nanmax( ds0[ncvars[n]].data[k3,:,:]*mask ) * multiplier[n] )

        # Attach new columns to cyclone data frame
        data2['year'] = np.repeat(mt[0],len(avglists[0]))
        data2['month'] = mt[1]
        data2['sid'] = ct.sid
        data2['age'] =  (np.array(ct.data.time)[:] - np.array(ct.data.time)[0]) / ct.lifespan()

        for n in range(len(ncvars)):
            data2[ncvars[n]+"Avg"] = avglists[n] # Add average
            data2[ncvars[n]+"Max"] = maxlists[n] # Add maximum

        ### Attach cyclone data frame to main data frame ###
        pdf = pd.concat([pdf,data2],ignore_index=True,sort=False)

    # Write to File
    pdf.to_csv(outpath+"/Agg"+str(rad)+"km_"+dataset+"_"+filevar.split('_')[0]+str(level)+"_"+str(mt[0])+md.dd[mt[1]-1]+"_"+V+".csv",index=False)

    # Advance time step & and shift current month data to prior month status
    mt0 = mt+[]
    mt = md.timeAdd(mt,filetimestep)
    ds0 = [ds,0][0]

