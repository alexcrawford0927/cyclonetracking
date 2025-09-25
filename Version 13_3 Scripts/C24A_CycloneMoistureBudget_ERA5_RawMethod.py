#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Date Created: 2023 May 26
Date Modified: 2023 Aug 30
Author: Alex Crawford

Purpose: Create a workflow to calculate a Lagrangian moisture budget around
A moving column of air, with the center of that moving column fixed to a 
cyclone center from a cyclone detection and tracking algorithm

New in V5: Using Evaportion instead of Latent Heat
New in V6: Added an if statement for using different EASE2 grids depending on 
cyclone data version (2023 Aug 30)
"""

'''**********
Load Modules
**********'''
import xarray as xr
import xesmf as xe
import CycloneModule_13_3 as md
import pandas as pd
import numpy as np
import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

'''**********
Define Variables
**********'''
starttime = [2023,2,1,0,0,0] # inclusive
endtime = [2025,1,1,0,0,0] # exclusive
filetimestep = [0,1,0,0,0,0] # time step between cyclone files
inittime = [1940,1,1,0,0,0] # Initiation time for cyclone tracking dataset

dataset = 'ERA5'
cycver = '13_2E5R'
subset = 'BBox10'
typ = 'System'

V = 'V7'
rad = 800 # radius for constant kernel size (units: km)
minlat = 20 # Cyclone must reach this latitude at least once
minlat2 = 15 # Trim any location that is south of this latitude

path = '/media/alex/Datapool'
apath = path+'/'+dataset #'/Volumes/Telemachus/'+dataset #
apath2 =path+'/'+dataset #'/Volumes/Prospero/'+dataset #
ppath = path+'/Projections' # '/Volumes/Cressida/Projections'
cpath = path+'/CycloneTracking/tracking'+cycver
# cpath = '/Volumes/Cressida/CycloneTracking/tracking'+cycver+'/'+subset+'/'+typ+'Tracks'
outpath = path+"/CycloneTracking/tracking"+cycver+"/"+subset+"/Aggregation"+typ+"/SpatialAvgEnv_"+str(rad)+'km'
# outpath = "/Volumes/Cressida/CycloneTracking/tracking"+cycver+"/"+subset+"/Aggregation"+typ+"/MoistureBudget_"+str(rad)

# Use the _uv suffix if cycver < 14
if int(cycver.split('_')[0]) < 14:
    regridname = 'ERA5_NH_to_EASE_NH_25km_uv.nc'
    regridnnname = 'ERA5_NH_to_EASE_NH_25km_uv_nearestneighbor.nc'
    outprjname = 'EASE2_N0_25km_Projection_uv.nc'
else:
    regridname = 'ERA5_NH_to_EASE_NH_25km.nc'
    regridnnname = 'ERA5_NH_to_EASE_NH_25km_nearestneighbor.nc'
    outprjname = 'EASE2_N0_25km_Projection.nc'
    
inprjname = 'ERA5_NH_Projection.nc'
landseamaskname = apath2+'/Invariant/'+dataset+'_LandSeaMask.nc'

# Variable Names
# filevars = ['TCW','TCWV','CloudWater','Precipitation','Evaporation','MoistureFluxDiv','SIC']
# filetres = ['3h','3h','3h','Hourly','Hourly','Hourly','3h']
# ncvars = [['tcw'],['tcwv'],['tclw','tciw'],['tp'],['e'],['vivwd'],['siconc']]
# diffvari = [0,1,2]
# multiplier = [[1],[1],[1,1],[1000],[1000],[3600],[100]] # to make the units mm (i.e., kg/m^2) of water -- and summed over an hour for fluxes

filevars = ['TCW','Precipitation','Evaporation','MoistureFluxDiv','SIC']
filetres = ['3h','Hourly','Hourly','Hourly','3h']
ncvars = [['tcw'],['tp'],['e'],['viwvd'],['siconc']]
diffvari = [0]
multiplier = [[1],[1000],[1000],[3600],[100]] # to make the units mm (i.e., kg/m^2) of water -- and summed over an hour for fluxes

cvars = ['x','y','lat','lon','p_cent','depth','radius','p_grad','DsqP','uv','DpDt']

'''**********
Main Analysis
**********'''
print("Setting up regridders")
# Set up a regridder (load or create as necessary)
outprj = xr.open_dataset(ppath+"/"+outprjname)
inprj = xr.open_dataset(ppath+"/"+inprjname)
try:
    regridder = xe.Regridder(inprj, outprj, 'bilinear', weights=xr.open_dataset(ppath+"/Regridders/"+regridname))

except:
    regridder = xe.Regridder(inprj, outprj, 'bilinear')
    regridder.to_netcdf(ppath+"/"+regridname)

try:
    regridder_nn = xe.Regridder(inprj, outprj, 'nearest_s2d', weights=xr.open_dataset(ppath+"/Regridders/"+regridnnname))
except:
    regridder_nn = xe.Regridder(inprj, outprj, 'nearest_s2d')
    regridder_nn.to_netcdf(ppath+'/'+regridnnname)

# Define kernel
spres = pd.read_pickle(cpath+"/cycloneparams.pkl")['spres']
dateref = pd.read_pickle(cpath+"/cycloneparams.pkl")['dateref']
kernel = md.circleKernel(int(rad/spres))
nanmask = outprj['z'].data*np.nan

# Load and reproject invariant data
landseamask = xr.open_dataset(landseamaskname)
landseamask = regridder_nn( landseamask.where(landseamask['latitude'] >= 0, drop=True) )['lsm'].data

# Check if there is a prior month, and if so, load and reproject those fields
if starttime != inittime:
    print("Loading prior month's data for reference")
    mt0 = md.timeAdd(starttime,[-1*f for f in filetimestep])
    xdlist0 = []
    for vi in range(len(filevars)):
        try:
            ds = xr.open_dataset(apath+"/"+filevars[vi]+"/"+"_".join([dataset,filevars[vi],filetres[vi],str(mt0[0])+md.dd[mt0[1]-1]+'.nc']))
        except:
            ds = xr.open_dataset(apath2+"/"+filevars[vi]+"/"+"_".join([dataset,filevars[vi],filetres[vi],str(mt0[0])+md.dd[mt0[1]-1]+'.nc']))
        xdlist0.append( regridder(ds) )
del ds

mt = starttime+[]
xdlist = []
while mt != endtime:
    Y, M = str(mt[0]), md.dd[mt[1]-1]
    print('Loading ' +str(mt))

    # Load fields for the given month -- and reproject
    for vi in range(len(filevars)):
        try:
            ds = xr.open_dataset(apath+"/"+filevars[vi]+"/"+"_".join([dataset,filevars[vi],filetres[vi],str(mt[0])+md.dd[mt[1]-1]+'.nc']))
        except:
            ds = xr.open_dataset(apath2+"/"+filevars[vi]+"/"+"_".join([dataset,filevars[vi],filetres[vi],str(mt[0])+md.dd[mt[1]-1]+'.nc']))
        xdlist.append( regridder(ds) )
    del ds

    # Load cyclone for current month
    cts  = pd.read_pickle(cpath+'/'+subset+'/'+typ+'Tracks'+"/"+Y+"/"+subset+typ.lower()+"tracks"+Y+M+".pkl")
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
        avglists = [[] for ii in range(len(sum(ncvars, []))+1)]
        difflists = [[] for ii in range(len(sum(ncvars, []))+1)]
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

            # Aggregate time-invariant fields
            avglists[0].append( np.nanmean(landseamask[0,:,:]*mask) )

            # Aggregate time-varying fields
            date = datelist[i] # Calendar date for current synoptic time
            if mt[:2] == date[:2]: # If the cyclone time is in the current month
                # Identitfy time indices
                k3 = int(md.daysBetweenDates(mt, date)*8) # Index for 3h data
                k1 = int(k3*3) # for hourly data

                # Aggregate Integrated and Instantaneous Fields
                j = 1
                for v in range(len(filevars)):
                    if filetres[v] == 'Hourly':
                        for n in range(len(ncvars[v])):
                            avglists[j].append( (np.nanmean( xdlist[v][ncvars[v][n]].data[k1,:,:]*mask ) + np.nanmean( xdlist[v][ncvars[v][n]].data[k1-1,:,:]*mask1 ) + np.nanmean( xdlist[v][ncvars[v][n]].data[k1-2,:,:]*mask2 )) * multiplier[v][n] )

                            j += 1
                    else:
                        for n in range(len(ncvars[v])):
                            avglists[j].append( np.nanmean( xdlist[v][ncvars[v][n]].data[k3,:,:]*mask ) * multiplier[v][n] )

                            j += 1

                # Aggregate Differenced Fields
                j = 1
                for v in diffvari:
                    if filetres[v] == 'Hourly':
                        for n in range(len(ncvars[v])):
                            difflists[j].append( avglists[j][-1] - (np.nanmean( xdlist[v][ncvars[v][n]].data[k1,:,:]*mask3 ) * multiplier[v][n]) )
                            j += 1
                    else:
                        for n in range(len(ncvars[v])):
                            difflists[j].append( avglists[j][-1] - (np.nanmean( xdlist[v][ncvars[v][n]].data[k3,:,:]*mask3 ) * multiplier[v][n]) )
                            j += 1


            else: # If the cyclone time is in the prior month
                # Identitfy time indices
                k3 = int(md.daysBetweenDates(mt0, date)*8) # Index for 3h data
                k1 = int(k3*3) # for hourly data

                # Aggregate Integrated and Instantaneous Fields
                j = 1
                for v in range(len(filevars)):
                    if filetres[v] == 'Hourly':
                        for n in range(len(ncvars[v])):
                            avglists[j].append( (np.nanmean( xdlist0[v][ncvars[v][n]].data[k1,:,:]*mask ) + np.nanmean( xdlist0[v][ncvars[v][n]].data[k1-1,:,:]*mask1 ) + np.nanmean( xdlist0[v][ncvars[v][n]].data[k1-2,:,:]*mask2 )) * multiplier[v][n] )

                            j += 1
                    else:
                        for n in range(len(ncvars[v])):
                            avglists[j].append( np.nanmean( xdlist0[v][ncvars[v][n]].data[k3,:,:]*mask ) * multiplier[v][n] )

                            j += 1

                # Aggregate Differenced Fields
                j = 1
                for v in diffvari:
                    if filetres[v] == 'Hourly':
                        for n in range(len(ncvars[v])):
                            difflists[j].append( avglists[j][-1] - (np.nanmean( xdlist0[v][ncvars[v][n]].data[k1,:,:]*mask3 ) * multiplier[v][n]) )
                            j += 1
                    else:
                        for n in range(len(ncvars[v])):
                            difflists[j].append( avglists[j][-1] - (np.nanmean( xdlist0[v][ncvars[v][n]].data[k3,:,:]*mask3 ) * multiplier[v][n]) )
                            j += 1

        # Attach new columns to cyclone data frame
        data2['year'] = np.repeat(mt[0],len(avglists[0]))
        data2['month'] = mt[1]
        data2['sid'] = ct.sid
        data2['time'] = ct.data.time.values
        data2['age'] =  (np.array(ct.data.time)[:] - np.array(ct.data.time)[0]) / ct.lifespan()
        data2['landfraction'] = avglists[0]
        
        for cvar in cvars:
            data2[cvar] = ct.data[cvar].values

        j = 1
        for v in range(len(filevars)):
            for n in range(len(ncvars[v])):
                data2[ncvars[v][n]] = avglists[j] # Add average

                if len(difflists[j]) > 0:
                    data2[ncvars[v][n]+"_Adv"] = difflists[j] # Add differencing

                j += 1

        ### Attach cyclone data frame to main data frame ###
        pdf = pd.concat([pdf,data2],ignore_index=True,sort=False)

    # Write to File
    pdf.to_csv(outpath+"/SpatialAvgEnv"+str(rad)+"km_"+str(mt[0])+md.dd[mt[1]-1]+".csv",index=False)

    # Advance time step & and shift current month data to prior month status
    mt0 = mt+[]
    mt = md.timeAdd(mt,filetimestep)
    xdlist0 = xdlist+[]
    xdlist = []

# TEST
# from scipy.stats import spearmanr as spearman

# rad = 1200
# V = '_V6' # '' #
# pdf = pd.read_csv("/Volumes/Cressida/CycloneTracking/tracking"+cycver+"/"+subset+"/Aggregation"+typ+"/MoistureBudget"+str(rad)+"/MoistureBudget_"+str(rad)+"km_"+str(starttime[0])+md.dd[starttime[1]-1]+V+".csv")
# # pdf = pd.read_csv("/Volumes/Cressida/CycloneTracking/tracking"+cycver+"/"+subset+"/Aggregation"+typ+"/MoistureBudgetAmly"+str(rad)+"/MoistureBudgetAmly_"+str(rad)+"km_"+str(starttime[0])+md.dd[starttime[1]-1]+V+".csv")
# pdf['tcw_tendency'] = np.concatenate( (np.array([np.nan]),np.array(pdf['tcw'])[1:] - np.array(pdf['tcw'])[:-1]) )

# print('\nUsing Integrated Latent Heat Flux')
# pdf['moisturefluxsum'] = -1 * (pdf['vimd'] + pdf['e'] + pdf['tp'])
# pdf['moisturefluxsum2'] = -1 * (pdf['vimd'] + pdf['e'] + pdf['tp']) + pdf['tcw_Adv']
# pdf['residual'] = pdf['tcw_tendency'] - pdf['moisturefluxsum2']
# valid = (np.isfinite(pdf['moisturefluxsum2'] + pdf['tcw_tendency']) & (pdf['age'] > 0) & (pdf['age'] < 1))

# print('w/ Advection: ' + str( np.round(spearman(pdf[valid]['tcw_tendency'],pdf[valid]['moisturefluxsum2'])[0],3)))
# print('w/out Advection: ' + str( np.round(spearman(pdf[valid]['tcw_tendency'],pdf[valid]['moisturefluxsum'])[0],3)))
