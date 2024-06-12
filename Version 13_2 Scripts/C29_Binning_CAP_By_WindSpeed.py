#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Date Created: 2023 Nov 2
Date Modified: 2023 Nov 2
Author: Alex Crawford

Purpose: 
(1) Calculate the CAP falling in a gridcell for each bin of cyclone intensity, 
as measured by max or avg wind speed around the cyclone
"""
'''**********
Load Modules
**********'''
import CycloneModule_13_2 as md
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import linregress 
import xarray as xr

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

'''**********
Define Variables
**********'''
dataset = 'ERA5'
cycver = '13_2E5R'
subset = 'BBox10'
typ = 'System'

VW = 'V6'
VM = 'V7'
V = 'V1'
rad = 400 # radius for constant kernel size (units: km)
windlev = 850 # pressure level for wind in hPa
spres = 25 # spatial resolution in km

capvar = 'tp'
windvar = 'uvAvg'
binwidth = 1.0
binmins = np.array([0] + list(np.arange(6.5,18.5,binwidth)))
minlat = 20 # Cyclone must reach this latitude at least once
minlat2 = 15 # Trim any location that is south of this latitude

starttime = [1978,11,1,0,0,0] # inclusive
endtime = [2022,1,1,0,0,0] # exclusive
filetimestep = [0,1,0,0,0,0] # time step between cyclone files
inittime = [1940,1,1,0,0,0] # Initiation time for cyclone tracking dataset
dateref = [1900,1,1,0,0,0] # Reference date for input data
mons = np.arange(1,13)

path = "/Volumes/Cressida/CycloneTracking/tracking"+cycver+"/"+subset
ctpath = "/Volumes/Cressida/CycloneTracking/tracking"+cycver+"/"+subset+"/"+typ+"Tracks"
mbpath = path+"/Aggregation"+typ+"/MoistureBudget_"+str(rad)+"/"+VM
windpath = path+"/Aggregation"+typ+"/HorizontalWind"+str(windlev)+"_"+str(rad)
outpath = path+"/Aggregation"+typ+"/"+str(int(rad*2))+"km/pr_BinBy_uv_"+V
outpath2 = path+"/Aggregation"+typ+"/"+str(int(rad*2))+"km"

if int(cycver.split('_')[0]) < 14:
    prjpath = "/Volumes/Cressida/Projections/EASE2_N0_"+str(spres)+"km_Projection_uv.nc"
else:
    prjpath = "/Volumes/Cressida/Projections/EASE2_N0_"+str(spres)+"km_Projection.nc"

'''**********
Main Analysis
**********'''
binmaxs = np.concatenate((binmins[1:],np.array([np.inf])))

# Load projection
prjnc = xr.open_dataset(prjpath)

# Make kernel
kernel = md.circleKernel(int(rad/spres), masked_value=0)
ky, kx = (np.array(kernel.shape)/2).astype(int)

# Prep outputs
yearlist, monthlist, nlist, plist = [], [], [], []

mt = starttime+[]
while mt != endtime:
    Y, M = str(mt[0]), md.dd[mt[1]-1]
    if M == '01':
        print(Y)
    
    ### Create output arrays ###
    narrs = [ np.zeros_like(prjnc['lat']) for i in range(binmins.shape[0]) ]
    parrs = [ np.zeros_like(prjnc['lat']) for i in range(binmins.shape[0]) ]
    
    ### Load monthly files ###
    cts = pd.read_pickle(ctpath+"/"+Y+"/"+subset+typ.lower()+'tracks'+Y+M+'.pkl')
    cts = [ct for ct in cts if ct.data.lat.max() >= minlat]
    sids = [ct.sid for ct in cts]
    mb = pd.read_csv(mbpath+"/MoistureBudget_"+str(rad)+"km_"+Y+M+"_"+VM+".csv")
    wind = pd.read_csv(windpath+"/Agg"+str(rad)+"km_"+dataset+"_HorizontalWind"+str(windlev)+"_"+Y+M+"_"+VW+".csv")

    ### Combine x, y, wind, and CAP ###
    wind['x'], wind['y'], wind[capvar] = np.nan, np.nan, np.nan
    
    for ct in cts:        
        wind.loc[wind['sid'] == ct.sid,'x'] = ct.data.loc[ct.data.lat >= minlat2].x.values
        wind.loc[wind['sid'] == ct.sid,'y'] = ct.data.loc[ct.data.lat >= minlat2].y.values
        wind.loc[wind['sid'] == ct.sid,capvar] = mb.loc[mb['sid'] == ct.sid,capvar].values
        
    # Ensure indices are integers
    wind['x'] = wind['x'].astype(int)
    wind['y'] = wind['y'].astype(int)

    ### Apply CAP and Wind vars to arrays based on bins ###
    for w in range(len(binmins)):
        # Subset by wind bin
        windbin = wind.loc[(wind[windvar] >= binmins[w]) & (wind[windvar] < binmaxs[w])]
        
        # Used kernel to apply to arrays
        for row in windbin.index:
            narrs[w][(windbin.loc[row,'y']-ky):(windbin.loc[row,'y']+ky+1),(windbin.loc[row,'x']-kx):(windbin.loc[row,'x']+kx+1)] += kernel
            parrs[w][(windbin.loc[row,'y']-ky):(windbin.loc[row,'y']+ky+1),(windbin.loc[row,'x']-kx):(windbin.loc[row,'x']+kx+1)] += kernel*windbin.loc[row,capvar]
    
    ### Append arrays to monthly lists ###
    yearlist.append(mt[0])
    monthlist.append(mt[1])
    nlist.append(narrs)
    plist.append(parrs)

    mt = md.timeAdd(mt,filetimestep)


### Write to File ###
yearlist = np.array(yearlist)
monthlist = np.array(monthlist)
for m in range(0,12):
    mask = np.where(monthlist == m+1)[0]

    if int(cycver.split('_')[0]) < 14:
        xrd = xr.Dataset(coords=dict(time=yearlist[mask], windbinmin=binmins, windbinmax=binmaxs, x=prjnc['u'].values, y=prjnc['v'].values))
    else:
        xrd = xr.Dataset(coords=dict(time=yearlist[mask], windbinmin=binmins, windbinmax=binmaxs, x=prjnc['x'], y=prjnc['y']))
        
    xrd = xrd.assign(bincount=(['time','windbinmin','y','x'], np.array(nlist)[mask,:]))
    xrd = xrd.assign(capsum=(['time','windbinmin','y','x'], np.array(plist)[mask,:]))
    
    xrd.to_netcdf(outpath+"/"+dataset+"_"+cycver+"_"+subset+"_"+capvar+"_BinBy_"+windvar+"_Monthly-"+md.dd[m]+"_"+str(yearlist[mask][0])+"-"+str(yearlist[mask][-1])+".nc")

xrd.close()

###############################

### Seasonal Averaging ###
ymin, ymax = 1979, 2021
years = np.arange(ymin,ymax+1)
mons = np.arange(1,13)
for em in [2,8]:
    yminadj = np.where(np.arange(em-1,em-4,-1) < 0, -1, 0)
    xr1 = xr.open_dataset(outpath+"/"+dataset+"_"+cycver+"_"+subset+"_"+capvar+"_BinBy_"+windvar+"_Monthly-"+md.dd[mons[em-1]-1]+"_"+str(ymin+yminadj[0])+"-"+str(ymax)+".nc")
    xr2 = xr.open_dataset(outpath+"/"+dataset+"_"+cycver+"_"+subset+"_"+capvar+"_BinBy_"+windvar+"_Monthly-"+md.dd[mons[em-2]-1]+"_"+str(ymin+yminadj[1])+"-"+str(ymax)+".nc")
    xr3 = xr.open_dataset(outpath+"/"+dataset+"_"+cycver+"_"+subset+"_"+capvar+"_BinBy_"+windvar+"_Monthly-"+md.dd[mons[em-3]-1]+"_"+str(ymin+yminadj[2])+"-"+str(ymax)+".nc")
    
    bincount = (xr1['bincount'].values[np.where((xr1['time'].values >= ymin+yminadj[0]) & (xr1['time'].values <= ymax+yminadj[0]))] + 
                xr2['bincount'].values[np.where((xr2['time'].values >= ymin+yminadj[1]) & (xr2['time'].values <= ymax+yminadj[1]))] +
                xr3['bincount'].values[np.where((xr3['time'].values >= ymin+yminadj[2]) & (xr3['time'].values <= ymax+yminadj[2]))])

    capsum = (xr1['capsum'].values[np.where((xr1['time'].values >= ymin+yminadj[0]) & (xr1['time'].values <= ymax+yminadj[0]))] + 
              xr2['capsum'].values[np.where((xr2['time'].values >= ymin+yminadj[1]) & (xr2['time'].values <= ymax+yminadj[1]))] +
              xr3['capsum'].values[np.where((xr3['time'].values >= ymin+yminadj[2]) & (xr3['time'].values <= ymax+yminadj[2]))])
        
    caprate = capsum / bincount
    
    caprate2 = capsum.sum(axis=1) / bincount.sum(axis=1)
    
    ### Climatology Analysis ###
    bincountavg = bincount.mean(axis=0)
    capsumavg = capsum.mean(axis=0)
    caprate2avg = caprate2.mean(axis=0)
    
    if int(cycver.split('_')[0]) < 14:
        xrclim = xr.Dataset(coords=dict(windbinmin=binmins, windbinmax=binmaxs, x=prjnc['u'].values, y=prjnc['v'].values))
    else:
        xrclim = xr.Dataset(coords=dict(windbinmin=binmins, windbinmax=binmaxs, x=prjnc['x'], y=prjnc['y']))
    
    xrclim = xrclim.assign(bincount_avg=(['windbinmin','y','x'], bincountavg))
    xrclim = xrclim.assign(caprate_avg=(['windbinmin','y','x'], capsumavg/bincountavg))
    xrclim = xrclim.assign(caprate_allbins_avg=(['y','x'], caprate2avg))

    xrclim.to_netcdf(outpath2+"/pr_BinBy_uv_"+V+"-SeasonalAvg/"+dataset+"_"+cycver+"_"+subset+"_"+capvar+"_BinBy_"+windvar+"_SeasonalAvg_"+md.sss[em-1]+"_"+str(ymin)+"-"+str(ymax)+".nc")
    
    ### Trend Analysis ###
    ntrendarr, nyintarr, nrarr = np.zeros_like(bincount[0,:])*np.nan, np.zeros_like(bincount[0,:])*np.nan, np.zeros_like(bincount[0,:])
    nparr, nsetrendarr = np.ones_like(bincount[0,:]), np.zeros_like(bincount[0,:])*np.nan
    
    ptrendarr, pyintarr, prarr = np.zeros_like(bincount[0,:])*np.nan, np.zeros_like(bincount[0,:])*np.nan, np.zeros_like(bincount[0,:])
    pparr, psetrendarr = np.ones_like(bincount[0,:]), np.zeros_like(bincount[0,:])*np.nan

    p2trendarr, p2yintarr, p2rarr = np.zeros_like(caprate2[0,:])*np.nan, np.zeros_like(caprate2[0,:])*np.nan, np.zeros_like(caprate2[0,:])
    p2parr, p2setrendarr = np.ones_like(caprate2[0,:]), np.zeros_like(caprate2[0,:])*np.nan
        
    for bi in range(bincount.shape[1]):
        validy, validx = np.where(np.isfinite(caprate[:,bi,:,:]).sum(axis=0) > 20)
        for i in range(len(validy)):
            # Extract location
            y, x = validy[i], validx[i]
            
            # Identify regression inputs
            N = bincount[:,bi,y,x]
            P = caprate[:,bi,y,x]
            validp = np.isfinite(P)
            
            # Perform regression
            lmn = linregress(years[validp],N[validp])
            lmp = linregress(years[validp],P[validp])
            
            # Extract outputs
            ntrendarr[bi,y,x], nyintarr[bi,y,x], nrarr[bi,y,x], nparr[bi,y,x], nsetrendarr[bi,y,x] = lmn
            ptrendarr[bi,y,x], pyintarr[bi,y,x], prarr[bi,y,x], pparr[bi,y,x], psetrendarr[bi,y,x] = lmp
    
    # Overall Value
    validy, validx = np.where(np.isfinite(caprate2).sum(axis=0) > 20)
    for i in range(len(validy)):
        # Extract location
        y, x = validy[i], validx[i]
        
        # Identify regression inputs
        validp = np.isfinite(caprate2[:,y,x])
        
        # Perform regression
        lmp = linregress(years[validp],caprate2[:,y,x][validp])
        
        # Extract outputs
        p2trendarr[y,x], p2yintarr[y,x], p2rarr[y,x], p2parr[y,x], p2setrendarr[y,x] = lmp
    
    
    if int(cycver.split('_')[0]) < 14:
        xrseas = xr.Dataset(coords=dict(windbinmin=binmins, windbinmax=binmaxs, x=prjnc['u'].values, y=prjnc['v'].values))
    else:
        xrseas = xr.Dataset(coords=dict(windbinmin=binmins, windbinmax=binmaxs, x=prjnc['x'], y=prjnc['y']))
    
    xrseas = xrseas.assign(bincount_trend=(['windbinmin','y','x'], ntrendarr))
    xrseas = xrseas.assign(bincount_yint=(['windbinmin','y','x'], nyintarr))
    xrseas = xrseas.assign(bincount_r2=(['windbinmin','y','x'], nrarr*nrarr))
    xrseas = xrseas.assign(bincount_pval=(['windbinmin','y','x'], nparr))
    xrseas = xrseas.assign(bincount_se=(['windbinmin','y','x'], nsetrendarr))
    
    xrseas = xrseas.assign(caprate_trend=(['windbinmin','y','x'], ptrendarr))
    xrseas = xrseas.assign(caprate_yint=(['windbinmin','y','x'], pyintarr))
    xrseas = xrseas.assign(caprate_r2=(['windbinmin','y','x'], prarr*prarr))
    xrseas = xrseas.assign(caprate_pval=(['windbinmin','y','x'], pparr))
    xrseas = xrseas.assign(caprate_se=(['windbinmin','y','x'], psetrendarr))

    xrseas = xrseas.assign(caprate_allbins_trend=(['y','x'], p2trendarr))
    xrseas = xrseas.assign(caprate_allbins_yint=(['y','x'], p2yintarr))
    xrseas = xrseas.assign(caprate_allbins_r2=(['y','x'], p2rarr*p2rarr))
    xrseas = xrseas.assign(caprate_allbins_pval=(['y','x'], p2parr))
    xrseas = xrseas.assign(caprate_allbins_se=(['y','x'], p2setrendarr))
        
    xrseas.to_netcdf(outpath2+"/pr_BinBy_uv_"+V+"-SeasonalTrend"+dataset+"_"+cycver+"_"+subset+"_"+capvar+"_BinBy_"+windvar+"_SeasonalTrend-"+md.sss[em-1]+"_"+str(ymin)+"-"+str(ymax)+".nc")
