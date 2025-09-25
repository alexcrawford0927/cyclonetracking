#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Date Created: 2023 Aug 29
Date Modified: 2023 Aug 30
Author: Alex Crawford

Purpose: Calculate regional trends in seasonally averaged cyclone moisture 
budget parameters
"""

import pandas as pd
import numpy as np
from scipy.stats import linregress
import CycloneModule_13_3 as md

'''**********
Define Variables
**********'''

# Cyclone Data Inputs
dataset = 'ERA5'
cycver = '13_2E5R'
subset = 'BBox10'
typ = 'System'

# Region Data Inputs
regtype = 'cross' # 'maxint' # 'gen' # 'cross' # 'lys'

# Moisture Budget Inputs
V = 'V7'
rad = 1200 # radius for constant kernel size (units: km)
MYM = '197811-202412'
meanvars = ['Peff','latAvg', 'lonAvg', 'PratioAvg', 'p_centAvg', 'depthAvg', 'radiusAvg',\
       'p_gradAvg', 'DsqPAvg', 'uvAvg', 'DpDtAvg', 'tcwAvg', 'siconcAvg',\
       'tisrAvg', 'tsrAvg', 'ttrAvg', 'tnetradAvg', 'latMax', 'depthMax',\
       'radiusMax', 'p_gradMax', 'DsqPMax', 'tpMax', 'eMax', 'viwvdMax',\
       'tcwMax', 'siconcMax', 'latMin', 'p_centMin', 'DpDtMin', 'ttrMin',\
       'tcwFirst', 'tcwLast']
sumvars = ['tp','e','viwvd','tcw_Adv','tcwDiff','hours','count']
ratevars = ['tp','e','viwvd','tcw_Adv']

# Trend Calculation Inputs
tymin, tymax = 1979, 2024
cymin, cymax = 1981, 2010

mons = [[11,12,1],[12,1,2],[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],\
        [11,12,1,2,3],[5,6,7,8,9],[1,2,3,4,5,6,7,8,9,10,11,12]]
SSS = md.sss + ['NDJFM','MJJAS','Annual']

# Paths
# inpath = '/Volumes/Cressida/CycloneTracking/tracking'+cycver+"/"+subset+'/Aggregation'+typ
inpath = "/media/alex/Datapool/CycloneTracking/tracking"+cycver+"/"+subset+"/Aggregation"+typ+"/MoistureBudget"+V

'''**********
Main Analysis
**********'''

# Load File
pdf = pd.read_csv(inpath+'/MoistureBudget_TrackSummary_'+str(rad)+'km_'+MYM+'_'+V+'.csv')
pdf['count'] = 1

##### GLOBAL ANALYSIS #####

### CONVERT TO SEASONS ###
smdfs = []

for si in range(len(mons)):
    # Subset to Season
    sdf = pdf.loc[np.in1d(pdf['month'],mons[si])]
    sdf['season'] = SSS[si]
    
    # Aggregate Within Season - Mean Vars
    smdf = sdf.loc[:,['year','season']+meanvars].groupby(by=['year','season']).mean().reset_index()
    
    # Aggregate Within Season - Sum Vars
    for var in sumvars:
        smdf[var] = sdf.loc[:,['year','season',var]].groupby(by=['year','season']).sum()[var].values
    
    for var in ratevars:
        smdf[var+'rate'] = smdf[var] / smdf['hours']
        
    smdfs.append(smdf)

smdf = pd.concat(smdfs)

# Convert direction of of fluxes to be positive = more moisture
for var in ['e','viwvd','tcwLast','erate','viwvdrate']:
    smdf[var] *= -1

# Subset to years for the trend #
smdft = smdf.loc[(smdf['year'] >= tymin) & (smdf['year'] <= tymax)]

### TRENDS & CLIMATOLOGY ###

# Global Trend Analysis
trenddf = pd.DataFrame({'region':'Global','season':SSS})
for var in meanvars+sumvars+[r+'rate' for r in ratevars]: 
    slope, yint, rval, pval, slope_se  =  [ np.repeat(0.0,len(mons)) for i in range(5) ]
    for si, s in enumerate(SSS):
        results = linregress(smdft.loc[smdft['season'] == s,'year'],smdft.loc[smdft['season'] == s,var])
        slope[si], yint[si], rval[si], pval[si], slope_se[si] = results
    trenddf[var+"_trend"] = slope
    trenddf[var+"_yint"] = yint
    trenddf[var+"_r2"] = rval**2  
    trenddf[var+"_pval"] = pval
    trenddf[var+"_se"] = slope_se      

# Global Climatology
climdf = pd.DataFrame({'region':'Global','season':SSS})
for var in meanvars+sumvars+[r+'rate' for r in ratevars]: 
    climdf[var] = smdf.loc[(smdf['year'] >= cymin) & (smdf['year'] <= cymax),('season',var)].groupby(by='season').mean()[var].values

##### REGIONAL ANALYSIS #####
for reg in np.unique([col for col in pdf.columns if regtype in col]):
    pdsub = pdf.loc[pdf[reg] == 1]
    
    smdfs = []

    for si in range(len(mons)):
        # Subset to Season
        sdf = pdsub.loc[np.in1d(pdsub['month'],mons[si])]
        sdf['season'] = SSS[si]
        
        # Aggregate Within Season - Mean Vars
        smdf = sdf.loc[:,['year','season']+meanvars].groupby(by=['year','season']).mean().reset_index()
        
        # Aggregate Within Season - Sum Vars
        for var in sumvars:
            smdf[var] = sdf.loc[:,['year','season',var]].groupby(by=['year','season']).sum()[var].values
        
        for var in ratevars:
            smdf[var+'rate'] = smdf[var] / smdf['hours']
            
        smdfs.append(smdf)

    smdf = pd.concat(smdfs)

    # Convert direction of of fluxes to be positive = more moisture
    for var in ['e','viwvd','tcwLast','erate','viwvdrate']:
        smdf[var] *= -1

    # Subset to years for the trend #
    smdft = smdf.loc[(smdf['year'] >= tymin) & (smdf['year'] <= tymax)]
    
    ### TRENDS & CLIMATOLOGY ###

    # Regional Trend Analysis
    regtrenddf = pd.DataFrame({'region':reg,'season':SSS})
    for var in meanvars+sumvars+[r+'rate' for r in ratevars]: 
        slope, yint, rval, pval, slope_se  =  [ np.repeat(0.0,len(mons)) for i in range(5) ]
        for si, s in enumerate(SSS):
            results = linregress(smdft.loc[smdft['season'] == s,'year'],smdft.loc[smdft['season'] == s,var])
            slope[si], yint[si], rval[si], pval[si], slope_se[si] = results
        regtrenddf[var+"_trend"] = slope
        regtrenddf[var+"_yint"] = yint
        regtrenddf[var+"_r2"] = rval**2  
        regtrenddf[var+"_pval"] = pval
        regtrenddf[var+"_se"] = slope_se      

    # Regional Climatology
    regclimdf = pd.DataFrame({'region':reg,'season':SSS})
    for var in meanvars+sumvars+[r+'rate' for r in ratevars]: 
        regclimdf[var] = smdf.loc[(smdf['year'] >= cymin) & (smdf['year'] <= cymax),('season',var)].groupby(by='season').mean()[var].values
    
    # Concatenate
    trenddf = pd.concat((trenddf,regtrenddf),axis=0, join='inner')
    climdf = pd.concat((climdf,regclimdf),axis=0, join='inner')

### Write to File ###
trenddf.to_csv(inpath+'/MoistureBudget_'+str(rad)+'km_Regional-Sesonal-Trends_'+regtype+'_'+str(tymin)+'-'+str(tymax)+'_'+V+'.csv', index=False)
climdf.to_csv(inpath+'/MoistureBudget_'+str(rad)+'km_Regional-Sesonal-Climatology_'+regtype+'_'+str(cymin)+'-'+str(cymax)+'_'+V+'.csv', index=False)
