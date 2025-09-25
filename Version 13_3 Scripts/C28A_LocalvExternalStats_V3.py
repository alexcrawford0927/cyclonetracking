#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 15:36:04 2023
Modified 30 Jan 2025

Summarize moisture budget of storms that enter CAO by breaking down the
evaopration and convergence and advection terms by within CAO and before being
within CAO parts -- also keep track of siconc and other variables
"""

'''**********
Load Modules
**********'''
import CycloneModule_13_3 as md
import pandas as pd
import numpy as np
from scipy import stats
# from sklearn.linear_model import LinearRegression
from statsmodels.regression import linear_model as lm
import netCDF4 as nc

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

'''**********
Define Variables
**********'''
dataset = 'ERA5'
cycver = '13_2E5R'
subset = 'BBox10'
typ = 'System'
regvar = 'CAO2' # 'reg' # 'reg2' #  
spres = 25 # in km
regvals = [1] # [10,11] #   [1,4,5,6,7,8,9,10,11] #
minlat2 = 15

# regname = ['Other','CAO','ILBK','MLNAtl','NPac','Med','Euro','Sib','EAsia','MLNAmer','NCan','Baf']
# regname = ['BK'] # ['CAOBK'] # w/ regvar = 'reg'
regname = ['CAO2'] # w/ regvar = 'CAO2' 

V = 'V7'
rad = 1200 # radius for constant kernel size (units: km)
# windlev = 850 # pressure level for wind in hPa
tres = 3 # temporal resolution of cyclone data

starttime = [1978,11,1,0,0,0] # inclusive
endtime = [2025,1,1,0,0,0] # exclusive
filetimestep = [0,1,0,0,0,0] # time step between cyclone files
inittime = [1940,1,1,0,0,0] # Initiation time for cyclone tracking dataset
dateref = [1900,1,1,0,0,0] # Reference date for input data
mons = [[11,12,1],[12,1,2],[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],\
        [11,12,1,2,3],[5,6,7,8,9],[10,11,12,1,2,3],[4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9,10,11,12]]
SSS = md.sss + ['NDJFM','MJJAS','ONDJFM','AMJJAS','Annual']

varlistam = ['tp','e','viwvd','tcw_Adv','siconc','tcw','tisr', 'tsr', 'ttr','tnetrad']
varlistas = ['hours','tp','e','viwvd','tcw_Adv','siconc','tcw', 'tisr', 'tsr', 'ttr','tnetrad']

varsumlist = ['tp','e','viwvd','tcw_Adv','hours']
varavglist = ['lat','lon','Pratio','p_cent','depth','radius','p_grad','DsqP','uv','DpDt','tcw','siconc','tisr', 'tsr', 'ttr','tnetrad']
varmaxlist = ['lat','depth','radius','p_grad','DsqP','tp','e','viwvd','tcw','siconc']
varminlist = ['lat','p_cent','DpDt','ttr']

try:
    REG = '-'.join([regname[r] for r in regvals])
except:
    REG = regname[0]
    
path = "/media/alex/Datapool/CycloneTracking/tracking"+cycver+"/"+subset+"/Aggregation"+typ
mbpath = path+"/SpatialAvgEnv_"+str(rad)+'km'
mbapath = path+"/SpatialAvgEnvAmly_"+str(rad)+'km'
# windpath = path+"/Aggregation"+typ+"/HorizontalWind"+str(windlev)+"_"+str(rad)
# windapath = path+"/Aggregation"+typ+"/HorizontalWind"+str(windlev)+"Amly_"+str(rad)
prjpath = "/media/alex/Datapool/Projections/EASE2_N0_"+str(spres)+"km_GenesisRegions.nc"
outpath = path+"/MoistureBudget"+V+'/'+REG

'''**********
Track-by-Track Summaries
**********'''
endtime2 = md.timeAdd(endtime,[-1*t for t in filetimestep])
print('Summaries')

# Read in region mask
if int(cycver.split('_')[0]) < 14:
    regs = np.flipud(nc.Dataset(prjpath)[regvar][:].data)
else:
    regs = nc.Dataset(prjpath)[regvar][:].data


try:
    pdfmb = pd.read_csv(outpath+"/SpatialAvgEnv_"+REG+"_TrackSummary_"+str(rad)+"km_"+str(starttime[0])+md.dd[starttime[1]-1]+"-"+str(endtime2[0])+md.dd[endtime2[1]-1]+"_"+V+".csv")
    pdfmbas = pd.read_csv(outpath+"/SpatialAvgEnvAmlySum_"+REG+"_TrackSummary_"+str(rad)+"km_"+str(starttime[0])+md.dd[starttime[1]-1]+"-"+str(endtime2[0])+md.dd[endtime2[1]-1]+"_"+V+".csv")
    # pdfmbaa = pd.read_csv(outpath+"/SpatialAvgEnvAmlyAvg_"+REG+"TrackSummary_"+str(rad)+"km_"+str(starttime[0])+md.dd[starttime[1]-1]+"-"+str(endtime2[0])+md.dd[endtime2[1]-1]+"_"+V+".csv")
    
    Y, M = str(endtime2[0]), md.dd[endtime2[1]-1]
    
except:
    mblist, mbasumlist, mbaalist = [], [], []
    mt = starttime+[]
    while mt != endtime:
        Y, M = str(mt[0]), md.dd[mt[1]-1]
        
        # Load monthly CSV files
        mb = pd.read_csv(mbpath+"/SpatialAvgEnv"+str(rad)+"km_"+Y+M+".csv")
        mba = pd.read_csv(mbapath+"/SpatialAvgEnvAmly"+str(rad)+"km_"+Y+M+".csv")

        ### Take track-by-track summaries ###
        
        # Raw Values
        mb['hours'], mba['hours'] = tres, tres
        mb['Pratio'] = mb['tp'] / mb['tcw']
        
        # Identify rows for which storm is in region of interest
        for sid in np.unique(mb['sid']):
            # Assign region
            mb.loc[mb['sid'] == sid,'reg'] = regs[mb.loc[(mb['sid'] == sid),'y'],mb.loc[(mb['sid'] == sid),'x']]
            mba.loc[mba['sid'] == sid,'reg'] = regs[mb.loc[(mb['sid'] == sid),'y'],mb.loc[(mb['sid'] == sid),'x']]

        # Boolean creation
        mb['regflag'] = np.in1d(mb['reg'], regvals)
        mba['regflag'] = np.in1d(mba['reg'], regvals)
        
        # Subset to only sids that at some point are in the region of interest
        mb2 = mb.loc[np.in1d(mb['sid'],np.unique(mb.loc[mb['regflag'] == 1,'sid']))]
        mba2 = mba.loc[np.in1d(mba['sid'],np.unique(mba.loc[mba['regflag'] == 1,'sid']))]
        
        # Remove any rows that occur after the last time cyclone is observed in the region of interest
        mbage = mb2.loc[:,('sid','regflag','age')].groupby(by=['sid','regflag']).max() # Find max age
        mbage.reset_index(inplace=True) # reset indices
        mbage = mbage.loc[mbage['regflag'] == True] # subset so that there's one row per sid
        mbage = mbage.rename(columns={'age':'maxage'}) # rename column

        mb2 = pd.merge(mb2,mbage.loc[:,('sid','maxage')], on='sid') # merge with original dataframe
        mb2 = mb2.loc[mb2['age'] <= mb2['maxage']] # subset to rows that are at/before the max age

        mba2 = pd.merge(mba2,mbage.loc[:,('sid','maxage')], on='sid') # merge with original dataframe
        mba2 = mba2.loc[mba2['age'] <= mba2['maxage']] # subset to rows that are at/before the max age
        
        # Group by track and region boolean -- raw
        mb3 = mb2.loc[:,('sid','regflag','year','month')].groupby(by=['sid','regflag']).first().reset_index()
        for var in varsumlist:
            mb3[var] = mb2.loc[:,('sid','regflag',var)].groupby(by=['sid','regflag']).sum()[var].values
        mb3['Peff'] = mb3['tp'] / (-1*(mb3['e'] + mb3['viwvd']))
        for var in ['tcw','tcw_Adv']:
            mb3[var+'Last'] = mb2.loc[:,('sid','regflag',var)].groupby(by=['sid','regflag']).last()[var].values
            mb3[var+'First'] = mb2.loc[:,('sid','regflag',var)].groupby(by=['sid','regflag']).first()[var].values
        
        for var in varavglist:
            mb3[var+'Avg'] = mb2.loc[:,('sid','regflag',var)].groupby(by=['sid','regflag']).mean()[var].values
        for var in varmaxlist:
            mb3[var+'Max'] = mb2.loc[:,('sid','regflag',var)].groupby(by=['sid','regflag']).max()[var].values
        for var in varminlist:
            mb3[var+'Min'] = mb2.loc[:,('sid','regflag',var)].groupby(by=['sid','regflag']).min()[var].values

        # Group by track and region boolean -- amly
        mba3 = mba2.loc[:,['sid','regflag','year','month']+varlistas].groupby(by=['sid','regflag']).sum().reset_index()
        for var in ['tcw','tcw_Adv']:
            mba3[var+'Last'] = mba2.loc[:,('sid','regflag',var)].groupby(by=['sid','regflag']).last()[var].values
            mba3[var+'First'] = mba2.loc[:,('sid','regflag',var)].groupby(by=['sid','regflag']).first()[var].values
        mba3['year'] = mt[0]
        mba3['month'] = mt[1]
        
        # Reformat so that there are separate columns for in and out of the region of interest
        mb4 = pd.merge(mb3.loc[mb3['regflag'] == 1], mb3.loc[mb3['regflag'] == 0], on='sid', how='left', suffixes=('_in', '_out'))        
        for var in varsumlist+['tcw_AdvFirst']:
           mb4[var+"_out"] = mb4[var+"_out"].fillna(0).values
        mb4.loc[np.isnan(mb4['tcwLast_out']),'tcwLast_out'] = mb4.loc[np.isnan(mb4['tcwLast_out']),'tcwFirst_in']
        
        mba4 = pd.merge(mba3.loc[mba3['regflag'] == 1], mba3.loc[mba3['regflag'] == 0], on='sid', how='left', suffixes=('_in', '_out'))
        mba4.loc[np.isnan(mba4['tcwLast_out']),'tcwLast_out'] = mba4.loc[np.isnan(mba4['tcwLast_out']),'tcwFirst_in']

        # Append
        mblist.append(mb4), mbasumlist.append(mba4)
        
        mt = md.timeAdd(mt,filetimestep)
    
    # Concatenate
    pdfmb = pd.concat(mblist, ignore_index=True)
    pdfmbas = pd.concat(mbasumlist, ignore_index=True)
    
    # Fix time
    pdfmb = pdfmb.rename(columns={'month_in':'month','year_in':'year'})
    pdfmb = pdfmb.drop(columns=['month_out','year_out'])
    pdfmbas = pdfmbas.rename(columns={'month_in':'month','year_in':'year'})
    pdfmbas = pdfmbas.drop(columns=['month_out','year_out'])

    # Write to File
    pdfmb.to_csv(outpath+"/SpatialAvgEnv_"+REG+"_TrackSummary_"+str(rad)+"km_"+str(starttime[0])+md.dd[starttime[1]-1]+"-"+Y+M+"_"+V+".csv", index=False)
    pdfmbas.to_csv(outpath+"/SpatialAvgEnvAmlySum_"+REG+"_TrackSummary_"+str(rad)+"km_"+str(starttime[0])+md.dd[starttime[1]-1]+"-"+Y+M+"_"+V+".csv", index=False)

'''**********
Correlations & Regressions
**********''' 
print("Correlations & Regressions")
varlist = [v[:-3] for v in pdfmb.columns[5:]]
varlist = [v for v in varlist if v.endswith('_') == False]

# Add constant
# Convert direction of of fluxes to be positive = more moisture
for var in ['e_in','viwvd_in','e_out','viwvd_out','tcw_AdvFirst_in']:
    pdfmb[var] *= -1
    
pdfmb['max_out'] = pdfmb['tp_in'] - pdfmb['e_in']
pdfmb['most_out'] = pdfmb['tp_in'] - pdfmb['e_in'] - pdfmb['viwvd_in']
pdfmb['min_out'] = pdfmb['e_out'] + pdfmb['viwvd_out'] - pdfmb['tp_out']

pdfmb['tcwDiff'] = pdfmb['tcwLast_out'] + pdfmb['tcwLast_in']
pdfmb['tcwDiff_out'] = pdfmb['tcwLast_out'] + pdfmb['tcwLast_in'] + pdfmb['viwvd_in']
pdfmb['tcwDiff_resid'] = pdfmb['tp_in'] - (pdfmb['tcwDiff'] + pdfmb['e_in'] + pdfmb['viwvd_in'])

pdfmb['final_in'] = pdfmb['tp_in'] - pdfmb['tcwLast_in']
pdfmb['final_out'] = pdfmb['viwvd_in'] + pdfmb['tcwLast_out']
pdfmb['final_resid'] = pdfmb['final_in'] - (pdfmb['e_in'] + pdfmb['final_out'])

pdfmb['tp_resid'] = pdfmb['tp_in'] - (pdfmb['e_in'] + pdfmb['viwvd_in'] + pdfmb['tcwLast_out'] + pdfmb['tcwLast_in'])

# Add precip rate
pdfmb['tprate_in'] = pdfmb['tp_in'] / pdfmb['hours_in']
pdfmb['erate_in'] = pdfmb['e_in'] / pdfmb['hours_in']
pdfmb['viwvdrate_in'] = pdfmb['viwvd_in'] / pdfmb['hours_in']

# Add constant
pdfmb['count'] = 1
pdfmb['constant'] = 1

# ''' Correlatons with tp '''
# # Prep output lists
# mout, ioout, rout, pout = [], [], [[] for vi in varlist], [[] for vi in varlist]
# routas, poutas = [[] for vi in varlistas], [[] for vi in varlistas]

# # Loop through each crossing region and season
# for mi in range(len(mons)):
#     # Subset by region and season
#     pdsub = pdfmb.loc[np.in1d(pdfmb['month'],mons[mi]) & (pdfmb['regflag_out'] == False)]
#     pdsubas = pdfmbas.loc[np.in1d(pdfmbas['month'],mons[mi]) & (pdfmbas['regflag_out'] == False)]

#     # RAW DATA
#     for vi in range(len(varlist)):
#         for io in ['_in','_out']:
#             # Correlation
#             rval, pval = stats.spearmanr(pdsub['tp_in'],pdsub[varlist[vi]+io])
            
#             # Append to lists
#             rout[vi].append(rval), pout[vi].append(pval)

#     # SUMMED ANOMALIES
#     for vi in range(len(varlistas)):
#         for io in ['_in','_out']:
#             # Correlation
#             rvalas, pvalas = stats.spearmanr(pdsubas['tp_in'],pdsubas[varlistas[vi]+io])
            
#             # Append to lists
#             routas[vi].append(rvalas), poutas[vi].append(pvalas)
            
#     # Append dimensions
#     for io in ['in','out']:    
#         mout.append(SSS[mi]), ioout.append(io)

# # Compile outputs and write files
# pdout = pd.DataFrame({'Season':mout,'Region':ioout})
# for vi in range(len(varlist)):
#     pdout["r_tp_in-"+varlist[vi]] = rout[vi]
#     pdout['p_tp_in-'+varlist[vi]] = pout[vi]
# pdout.to_csv(outpath+"/Spearman_Track_"+REG+"_Precipitation_"+str(rad)+"km_"+str(starttime[0])+md.dd[starttime[1]-1]+"-"+str(endtime2[0])+md.dd[endtime2[1]-1]+"_"+V+".csv", index=False)

# pdoutas = pd.DataFrame({'Season':mout, 'Region':ioout})
# for vi in range(len(varlistas)):
#     pdoutas["r_tp_in-"+varlistas[vi]] = routas[vi]
#     pdoutas['p_tp_in-'+varlistas[vi]] = poutas[vi]
# pdoutas.to_csv(outpath+"/Spearman_Track_"+REG+"_PrecipitationAmlySum_"+str(rad)+"km_"+str(starttime[0])+md.dd[starttime[1]-1]+"-"+str(endtime2[0])+md.dd[endtime2[1]-1]+"_"+V+".csv", index=False)

# ''' Correlatons with tprate '''
# for var in ['tp','e','viwvd','tcw_Adv']:
#     for io in ['_in','_out']:
#         pdfmb[var+'rate'+io] = pdfmb[var+io]/pdfmb['hours'+io]*24 # units of per day
# varlistrate = ['erate','viwvdrate','tcw_Advrate','tcwAvg','p_centAvg','depthAvg','radiusAvg','p_gradAvg','DsqPAvg','uvAvg','DpDtAvg','siconcAvg','tisrAvg','ttrAvg','tnetradAvg','latAvg','latMin']

# # Prep output lists
# mout, ioout, rout, pout = [], [], [[] for vi in varlistrate], [[] for vi in varlistrate]

# # Loop through each crossing region and season
# for mi in range(len(mons)):
#     # Subset by region and season
#     pdsub = pdfmb.loc[np.in1d(pdfmb['month'],mons[mi]) & (pdfmb['regflag_out'] == False)]
#     pdsubas = pdfmbas.loc[np.in1d(pdfmbas['month'],mons[mi]) & (pdfmbas['regflag_out'] == False)]

#     # RAW DATA
#     for vi in range(len(varlistrate)):
#         for io in ['_in','_out']:
#             # Correlation
#             rval, pval = stats.spearmanr(pdsub['tp_in'],pdsub[varlistrate[vi]+io])
            
#             # Append to lists
#             rout[vi].append(rval), pout[vi].append(pval)
#     # Append dimensions
#     for io in ['in','out']:    
#         mout.append(SSS[mi]), ioout.append(io)

# # Compile outputs and write files
# pdout = pd.DataFrame({'Season':mout,'Region':ioout})
# for vi in range(len(varlistrate)):
#     pdout["r_tp_in-"+varlistrate[vi]] = rout[vi]
#     pdout['p_tp_in-'+varlistrate[vi]] = pout[vi]
# pdout.to_csv(outpath+"/Spearman_Track_"+REG+"_PrecipitationRate_"+str(rad)+"km_"+str(starttime[0])+md.dd[starttime[1]-1]+"-"+str(endtime2[0])+md.dd[endtime2[1]-1]+"_"+V+".csv", index=False)


# ''' Correlatons with evap '''
# # Prep output lists
# mout, ioout, rout, pout = [], [], [[] for vi in varlist], [[] for vi in varlist]
# routas, poutas = [[] for vi in varlistas], [[] for vi in varlistas]

# # Loop through each crossing region and season
# for mi in range(len(mons)):
#     # Subset by region and season
#     pdsub = pdfmb.loc[np.in1d(pdfmb['month'],mons[mi]) & (pdfmb['regflag_out'] == False)]
#     pdsubas = pdfmbas.loc[np.in1d(pdfmbas['month'],mons[mi]) & (pdfmbas['regflag_out'] == False)]

#     # RAW DATA
#     for vi in range(len(varlist)):
#         for io in ['_in','_out']:
#             # Correlation
#             rval, pval = stats.spearmanr(pdsub['e_in'],pdsub[varlist[vi]+io])
            
#             # Append to lists
#             rout[vi].append(rval), pout[vi].append(pval)

#     # SUMMED ANOMALIES
#     for vi in range(len(varlistas)):
#         for io in ['_in','_out']:
#             # Correlation
#             rvalas, pvalas = stats.spearmanr(pdsubas['e_in'],pdsubas[varlistas[vi]+io])
            
#             # Append to lists
#             routas[vi].append(rvalas), poutas[vi].append(pvalas)
            
#     # Append dimensions
#     for io in ['in','out']:    
#         mout.append(SSS[mi]), ioout.append(io)

# # Compile outputs and write files
# pdout = pd.DataFrame({'Season':mout,'Region':ioout})
# for vi in range(len(varlist)):
#     pdout["r_e_in-"+varlist[vi]] = rout[vi]
#     pdout['p_e_in-'+varlist[vi]] = pout[vi]
# pdout.to_csv(outpath+"/Spearman_Track_"+REG+"_Evaporation_"+str(rad)+"km_"+str(starttime[0])+md.dd[starttime[1]-1]+"-"+str(endtime2[0])+md.dd[endtime2[1]-1]+"_"+V+".csv", index=False)

# pdoutas = pd.DataFrame({'Season':mout, 'Region':ioout})
# for vi in range(len(varlistas)):
#     pdoutas["r_e_in-"+varlistas[vi]] = routas[vi]
#     pdoutas['p_e_in-'+varlistas[vi]] = poutas[vi]
# pdoutas.to_csv(outpath+"/Spearman_Track_"+REG+"_EvaporationAmlySum_"+str(rad)+"km_"+str(starttime[0])+md.dd[starttime[1]-1]+"-"+str(endtime2[0])+md.dd[endtime2[1]-1]+"_"+V+".csv", index=False)

# ''' Correlatons with tcwLast_out '''
# # Prep output lists
# mout, ioout, rout, pout = [], [], [[] for vi in varlist], [[] for vi in varlist]
# routas, poutas = [[] for vi in varlistas], [[] for vi in varlistas]

# # Loop through each crossing region and season
# for mi in range(len(mons)):
#     # Subset by region and season
#     pdsub = pdfmb.loc[np.in1d(pdfmb['month'],mons[mi]) & (pdfmb['regflag_out'] == False)]
#     pdsubas = pdfmbas.loc[np.in1d(pdfmbas['month'],mons[mi]) & (pdfmbas['regflag_out'] == False)]

#     # RAW DATA
#     for vi in range(len(varlist)):
#         for io in ['_in','_out']:
#             # Correlation
#             rval, pval = stats.spearmanr(pdsub['tcwLast_out'],pdsub[varlist[vi]+io])
            
#             # Append to lists
#             rout[vi].append(rval), pout[vi].append(pval)

#     # SUMMED ANOMALIES
#     for vi in range(len(varlistas)):
#         for io in ['_in','_out']:
#             # Correlation
#             rvalas, pvalas = stats.spearmanr(pdsubas['tcwLast_out'],pdsubas[varlistas[vi]+io])
            
#             # Append to lists
#             routas[vi].append(rvalas), poutas[vi].append(pvalas)
            
#     # Append dimensions
#     for io in ['in','out']:    
#         mout.append(SSS[mi]), ioout.append(io)

# # Compile outputs and write files
# pdout = pd.DataFrame({'Season':mout,'Region':ioout})
# for vi in range(len(varlist)):
#     pdout["r_tcwLast_out-"+varlist[vi]] = rout[vi]
#     pdout['p_tcwLast_out-'+varlist[vi]] = pout[vi]
# pdout.to_csv(outpath+"/Spearman_Track_"+REG+"_tcwLastout_"+str(rad)+"km_"+str(starttime[0])+md.dd[starttime[1]-1]+"-"+str(endtime2[0])+md.dd[endtime2[1]-1]+"_"+V+".csv", index=False)

# pdoutas = pd.DataFrame({'Season':mout, 'Region':ioout})
# for vi in range(len(varlistas)):
#     pdoutas["r_tcwLast_out-"+varlistas[vi]] = routas[vi]
#     pdoutas['p_tcwLast_out-'+varlistas[vi]] = poutas[vi]
# pdoutas.to_csv(outpath+"/Spearman_Track_"+REG+"_tcwLastoutAmlySum_"+str(rad)+"km_"+str(starttime[0])+md.dd[starttime[1]-1]+"-"+str(endtime2[0])+md.dd[endtime2[1]-1]+"_"+V+".csv", index=False)


# ''' Model 0: tp = e + viwvd + res'''
# # Prep output lists
# mout, r2out, bout, pout, bminout, bmaxout = [[]for i in range(6)]
# m1vars = ['e','viwvd','res']

# # Loop through each crossing region and season
# for mi in range(len(mons)):
#     # Subset by region and season
#     pdsub = pdfmb.loc[np.in1d(pdfmb['month'],mons[mi]) & (pdfmb['regflag_out'] == False)]
#     # pdsubas = pdfmbas.loc[np.in1d(pdfmbas['month'],mons[mi]) & (pdfmbas['regflag_out'] == False)]
        
#     # Change sign of e and vimd and tcwLast_in (but NOT tcwLast_out)
#     for var in ['e_in','e_out','viwvd_in','viwvd_out']:
#         pdsub[var] = -1*pdsub[var]
    
#     # Create a net external variable
#     pdsub['tp'] = pdsub['tp_in'] + pdsub['tp_out'] 
#     pdsub['e'] = pdsub['e_in'] + pdsub['e_out']
#     pdsub['viwvd'] = pdsub['viwvd_in'] + pdsub['viwvd_out']
#     pdsub['res'] = pdsub['tp'] - pdsub['e'] - pdsub['viwvd']

#     # Normalize
#     for var in ['tp']+m1vars:
#         pdsub[var] = ( pdsub[var] - np.mean(pdsub[var]) ) / np.std(pdsub[var])

#     # Make Model
#     model1 = lm.OLS(pdsub['tp'], pdsub.loc[:,m1vars]).fit()
    
#     # Extract Outputs
#     r2out.append(model1.rsquared_adj)
#     bout.append(np.array(model1.params))
#     bminout.append(np.array(model1.conf_int())[:,0])
#     bmaxout.append(np.array(model1.conf_int())[:,1])
#     pout.append(np.array(model1.pvalues))

#     # Append dimensions
#     mout.append(SSS[mi])
    
# # Compile outputs and write files
# pdout = pd.DataFrame({'season':mout, 'r2':r2out})
# for i in range(len(m1vars)):
#     pdout['beta_'+m1vars[i]] = np.array(bout)[:,i]
#     pdout['bmin_'+m1vars[i]] = np.array(bminout)[:,i]
#     pdout['bmax_'+m1vars[i]] = np.array(bmaxout)[:,i]
#     pdout['pval_'+m1vars[i]] = np.array(pout)[:,i]
    
# pdout.to_csv(outpath+"/Regress_"+REG+"_TotalPrecip_"+'-'.join(m1vars)+"_"+str(rad)+"km_"+str(starttime[0])+md.dd[starttime[1]-1]+"-"+str(endtime2[0])+md.dd[endtime2[1]-1]+"_"+V+".csv", index=False)


# ''' Model 1: tp_in = e_in + viwvd_in + ext'''
# # Prep output lists
# mout, r2out, bout, pout, bminout, bmaxout = [[]for i in range(6)]
# m1vars = ['e_in','viwvd_in','most_out']

# # Loop through each crossing region and season
# for mi in range(len(mons)):
#     # Subset by region and season
#     pdsub = pdfmb.loc[np.in1d(pdfmb['month'],mons[mi]) & (pdfmb['regflag_out'] == False)]
#     # pdsubas = pdfmbas.loc[np.in1d(pdfmbas['month'],mons[mi]) & (pdfmbas['regflag_out'] == False)]
        
#     # Change sign of e and vimd and tcwLast_in (but NOT tcwLast_out)
#     for var in ['e_in','e_out','viwvd_in','viwvd_out','tcwLast_in']:
#         pdsub[var] = -1*pdsub[var]
    
#     # Create a net external variable
#     pdsub['most_out'] = pdsub['tp_in'] - pdsub['e_in'] - pdsub['viwvd_in']

#     # Normalize
#     for var in ['tp_in'] + m1vars:
#         pdsub[var] = ( pdsub[var] - np.mean(pdsub[var]) ) / np.std(pdsub[var])

#     # Make Model
#     model1 = lm.OLS(pdsub['tp_in'], pdsub.loc[:,m1vars]).fit()
    
#     # Extract Outputs
#     r2out.append(model1.rsquared_adj)
#     bout.append(np.array(model1.params))
#     bminout.append(np.array(model1.conf_int())[:,0])
#     bmaxout.append(np.array(model1.conf_int())[:,1])
#     pout.append(np.array(model1.pvalues))

#     # Append dimensions
#     mout.append(SSS[mi])
    
# # Compile outputs and write files
# pdout = pd.DataFrame({'season':mout, 'r2':r2out})
# for i in range(len(m1vars)):
#     pdout['beta_'+m1vars[i]] = np.array(bout)[:,i]
#     pdout['bmin_'+m1vars[i]] = np.array(bminout)[:,i]
#     pdout['bmax_'+m1vars[i]] = np.array(bmaxout)[:,i]
#     pdout['pval_'+m1vars[i]] = np.array(pout)[:,i]
    
# pdout.to_csv(outpath+"/Regress_"+REG+"_TotalPrecip_"+'-'.join(m1vars)+"_"+str(rad)+"km_"+str(starttime[0])+md.dd[starttime[1]-1]+"-"+str(endtime2[0])+md.dd[endtime2[1]-1]+"_"+V+".csv", index=False)


# ''' Model 2: tpAvg = tcwAvg * p_gradAvg '''
# # Prep output lists
# mout, r2out, bout, pout, bminout, bmaxout = [[]for i in range(6)]

# cycvar = 'p_gradAvg_in'

# # Calculate Average Precipitation Rate
# pdfmb['tpAvg_in'] = pdfmb['tp_in'] / pdfmb['hours_in']
        
# # Loop through each crossing region and season
# for mi in range(len(mons)):
#     # Subset by region and season
#     pdsub = pdfmb.loc[np.in1d(pdfmb['month'],mons[mi]) & (pdfmb['regflag_out'] == False)]
#     # pdsubas = pdfmbas.loc[np.in1d(pdfmbas['month'],mons[mi]) & (pdfmbas['regflag_out'] == False)]
    
#     # Normalize
#     for var in ['tpAvg_in','tcwAvg_in',cycvar]:
#         pdsub[var] = ( pdsub[var] - np.mean(pdsub[var]) ) / np.std(pdsub[var])
        
#     # Calculate Interaction
#     pdsub['Interaction'] = pdsub['tcwAvg_in'] * pdsub[cycvar]

#     # Make Model
#     X = pdsub.loc[:,('tcwAvg_in',cycvar,'Interaction','constant')]
#     model2 = lm.OLS(pdsub['tpAvg_in'], X).fit()
    
#     # Extract Outputs
#     r2out.append(model2.rsquared_adj)
#     bout.append(np.array(model2.params))
#     bminout.append(np.array(model2.conf_int())[:,0])
#     bmaxout.append(np.array(model2.conf_int())[:,1])
#     pout.append(np.array(model2.pvalues))

#     # Append dimensions
#     mout.append(SSS[mi])
    
# # Compile outputs and write files
# pdout = pd.DataFrame({'season':mout, 'r2':r2out})
# varout = ('tcwAvg_in',cycvar,'Interaction')
# for i in range(3):
#     pdout['beta_'+varout[i]] = np.array(bout)[:,i]
#     pdout['bmin_'+varout[i]] = np.array(bminout)[:,i]
#     pdout['bmax_'+varout[i]] = np.array(bmaxout)[:,i]
#     pdout['pval_'+varout[i]] = np.array(pout)[:,i]

# pdout.to_csv(outpath+"/Regress_"+REG+"_PrecipRate_in_TCW_in_"+cycvar+"_"+str(rad)+"km_"+str(starttime[0])+md.dd[starttime[1]-1]+"-"+str(endtime2[0])+md.dd[endtime2[1]-1]+"_"+V+".csv", index=False)


# ''' Model 3: tp_in_tot = tp_in * hours_in'''
# # Prep output lists
# mout, r2out, bout, pout, bminout, bmaxout = [[]for i in range(6)]

# # Calculate Average Precipitation Rate
# pdfmb['tpAvg_in'] = pdfmb['tp_in'] / pdfmb['hours_in']
        
# # Loop through each crossing region and season
# for mi in range(len(mons)):
#     # Subset by region and season
#     pdsub = pdfmb.loc[np.in1d(pdfmb['month'],mons[mi]) & (pdfmb['regflag_out'] == False)]
#     # pdsubas = pdfmbas.loc[np.in1d(pdfmbas['month'],mons[mi]) & (pdfmbas['regflag_out'] == False)]
    
#     # Normalize
#     for var in ['tpAvg_in','tp_in','hours_in']:
#         pdsub[var] = ( pdsub[var] - np.mean(pdsub[var]) ) / np.std(pdsub[var])
        
#     # Calculate Interaction
#     pdsub['Interaction'] = pdsub['tpAvg_in'] * pdsub['hours_in']

#     # Make Model
#     X = pdsub.loc[:,('tpAvg_in','hours_in','Interaction','constant')]
#     model2 = lm.OLS(pdsub['tp_in'], X).fit()
    
#     # Extract Outputs
#     r2out.append(model2.rsquared_adj)
#     bout.append(np.array(model2.params))
#     bminout.append(np.array(model2.conf_int())[:,0])
#     bmaxout.append(np.array(model2.conf_int())[:,1])
#     pout.append(np.array(model2.pvalues))

#     # Append dimensions
#     mout.append(SSS[mi])
    
# # Compile outputs and write files
# pdout = pd.DataFrame({'season':mout, 'r2':r2out})
# varout = ('tpAvg_in','hours_in','Interaction')
# for i in range(3):
#     pdout['beta_'+varout[i]] = np.array(bout)[:,i]
#     pdout['bmin_'+varout[i]] = np.array(bminout)[:,i]
#     pdout['bmax_'+varout[i]] = np.array(bmaxout)[:,i]
#     pdout['pval_'+varout[i]] = np.array(pout)[:,i]

# pdout.to_csv(outpath+"/Regress_"+REG+"_TotalPrecip_tp_in-hours_in_"+str(rad)+"km_"+str(starttime[0])+md.dd[starttime[1]-1]+"-"+str(endtime2[0])+md.dd[endtime2[1]-1]+"_"+V+".csv", index=False)

# ''' Model X1: (tp_in + tcwLast_in) = e_in + viwvd_in + tcwLast_out '''
# # Prep output lists
# mout, r2out, bout, pout, bminout, bmaxout = [[]for i in range(6)]
# m1vars = ['e_in','viwvd_in','tcwLast_out']

# # Loop through each crossing region and season
# for mi in range(len(mons)):
#     # Subset by region and season
#     pdsub = pdfmb.loc[np.in1d(pdfmb['month'],mons[mi]) & (pdfmb['regflag_out'] == False)]
#     # pdsubas = pdfmbas.loc[np.in1d(pdfmbas['month'],mons[mi]) & (pdfmbas['regflag_out'] == False)]
    
#     # Normalize
#     for var in ['final_in'] + m1vars:
#         pdsub[var] = ( pdsub[var] - np.mean(pdsub[var]) ) / np.std(pdsub[var])

#     # Make Model
#     model1 = lm.OLS(pdsub['final_in'], pdsub.loc[:,m1vars]).fit()
    
#     # Extract Outputs
#     r2out.append(model1.rsquared_adj)
#     bout.append(np.array(model1.params))
#     bminout.append(np.array(model1.conf_int())[:,0])
#     bmaxout.append(np.array(model1.conf_int())[:,1])
#     pout.append(np.array(model1.pvalues))

#     # Append dimensions
#     mout.append(SSS[mi])
    
# # Compile outputs and write files
# pdout = pd.DataFrame({'season':mout, 'r2':r2out})
# for i in range(len(m1vars)):
#     pdout['beta_'+m1vars[i]] = np.array(bout)[:,i]
#     pdout['bmin_'+m1vars[i]] = np.array(bminout)[:,i]
#     pdout['bmax_'+m1vars[i]] = np.array(bmaxout)[:,i]
#     pdout['pval_'+m1vars[i]] = np.array(pout)[:,i]
    
# pdout.to_csv(outpath+"/Regress_"+REG+"_final_in_"+'-'.join(m1vars)+"_"+str(rad)+"km_"+str(starttime[0])+md.dd[starttime[1]-1]+"-"+str(endtime2[0])+md.dd[endtime2[1]-1]+"_"+V+".csv", index=False)

# ''' Model X2: tp_in = e_in + viwvd_in + tcwLast_out + tcwLast_in '''
# # Prep output lists
# mout, r2out, bout, pout, bminout, bmaxout = [[]for i in range(6)]
# m1vars = ['e_in','viwvd_in','tcwLast_out', 'tcwLast_in']

# # Loop through each crossing region and season
# for mi in range(len(mons)):
#     # Subset by region and season
#     pdsub = pdfmb.loc[np.in1d(pdfmb['month'],mons[mi]) & (pdfmb['regflag_out'] == False)]
#     # pdsubas = pdfmbas.loc[np.in1d(pdfmbas['month'],mons[mi]) & (pdfmbas['regflag_out'] == False)]
    
#     # Normalize
#     for var in ['tp_in'] + m1vars:
#         pdsub[var] = ( pdsub[var] - np.mean(pdsub[var]) ) / np.std(pdsub[var])

#     # Make Model
#     model1 = lm.OLS(pdsub['tp_in'], pdsub.loc[:,m1vars]).fit()
    
#     # Extract Outputs
#     r2out.append(model1.rsquared_adj)
#     bout.append(np.array(model1.params))
#     bminout.append(np.array(model1.conf_int())[:,0])
#     bmaxout.append(np.array(model1.conf_int())[:,1])
#     pout.append(np.array(model1.pvalues))

#     # Append dimensions
#     mout.append(SSS[mi])
    
# # Compile outputs and write files
# pdout = pd.DataFrame({'season':mout, 'r2':r2out})
# for i in range(len(m1vars)):
#     pdout['beta_'+m1vars[i]] = np.array(bout)[:,i]
#     pdout['bmin_'+m1vars[i]] = np.array(bminout)[:,i]
#     pdout['bmax_'+m1vars[i]] = np.array(bmaxout)[:,i]
#     pdout['pval_'+m1vars[i]] = np.array(pout)[:,i]
    
# pdout.to_csv(outpath+"/Regress_"+REG+"_tp_in_"+'-'.join(m1vars)+"_"+str(rad)+"km_"+str(starttime[0])+md.dd[starttime[1]-1]+"-"+str(endtime2[0])+md.dd[endtime2[1]-1]+"_"+V+".csv", index=False)


# ''' Model X3: tp_in = e_in + viwvd_in + tcwDiff '''
# # Prep output lists
# mout, r2out, bout, pout, bminout, bmaxout = [[]for i in range(6)]
# m1vars = ['e_in','viwvd_in','tcwDiff']

# # Loop through each crossing region and season
# for mi in range(len(mons)):
#     # Subset by region and season
#     pdsub = pdfmb.loc[np.in1d(pdfmb['month'],mons[mi]) & (pdfmb['regflag_out'] == False)]
#     # pdsubas = pdfmbas.loc[np.in1d(pdfmbas['month'],mons[mi]) & (pdfmbas['regflag_out'] == False)]
    
#     # Normalize
#     for var in ['tp_in'] + m1vars:
#         pdsub[var] = ( pdsub[var] - np.mean(pdsub[var]) ) / np.std(pdsub[var])

#     # Make Model
#     model1 = lm.OLS(pdsub['tp_in'], pdsub.loc[:,m1vars]).fit()
    
#     # Extract Outputs
#     r2out.append(model1.rsquared_adj)
#     bout.append(np.array(model1.params))
#     bminout.append(np.array(model1.conf_int())[:,0])
#     bmaxout.append(np.array(model1.conf_int())[:,1])
#     pout.append(np.array(model1.pvalues))

#     # Append dimensions
#     mout.append(SSS[mi])
    
# # Compile outputs and write files
# pdout = pd.DataFrame({'season':mout, 'r2':r2out})
# for i in range(len(m1vars)):
#     pdout['beta_'+m1vars[i]] = np.array(bout)[:,i]
#     pdout['bmin_'+m1vars[i]] = np.array(bminout)[:,i]
#     pdout['bmax_'+m1vars[i]] = np.array(bmaxout)[:,i]
#     pdout['pval_'+m1vars[i]] = np.array(pout)[:,i]
    
# pdout.to_csv(outpath+"/Regress_"+REG+"_tp_in_"+'-'.join(m1vars)+"_"+str(rad)+"km_"+str(starttime[0])+md.dd[starttime[1]-1]+"-"+str(endtime2[0])+md.dd[endtime2[1]-1]+"_"+V+".csv", index=False)

''' Model R2: tprate_in = erate_in + viwvdrate_in + tcwLast_out + tcwLast_in  '''
''' Model R3: tprate_in = erate_in + viwvdrate_in + tcwDiff '''
# Prep output lists
m1varslist = [['erate_in','viwvdrate_in','tcwDiff'],['erate_in','viwvdrate_in','tcwLast_out','tcwLast_in']]
yvarlist = ['tprate_in','tprate_in']

for i in range(len(yvarlist)):
    m1vars = m1varslist[i]
    yvar = yvarlist[i]

    mout, r2out, bout, pout, bminout, bmaxout = [[]for i in range(6)]
    
    # Loop through each crossing region and season
    for mi in range(len(mons)):
        # Subset by region and season
        pdsub = pdfmb.loc[np.in1d(pdfmb['month'],mons[mi]) & (pdfmb['regflag_out'] == False)]
        # pdsubas = pdfmbas.loc[np.in1d(pdfmbas['month'],mons[mi]) & (pdfmbas['regflag_out'] == False)]
        
        # Normalize
        for var in [yvar] + m1vars:
            pdsub[var] = ( pdsub[var] - np.mean(pdsub[var]) ) / np.std(pdsub[var])
    
        # Make Model
        model1 = lm.OLS(pdsub[yvar], pdsub.loc[:,m1vars]).fit()
        
        # Extract Outputs
        r2out.append(model1.rsquared_adj)
        bout.append(np.array(model1.params))
        bminout.append(np.array(model1.conf_int())[:,0])
        bmaxout.append(np.array(model1.conf_int())[:,1])
        pout.append(np.array(model1.pvalues))
    
        # Append dimensions
        mout.append(SSS[mi])
        
    # Compile outputs and write files
    pdout = pd.DataFrame({'season':mout, 'r2':r2out})
    for i in range(len(m1vars)):
        pdout['beta_'+m1vars[i]] = np.array(bout)[:,i]
        pdout['bmin_'+m1vars[i]] = np.array(bminout)[:,i]
        pdout['bmax_'+m1vars[i]] = np.array(bmaxout)[:,i]
        pdout['pval_'+m1vars[i]] = np.array(pout)[:,i]
        
    pdout.to_csv(outpath+"/Regress_"+REG+"_"+yvar+'_'+'-'.join(m1vars)+"_"+str(rad)+"km_"+str(starttime[0])+md.dd[starttime[1]-1]+"-"+str(endtime2[0])+md.dd[endtime2[1]-1]+"_"+V+".csv", index=False)
