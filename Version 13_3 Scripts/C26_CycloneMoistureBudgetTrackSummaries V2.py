#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Date Created:  2023 Aug 23
Date Modified: 2023 Aug 30
Author: Alex Crawford

Purpose: 
(1) Calculate track-wise correlations between total precipitation produced 
by a cyclone and additional moisture budget properties
(2) Calculate linear regression coefficients for track-wise decomposition of
total precipitation and precipitation rate
"""
'''**********
Load Modules
**********'''
import CycloneModule_13_3 as md
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from statsmodels.regression import linear_model as lm

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

'''**********
Define Variables
**********'''
dataset = 'ERA5'
cycver = '13_2E5R'
subset = 'BBox10'
typ = 'System'
regvar = 'reg2'
regtype = 'crossReg' # 'maxintReg' # 'genReg', 'lysReg', 
regs = np.arange(1,12)

V = 'V7'
rad = 800 # radius for constant kernel size (units: km)
tres = 3 # temporal resolution of cyclone data

starttime = [1978,11,1,0,0,0] # inclusive
endtime = [2025,1,1,0,0,0] # exclusive
filetimestep = [0,1,0,0,0,0] # time step between cyclone files
inittime = [1940,1,1,0,0,0] # Initiation time for cyclone tracking dataset
dateref = [1900,1,1,0,0,0] # Reference date for input data
mons = np.arange(1,13)

varlistam = ['tp','e','viwvd','tcw_Adv','siconc','tcw','tisr', 'tsr', 'ttr','tnetrad']
varlistas = ['hours','tp','e','viwvd','tcw_Adv','siconc','tcw', 'tisr', 'tsr', 'ttr','tnetrad']

varsumlist = ['tp','e','viwvd','tcw_Adv','hours']
varavglist = ['lat','lon','Pratio','p_cent','depth','radius','p_grad','DsqP','uv','DpDt','tcw','siconc','tisr', 'tsr', 'ttr','tnetrad']
varmaxlist = ['lat','depth','radius','p_grad','DsqP','tp','e','viwvd','tcw','siconc']
varminlist = ['lat','p_cent','DpDt','ttr']
varfirstlist = ['tcw']
varlastlist = ['tcw']

path = "/media/alex/Datapool/CycloneTracking/tracking"+cycver+"/"+subset+"/Aggregation"+typ
mbpath = path+"/SpatialAvgEnv_"+str(rad)+"km"
mbapath = path+"/SpatialAvgEnvAmly_"+str(rad)+"km"
outpath = path+"/MoistureBudget"+V

'''**********
Track-by-Track Summaries
**********'''
endtime2 = md.timeAdd(endtime,[-1*t for t in filetimestep])
print('Summaries')
try:
    pdfmb = pd.read_csv(outpath+"/MoistureBudget_TrackSummary_"+str(rad)+"km_"+str(starttime[0])+md.dd[starttime[1]-1]+"-"+str(endtime2[0])+md.dd[endtime2[1]-1]+"_"+V+".csv")
    pdfmbas = pd.read_csv(outpath+"/MoistureBudgetAmlySum_TrackSummary_"+str(rad)+"km_"+str(starttime[0])+md.dd[starttime[1]-1]+"-"+str(endtime2[0])+md.dd[endtime2[1]-1]+"_"+V+".csv")
    pdfmbaa = pd.read_csv(outpath+"/MoistureBudgetAmlyAvg_TrackSummary_"+str(rad)+"km_"+str(starttime[0])+md.dd[starttime[1]-1]+"-"+str(endtime2[0])+md.dd[endtime2[1]-1]+"_"+V+".csv")
    
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
        
        mbs = mb.loc[(mb['age'] > 0) & (mb['age'] < 1),('sid','year','month')].groupby(by=('sid')).first().reset_index()
        for var in varsumlist:
            mbs[var] = mb.loc[(mb['age'] > 0) & (mb['age'] < 1),('sid',var)].groupby(by=('sid')).sum()[var].values
        mbs['Peff'] = mbs['tp'] / (-1*(mbs['e'] + mbs['viwvd']))
        
        for var in varavglist:
            mbs[var+'Avg'] = mb.loc[(mb['age'] > 0) & (mb['age'] < 1),('sid',var)].groupby(by=('sid')).mean()[var].values
        for var in varmaxlist:
            mbs[var+'Max'] = mb.loc[(mb['age'] > 0) & (mb['age'] < 1),('sid',var)].groupby(by=('sid')).max()[var].values
        for var in varminlist:
            mbs[var+'Min'] = mb.loc[(mb['age'] > 0) & (mb['age'] < 1),('sid',var)].groupby(by=('sid')).min()[var].values
        for var in varfirstlist:
            mbs[var+'First'] = mb.loc[:,('sid',var)].groupby(by=('sid')).first()[var].values
        for var in varfirstlist:
            mbs[var+'Last'] = mb.loc[:,('sid',var)].groupby(by=('sid')).last()[var].values
        mbs['tcwDiff'] = mbs['tcwFirst'] - mbs['tcwLast']
    
        # Anomalies - By Sum
        mbas = mba.loc[(mba['age'] > 0) & (mba['age'] < 1),['sid','year','month']+varlistas].groupby(by=('sid')).sum().reset_index()
        mbas['year'] = mt[0]
        mbas['month'] = mt[1]
        
        # Anomalies - By Avg
        mbaa = mba.loc[(mba['age'] > 0) & (mba['age'] < 1),['sid','year','month']+varlistam].groupby(by=('sid')).mean().reset_index()

        # Append        
        mblist.append(mbs), mbasumlist.append(mbas), mbaalist.append(mbaa)
        
        mt = md.timeAdd(mt,filetimestep)
    
    pdfmb = pd.concat(mblist, ignore_index=True)
    pdfmbas = pd.concat(mbasumlist, ignore_index=True)
    pdfmbaa = pd.concat(mbaalist, ignore_index=True)
    
    # Load regions
    pdfreg = pd.read_csv(path+"/Regions-"+regvar+"_"+cycver+"_"+subset+"_"+typ+"_"+str(starttime[0])+md.dd[starttime[1]-1]+"-"+str(endtime2[0])+md.dd[endtime2[1]-1]+".csv")
    regvarlist = [regvar for regvar in pdfreg.columns if 'Reg' in regvar]
    
    # Merge datasets
    pdfmb = pd.merge(pdfmb, pdfreg.loc[:,['year','month','sid']+regvarlist], on=['year','month','sid'])
    pdfmbas = pd.merge(pdfmbas, pdfreg.loc[:,['year','month','sid']+regvarlist], on=['year','month','sid'])
    pdfmbaa = pd.merge(pdfmbaa, pdfreg.loc[:,['year','month','sid']+regvarlist], on=['year','month','sid'])
    
    # Add N70 region
    pdfmb['crossN70'] = np.where(pdfmb['latMax'] >= 70, 1, 0)
    pdfmbas['crossN70'] = np.where(pdfmb['latMax'] >= 70, 1, 0)
    pdfmbaa['crossN70'] = np.where(pdfmb['latMax'] >= 70, 1, 0)
    
    # Write to File
    pdfmb.to_csv(outpath+"/MoistureBudget_TrackSummary_"+str(rad)+"km_"+str(starttime[0])+md.dd[starttime[1]-1]+"-"+Y+M+"_"+V+".csv", index=False)
    pdfmbas.to_csv(outpath+"/MoistureBudgetAmlySum_TrackSummary_"+str(rad)+"km_"+str(starttime[0])+md.dd[starttime[1]-1]+"-"+Y+M+"_"+V+".csv", index=False)
    pdfmbaa.to_csv(outpath+"/MoistureBudgetAmlyAvg_TrackSummary_"+str(rad)+"km_"+str(starttime[0])+md.dd[starttime[1]-1]+"-"+Y+M+"_"+V+".csv", index=False)

'''**********
Correlations & Regressions
**********'''
print("Correlations & Regressions")
varlist = pdfmb.columns[4:-15]

''' Correlatons '''
# Prep output lists
mout, regout, rout, pout = [], [], [[] for vi in varlist], [[] for vi in varlist]
routas, poutas = [[] for vi in varlistas], [[] for vi in varlistas]
routaa, poutaa = [[] for vi in varlistam], [[] for vi in varlistam]

# Loop through each crossing region and season
for mi in [0,1,2,3,4,5,6,7,8,9,10,11]:
    for r in regs:
        # Subset by region and season
        if regtype == 'crossReg':
            pdsub = pdfmb.loc[np.in1d(pdfmb['month'],[mons[mi],mons[mi-1],mons[mi-2]]) & (pdfmb[regtype+str(r)] == 1)]
            pdsubas = pdfmbas.loc[np.in1d(pdfmbas['month'],[mons[mi],mons[mi-1],mons[mi-2]]) & (pdfmbas[regtype+str(r)] == 1)]
            pdsubaa = pdfmbaa.loc[np.in1d(pdfmbaa['month'],[mons[mi],mons[mi-1],mons[mi-2]]) & (pdfmbaa[regtype+str(r)] == 1)]

        else:
            pdsub = pdfmb.loc[np.in1d(pdfmb['month'],[mons[mi],mons[mi-1],mons[mi-2]]) & (pdfmb[regtype] == r)]
            pdsubas = pdfmbas.loc[np.in1d(pdfmbas['month'],[mons[mi],mons[mi-1],mons[mi-2]]) & (pdfmbas[regtype] == r)]
            pdsubaa = pdfmbaa.loc[np.in1d(pdfmbaa['month'],[mons[mi],mons[mi-1],mons[mi-2]]) & (pdfmbaa[regtype] == r)]
        
        # RAW DATA
        for vi in range(len(varlist)):
            # Correlation
            rval, pval = stats.spearmanr(pdsub['tp'],pdsub[varlist[vi]])
            
            # Append to lists
            rout[vi].append(rval), pout[vi].append(pval)

        # SUMMED ANOMALIES
        for vi in range(len(varlistas)):
            # Correlation
            rvalas, pvalas = stats.spearmanr(pdsubas['tp'],pdsubas[varlistas[vi]])
            
            # Append to lists
            routas[vi].append(rvalas), poutas[vi].append(pvalas)
        
        # AVG ANOMALIES
        for vi in range(len(varlistam)):
            # Correlation
            rvalaa, pvalaa = stats.spearmanr(pdsubaa['tp'],pdsubaa[varlistam[vi]])
            
            # Append to lists
            routaa[vi].append(rvalaa), poutaa[vi].append(pvalaa)
        
        # Append dimensions
        mout.append(md.sss[mi]), regout.append(r)

# Compile outputs and write files
pdout = pd.DataFrame({'Season':mout, 'Region':regout})
for vi in range(len(varlist)):
    pdout["r_tp-"+varlist[vi]] = rout[vi]
    pdout['p_tp-'+varlist[vi]] = pout[vi]
pdout.to_csv(outpath+"/Spearman_Track-"+regtype+"-tp_"+str(rad)+"km_"+str(starttime[0])+md.dd[starttime[1]-1]+"-"+str(endtime2[0])+md.dd[endtime2[1]-1]+"_"+V+".csv", index=False)

pdoutas = pd.DataFrame({'Season':mout, 'Region':regout})
for vi in range(len(varlistas)):
    pdoutas["r_tp-"+varlistas[vi]] = routas[vi]
    pdoutas['p_tp-'+varlistas[vi]] = poutas[vi]
pdoutas.to_csv(outpath+"/Spearman_Track-"+regtype+"-tpAmlySum_"+str(rad)+"km_"+str(starttime[0])+md.dd[starttime[1]-1]+"-"+str(endtime2[0])+md.dd[endtime2[1]-1]+"_"+V+".csv", index=False)

pdoutaa = pd.DataFrame({'Season':mout, 'Region':regout})
for vi in range(len(varlistam)):
    pdoutaa["r_tp-"+varlistam[vi]] = routaa[vi]
    pdoutaa['p_tp-'+varlistam[vi]] = poutaa[vi]
pdoutaa.to_csv(outpath+"/Spearman_Track-"+regtype+"-tpAmlyAvg_"+str(rad)+"km_"+str(starttime[0])+md.dd[starttime[1]-1]+"-"+str(endtime2[0])+md.dd[endtime2[1]-1]+"_"+V+".csv", index=False)
   
''' Model 1: tp = e + viwvd + adv '''
# Prep output lists
mout, regout, r2out, bout, pout, bminout, bmaxout = [[]for i in range(7)]

# Loop through each crossing region and season
for mi in [0,1,2,3,4,5,6,7,8,9,10,11]:
    for r in regs:
        # Subset by crossing and season
        if regtype == 'crossReg':
            pdsub = pdfmb.loc[np.in1d(pdfmb['month'],[mons[mi],mons[mi-1],mons[mi-2]]) & (pdfmb[regtype+str(r)] == 1)]
        else:
            pdsub = pdfmb.loc[np.in1d(pdfmb['month'],[mons[mi],mons[mi-1],mons[mi-2]]) & (pdfmb[regtype] == r)]
        
        # Change sign of e and viwvd
        pdsub['e'] = -1*pdsub['e']
        pdsub['viwvd'] = -1*pdsub['viwvd']
        
        # Normalize
        for var in ['tp','e','viwvd','tcw_Adv']:
            pdsub[var] = ( pdsub[var] - np.mean(pdsub[var]) ) / np.std(pdsub[var])

        # Make Model
        X = pdsub.loc[:,('e','viwvd','tcw_Adv')]
        model1 = lm.OLS(pdsub['tp'], X).fit()
        
        # Extract Outputs
        r2out.append(model1.rsquared_adj)
        bout.append(np.array(model1.params))
        bminout.append(np.array(model1.conf_int())[:,0])
        bmaxout.append(np.array(model1.conf_int())[:,1])
        pout.append(np.array(model1.pvalues))

        # Append dimensions
        mout.append(md.sss[mi]), regout.append(r) 
        
# Compile outputs and write files
pdout = pd.DataFrame({'season':mout, 'region':regout, 'r2':r2out})
varout = ('e','viwvd','tcw_Adv')
for i in range(3):
    pdout['beta_'+varout[i]] = np.array(bout)[:,i]
    pdout['bmin_'+varout[i]] = np.array(bminout)[:,i]
    pdout['bmax_'+varout[i]] = np.array(bmaxout)[:,i]
    pdout['pval_'+varout[i]] = np.array(pout)[:,i]

pdout.to_csv(outpath+"/Regress-"+regtype+"_TotalPrecip_Evap_Div_"+str(rad)+"km_"+str(starttime[0])+md.dd[starttime[1]-1]+"-"+str(endtime2[0])+md.dd[endtime2[1]-1]+"_"+V+".csv", index=False)

''' Model 2: tpAvg = tcwAvg * p_gradAvg '''
# Prep output lists
mout, regout, r2out, bout, pout, bminout, bmaxout = [[]for i in range(7)]

# Calculate Average Precipitation Rate
pdfmb['tpAvg'] = pdfmb['tp'] / pdfmb['hours']
        
# Loop through each crossing region and season
for mi in [0,1,2,3,4,5,6,7,8,9,10,11]:
    for r in regs:
        # Subset by crossing and season
        if regtype == 'crossReg':
            pdsub = pdfmb.loc[np.in1d(pdfmb['month'],[mons[mi],mons[mi-1],mons[mi-2]]) & (pdfmb[regtype+str(r)] == 1)]
        else:
            pdsub = pdfmb.loc[np.in1d(pdfmb['month'],[mons[mi],mons[mi-1],mons[mi-2]]) & (pdfmb[regtype] == r)]

        # Normalize
        for var in ['tpAvg','tcwAvg','p_gradAvg']:
            pdsub[var] = ( pdsub[var] - np.mean(pdsub[var]) ) / np.std(pdsub[var])
            
        # Calculate Interaction
        pdsub['Interaction'] = pdsub['tcwAvg'] * pdsub['p_gradAvg']

        # Make Model
        X = pdsub.loc[:,('tcwAvg','p_gradAvg','Interaction')]
        model2 = lm.OLS(pdsub['tpAvg'], X).fit()
        
        # Extract Outputs
        r2out.append(model2.rsquared_adj)
        bout.append(np.array(model2.params))
        bminout.append(np.array(model2.conf_int())[:,0])
        bmaxout.append(np.array(model2.conf_int())[:,1])
        pout.append(np.array(model2.pvalues))

        # Append dimensions
        mout.append(md.sss[mi]), regout.append(r) 
        
# Compile outputs and write files
pdout = pd.DataFrame({'season':mout, 'region':regout, 'r2':r2out})
varout = ('tcwAvg','p_gradAvg','Interaction')
for i in range(3):
    pdout['beta_'+varout[i]] = np.array(bout)[:,i]
    pdout['bmin_'+varout[i]] = np.array(bminout)[:,i]
    pdout['bmax_'+varout[i]] = np.array(bmaxout)[:,i]
    pdout['pval_'+varout[i]] = np.array(pout)[:,i]

pdout.to_csv(outpath+"/Regress-"+regtype+"_PrecipRate_TCW_p_gradAvg_"+str(rad)+"km_"+str(starttime[0])+md.dd[starttime[1]-1]+"-"+str(endtime2[0])+md.dd[endtime2[1]-1]+"_"+V+".csv", index=False)

''' Model 3: tp = tpAvg * hours'''
# Prep output lists
mout, regout, r2out, bout, pout, bminout, bmaxout = [[]for i in range(7)]

# Calculate Average Precipitation Rate
pdfmb['tpAvg'] = pdfmb['tp'] / pdfmb['hours']
        
# Loop through each crossing region and season
for mi in [0,1,2,3,4,5,6,7,8,9,10,11]:
    for r in regs:
        # Subset by crossing and season
        if regtype == 'crossReg':
            pdsub = pdfmb.loc[np.in1d(pdfmb['month'],[mons[mi],mons[mi-1],mons[mi-2]]) & (pdfmb[regtype+str(r)] == 1)]
        else:
            pdsub = pdfmb.loc[np.in1d(pdfmb['month'],[mons[mi],mons[mi-1],mons[mi-2]]) & (pdfmb[regtype] == r)]

        # Normalize
        for var in ['tpAvg','tp','hours']:
            pdsub[var] = ( pdsub[var] - np.mean(pdsub[var]) ) / np.std(pdsub[var])
            
        # Calculate Interaction
        pdsub['Interaction'] = pdsub['tpAvg'] * pdsub['hours']

        # Make Model
        X = pdsub.loc[:,('tpAvg','hours','Interaction')]
        model2 = lm.OLS(pdsub['tp'], X).fit()
        
        # Extract Outputs
        r2out.append(model2.rsquared_adj)
        bout.append(np.array(model2.params))
        bminout.append(np.array(model2.conf_int())[:,0])
        bmaxout.append(np.array(model2.conf_int())[:,1])
        pout.append(np.array(model2.pvalues))

        # Append dimensions
        mout.append(md.sss[mi]), regout.append(r) 
        
# Compile outputs and write files
pdout = pd.DataFrame({'season':mout, 'region':regout, 'r2':r2out})
varout = ('tpAvg','hours','Interaction')
for i in range(3):
    pdout['beta_'+varout[i]] = np.array(bout)[:,i]
    pdout['bmin_'+varout[i]] = np.array(bminout)[:,i]
    pdout['bmax_'+varout[i]] = np.array(bmaxout)[:,i]
    pdout['pval_'+varout[i]] = np.array(pout)[:,i]

pdout.to_csv(outpath+"/Regress-"+regtype+"_TotalPrecip_PrecipRate_CycHours_"+str(rad)+"km_"+str(starttime[0])+md.dd[starttime[1]-1]+"-"+str(endtime2[0])+md.dd[endtime2[1]-1]+"_"+V+".csv", index=False)


print("Complete")
