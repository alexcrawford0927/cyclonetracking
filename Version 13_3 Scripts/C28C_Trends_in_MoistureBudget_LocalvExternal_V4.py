#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:47:13 2023
Modified 31 Jan 2025

@author: acrawfora
"""
from scipy import stats
import numpy as np
import pandas as pd
from statsmodels.regression import linear_model as lm
from scipy.stats import linregress
import CycloneModule_13_3 as md

'''**********
Define Variables
**********'''
dataset = 'ERA5'
cycver = '13_2E5R'
subset = 'BBox10'
typ = 'System'

V = 'V7'
rad = 800 # radius for constant kernel size (units: km)
regvals = [3] # [0] # [1]
# regname = ['Other','CAO','ILBK','MLNAtl','NPac','Med','Euro','Sib','EAsia','MLNAmer','NCan','Baf']
regname = ['CAOBK','N70','BK','CAO2']
gentype = 'external' # 'all' , local' , or 'external'

TIME = '197811-202412'
ymin, ymax = 1979, 2024
mons = [[11,12,1],[12,1,2],[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],\
        [11,12,1,2,3],[5,6,7,8,9],[10,11,12,1,2,3],[4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9,10,11,12]]
SSS = md.sss + ['NDJFM','MJJAS','Oct-Mar','Apr-Sep','Annual']
    
REG = '-'.join([regname[r] for r in regvals])
path = "/media/alex/Datapool/CycloneTracking/tracking"+cycver+"/"+subset
inpath = path+"/Aggregation"+typ+"/MoistureBudget"+V
outpath = inpath+"/"+REG

varlist1 = ['tp_in','e_in','viwvd_in','tcw_Adv_in','hours_in','Peff_in','tcwLast_in','tcwFirst_in',
            'tcw_AdvLast_in','tcw_AdvFirst_in','latAvg_in','lonAvg_in','PratioAvg_in',
            'p_centAvg_in','depthAvg_in','radiusAvg_in','p_gradAvg_in','DsqPAvg_in','uvAvg_in','DpDtAvg_in',
            'tcwAvg_in','siconcAvg_in','tisrAvg_in','tsrAvg_in','ttrAvg_in',
            'tnetradAvg_in','latMax_in','depthMax_in','radiusMax_in','p_gradMax_in',
            'DsqPMax_in','tpMax_in','eMax_in','viwvdMax_in','tcwMax_in','tp_resid',
            'siconcMax_in','latMin_in','p_centMin_in','DpDtMin_in','ttrMin_in',
            'max_out','most_out','min_out','final_in','final_out','final_resid',
            'tprate_in','erate_in','viwvdrate_in','tcwDiff','tcwDiff_resid','tcwDiff_out','count']
varlist2 = ['tp_out','e_out','viwvd_out','tcw_Adv_out','hours_out','Peff_out',
            'tcwLast_out','tcwFirst_out','tcw_AdvLast_out','tcw_AdvFirst_out',
            'latAvg_out','lonAvg_out','PratioAvg_out','p_centAvg_out',
            'depthAvg_out','radiusAvg_out','p_gradAvg_out','DsqPAvg_out','uvAvg_out',
            'DpDtAvg_out','tcwAvg_out','siconcAvg_out','tisrAvg_out','tsrAvg_out',
            'ttrAvg_out','tnetradAvg_out','latMax_out','depthMax_out','radiusMax_out',
            'p_gradMax_out','DsqPMax_out','tpMax_out','eMax_out','viwvdMax_out',
            'tcwMax_out','siconcMax_out','latMin_out','p_centMin_out','DpDtMin_out',
            'ttrMin_out']
varlist3 = ['tp_in_tot','e_in_tot','tcwLast_in_tot','tcwLast_out_tot','tcwFirst_in_tot',
            'viwvd_in_tot','hours_in_tot','max_out_tot','most_out_tot','min_out_tot','tp_resid_tot',
            'final_in_tot','final_out_tot','final_resid_tot','tcwDiff_tot','tcwDiff_resid_tot','tcwDiff_out_tot']


# varlist2 variables are only valid if the variable regflag_out is not NaN

'''**********
Main Analysis
**********'''
# Load data
pdfmb = pd.read_csv(outpath+"/SpatialAvgEnv_"+REG+"_TrackSummary_"+str(rad)+"km_"+TIME+"_"+V+".csv")

# Convert direction of of fluxes to be positive = more moisture
for var in ['e_in','viwvd_in','e_out','viwvd_out','tcwLast_in','tcw_AdvFirst_in']:
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

# Add seasonal year
pdfmb['syear'] = np.where(pdfmb['month'] >= 11, pdfmb['year']+1, pdfmb['year'])
pdfmb = pdfmb.loc[(pdfmb['syear'] <= ymax) & (pdfmb['syear'] >= ymin)]

if gentype == 'local':
    pdfmb = pdfmb.loc[np.isnan(pdfmb['latAvg_out'])]
    varlistA  = varlist1+ []
elif gentype == 'external':
    pdfmb = pdfmb.loc[np.isfinite(pdfmb['latAvg_out'])]
    varlistA = varlist1+varlist2
else:
    varlistA = varlist1+varlist2

# Prep Outputs
# varlist = list(pdfmb.columns[4:-2])
meanlist = []
meanlistm = []

outdf = pd.DataFrame({'season':SSS})
r2out, bout, bminout, bmaxout, pout = [[[] for i in range(len(varlistA))] for i in range(5)]

outdfm = pd.DataFrame({'season':SSS})
r2outm, boutm, bminoutm, bmaxoutm, poutm = [[[] for i in range(len(varlistA+varlist3))] for i in range(5)]

for m in mons:
    pdsub = pdfmb.loc[np.in1d(pdfmb['month'],m)]

    pdsubm = pdsub.loc[:,['syear','constant']+varlistA].groupby(by=['syear']).mean().reset_index()
    for var in varlist3:
        pdsubm[var] = pdsub.loc[:,['syear',var[:-4]]].groupby(by=['syear']).sum()[var[:-4]].values
    pdsubm['count'] = pdsub.loc[:,['syear','count']].groupby(by=['syear']).sum()['count'].values
    
    
    # Climatology
    meanlist.append( pdsub.mean() )
    meanlistm.append( pdsubm.mean() )
    
    # Trend - Track by Track
    for v, var in enumerate(varlistA):
        
        if v in varlist2:
            # Make Model
            X = pdsub.loc[(pdsub['regflag_out'] == False),['constant','year']]
            model2 = lm.OLS(pdsub.loc[(pdsub['regflag_out'] == False),var], X, missing='drop').fit()        
            
        else:
            # Make Model
            X = pdsub.loc[:,['constant','year']]
            model2 = lm.OLS(pdsub[var], X, missing='drop').fit()
        
        model2.summary()
                
        # Extract Outputs
        r2out[v].append(model2.rsquared_adj)
        bout[v].append(np.array(model2.params))
        bminout[v].append(np.array(model2.conf_int())[:,0])
        bmaxout[v].append(np.array(model2.conf_int())[:,1])
        pout[v].append(np.array(model2.pvalues))
    
    # Trend - Season by Season
    for v, var in enumerate(varlistA+varlist3):
        # Make Model
        X = pdsubm.loc[:,['constant','syear']]
        modelm = lm.OLS(pdsubm[var], X, missing='drop').fit()
        modelm.summary()
        
        # linregress(pdsubm['syear'], pdsubm[var])
        
        # Extract Outputs
        r2outm[v].append(modelm.rsquared_adj)
        boutm[v].append(np.array(modelm.params))
        bminoutm[v].append(np.array(modelm.conf_int())[:,0])
        bmaxoutm[v].append(np.array(modelm.conf_int())[:,1])
        poutm[v].append(np.array(modelm.pvalues))    
    
# Append trend data to data frame
for v, var in enumerate(varlistA+varlist3):
    if var in varlistA:
        outdf[var+"_trend"] = np.array(bout[v])[:,1]
        outdf[var+"_trendmax"] = np.array(bmaxout[v])[:,1]
        outdf[var+"_trendmin"] = np.array(bminout[v])[:,1]
        outdf[var+"_r2"] = np.array(r2out[v])
        outdf[var+"_pval"] = np.array(pout[v])[:,1]
    
    outdfm[var+"_trend"] = np.array(boutm[v])[:,1]
    outdfm[var+"_trendmax"] = np.array(bmaxoutm[v])[:,1]
    outdfm[var+"_trendmin"] = np.array(bminoutm[v])[:,1]
    outdfm[var+"_r2"] = np.array(r2outm[v])
    outdfm[var+"_pval"] = np.array(poutm[v])[:,1]

outdf.to_csv(outpath+"/SpatialAvgEnv_"+gentype+"TrackSummary_Trends_"+REG+"_"+str(rad)+"km_"+TIME+"_"+V+".csv", index=False)
outdfm.to_csv(outpath+"/SpatialAvgEnv_"+gentype+"SeasonSummary_Trends_"+REG+"_"+str(rad)+"km_"+TIME+"_"+V+".csv", index=False)
  
# Combine mean data into data frame
meandf = pd.DataFrame(meanlist)
meandf['season'] = SSS
meandf = meandf.loc[:,[col for col in meandf.columns if col not in ('month','constant','year','syear')]]
meandf.to_csv(outpath+"/SpatialAvgEnv_"+gentype+"TrackSummary_Climatology_"+REG+"_"+str(rad)+"km_"+TIME+"_"+V+".csv", index=False)

meandfm = pd.DataFrame(meanlistm)
meandfm['season'] = SSS
meandfm = meandfm.loc[:,[col for col in meandfm.columns if col not in ('month','constant','year','syear')]]
meandfm.to_csv(outpath+"/SpatialAvgEnv_"+gentype+"SeasonSummary_Climatology_"+REG+"_"+str(rad)+"km_"+TIME+"_"+V+".csv", index=False)
