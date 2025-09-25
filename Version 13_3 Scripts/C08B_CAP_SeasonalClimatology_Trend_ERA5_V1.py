"""
Author: Alex Crawford
Date Created: 15 Oct 2021
Date Modified: 19 Oct 2021

Purpose: Calculate averages and trends in CAP, including its decomposition
into precipitation rate, cyclone frequency, cyclone speed, and a residual term
"""

import numpy as np
from scipy import ndimage
import netCDF4 as nc
import CycloneModule_13_3 as md


def uniquelen(arr,adjust=0):
    '''Identifies the number of unique values in an array.
    Use a negative "adjust" value if there are placeholder values  (e.g., -9999).'''
    return len(np.unique(arr)) + adjust

'''*******************************************
Set up Environment & Variables
*******************************************'''
dataset = "ERA5"
verd, vert = "13test", "P"
subset = ""
typ = "System"

capfactor = 1000 # For converting from m to mm
tres = 3 # Temporal resolution in hours
shape = (0,720,720)
minN = 10 # minimum number of years needed to calculate a trend

path = "/media/alex/Datapool/"+dataset # "/Volumes/Prospero/"+dataset
cycprjpath = "/media/alex/Datapool/Projections" # "/Volumes/Cressida/Projections"
precippath = path+"/CAP/tracking"+verd+vert+"/"+subset

ymin, ymax = 1981, 2010
reftime = [1900,1,1,0,0,0]

trendnames = ['trend','yint','r2','pvalue','stderr']
trendunits = ['Trend', 'Y-Intercept of the trend','R^2 value for the trend','P-value for the trend','Standard error in the trend']

'''*******************************************
Main Analysis
*******************************************'''

years = np.arange(ymin,ymax+1)
YY = str(ymin) + "-" + str(ymax)

# Initiate outputs
nCAP_trend, captot_trend, caprate_trend, nCyc_trend, hCyc_trend, uv_trend, uvinv_trend, capresid_trend = [], [], [], [] ,[] ,[] ,[], []
nCAP_sd, captot_sd, caprate_sd, nCyc_sd, hCyc_sd, uv_sd, uvinv_sd, capresid_sd = [], [], [], [] ,[] ,[] ,[], []
nCAP_clim, captot_clim, caprate_clim, nCyc_clim, hCyc_clim, uv_clim, uvinv_clim, capresid_clim = [], [], [], [], [], [], [], []

for m in range(12):
    M = md.dd[m]
    print(M)
    
    # Initiate multi-month arrays
    nCAPM, captotM, caprateM,nCycM, hCycM,  = np.empty(shape), np.empty(shape), np.empty(shape),np.empty(shape), np.empty(shape)
    uvM, uvinvM, capresidM = np.empty(shape), np.empty(shape), np.empty(shape)
    
    for y in years:
        Y = str(y)
        
        # Set times
        mt2 = [y,m+1,1,0,0,0]
        mt1 = md.timeAdd(mt2,[0,-1,0,0,0,0])
        mt0 = md.timeAdd(mt1,[0,-1,0,0,0,0])
        
        ### Load Data ###
        pnc2 = nc.Dataset(precippath+"/"+dataset+"_CAP2_"+Y+M+".nc")
        pnc1 = nc.Dataset(precippath+"/"+dataset+"_CAP2_"+str(mt1[0])+md.dd[mt1[1]-1]+".nc")
        pnc0 = nc.Dataset(precippath+"/"+dataset+"_CAP2_"+str(mt0[0])+md.dd[mt0[1]-1]+".nc")

        sids = np.concatenate( ( pnc0['sid'][:].data, pnc1['sid'][:].data, pnc2['sid'][:].data ) )
        cap = np.concatenate( ( pnc0['cap'][:].data, pnc1['cap'][:].data, pnc2['cap'][:].data ) )
        uvraw = np.concatenate( ( pnc0['uv'][:].data, pnc1['uv'][:].data, pnc2['uv'][:].data ) )
        
        ### Decompose ###
        # Calculate number of times in a month with CAP (SUM(ti))
        nCAP = (sids != -1).sum(axis=0)*tres # units: h
        
        # Calculate total CAP (∆P/∆t = SUM(Pi)*SUM(ti))
        captot = cap.sum(axis=0)*capfactor # units: m --> mm
        
        # Calculate average CAP (Pi/ti)
        caprate = captot/nCAP # units: mm/h
        
        # Calculate number of unique cyclones in a month that contribute to CAP (C/∆t)
        nCyc = np.apply_along_axis(uniquelen,0,sids,adjust=-1) # units: # of Cyc
        
        # Calculate number of CAP hours per cyclone (SUM(ti)/C)
        hCyc = nCAP/nCyc # units: h / 1 Cyc
        
        # Calculate average cyclone propagation (Xi/ti), then take inverse (ti/Xi)
        nuv = np.isfinite(uvraw)
        uv = np.where(nuv, uvraw, 0).sum(axis=0) / nuv.sum(axis=0) # units: km/h
        uvinv = 1 / uv  # units: km/h --> h/km
        uvinv[np.isfinite(uvinv) == 0] = np.nan
        
        # Calculate Residual (XC/C)
        capresid = captot / (caprate * nCyc * uvinv) # units: km / 1 Cyc
        
        ### Concatenate to multi-month arrays ###
        nCAPM = np.concatenate((nCAPM,nCAP[np.newaxis,:,:]))
        captotM = np.concatenate((captotM,captot[np.newaxis,:,:]))
        caprateM = np.concatenate((caprateM,caprate[np.newaxis,:,:]))
        nCycM = np.concatenate((nCycM,nCyc[np.newaxis,:,:]))
        hCycM = np.concatenate((hCycM,hCyc[np.newaxis,:,:]))
        uvM = np.concatenate((uvM,uv[np.newaxis,:,:]))
        uvinvM = np.concatenate((uvinvM,uvinv[np.newaxis,:,:]))
        capresidM = np.concatenate((capresidM,capresid[np.newaxis,:,:]))
    
    ### SD ###
    nCAP_sd.append( md.sd(nCAPM,minN) )
    captot_sd.append( md.sd(captotM,minN) )
    caprate_sd.append( md.sd(caprateM,minN) )
    nCyc_sd.append( md.sd(nCycM,minN) )
    hCyc_sd.append( md.sd(hCycM,minN) )
    uv_sd.append( md.sd(uvM,minN) )
    uvinv_sd.append( md.sd(uvinvM,minN) )
    capresid_sd.append( md.sd(capresidM,minN) )  

    ### Trend ###
    nCAP_trend.append( md.lm(years,nCAPM,minN) )
    captot_trend.append( md.lm(years,captotM,minN) )
    caprate_trend.append( md.lm(years,caprateM,minN) )
    nCyc_trend.append( md.lm(years,nCycM,minN) )
    hCyc_trend.append( md.lm(years,hCycM,minN) )
    uv_trend.append( md.lm(years,uvM,minN) )
    uvinv_trend.append( md.lm(years,uvinvM,minN) )
    capresid_trend.append( md.lm(years,capresidM,minN) ) 
    
    ### Climatology ###
    # Set NaNs back to 0 for averaged values
    captotM[ np.isnan(captotM) ] = 0
    caprateM[ np.isnan(caprateM) ] = 0
    hCycM[ np.isnan(hCycM) ] = 0
    uvM[ np.isnan(uvM) ] = 0
    uvinvM[ np.isnan(uvinvM) ] = 0
    capresidM[ np.isnan(capresidM) ] = 0
    
    # Calculate averages
    nCAPC = nCAPM.sum(axis=0) 
    
    nCAP_clim.append( nCAPC )
    captot_clim.append( (captotM*nCAPM).sum(axis=0) / nCAPC )
    caprate_clim.append( (caprateM*nCAPM).sum(axis=0) / nCAPC )
    nCyc_clim.append( nCycM.mean(axis=0) )
    hCyc_clim.append( hCycM.mean(axis=0) )
    uv_clim.append( (uvM*nCAPM).sum(axis=0) / nCAPC )
    uvinv_clim.append( (uvinvM*nCAPM).sum(axis=0) / nCAPC )
    capresid_clim.append( (capresidM*nCAPM).sum(axis=0) / nCAPC )
        
print("---Write to File")
### WRITE SD TO FILE ###
ncsd = nc.Dataset(path+"/SeasonalSD/"+YY+"/"+dataset+"_"+verd+vert+"_"+subset+"CAP2_SD_"+YY+".nc",'w')
ncsd.createDimension('x', pnc2['x'].shape[0])
ncsd.createDimension('y', pnc2['y'].shape[0])
ncsd.createDimension('time', 12)
ncsd.description = '''Standard deviation of values (''' + YY + ''') for cyclone-associated precipitation for tracks in the 
dataset version ''' + verd+vert + '''. \n Timestep: ''' + str(tres) + '''h\nOutput Units: mm/h'''

# Create Dimension Variables
nctime = ncsd.createVariable('time',int,('time'))
nctime.units = 'month of year'
nctime[:] = np.arange(1,12+1)

ncx = ncsd.createVariable('x',int,('x'))
ncy = ncsd.createVariable('y',int,('y'))
ncx.units, ncy.units = 'm', 'm'
ncx[:] = pnc2['x'][:].data
ncy[:] = pnc2['y'][:].data

nclat = ncsd.createVariable('lat', np.float32, ('y','x'))
nclon = ncsd.createVariable('lon', np.float32, ('y','x'))
nclat.units = 'degrees'
nclon.units = 'degrees'
nclat[:] = pnc2['lat'][:].data
nclon[:] = pnc2['lon'][:].data

# Create Variables
ncnCAP = ncsd.createVariable('nCAP',np.int32,('time','y','x'))
ncnCAP.units = 'CAP hours per month'
ncnCAP[:] = np.array(nCAP_sd) / len(years)

nccaptot = ncsd.createVariable('captot',float,('time','y','x'))
nccaptot.units = 'mm of CAP per month'
nccaptot[:] = np.array(captot_sd)

nccaprate = ncsd.createVariable('caprate',float,('time','y','x'))
nccaprate.units = 'mm/h'
nccaprate[:] = np.array(caprate_sd)

ncnCyc = ncsd.createVariable('nCyc',np.int32,('time','y','x'))
ncnCyc.units = '# of unique CAP-producing cyclones per month'
ncnCyc[:] = np.array(nCyc_sd)

nchCyc = ncsd.createVariable('hCyc',np.int32,('time','y','x'))
nchCyc.units = 'CAP hours per CAP-producing cyclone'
nchCyc[:] = np.array(hCyc_sd)

ncuv = ncsd.createVariable('uv',float,('time','y','x'))
ncuv.units = 'km/h for CAP-producing cyclones'
ncuv[:] = np.array(uv_sd)

ncuvinv = ncsd.createVariable('uvinv',float,('time','y','x'))
ncuvinv.units = 'h/km for CAP-producing cyclones'
ncuvinv[:] = np.array(uvinv_sd)

nccapresid = ncsd.createVariable('capresid',float,('time','y','x'))
nccapresid.units = 'km per CAP-producing cyclone'
nccapresid[:] = np.array(capresid_sd)

ncsd.close()

### WRITE TREND TO FILE ###
nct = nc.Dataset(path+"/SeasonalTrend/"+YY+"/"+dataset+"_"+verd+vert+"_"+subset+"_CAP2_Trend_"+YY+".nc",'w')
nct.createDimension('x', pnc2['x'].shape[0])
nct.createDimension('y', pnc2['y'].shape[0])
nct.createDimension('time', 12)
nct.description = '''Trend (''' + YY + ''') for cyclone-associated precipitation for tracks in the 
dataset version ''' + verd+vert + '''. \n Timestep: ''' + str(tres) + '''h\nOutput Units: mm/h'''

# Create Dimension Variables
nctime = nct.createVariable('time',int,('time'))
nctime.units = 'end month in season (e.g., DJF = 2)'
nctime[:] = np.arange(1,12+1)

ncx = nct.createVariable('x',int,('x'))
ncy = nct.createVariable('y',int,('y'))
ncx.units, ncy.units = 'm', 'm'
ncx[:] = pnc2['x'][:].data
ncy[:] = pnc2['y'][:].data

nclat = nct.createVariable('lat', np.float32, ('y','x'))
nclon = nct.createVariable('lon', np.float32, ('y','x'))
nclat.units = 'degrees'
nclon.units = 'degrees'
nclat[:] = pnc2['lat'][:].data
nclon[:] = pnc2['lon'][:].data

# Create Variables
for i in range(len(trendnames)):
    ncnCAP = nct.createVariable('nCAP_'+trendnames[i],float,('time','y','x'))
    ncnCAP.units = trendunits[i] + ' in CAP hours per month'
    ncnCAP[:] = np.array(nCAP_trend)[:,i,:,:]

for i in range(len(trendnames)):
    nccaptot = nct.createVariable('captot_'+trendnames[i],float,('time','y','x'))
    nccaptot.units = trendunits[i] + ' mm of CAP per month'
    nccaptot[:] = np.array(captot_trend)[:,i,:,:]
    
for i in range(len(trendnames)):
    nccaprate = nct.createVariable('caprate_'+trendnames[i],float,('time','y','x'))
    nccaprate.units = trendunits[i] + ' in mm/h'
    nccaprate[:] = np.array(caprate_trend)[:,i,:,:]

for i in range(len(trendnames)):
    ncnCyc = nct.createVariable('nCyc_'+trendnames[i],float,('time','y','x'))
    ncnCyc.units = trendunits[i] + ' # of unique CAP-producing cyclones per month'
    ncnCyc[:] = np.array(nCyc_trend)[:,i,:,:]

for i in range(len(trendnames)):
    nchCyc = nct.createVariable('hCyc_'+trendnames[i],float,('time','y','x'))
    nchCyc.units = trendunits[i] + ' CAP hours per CAP-producing cyclone'
    nchCyc[:] = np.array(hCyc_trend)[:,i,:,:]

for i in range(len(trendnames)):
    ncuv = nct.createVariable('uv_'+trendnames[i],float,('time','y','x'))
    ncuv.units = trendunits[i] + ' in km/h for CAP-producing cyclones'
    ncuv[:] = np.array(uv_trend)[:,i,:,:]

for i in range(len(trendnames)):
    ncuvinv = nct.createVariable('uvinv_'+trendnames[i],float,('time','y','x'))
    ncuvinv.units = trendunits[i] + ' in h/km for CAP-producing cyclones'
    ncuvinv[:] = np.array(uvinv_trend)[:,i,:,:]

for i in range(len(trendnames)):
    nccapresid = nct.createVariable('capresid_'+trendnames[i],float,('time','y','x'))
    nccapresid.units = trendunits[i] + ' in km per CAP-producing cyclone'
    nccapresid[:] = np.array(capresid_trend)[:,i,:,:]

nct.close()

### WRITE CLIMATOLOGY TO FILE ###
ncavg = nc.Dataset(path+"/SeasonalClimatology/"+YY+"/"+dataset+"_"+verd+vert+"_"+subset+"_CAP2_Climatology_"+YY+".nc",'w')
ncavg.createDimension('x', pnc2['x'].shape[0])
ncavg.createDimension('y', pnc2['y'].shape[0])
ncavg.createDimension('time', 12)
ncavg.description = '''Average values (''' + YY + ''') for cyclone-associated precipitation for tracks in the 
dataset version ''' + verd+vert + '''. \n Timestep: ''' + str(tres) + '''h\nOutput Units: mm/h'''

# Create Dimension Variables
nctime = ncavg.createVariable('time',int,('time'))
nctime.units = 'month of year'
nctime[:] = np.arange(1,12+1)

ncx = ncavg.createVariable('x',int,('x'))
ncy = ncavg.createVariable('y',int,('y'))
ncx.units, ncy.units = 'm', 'm'
ncx[:] = pnc2['x'][:].data
ncy[:] = pnc2['y'][:].data

nclat = ncavg.createVariable('lat', np.float32, ('y','x'))
nclon = ncavg.createVariable('lon', np.float32, ('y','x'))
nclat.units = 'degrees'
nclon.units = 'degrees'
nclat[:] = pnc2['lat'][:].data
nclon[:] = pnc2['lon'][:].data

# Create Variables
ncnCAP = ncavg.createVariable('nCAP',np.int32,('time','y','x'))
ncnCAP.units = 'CAP hours per month'
ncnCAP[:] = np.array(nCAP_clim) / len(years)

nccaptot = ncavg.createVariable('captot',float,('time','y','x'))
nccaptot.units = 'mm of CAP per month'
nccaptot[:] = np.array(captot_clim)

nccaprate = ncavg.createVariable('caprate',float,('time','y','x'))
nccaprate.units = 'mm/h'
nccaprate[:] = np.array(caprate_clim)

ncnCyc = ncavg.createVariable('nCyc',np.int32,('time','y','x'))
ncnCyc.units = '# of unique CAP-producing cyclones per month'
ncnCyc[:] = np.array(nCyc_clim)

nchCyc = ncavg.createVariable('hCyc',np.int32,('time','y','x'))
nchCyc.units = 'CAP hours per CAP-producing cyclone'
nchCyc[:] = np.array(hCyc_clim)

ncuv = ncavg.createVariable('uv',float,('time','y','x'))
ncuv.units = 'km/h for CAP-producing cyclones'
ncuv[:] = np.array(uv_clim)

ncuvinv = ncavg.createVariable('uvinv',float,('time','y','x'))
ncuvinv.units = 'h/km for CAP-producing cyclones'
ncuvinv[:] = np.array(uvinv_clim)

nccapresid = ncavg.createVariable('capresid',float,('time','y','x'))
nccapresid.units = 'km per CAP-producing cyclone'
nccapresid[:] = np.array(capresid_clim)

ncavg.close()

