'''
Author: Alex Crawford
Date Created: 24 Jan 2017
Date Modified: 13 May 2020 - Updated for Python 3
               13 Nov 2020 - Updated for netcdf instead of gdal
               13 Oct 2021 - Added tracking of the smoothing size and minimum number of years to outputs
Purpose: Calculate trends for aggergated cyclone fields (from C4 script) on 
monthly and seasonal scales.

User inputs:
    Path Variables
    Track Type (typ): Cyclone or System
    Bounding Box ID (bboxnum): 2-digit character string
    Time Variables: when to start, end, the time step of the data
'''

'''********************
Import Modules
********************'''
print("Loading modules.")
import os
import numpy as np
import netCDF4 as nc
import xarray as xr
import CycloneModule_13_3 as md
import scipy.stats as stats
from scipy import ndimage

'''*******************************************
Set up Environment
*******************************************'''
print("Setting up environment.")
bboxnum = "" # use "" if performing on all cyclones; or BBox##
typ = "System"
newspres = '' # Leave empty unless using a reduced-resolution version, e.g., "_100km", "_50km", etc.
ver = "13testP"
rad = 800 # Radius of smoothing (folder is 2*this value)

path = "/media/alex/Datapool" #"/Volumes/Cressida"
inpath = path+"/CycloneTracking/tracking"+ver+"/"+bboxnum+"/Aggregation"+typ+"/"+str(rad*2)+"km"

'''*******************************************
Define Variables
*******************************************'''
print("Defining variables")
# File Variables
minN = 10

# Time Variables 
mos = range(1,12+1)
ymin, ymax = 1981 , 2010

# Variables
# vNames = ['cychours'] + ['tcw','siconc','ttr'] + ['tp','e','viwvd'] #,'tnetrad']
# ncvars = [['cychours']] + [['tcw'],['siconc'],['ttr']] + [['tp-cyc','tp-all'],['e-pos-cyc','e-neg-cyc','e-pos-all','e-pos-all'],['viwvd-pos-cyc','viwvd-pos-all','viwvd-neg-cyc','viwvd-neg-all']]
# aggtype = [['sum']] + [['mean'],['mean'],['mean']] + [['sum','sum'],['sum','sum','sum'],['sum','sum','sum']]

vNames =  ['tp','e','viwvd'] #
ncvars =  [['tp-cyc','tp-all'],['e-pos-cyc','e-neg-cyc','e-pos-all','e-pos-all'],['viwvd-pos-cyc','viwvd-pos-all','viwvd-neg-cyc','viwvd-neg-all']]
aggtype = [['sum','sum'],['sum','sum','sum'],['sum','sum','sum']]

'''*******************************************
Main Analysis
*******************************************'''
print("Main Analysis")   
YY = "_"+str(ymin) + "-" + str(ymax)+"_"

### SEASONAL TRENDS ###

print("Step 2. Trends By 3-Month Season")
ncvarsflat = np.concatenate([np.array(ncvar) for ncvar in ncvars])
vlist, vunits = [[] for i in range(len(ncvarsflat))], [[] for i in range(len(ncvarsflat))]
vv= 0
for v in range(len(vNames)):
    print("--Variable: "+vNames[v])
    ncf = xr.open_dataset(inpath+"/"+ver+"_AggregationFields_Monthly"+newspres+"_"+vNames[v]+".nc", decode_times=False)
    times = ncf['time'][:]
    
    for i in range(len(ncvars[v])):
        vunits[vv] = ncf[ncvars[v][i]].units
        
        blist, alist, rlist, plist, elist = [], [], [], [], []
        for m in mos: # by ending month
            tsub = np.where( (np.round((times-((m-1)/12))%1,3) == 0) & (times >= ymin) & (times < ymax+1) )[0]
        
            varr3 = ncf[ncvars[v][i]][tsub,:,:].data
            varr2 = ncf[ncvars[v][i]][tsub-1,:,:].data
            varr1 = ncf[ncvars[v][i]][tsub-2,:,:].data
            
            # Aggregate month to season
            if aggtype[v] == 'sum':
                svarr = varr1 + varr2 + varr3
            else:
                varr = np.array([varr1,varr2,varr3])
                narr = np.isfinite( varr )
                varr[narr == 0] = 0
                
                svarr = varr.sum(axis=0) / narr.sum(axis=0)
    
            # Perform a linear regression
            b, a, r2, pvalue, stderr, stderra = md.lm(np.arange(ymin,ymax+1,1),svarr,minN)
            
            blist.append(b), alist.append(a), rlist.append(r2), plist.append(pvalue), elist.append(stderr)
        
        vlist[vv] = [blist,alist,rlist,plist,elist]
        vv += 1

## Write to File ##
sname = ver+"_Trend"+YY+"Fields_Seasonal"+newspres+"_MinYrs"+str(minN)+".nc"
if sname in os.listdir(inpath):
    snc = nc.Dataset(inpath+"/"+sname,'r+')

else:
    snc = nc.Dataset(inpath+"/"+sname,'w')
    snc.createDimension('y', ncf['y'].shape[0])
    snc.createDimension('x', ncf['x'].shape[0])
    snc.createDimension('time', 12)
    snc.description = 'Trend ('+YY+') of aggregation of cyclone track characteristics on seasonal time scale (Notes: minimum number of years required: ' + str(minN) + '; count data is summed for each season).'
    
    ncy = snc.createVariable('y', np.float32, ('y',))
    ncx = snc.createVariable('x', np.float32, ('x',))
    ncy.units, ncx.units = 'm', 'm'
    ncy[:] = ncf['y'][:].data
    ncx[:] = ncf['x'][:].data
    
    # Add times, lats, and lons
    nctime = snc.createVariable('time', np.float32, ('time',))
    nctime.units = 'ending months'
    nctime[:] = np.arange(1,12+1,1)
    
    nclon = snc.createVariable('lon', np.float32, ('y','x'))
    nclon.units = 'degrees'
    nclon[:] = ncf['lon'][:].data
    
    nclat = snc.createVariable('lat', np.float32, ('y','x'))
    nclat.units = 'degrees'
    nclat[:] = ncf['lat'][:].data

ncf.close()

for v in range(len(ncvarsflat)):
    # Trend
    try:
        ncvar = snc.createVariable(ncvarsflat[v]+"_trend", np.float64, ('time','y','x'))
        ncvar.units = vunits[v]+"/yr"
        ncvar[:] = np.array(vlist[v][0])
    except:
        snc[ncvarsflat[v]+"_trend"][:] = np.array(vlist[v][0])
    
    # Y-Intercept
    try:
        ncvar = snc.createVariable(ncvarsflat[v]+"_yint", np.float64, ('time','y','x'))
        ncvar.units = vunits[v]
        ncvar[:] = np.array(vlist[v][1])
    except:
        snc[ncvarsflat[v]+"_yint"][:] = np.array(vlist[v][1])

    # R^2
    try:
        ncvar = snc.createVariable(ncvarsflat[v]+"_r2", np.float64, ('time','y','x'))
        ncvar.units = "ratio"
        ncvar[:] = np.array(vlist[v][2])
    except:
        snc[ncvarsflat[v]+"_r2"][:] = np.array(vlist[v][2])

    # P-Value
    try:
        ncvar = snc.createVariable(ncvarsflat[v]+"_pvalue", np.float64, ('time','y','x'))
        ncvar.units = "ratio"
        ncvar[:] = np.array(vlist[v][3])
    except:
        snc[ncvarsflat[v]+"_pvalue"][:] = np.array(vlist[v][3])

    # Std Err
    try:
        ncvar = snc.createVariable(ncvarsflat[v]+"_sdtrend", np.float64, ('time','y','x'))
        ncvar.units = vunits[v]+"/yr"
        ncvar[:] = np.array(vlist[v][4])
    except:
        snc[ncvarsflat[v]+"_sdtrend"][:] = np.array(vlist[v][4])

snc.close()

print('Complete')

# snc = xr.open_dataset(inpath+"/"+sname)
# snc = snc.drop_vars(['n_trend', 'n_yint', 'n_r2', 'n_pvalue', 'n_sdtrend', 'tcw_Adv_trend', 'tcw_Adv_yint', 'tcw_Adv_r2', 'tcw_Adv_pvalue', 'tcw_Adv_sdtrend', 'tp_trend', 'tp_yint', 'tp_r2', 'tp_pvalue', 'tp_sdtrend', 'e_trend', 'e_yint', 'e_r2', 'e_pvalue', 'e_sdtrend', 'viwvd_trend', 'viwvd_yint', 'viwvd_r2', 'viwvd_pvalue', 'viwvd_sdtrend', 'peff_trend', 'peff_yint', 'peff_r2', 'peff_pvalue', 'peff_sdtrend','tp_cycratio_trend','tp_cycratio_yint','tp_cycratio_r2','tp_cycratio_pvalue','tp_cycratio_sdtrend','e_cycratio_trend','e_cycratio_yint','e_cycratio_r2','e_cycratio_pvalue','e_cycratio_sdtrend','viwvd_cycratio_trend','viwvd_cycratio_yint','viwvd_cycratio_r2','viwvd_cycratio_pvalue','viwvd_cycratio_sdtrend'])
# snc.to_netcdf(inpath+"/NEW"+sname)
