'''
Author: Alex Crawford
Date Created: 10/28/16
Date Modified: 5/6/20 (Updated for Python 3)
    18 Jan 2021 --> Built in removal of DpDr and precip columns if present
    11 Apr 2022 --> Fixed a bug in the removal of DpDr and precip columns and unit conversion
    23 Jan 2023 --> Changed order of columns, added year/month/day/hour columns, changed units to be
        hPa instead of Pa
    23 Feb 2025 --> Changed the file structure to be per month (with an sid/tid column) instead of per cyclone
Purpose: Export Cyclone objects' data table as CSV files with year, month, and TID in the filename.
'''

'''********************
Import Modules
********************'''
print("Loading modules.")
import pandas as pd
import numpy as np
import CycloneModule_13_3 as md
import os
# import pickle5

'''*******************************************
Define Variables
*******************************************'''
print("Defining variables")

vers = "13testP" # "13_4E5P" # "11_1E5J"
bbox = "" # Use BBox## for subsets
typ = "System"
path = "/Volumes/Cressida/CycloneTracking/tracking"+vers

# Time Variables
years = range(1981,2010+1)
mos = range(1,12+1)
dateref = [1900,1,1,0,0,0]

# Cyclone Parameters
# Aggregation Parameters
minls = 1 # minimum lifespan (in days) for a track to be considered
mintl = 1000 # in km for version 11 and beyond
minlat = 0 # minimum latitude that must be acheived at some point in the track

'''*******************************************
Main Analysis
*******************************************'''
print("Main Analysis")

# Load parameters
params = pd.read_pickle(path+"/cycloneparams.pkl")
# params = pickle5.load(open(path+"/cycloneparams.pkl",'rb'))
try:
    spres = params['spres']
except:
    spres = 100

# Set Up Output Directories
try:
    os.chdir(path+"/"+bbox+"/CSV"+typ)
except:
    os.mkdir(path+"/"+bbox+"/CSV"+typ)

for y in years:
    Y = str(y)
    #Create directories
    try:
        os.chdir(path+"/"+bbox+"/CSV"+typ+"/"+Y)
    except:
        os.mkdir(path+"/"+bbox+"/CSV"+typ+"/"+Y)
        os.chdir(path+"/"+bbox+"/CSV"+typ+"/"+Y)

# Write CSV for Systems
if typ == "System":
    for y in years:
        Y = str(y)
        print(Y)
        for m in mos:
            M = md.dd[m-1]
            
            trdatalist = []

            # Load data
            trs = pd.read_pickle(path+"/"+bbox+"/"+typ+"Tracks"+"/"+Y+"/"+bbox+typ.lower()+"tracks"+Y+M+".pkl")
            # trs = pickle5.load(open(path+"/"+bbox+"/"+typ+"Tracks"+"/"+Y+"/"+bbox+typ.lower()+"tracks"+Y+M+".pkl",'rb'))

            # Write CSVs
            for tr in trs:
                if tr.lifespan() >= minls and tr.trackLength() >= mintl and np.max(tr.data.lat) >= minlat:
                    trdata = tr.data
                    if len(np.intersect1d(trdata.columns,['precip','precipArea','DpDr'])) == 3:
                        trdata = trdata.drop(columns=['precip','precipArea','DpDr'])

                    dates = [md.timeAdd(dateref,[0,0,t,0,0,0]) for t in trdata['time']]

                    trdata['sid'] = tr.sid
                    trdata['year'] = np.array([date[0] for date in dates])
                    trdata['month'] = np.array([date[1] for date in dates])
                    trdata['day'] = np.array([date[2] for date in dates])
                    trdata['hour'] = np.array([date[3] for date in dates])
                    trdata['p_cent'] = trdata['p_cent'] / 100 # Pa --> hPa
                    trdata['p_edge'] = trdata['p_edge'] / 100 # Pa --> hPa
                    trdata['depth'] = trdata['depth'] / 100 # Pa --> hPa
                    trdata['radius'] = trdata['radius'] * spres # to units of km
                    trdata['area'] = trdata['area'] * spres * spres # to units of km^2
                    trdata['DsqP'] = trdata['DsqP'] / spres / spres * 100 * 100 / 100 # to units of hPa/[100 km]^2
                    trdata['p_grad'] = trdata['p_grad'] / 100 * 1000 * 1000 # to units of hPa / [1000 km]
                    trdata['Dp'] = trdata['Dp'] / 100 # Pa --> hPa
                    trdata['DpDt'] = trdata['DpDt'] / 100 # Pa/day --> hPa/day
                    
                    trdatalist.append( trdata.loc[:,['sid','year','month','day','hour','time','lat','lon','x','y','p_cent','p_edge','depth','p_grad','DsqP','radius','area','Dp','DpDt','u','v','uv','Ege','Ely','Emg','Esp','Erg','id','pid','type','centers']] )
            
            mtrdata = pd.concat(trdatalist)
            mtrdata.to_csv(path+"/"+bbox+"/CSV"+typ+"/"+Y+"/"+typ+vers+"_"+bbox+"_"+Y+M+".csv",index=False)

# Write CSV for Cyclones
else:
    for y in years:
        Y = str(y)
        print(Y)
        for m in mos:
            M = md.dd[m-1]
            trdatalist = []
            
            # trs = pickle5.load(open(path+"/"+bbox+"/"+typ+"Tracks"+"/"+Y+"/"+bbox+typ.lower()+"tracks"+Y+M+".pkl",'rb'))
            trs = pd.read_pickle(path+"/"+bbox+"/"+typ+"Tracks"+"/"+Y+"/"+bbox+typ.lower()+"tracks"+Y+M+".pkl")

            for tr in trs:
                if tr.lifespan() >= minls and tr.trackLength() >= mintl and np.max(tr.data.lat) >= minlat:
                    trdata= tr.data
                    if len(np.intersect1d(trdata.columns,['precip','precipArea','DpDr'])) == 3:
                        trdata = trdata.drop(columns=['precip','precipArea','DpDr'])
                    
                    dates = [md.timeAdd(dateref,[0,0,t,0,0,0]) for t in trdata['time']]

                    trdata['tid'] = tr.tid
                    trdata['year'] = np.array([date[0] for date in dates])
                    trdata['month'] = np.array([date[1] for date in dates])
                    trdata['day'] = np.array([date[2] for date in dates])
                    trdata['hour'] = np.array([date[3] for date in dates])
                    trdata['p_cent'] = trdata['p_cent'] / 100 # Pa --> hPa
                    trdata['p_edge'] = trdata['p_edge'] / 100 # Pa --> hPa
                    trdata['depth'] = trdata['depth'] / 100 # Pa --> hPa
                    trdata['radius'] = trdata['radius'] * spres # to units of km
                    trdata['area'] = trdata['area'] * spres * spres # to units of km^2
                    trdata['DsqP'] = trdata['DsqP'] / spres / spres * 100 * 100 / 100 # to units of hPa/[100 km]^2
                    trdata['p_grad'] = trdata['p_grad'] / 100 * 1000 * 1000 # to units of hPa / [1000 km]
                    trdata['Dp'] = trdata['Dp'] / 100 # Pa --> hPa
                    trdata['DpDt'] = trdata['DpDt'] / 100 # Pa/day --> hPa/day 

                    trdata['year'] = np.array([date[0] for date in dates])
                    trdata['month'] = np.array([date[1] for date in dates])
                    trdata['day'] = np.array([date[2] for date in dates])
                    trdata['hour'] = np.array([date[3] for date in dates])
                    trdata['p_cent'] = trdata['p_cent'] / 100 # Pa --> hPa
                    trdata['p_edge'] = trdata['p_edge'] / 100 # Pa --> hPa
                    trdata['depth'] = trdata['depth'] / 100 # Pa --> hPa
                    trdata['radius'] = trdata['radius'] * spres # to units of km
                    trdata['area'] = trdata['area'] * spres * spres # to units of km^2
                    trdata['DsqP'] = trdata['DsqP'] / spres / spres * 100 * 100 / 100 # to units of hPa/[100 km]^2
                    trdata['p_grad'] = trdata['p_grad'] / 100 * 1000 * 1000 # to units of hPa / [1000 km]
                    trdata['Dp'] = trdata['Dp'] / 100 # Pa --> hPa
                    trdata['DpDt'] = trdata['DpDt'] / 100 # Pa/day --> hPa/day

                    
                    trdatalist.append( trdata.loc[:,['tid','year','month','day','hour','time','lat','lon','x','y','p_cent','p_edge','depth','p_grad','DsqP','radius','area','Dp','DpDt','u','v','uv','Ege','Ely','Emg','Esp','Erg','id','pid','type','centers']] )

            mtrdata = pd.concat(trdatalist)
            mtrdata.to_csv(path+"/"+bbox+"/CSV"+typ+"/"+Y+"/"+typ+vers+"_"+bbox+"_"+Y+M+".csv",index=False)

#####

# Combine into one master file
# starttime = [1940,1,1,0,0,0] # inclusive
# endtime = [2025,1,1,0,0,0] # exclusive
# mstep = [0,1,0,0,0,0]

# outdata = pd.read_csv(path+"/"+bbox+"/CSV"+typ+"/"+str(starttime[0])+"/"+typ+vers+"_"+bbox+"_"+str(starttime[0])+md.dd[starttime[1]-1]+".csv")

# time = starttime + []
# while time != endtime:
#     Y, M = str(time[0]), md.dd[time[1]-1]
#     mtrdata = pd.read_csv(path+"/"+bbox+"/CSV"+typ+"/"+Y+"/"+typ+vers+"_"+bbox+"_"+Y+M+".csv")
    
#     outdata = pd.concat((outdata,mtrdata),ignore_index=True)
    
#     time = md.timeAdd(time,mstep)

# outdata.to_csv(path+"/"+bbox+"/"+typ+vers+"_"+bbox+"_AllTracks_"+str(starttime[0])+"-"+str(time[0])+".csv")
