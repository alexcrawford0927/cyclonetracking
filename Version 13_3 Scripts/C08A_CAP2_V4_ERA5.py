'''
Author: Alex Crawford
Date Created: 15 Apr 2021
Date Modified: 15 Oct 2021 --> Switch from original findCAP function to new findCAP2
            function that also records as output which cyclone the CAP in a gridcell
            is associated with (i.e., in an Eulerian sense).
            --> Additionally, records the propagation speed of the cyclone
            associated with the precipitation
            24 May 2023 --> Adapted to use pandas 1.0.1 and xarray (more efficient reprojection)
            04 Feb 2025 --> Changed reprojection to make use of xesmf wrapping option; replaced np.int with int
Purpose: Calculate cyclone-associated precipitation using ERA5 precip data and
fields/tracks of cyclone from a cyclone detection an tracking algorithm.
'''

'''********************
Import Modules
********************'''
import pandas as pd
import numpy as np
# import scipy.ndimage.measurements # Imported in the cyclone module
import xesmf as xe
import xarray as xr
import netCDF4 as nc
import CycloneModule_13_3 as md
import warnings
warnings.filterwarnings("ignore")

'''*******************************************
Set up Environment
*******************************************'''
dataset = "ERA5"
verd, vert = "13test", "P"
subset = ""
typ = "System"

path = '/media/alex/Datapool'
cycpath = path+"/CycloneTracking/"
precippath = path+"/"+dataset+"/Precipitation"
outpath = path+"/"+dataset+"/CAP/tracking"+verd+vert+"/"+subset

cycprjpath = path+"/Projections"
inprjpath = path+"/Projections/"+dataset+"_NH_Projection.nc"

'''*******************************************
Define Variables
*******************************************'''
print("Defining variables")

# Time Variables
starttime = [1979,1,1,0,0,0] # Format: [Y,M,D,H,M,S] --> Note, first ERA5 valid time is 0700 UTC on 1 Jan 1979
endtime = [1979,2,1,0,0,0] # stop BEFORE this time (exclusive)
monthstep = [0,1,0,0,0,0] # A Time step that increases by 1 month [Y,M,D,H,M,S]
timestep = [0,0,0,6,0,0] # Format: [Y,M,D,H,M,S] -- this is the time step of the cyclone data, not the precip data

dateref = [1900,1,1,0,0,0] # [Y,M,D,H,M,S]
inittime = [1940,1,1,0,0,0] # time when cyclone detection dataset starts

lys, dpy = 1, 365 # calendar parameters: are there leap years, and number of days is non-leap years

# CAP Parameters
# Minimum precipitation rate (mm) per timestep (hr)
## Used to determine contiguous precipitation areas
pMin = 1.5*timestep[3]/24/1000 # input is: mm/day * timestep (hr) * 1 day/24 hr * 1 m/1000 mm --> units of m

# Minimum radius for cyclone area (m) (additive with algorithm's cyclone area calculation)
r = 250000.

'''*******************************************
Main Analysis
*******************************************'''
print("Main Analysis")

# Load Grids for cyclones, and then precipitation
params = pd.read_pickle(cycpath+"/tracking"+verd+vert+"/cycloneparams.pkl")

if int(verd.split('_')[0]) < 14:
    cycprj = nc.Dataset(cycprjpath+"/EASE2_N0_"+str(params['spres'])+"km_Projection_uv.nc")
else:
    cycprj = nc.Dataset(cycprjpath+"/EASE2_N0_"+str(params['spres'])+"km_Projection.nc")

outlat = cycprj['lat'][:].data
outlon = cycprj['lon'][:].data

inprj = nc.Dataset(inprjpath)
try:
    inlon, inlat = inprj['lon'][:].data, inprj['lat'][:].data
except:
    inlon, inlat = inprj['longitude'][:].data, inprj['latitude'][:].data

# Define Grids as Dictionaries
grid_in = {'lon': inlon, 'lat': inlat}
grid_out = {'lon': outlon, 'lat': outlat}

# Create Regridder
regridder = xe.Regridder(grid_in, grid_out, 'bilinear', periodic=True)

# Commence Main Loop (Monthly)
mt = starttime
while mt != endtime:
    # Extract date
    Y = str(mt[0])
    M = md.dd[mt[1]-1]
    MM = md.mmm[mt[1]-1]

    mt2 = md.timeAdd(mt,monthstep,lys,dpy)

    # Load Cyclone Tracks For Current Month and Final Month (if it exists)
    cs1 = pd.read_pickle(cycpath+"/tracking"+verd+vert+"/"+subset+"/"+typ+"Tracks/"+Y+"/"+subset+typ.lower()+"tracks"+Y+M+".pkl")
    sids1 = np.array([c.sid for c in cs1])
    try:
        cs2 = pd.read_pickle(cycpath+"/tracking"+verd+vert+"/"+subset+"/"+typ+"Tracks/"+str(mt2[0])+"/"+subset+typ.lower()+"tracks"+str(mt2[0])+md.dd[mt2[1]-1]+".pkl")
        sids2 = np.array([c.sid for c in cs2])
    except:
        cs2, sids2 = [], []

    # Load Precip File with netcdf4 to work with time variable
    precipnc = nc.Dataset(precippath+"/ERA5_Precipitation_Hourly_"+Y+M+".nc")
    times = precipnc['time'][:].data

    # Load Precip File with xarray to work with xesmf
    precipx = xr.open_dataset(precippath+"/ERA5_Precipitation_Hourly_"+Y+M+".nc", drop_variables=['sf'])

    # Reproject Precip File
    precipx = regridder(precipx)

    # Load Cyclone Fields
    cfs = pd.read_pickle(cycpath+"/detection"+verd+"/CycloneFields/CF"+Y+M+".pkl")
    ctimes = np.array([c.time for c in cfs])*24

    # Begin hour by hour loop
    caps, pAreas, uvs = [], [], []

    if mt == inittime: # Start time has to be later for first month
        t = md.timeAdd(mt,timestep)
    else:
        t = mt

    while t != mt2:
        # Extract Date
        D = md.dd[t[2]-1]
        H = md.hhmm[t[3]]
        h = md.daysBetweenDates(dateref,t)*24
        if h%120 == 0:
            print(t)

        # Identify Valid Storms to Consider
        cc = [ [c for c in cs1 if h in np.array(c.data.time[(c.data.type > 0)]*24)] , [c for c in cs2 if h in np.array(c.data.time[(c.data.type > 0)]*24)] ]
        mi = list(np.repeat(0,len(cc[0]))) + list(np.repeat(1,len(cc[1]))) # index for month in which the track exists
        sids = [c.sid for c in cc[0]+cc[1]]
        ids = [int(c.data.id[(c.data.time*24 == h)]) for c in cc[0]+cc[1]]

        # Extract & Modify Cyclone Field
        cf = cfs[np.where(ctimes == h)[0][0]]
        cyclones = [c for c in cf.cyclones if c.id in ids]
        ids2 = np.array([c.id for c in cyclones])

        # Extract Precip Fields
        plsc = precipx['lsp'][np.where( (times >= h) & (times < h+timestep[3]) )[0],:,:].data.sum(axis=0)
        ptot = precipx['tp'][np.where( (times >= h) & (times < h+timestep[3]) )[0],:,:].data.sum(axis=0)

        # Calculate CAP
        cap, pArea2 = md.findCAP2(cyclones, cf.fieldAreas, plsc, ptot, cycprj['yDistance'][:].data,
                         cycprj['xDistance'][:].data, outlat, outlon, pMin, r)
        caps.append(cap)

        # Record Cyclone Track Ids for CAP grid cells
        pArea3 = np.zeros_like(pArea2)-1
        for ci in np.unique(pArea2[np.isfinite(pArea2)]).astype(int):
            pArea3[pArea2 == ci] = sids[ci]
        pAreas.append(pArea3)

        # Record Cyclone Propagation Speed for CAP grid cells
        uv = np.zeros_like(pArea2)*np.nan
        for ci in np.unique(pArea2[np.isfinite(pArea2)]).astype(int):
            m = mi[ci]
            if m == 0:
                uv[pArea2 == ci] = float(cs1[np.where(sids1 == sids[ci])[0][0]].data.uv.loc[cs1[np.where(sids1 == sids[ci])[0][0]].data.time == h/24])
            else:
                uv[pArea2 == ci] = float(cs2[np.where(sids2 == sids[ci])[0][0]].data.uv.loc[cs2[np.where(sids2 == sids[ci])[0][0]].data.time == h/24])

        uvs.append(uv)

        # Modify Cyclone Track Dataframes
        for ci in range(len(ids)):
            m = mi[ci]
            if m == 0:
                cs1[np.where(sids1 == sids[ci])[0][0]].data.loc[cs1[np.where(sids1 == sids[ci])[0][0]].data.time == h/24, 'precip'] = cyclones[np.where(ids2 == ids[ci])[0][0]].precip
                cs1[np.where(sids1 == sids[ci])[0][0]].data.loc[cs1[np.where(sids1 == sids[ci])[0][0]].data.time == h/24, 'precipArea'] = cyclones[np.where(ids2 == ids[ci])[0][0]].precipArea
            else:
                cs2[np.where(sids2 == sids[ci])[0][0]].data.loc[cs2[np.where(sids2 == sids[ci])[0][0]].data.time == h/24, 'precip'] = cyclones[np.where(ids2 == ids[ci])[0][0]].precip
                cs2[np.where(sids2 == sids[ci])[0][0]].data.loc[cs2[np.where(sids2 == sids[ci])[0][0]].data.time == h/24, 'precipArea'] = cyclones[np.where(ids2 == ids[ci])[0][0]].precipArea

        # Advance time step
        t = md.timeAdd(t,timestep,lys,dpy)

    # Write netCDF File for CAP
    print("---Write to File")
    nc1 = nc.Dataset(outpath+"/"+dataset+"_CAP2_"+Y+M+".nc",'w')
    nc1.createDimension('x', outlat.shape[1])
    nc1.createDimension('y', outlat.shape[0])
    nc1.createDimension('time', len(caps))
    nc1.description = '''Cyclone-associated precipitation for tracks in the
    dataset version ''' + verd+vert + '''. \n Timestep: ''' + str(timestep[3]) + '''h
    Minimum Large-Scale Precip Rate: ''' + str(pMin*1000*24/timestep[3]) + ''' mm/day
    Augmented Radius: ''' + str(int(r/1000)) + '''km\n Output Units: ''' + str(precipnc['tp'].units) + " / " + str(timestep[3]) + ''' h'''

    # Create Dimension Variables
    nctime = nc1.createVariable('time',int,('time'))
    nctime.units = precipnc['time'].units
    nctime[:] = times[::int(timestep[3])]

    ncx = nc1.createVariable('x',int,('x'))
    ncy = nc1.createVariable('y',int,('y'))
    ncx.units, ncy.units = 'm', 'm'
    try:
        ncx[:] = cycprj['x'][:].data
        ncy[:] = cycprj['y'][:].data
    except:
        ncx[:] = cycprj['u'][:].data
        ncy[:] = cycprj['v'][:].data
        
    nclat = nc1.createVariable('lat', np.float32, ('y','x'))
    nclon = nc1.createVariable('lon', np.float32, ('y','x'))
    nclat.units = 'degrees'
    nclon.units = 'degrees'
    nclat[:] = outlat
    nclon[:] = outlon

    # Create Output Variables
    nccap = nc1.createVariable('cap',np.float32,('time','y','x'))
    nccap.units = str(precipnc['tp'].units) + " / " + str(timestep[3]) + " h"
    nccap[:] = np.array(caps)

    ncid = nc1.createVariable('sid',np.int16,('time','y','x'))
    ncid.units = 'system ID for cyclone track associated with the precipitation for each time/cell'
    ncid[:] = np.array(pAreas)

    ncuv = nc1.createVariable('uv',np.float32,('time','y','x'))
    ncuv.units = 'cyclone propagation speed (km/h)'
    ncuv[:] = np.array(uvs)

    nc1.close(), precipnc.close(), precipx.close()

    # Overwrite Cyclone Tracks for Current Month
    pd.to_pickle(cs1,cycpath+"/tracking"+verd+vert+"/"+subset+"/"+typ+"Tracks/"+Y+"/"+subset+typ.lower()+"tracks"+Y+M+".pkl")

    # Advance times step
    mt = md.timeAdd(mt,monthstep,lys,dpy)

    # Overwrite Cyclone Tracks for Next Month (if valid)
    if mt != endtime:
        pd.to_pickle(cs2,cycpath+"/tracking"+verd+vert+"/"+subset+"/"+typ+"Tracks/"+str(mt2[0])+"/"+subset+typ.lower()+"tracks"+str(mt2[0])+md.dd[mt2[1]-1]+".pkl")





