'''
Author: Alex Crawford
Date Created: 20 Jan 2015
Date Modified: 5 Oct 2015

This module contians the classes and functions used for a cyclone detection 
and tracking algorithm (the C3_CycloneDetection scripts).
'''
__version__ = "10.3"

import os
import pandas as pd
import cPickle
import copy
import numpy as np
from osgeo import gdal, gdalconst, gdalnumeric
import scipy.ndimage.measurements

'''
###############
### CLASSES ###
###############
'''

class minimum:
    '''This class stores the vital information about a minimum identified in a
    single field of data. It is used as a building block for more complicated 
    objets (cyclone, cyclonefield). Must provide the time, the x and y 
    locations of the minimum, and the value of the minimum (p).
    
    The type is used to identify the minimum as 0 = disgarded from analysis; 
    1 = identified as a system center; 2 = identified as a secondary minimum 
    within a larger cyclone system.
    '''    
    def __init__(self,time,y,x,p_cent,i=0,t=0):
        self.time = time
        self.y = y
        self.x = x
        self.lat = np.nan
        self.long = np.nan
        self.p_cent = p_cent
        self.id = i
        self.tid = np.nan
        self.p_edge = p_cent
        self.area = 0
        self.areaID = np.nan
        self.DsqP = np.nan
        self.type = t
        self.secondary = [] # stores the ids of any secondary minima
        self.parent = {"y":y,"x":x,"id":i}
        self.precip = 0
        self.precipArea = 0
    def radius(self):
        return (self.area/(np.pi))**0.5
    def depth(self):
        return self.p_edge-self.p_cent
    def Dp(self):
        return self.depth()/self.radius()
    def centerCount(self):
        if self.type == 1:
            return len(self.secondary)+1
        else:
            return 0
    def add_parent(self,yPar,xPar,iPar):
        self.type = 2
        self.parent = {"y":yPar,"x":xPar,"id":iPar}

class cyclonefield:
    '''This class stores a) binary fields showing presence/absence of cyclone 
    centers and cyclone areas and b) an object for each cyclone -- all at a 
    particular instant in time. Good for computing summary values of the 
    cyclone state at a particular time. By default, the elevations parameters 
    are set to 0 (ignored in analysis).
    
    To initiate, must define:\n
    time = any format, but [Y,M,D,H,M,S] is suggested
    field = a numpy array of SLP (usu. in Pa; floats or ints)\n
    
    May also include:\n
    max_elev = the maximum elevation at which SLP minima will be considered (if
        you intend on changing the default, must also load a DEM array)
    elevField = an elevation array that must have the same shape as field\n
    '''
    ##################
    # Initialization #
    ##################
    
    def __init__(self, time):
        self.time = time
        self.cyclones = []
       
    #############################
    # Cyclone Centers Detection #
    #############################
    def findMinima(self, fieldMask, kSize, nanthreshold=0.5):
        '''Identifies minima in the field using limits and kSize.
        '''
        self.fieldMinima = detectMinima(fieldMask,kSize,nanthreshold).astype(np.int)
    
    def findCenters(self, field, fieldMask, kSize, d_slp, d_dist, cellsize, lats, longs, nanthreshold=0.5):
        '''Uses the initialization parameters to identify potential cyclones
        centers. Identification begins with finding minimum values in the field
        and then uses a gradient parameter and elevation parameter to restrict
        the number of centers. A few extra characteristics about the minima are 
        recorded (e.g. lat, long, Laplacian). Lastly, it adds a minimum object 
        to the centers list for each center identified in the centers field.
        '''
        # STEP 1: Calculate Laplacian
        laplac = laplacian(field)        
        
        # STEP 2: Identify Centers
        self.fieldCenters = findCenters(fieldMask, kSize, d_slp, d_dist, cellsize, nanthreshold)
        
        # Identify center locations
        rows, cols = np.where(self.fieldCenters == 1)
        
        # STEP 3: Assign each as a minimum in the centers list:
        for c in range(np.sum(self.fieldCenters)):
            center = minimum(self.time,rows[c],cols[c],field[rows[c],cols[c]],c,0)
            center.lat = lats[rows[c],cols[c]]
            center.long = longs[rows[c],cols[c]]
            center.DsqP = laplac[rows[c],cols[c]]
            self.cyclones.append(center)
            
    ###############################
    # Cyclone Area Identification #
    ###############################
    def findAreas(self, fieldMask, contint, mcctol, mccdist, cellsize, kSize):
        # Identify maxima
        maxes = detectMaxima(fieldMask,kSize)
        
        # Define Areas, Identify Primary v. Secondary Cyclones
        self.fieldAreas, self.fieldCenters2, self.cyclones = \
            findAreas(fieldMask, self.fieldCenters, self.cyclones,\
            contint, mcctol, mccdist, cellsize, maxes)
        
        # Identify area id for each center
        cAreas, nC = scipy.ndimage.measurements.label(self.fieldAreas)
        for i in range(len(self.cyclones)):
            self.cyclones[i].areaID = cAreas[self.cyclones[i].y,self.cyclones[i].x]

    def findAreas2(self, fieldMask, contint, mcctol, mccdist, cellsize, kSize):
        # Identify maxima
        maxes = detectMaxima(fieldMask,kSize)
        
        # Define Areas, Identify Primary v. Secondary Cyclones
        self.fieldAreas, self.fieldCenters2, self.cyclones = \
            findAreas2(fieldMask, self.fieldCenters, self.cyclones,\
            contint, mcctol, mccdist, cellsize, maxes)
        
        # Identify area id for each center
        cAreas, nC = scipy.ndimage.measurements.label(self.fieldAreas)
        for i in range(len(self.cyclones)):
            self.cyclones[i].areaID = cAreas[self.cyclones[i].y,self.cyclones[i].x]
    
    ################################################
    # Cyclone-Associated Precipitation Calculation #
    ###############################################
    def findCAP(self,plsc,ptot,pMin=0.375,r=250000,cellsize=100000):
        self.CAP = findCAP(self,plsc,ptot,pMin,r,cellsize)
    
    ######################
    # Summary Statistics #
    ######################
    # Summary values:
    def cycloneCount(self):
        return len([c.type for c in self.cyclones if c.type == 1])
    def area_total(self):
        areas = self.area()
        counts = self.centerCount()
        return sum([areas[a] for a in range(len(areas)) if counts[a] > 0])
    
    # Reorganization of cyclone object values:    
    def x(self):
        return [c.x for c in self.cyclones]
    def y(self):
        return [c.y for c in self.cyclones]
    def lats(self):
        return [c.lat for c in self.cyclones]
    def longs(self):
        return [c.long for c in self.cyclones]
    def p_cent(self):
        return [c.p_cent for c in self.cyclones]
    def p_edge(self):
        return [c.p_edge for c in self.cyclones]
    def radius(self):
        return [c.radius for c in self.cyclones]
    def area(self):
        return [c.area for c in self.cyclones]
    def areaID(self):
        return [c.areaID for c in self.cyclones]
    def centerType(self):
        return [c.type for c in self.cyclones]
    def centerCount(self):
        return [c.centerCount() for c in self.cyclones]
    def tid(self):
        return [c.tid for c in self.cyclones]

class cyclonetrack:
    '''This class stores vital information about the track of a single cyclone.
    It contains a pandas dataframe built from objets of the class minimum 
    and provides summary statistics about the cyclone's track and life cycle. 
    To make sense, the cyclone objects should be entered in chronological order.
    
    UNITS:
    time = days
    x, y, dx, dy = grid cells (1 = 100 km)
    area = sq grid cells (1 = 100 km^2)
    p_cent, p_edge, depth = Pa
    radius = grid cells
    u, v, uv = km/hr
    DpDr = Pa/grid cell
    DsqP = Pa/(grid cell)^2
    DpDt = Pa/day
    precip = mm (implied: mm (time interval)^-1)
    id, tid, sid, ftid, otid, centers, type = no units
        id = a unique number for each center identified in a SLP field
        tid = a unique number for each center track in a given month
        ptid = the tid of the parent center in a MCC (ptid == tid is single-
            center cyclones)
        ftid = the tid of a center in the prior month (only applicable if a
            cyclone has genesis in a different month than its lysis)
        otid = the tid of a cyclone center that interacts with the given
            center (split, merge, re-genesis)
    ly, ge, rg, sp, mg = 0: no event occurred, 1: only a center-related event 
        occurred, 2: only an area-related event occurred, 3: both center- and 
        area-related events occurred
    '''
    ##############
    # Initialize #
    ##############
    def __init__(self,center,tid,Etype=3, ptid=np.nan, ftid=np.nan):
        self.tid = tid # A track id
        self.ftid = ftid # Former track id
        self.ptid = ptid # The most current parent track id
        
        # Create Main Data Frame
        self.data = pd.DataFrame(columns=["time","id","pid","ptid",\
            "x","y","lat","long","p_cent","p_edge","area","radius","depth",\
            "DpDr","DsqP","DpDt","u","v","uv","Dx","Dy","type","centers",\
            "Ege","Erg","Ely","Esp","Emg"])
        row0 = pd.DataFrame([{"time":center.time, "id":center.id, "pid":center.parent["id"],\
            "x":center.x, "y":center.y, "lat":center.lat, "long":center.long, \
            "p_cent":center.p_cent, "p_edge":center.p_edge, "area":center.area, \
            "radius":center.radius(), "depth":center.depth(),"DpDr":center.Dp(),\
            "DsqP":center.DsqP,"type":center.type, "centers":center.centerCount(),\
            "precip":center.precip,"precipArea":center.precipArea,\
            "Ege":Etype,"Erg":0,"Ely":0,"Esp":0,"Emg":0,"ptid":ptid},])
        self.data = self.data.append(row0, ignore_index=1)
        
        # Create Events Data Frame
        self.events = pd.DataFrame(columns=["time","id","event","Etype","otid","x","y"])
        event0 = pd.DataFrame([{"time":center.time,"id":center.id,"event":"ge",\
            "Etype":Etype,"otid":np.nan,"x":center.x,"y":center.y},])
        self.events = self.events.append(event0, ignore_index=1)
    
    ###############
    # Append Data #
    ###############   
    def addInstance(self,center,ptid=-1):
        row = pd.DataFrame([{"time":center.time, "id":center.id, "pid":center.parent["id"],\
            "x":center.x, "y":center.y, "lat":center.lat, "long":center.long, \
            "p_cent":center.p_cent, "p_edge":center.p_edge, "area":center.area, \
            "radius":center.radius(), "depth":center.depth(),"DpDr":center.Dp(),\
            "DsqP":center.DsqP,"type":center.type, "centers":center.centerCount(),\
            "precip":center.precip,"precipArea":center.precipArea,\
            "Ege":0,"Ely":0,"Esp":0,"Emg":0,"Erg":0},])
        self.data = self.data.append(row, ignore_index=1)
        r =  len(self.data.index)-1
        t = self.data.time[r]
        Dt = t - self.data.time[r-1]
        
        if Dt != 0:
            Dp = self.data.p_cent.irow(r) - self.data.p_cent[r-1]
            self.data.Dx[r] = self.data.x[r] - self.data.x[r-1]
            self.data.Dy[r] = self.data.y[r] - self.data.y[r-1]
            self.data.u[r] = haversine(self.data.lat[r],self.data.lat[r],self.data.long[r-1],self.data.long[r])/(Dt*1000*24)
            self.data.v[r] = haversine(self.data.lat[r-1],self.data.lat[r],self.data.long[r],self.data.long[r])/(Dt*1000*24)
            self.data.uv[r] = ( ((self.data.Dx[r]/Dt)**2 + (self.data.Dy[r]/Dt)**2)**0.5 )*100/24
            # Following Roebber (1984) and Serreze et al. (1997), scale the deepening rate by latitude
            self.data.DpDt[r] = (Dp/Dt) * (np.sin(np.pi/3)/np.sin(np.pi*center.lat/180))
        
        if ptid == -1:
            self.data.ptid[r] = self.ptid
        else:
            self.data.ptid[r] = ptid
    
    def addEvent(self,center,time,event,Etype,otid=np.nan):
        '''Events include genesis (ge), regenesis (rg) splitting (sp), merging
        (mg), and lysis (ly). Splitting and merging require the id of the 
        cyclone track being split from or merged with (otid). Note that lysis 
        is given the time step and location of the last instance of the 
        cyclone. For all types except rg, the event can be center-based, area-
        based, or both. Genesis occurs when a center/area doesn't exist in 
        time 1 but does exist in time 2. Lysis occurs when a center/area does 
        exist in time 1 but doesn't in time 2. A split occurs when one center/
        area in time 1 tracks to multiple centers/areas in time 2. A merge 
        occurs when multiple centers/areas in time 1 track to the same center/
        area in time 2. Regenesis is a special type of area genesis that occurs
        if the primary system of multiple centers experiences lysis but the
        system continues on from a secondary center.
        
        The occurrence of events is recorded both in an events data frame and
        the main tracking data frame.
        
        center = an object of class minimum that represents a cyclone center
        event = ge, rg, ly, sp, or mg
        eType = 1: center only, 2: area only, 3: both center and area
        otid = the track id of the other center involved for sp and mg events.
        '''
        row = pd.DataFrame([{"time":time,"id":center.id,"event":event,\
            "Etype":Etype,"otid":otid,"x":center.x,"y":center.y},])
        self.events = self.events.append(row, ignore_index=1)
        
        # Event Booleans for Main Data Frame
        if event == "ge":
            self.data.Ege[self.data.time == time] = Etype
        elif event == "ly":
            self.data.Ely[self.data.time == time] = Etype
        elif event == "sp":
            self.data.Esp[self.data.time == time] = Etype
        elif event == "mg":
            self.data.Emg[self.data.time == time] = Etype
        elif event == "rg":
            self.data.Erg[self.data.time == time] = Etype
    
    #############
    # Summarize #
    #############
    def lifespan(self):
        '''Subtracts the earliest time stamp from the latest.'''
        return max(self.data.time) - min(self.data.time[self.data.type != 0])
    def maxDpDt(self):
        '''Returns the maximum deepening rate in the track and the 
        time and location (row, col) in which it occurred.'''
        v = np.min(np.where(np.isfinite(list(self.data.DpDt)) == 1,list(self.data.DpDt),np.inf))
        t = list(self.data.time[self.data.DpDt == v])
        y = [int(i) for i in list(self.data.y[self.data.DpDt == v])]
        x = [int(i) for i in list(self.data.x[self.data.DpDt == v])]
        return v, t, y, x
    def maxDsqP(self):
        '''Returns the maximum intensity in the track and the time 
        and location (row, col) in which it occurred.'''
        v = np.max(np.where(np.isfinite(list(self.data.DsqP)) == 1,list(self.data.DsqP),-np.inf))
        t = list(self.data.time[self.data.DsqP == v])
        y = [int(i) for i in list(self.data.y[self.data.DsqP == v])]
        x = [int(i) for i in list(self.data.x[self.data.DsqP == v])]
        return v, t, y, x
    def minP(self):
        '''Returns the minimum pressure in the track and the 
        time and location (row, col) in which it occurred.'''
        v = np.min(np.where(np.isfinite(list(self.data.p_cent)) == 1,list(self.data.p_cent),np.inf))
        t = list(self.data.time[self.data.p_cent == v])
        y = [int(i) for i in list(self.data.y[self.data.p_cent == v])]
        x = [int(i) for i in list(self.data.x[self.data.p_cent == v])]
        return v, t, y, x
    def maxUV(self):
        '''Returns the maximum cyclone propagation speed in the track and the 
        time and location (row, col) in which it occurred.'''
        v = np.max(np.where(np.isfinite(list(self.data.uv)) == 1,list(self.data.uv),-np.inf))
        t = list(self.data.time[self.data.uv == v])
        y = [int(i) for i in list(self.data.y[self.data.uv == v])]
        x = [int(i) for i in list(self.data.x[self.data.uv == v])]
        return v, t, y, x
    def maxDepth(self):
        '''Returns the maximum depth in the track and the 
        time and location (row, col) in which it occurred.'''
        v = np.max(np.where(np.isfinite(list(self.data.depth)) == 1,list(self.data.depth),-np.inf))
        t = list(self.data.time[self.data.depth == v])
        y = [int(i) for i in list(self.data.y[self.data.depth == v])]
        x = [int(i) for i in list(self.data.x[self.data.depth == v])]
        return v, t, y, x
    def trackLength(self):
        '''Adds together the distance between each segment of the track to find
        the total distance traveled.'''
        return np.nansum([float((self.data.Dx[i]**2+self.data.Dy[i]**2)**0.5) \
            for i in list(self.data.index) if ((self.data.type[i] != 0) or (self.data.Ely[i] > 0))])
    def avgArea(self):
        '''Identifies the average area for the track and the time stamp for 
        when it occurred.'''
        areas = [float(i) for i in list(self.data.area[self.data.type != 0])]
        return float(sum(areas))/len(self.data[self.data.type != 0])
    def mcc(self):
        '''Returns a 1 if at any point along the track the cyclone system is
        a multi-center cyclone. Retruns a 0 otherwise.'''
        if np.nansum([int(c) != 1 for c in self.data.centers[self.data.type != 0]]) == 0:
            return 0
        else:
            return 1
    def CAP(self):
        '''Returns the total cyclone-associated precipitation for the cyclone center.'''
        return np.nansum(list(self.data.precip)[self.data.type != 0])

class systemtrack:
    '''This class stores vital information about the track of a single system.
    It contains a pandas dataframe built from objets of the class minimum 
    and provides summary statistics about the system's track and life cycle. 
    To make sense, the system track should be constructed directly from 
    finished cyclone tracks. The difference between a system track and a 
    cyclone track is that a cyclone track exists for each cyclone center, 
    whereas only one system track exists for each mcc.
    
    UNITS:
    time = days
    x, y, dx, dy = grid cells (1 = 100 km)
    area, precipArea = sq grid cells (1 = (100 km)^2)
    p_cent, p_edge, depth = Pa
    radius = grid cells
    u, v, uv = km/hr
    DpDr = Pa/grid cell
    DsqP = Pa/(grid cell)^2
    DpDt = Pa/day
    precip = mm (implied: mm (time interval)^-1)
    id, tid, sid, ftid, otid, centers, type = no units
        id = a unique number for each center identified in a SLP field
        tid = a unique number for each center track in a given month
        sid = a unique number for each system track in a given month
        ptid = the tid of the parent center in a MCC (ptid == tid is single-
            center cyclones)
        ftid = the tid of a center in the prior month (only applicable if a
            cyclone has genesis in a different month than its lysis)
        otid = the tid of a cyclone center that interacts with the given
            center (split, merge, re-genesis)
    ly, ge, rg, sp, mg = 0: no event occurred, 1: only a center-related event 
        occurred, 2: only an area-related event occurred, 3: both center- and 
        area-related events occurred
    '''
    ##############
    # Initialize #
    ##############
    def __init__(self,data,events,tid,sid,ftid=np.nan):
        self.tid = tid # A track id
        self.ftid = ftid # The former track id
        self.sid = sid # A system id
                
        # Create Main Data Frame
        self.data = copy.deepcopy(data)
        
        # Create Events Data Frame
        self.events = copy.deepcopy(events)
    
    #############
    # Summarize #
    #############
    def lifespan(self):
        '''Subtracts the earliest time stamp from the latest.'''
        return max(self.data.time) - min(self.data.time[self.data.type != 0])
    def maxDpDt(self):
        '''Returns the maximum deepening rate in the track and the 
        time and location (row, col) in which it occurred.'''
        v = np.min(np.where(np.isfinite(list(self.data.DpDt)) == 1,list(self.data.DpDt),np.inf))
        t = list(self.data.time[self.data.DpDt == v])
        y = [int(i) for i in list(self.data.y[self.data.DpDt == v])]
        x = [int(i) for i in list(self.data.x[self.data.DpDt == v])]
        return v, t, y, x
    def maxDsqP(self):
        '''Returns the maximum intensity in the track and the time 
        and location (row, col) in which it occurred.'''
        v = np.max(np.where(np.isfinite(list(self.data.DsqP)) == 1,list(self.data.DsqP),-np.inf))
        t = list(self.data.time[self.data.DsqP == v])
        y = [int(i) for i in list(self.data.y[self.data.DsqP == v])]
        x = [int(i) for i in list(self.data.x[self.data.DsqP == v])]
        return v, t, y, x
    def minP(self):
        '''Returns the minimum pressure in the track and the 
        time and location (row, col) in which it occurred.'''
        v = np.min(np.where(np.isfinite(list(self.data.p_cent)) == 1,list(self.data.p_cent),np.inf))
        t = list(self.data.time[self.data.p_cent == v])
        y = [int(i) for i in list(self.data.y[self.data.p_cent == v])]
        x = [int(i) for i in list(self.data.x[self.data.p_cent == v])]
        return v, t, y, x
    def maxUV(self):
        '''Returns the maximum cyclone propagation speed in the track and the 
        time and location (row, col) in which it occurred.'''
        v = np.max(np.where(np.isfinite(list(self.data.uv)) == 1,list(self.data.uv),-np.inf))
        t = list(self.data.time[self.data.uv == v])
        y = [int(i) for i in list(self.data.y[self.data.uv == v])]
        x = [int(i) for i in list(self.data.x[self.data.uv == v])]
        return v, t, y, x
    def maxDepth(self):
        '''Returns the maximum depth in the track and the 
        time and location (row, col) in which it occurred.'''
        v = np.max(np.where(np.isfinite(list(self.data.depth)) == 1,list(self.data.depth),-np.inf))
        t = list(self.data.time[self.data.depth == v])
        y = [int(i) for i in list(self.data.y[self.data.depth == v])]
        x = [int(i) for i in list(self.data.x[self.data.depth == v])]
        return v, t, y, x
    def trackLength(self):
        '''Adds together the distance between each segment of the track to find
        the total distance traveled.'''
        return np.nansum([float((self.data.Dx[i]**2+self.data.Dy[i]**2)**0.5) \
            for i in list(self.data.index) if ((self.data.type[i] != 0) or (self.data.Ely[i] > 0))])
    def avgArea(self):
        '''Identifies the average area for the track and the time stamp for 
        when it occurred.'''
        areas = [float(i) for i in list(self.data.area[self.data.type != 0])]
        return float(sum(areas))/len(self.data[self.data.type != 0])
    def mcc(self):
        '''Returns a 1 if at any point along the track the cyclone system is
        a multi-center cyclone. Retruns a 0 otherwise.'''
        if np.nansum([int(c) != 1 for c in self.data.centers[self.data.type != 0]]) == 0:
            return 0
        else:
            return 1
    def CAP(self):
        '''Returns the total cyclone-associated precipitation for the cyclone center.'''
        return np.nansum(list(self.data.precip)[self.data.type != 0])

class cycloneParameters:
    '''This class stores parameters used for cyclone detection and tracking. The
    only real purpose is to save the parameters being used for future reference.
    
    Will be removed in version 10_6.
    '''
    def __init__(self, outpath, timestart, timeend, timestep, timeref, reffile, \
    cellsize, surfMin, surfMax, kSize, d_slp, d_dist, elevMax, contInt, \
    mccTol, mccDist, speedMax, speedScalar, timecount, fieldElev=np.nan, \
    fieldLats= np.nan, fieldLongs=np.nan):
        self.outpath = outpath
        self.timeStart = timestart
        self.timeEnd = timeend
        self.timeStep = timestep
        self.timeCount = timecount
        self.timeRef = timeref
        self.fileRef = reffile
        self.cellSize = cellsize
        self.fieldElev = fieldElev
        self.fieldLats = fieldLats
        self.fieldLongs = fieldLongs
        self.surfMin = surfMin
        self.surfMax = surfMax
        self.kSize = kSize
        self.d_slp = d_slp
        self.d_dist = d_dist
        self.elevMax = elevMax
        self.contInt = contInt
        self.mccTol = mccTol
        self.mccDist = mccDist
        self.speedMax = speedMax
        self.speedScalar = speedScalar

'''        
#################
### FUNCTIONS ###
#################
'''

'''###########################
Find Nearest Value
###########################'''
def findNearest(array,value):
    '''
    Finds the gridcell of a numpy array that most closely matches the given
    value. Returns value and its index.
    '''
    idx = (np.abs(array-value)).argmin()
    return array[idx], idx

'''###########################
Find Nearest Point
###########################'''
def findNearestPoint(a,B,latlon=0):
    '''
    Finds the closest location in the array B to the point a, when a is an 
    ordered pair (row,col or lat,lon) and B is an array or list of ordered 
    pairs. All pairs should be in the same format (row,col) or (col,row).
    
    The optional paramter latlon is 0 by default, meaning that "closest" is
    calculated as a Euclidian distance of numpy array positions. If latlon=1, 
    the haversine formula is used to determine distance instead.
    
    Returns the index of the location in B that is closest and the minimum distance.
    '''
    if latlon == 0:
        dist = [( (b[0]-a[0])**2 + (b[1]-a[1])**2 )**0.5 for b in B]
    else:
        dist = [haversine(a[0],b[0],a[1],b[1]) for b in B]
    i = np.argmin(dist)
    
    return i , dist[i]

'''###########################
Find Nearest Area
###########################'''
def findNearestArea(a,B,b="all",latlon=[]):
    '''
    Finds the closest unique area in the array B to the point a, when a is an 
    ordered pair (row,col or lat,lon) and B is an array of contiguous areas 
    identified by a unique integer.* All pairs should be in the same format 
    (row,col). 
    
    The optional parameter b can be used to assess a subset of the areas in B 
    (in which case b should be a list, tuple, or 1-D arrray). By default, all 
    areas in B are assessed.
    
    The optional paramter latlon is [] by default, meaning that "closest" is
    calculated as a Euclidian distance of numpy array positions. Alternatively,
    latlon can be a list of two numpy arrays (lats,lons) with the same shape 
    as the input B. If so, the haversine formula is used to determine distance
    instead. If latitude and longitude are used, then a should be a tuple of
    latitude and longitude; otherwise, it should be a tuple of (row,col)
    
    Returns the ID of the area in B that is closest.
    
    *You can generate an array like this from a field of 0s and 1s using 
    scipy.ndimage.measurements.label(ARRAY).
    '''
    if b == "all":
        b = np.unique(B)[np.where(np.unique(B) != 0)]
    
    # First identify the shortest distance between point a and each area in B
    if latlon == []:
        dist = []
        for j in b:
            locs1 = np.where(B == j)
            locs2 = [(locs1[0][i],locs1[1][i]) for i in range(len(locs1[0]))]
            dist.append(findNearestPoint(a,locs2)[1])
    else:
        dist = []
        for j in b:
            locs1 = np.where(B == j)
            locs2 = [(latlon[0][locs1[0][i],locs1[1][i]],latlon[1][locs1[0][i],locs1[1][i]]) for i in range(len(locs1[0]))]
            dist.append(findNearestPoint(a,locs2,1)[1])
        
    # Then identify the shortest shortest distance
    return b[np.argmin(dist)]

'''###########################
Leap Year Boolean Creation
###########################'''
def leapyearBoolean(years):
    '''
    Given a list of years, this function will identify which years are leap 
    years and which years are not. Returns a list of 0s (not a leap year) and 
    1s (leap year). Each member of the year list must be an integer or float.
    
    Requires numpy.
    '''
    ly = [] # Create empty list
    for y in years: # For each year...
        if (y%4 == 0) and (y%100!= 0): # If divisible by 4 but not 100...
            ly.append(1) # ...it's a leap year
        elif y%400 == 0: # If divisible by 400...
            ly.append(1) # ...it's a leap year
        else: # Otherwise...
            ly.append(0) # ...it's NOT a leap year
    
    return ly

'''###########################
Calculate Days Between Two Dates
###########################'''
def daysBetweenDates(date1,date2,lys=1):
    '''
    Calculates the number of days between date1 (inclusive) and date2 (exclusive)
    when given dates in list format [year,month,day,hour,minute,second] or 
    [year,month,day]. Works even if one year is BC (BCE) and the other is AD (CE). 
    If hours are used, they must be 0 to 24. Requires numpy.
    
    date1 = the start date (earlier in time; entire day included in the count if time of day not specified)\n
    date2 = the end date (later in time; none of day included in count unless time of day is specified)
    '''
    db4 = [0,31,59,90,120,151,181,212,243,273,304,334] # Number of days in the prior months
    
    if date1[0] == date2[0]: # If the years are the same...
        # 1) No intervening years, so ignore the year value:        
        daysY = 0
    
    else: # B) If the years are different...
        
        # 1) Calculate the total number of days based on the years given:
        years = range(date1[0],date2[0]) # make a list of all years to count
        years = [yr for yr in years if yr != 0]
        if lys==1:
            lyb = leapyearBoolean(years) # annual boolean for leap year or not leap year
        else:
            lyb = [0]
        
        daysY = 365*len(years)+np.sum(lyb) # calculate number of days
    
    if lys == 1:
        ly1 = leapyearBoolean([date1[0]])[0]
        ly2 = leapyearBoolean([date2[0]])[0]
    else:
        ly1, ly2 = 0, 0
    
    # 2) Calcuate the total number of days to subtract from start year
    days1 = db4[date1[1]-1] + date1[2] -1 # days in prior months + prior days in current month - the day you're starting on
    # Add leap day if appropriate:    
    if date1[1] > 2:
        days1 = days1 + ly1
    
    # 3) Calculate the total number of days to add from end year
    days2 = db4[date2[1]-1] + date2[2] - 1 # days in prior months + prior days in current month - the day you're ending on
    # Add leap day if appropriate:    
    if date2[1] > 2:
        days2 = days2 + ly2
        
    # 4) Calculate fractional days (hours, minutes, seconds)
    day1frac, day2frac = 0, 0
    
    if len(date1) == 6:
        day1frac = (date1[5] + date1[4]*60 + date1[3]*3600)/86400.
    elif len(date1) != 3:
        raise Exception("date1 does not have the correct number of values.")
    
    if len(date2) == 6:
        day2frac = (date2[5] + date2[4]*60 + date2[3]*3600)/86400.
    elif len(date2) != 3:
        raise Exception("date2 does not have the correct number of values.")
    
    # 5) Final calculation
    days = daysY - days1 + days2 - day1frac + day2frac
    
    return days

'''###########################
Add Time
###########################'''
def timeAdd(time1,time2,lys=1):
    '''This function takes the sum of two times in the format [Y,M,D,H,M,S]. 
    The variable time1 should be a proper date (Months are 1 to 12, Hours are 0 to 23), 
    but time2 does not have to be a proper date. Note that if you use months or years in time2, 
    the algorithm will not discrimnate the number of days per month or year. To be precise,
    use only days, hours, minutes, and seconds. It can handle the BC/AD transition but 
    can only handle non-integers for days, hours, minutes, and seconds. To
    perform date subtraction, simply make the entries in time2 negative numbers.
    
    lys = Boolean to determine whether to recognize leap years (1; default) or to use
    a constant 365-day calendar.
    
    Addition Examples:
    [2012,10,31,17,44,54] + [0,0,0,6,15,30] = [2012,11,1,0,0,24] #basic example
    [2012,10,31,17,44,54] + [0,0,0,0,0,22530] = [2012,11,1,0,0,24] #time2 is improper time
    
    [1989,2,25,0,0,0] + [0,5,0,0,0,0] = [1989,7,25,0,0,0] #non-leap year months
    [1988,2,25,0,0,0] + [0,5,0,0,0,0] = [1988,7,25,0,0,0] #leap year months
    
    [1989,2,25,0,0,0] + [0,0,150,0,0,0] = [1989,7,25,0,0,0] #non-leap year days
    [1988,2,25,0,0,0] + [0,0,150,0,0,0] = [1988,7,24,0,0,0] #leap year days
    
    [1989,7,25,0,0,0] + [4,0,0,0,0] = [1993,7,25,0,0,0] #non-leap year years
    [1988,7,25,0,0,0] + [4,0,0,0,0] = [1992,7,25,0,0,0] #leap year years
    
    [-1,12,31,23,59,59] + [0,0,0,0,0,1] = [1,1,1,0,0,0] #crossing BC/AD with seconds
    [-2,1,1,0,0,0] + [4,0,0,0,0,0] = [3,1,1,0,0,0] #crossing BC/AD with years
    
    [1900,9,30,12,0,0] + [0,0,0.25,0,0,0] = [1900, 9, 30.0, 18, 0, 0.0] #fractional days
    [1900,9,30,12,0,0] + [0,0,0.5,0,0,0] = [1900, 10, 1.0, 0, 0, 0.0] #fractional days
    
    Subtraction Examples:
    [2012,10,31,17,44,54] - [0,0,0,-10,-50,-15] = [2012,10,31,6,54,39] #basic example
    [2012,10,31,17,44,54] - [0,0,0,0,0,-39015] = [2012,10,31,6,54,39] #time2 is imporper time
    
    [1989,7,25,0,0,0] + [0,-5,0,0,0,0] = [1989,2,25,0,0,0] #non-leap year months
    [1988,7,25,0,0,0] + [0,-5,0,0,0,0] = [1988,2,25,0,0,0] #leap year months
    
    [1989,7,25,0,0,0] + [0,0,-150,0,0,0] = [1989,2,25,0,0,0] #non-leap year days
    [1988,7,25,0,0,0] + [0,0,-150,0,0,0] = [1988,2,26,0,0,0] #leap year days
    
    [1993,2,25,0,0,0] + [-4,0,0,0,0] = [1989,2,25,0,0,0] #non-leap year years
    [1992,2,25,0,0,0] + [-4,0,0,0,0] = [1988,2,25,0,0,0] #leap year years
    
    [1,1,1,0,0,0] + [0,0,0,0,0,-1] = [-1,12,31,23,59,59] #crossing BC/AD with seconds
    [2,1,1,0,0,0] + [-4,0,0,0,0,0] = [-3,1,1,0,0,0] #crossing BC/AD with years 
    
    [1900,9,30,12,0,0] + [0,0,0.25,0,0,0] = [1900, 9, 30.0, 6, 0, 0.0] #fractional days
    [1900,9,30,12,0,0] + [0,0,0.5,0,0,0] = [1900, 9, 29.0, 0, 0, 0.0] #fractional days
    '''
    if len(time1) == 3:
        time1 = time1 + [0,0,0]
    if len(time2) == 3:
        time2 = time2 + [0,0,0]
    
    # Ensure that years and months are whole numbers:
    if time1[0]%1 != 0 or time1[1]%1 != 0 or time2[0]%1 != 0 or time2[1]%1 != 0:
        raise ValueError("The year and month entries are not all whole numbers.")
    
    else:
        # Identify Fractional days:
        day1F = time1[2]%1
        day2F = time2[2]%1
        
        # Initial Calculation: Add, transfer appropriate amount to next place, keep the remainder
        secR = (time1[5] + time2[5] + (day1F+day2F)*86400)%60
        minC = int((time1[5] + time2[5] + (day1F+day2F)*86400)/60)
        
        minR = (time1[4] + time2[4] + minC)%60
        hrsC = int((time1[4] + time2[4] + minC)/60)
        
        hrsR = (time1[3] + time2[3] + hrsC)%24
        dayC = int((time1[3] + time2[3] + hrsC)/24)
        
        dayA = (time1[2]-day1F) + (time2[2]-day2F) + dayC # Initially, just calculate days
        
        monA = (time1[1]-1 + time2[1])%12 + 1 # Because there is no month 0
        yrsC = int((time1[1]-1 + time2[1])/12)
        
        yrsA = time1[0] + time2[0] + yrsC
        
        ######################
        #### REFINEMENTS  ####
        dpm = [31,28,31,30,31,30,31,31,30,31,30,31] # days per month
        dpmA = [d for d in dpm] # make modifiable copy
        
        ### Gregorian Calendar ###
        if lys == 1:
            ### Deal with BC/AD ###
            if time1[0] < 0 and yrsA >= 0: # Going from BC to AD
                yrsR = yrsA + 1
            elif time1[0] > 0 and yrsA <= 0: # Going from AD to BC
                yrsR = yrsA - 1
            else:
                yrsR = yrsA
            
            ### Deal with Days ###
            dpmA[1] = dpmA[1] + leapyearBoolean([yrsR])[0] # days per month adjusted for leap year (if applicable)
            
            if dayA > 0: # if the number of days is positive
                if dayA <= dpmA[monA-1]: # if the number of days is positive and less than the full month...
                    dayR = dayA #...no more work needed
                    monR = monA
                
                elif dayA <= sum(dpmA[monA-1:]): # if the number of days is positive and over a full month but not enough to carry over to the next year...
                    monR = monA
                    dayR = dayA
                    while dayR > dpmA[monR-1]: # then walk through each month, subtracting days as you go until there's less than a month's worth
                        dayR = dayR - dpmA[monR-1]
                        monR = monR+1
                
                else: # if the number of days is positive and will carry over to another year...
                    dayR = dayA - sum(dpmA[monA-1:]) # go to Jan 1 of next year...
                    yrsR = yrsR+1
                    ly = leapyearBoolean([yrsR])[0]
                    while dayR > 365+ly: # and keep subtracting 365 or 366 (leap year dependent) until until no local possible
                        dayR = dayR - (365+ly)
                        yrsR = yrsR+1
                        if yrsR == 0: # Disallow 0-years
                            yrsR = 1
                        ly = leapyearBoolean([yrsR])[0]
                    
                    dpmB = [d for d in dpm]
                    dpmB[1] = dpmB[1] + ly
                    monR = 1
                    while dayR > dpmB[monR-1]: # then walk through each month
                        dayR = dayR - dpmB[monR-1]
                        monR = monR+1
            
            elif dayA == 0: # if the number of days is 0
                if monA > 1:
                    monR = monA-1
                    dayR = dpmA[monR-1]
                else:
                    monR = 12
                    dayR = 31
                    yrsR = yrsR - 1
            
            else: # if the number of days is negative
                if abs(dayA) < sum(dpmA[:monA-1]): # if the number of days will stay within the same year...
                    monR = monA
                    dayR = dayA
                    while dayR <= 0:
                        monR = monR-1
                        dayR = dpmA[monR-1] + dayR
                
                else: # if the number of days is negative and will cross to prior year...
                    dayR = dayA + sum(dpmA[:monA-1])
                    yrsR = yrsR-1
                    if yrsR == 0:
                        yrsR = -1
                    
                    ly = leapyearBoolean([yrsR])[0]
                    while abs(dayR) >= 365+ly:
                        dayR = dayR + (365+ly)
                        yrsR = yrsR-1
                        ly = leapyearBoolean([yrsR])[0]
                    
                    dpmB = [d for d in dpm]
                    dpmB[1] = dpmB[1] + ly
                    monR = 13
                    dayR = dayR
                    while dayR <= 0:
                        monR = monR-1
                        dayR = dpmB[monR-1] + dayR
        
        ### 365-Day Calendar ###
        else:            
            if dayA > 0: # if the number of days is positive
                if dayA <= dpmA[monA-1]: # if the number of days is positive and less than the full month...
                    dayR = dayA #...no more work needed
                    monR = monA
                
                elif dayA <= sum(dpmA[monA-1:]): # if the number of days is positive and over a full month but not enough to carry over to the next year...
                    monR = monA
                    dayR = dayA
                    while dayR > dpmA[monR-1]: # then walk through each month, subtracting days as you go until there's less than a month's worth
                        dayR = dayR - dpmA[monR-1]
                        monR = monR+1
                
                else: # if the number of days is positive and will carry over to another year...
                    dayR = dayA - sum(dpmA[monA-1:]) # go to Jan 1 of next year...
                    yrsA = yrsA+1
                    while dayR > 365: # and keep subtracting 365 or 366 (leap year dependent) until until no local possible
                        dayR = dayR - 365
                        yrsA = yrsA+1
                    
                    dpmB = [d for d in dpm]
                    dpmB[1] = dpmB[1]
                    monR = 1
                    while dayR > dpmB[monR-1]: # then walk through each month
                        dayR = dayR - dpmB[monR-1]
                        monR = monR+1
            
            elif dayA == 0: # if the number of days is 0
                if monA > 1:
                    monR = monA-1
                    dayR = dpmA[monR-1]
                else:
                    monR = 12
                    dayR = 31
                    yrsA = yrsA -1
            
            else: # if the number of days is negative
                if abs(dayA) < sum(dpmA[:monA-1]): # if the number of days will stay within the same year...
                    monR = monA
                    dayR = dayA
                    while dayR <= 0:
                        monR = monR-1
                        dayR = dpmA[monR-1] + dayR
                
                else: # if the number of days is negative and will cross to prior year...
                    dayR = dayA + sum(dpmA[:monA-1])
                    yrsA = yrsA-1
                    while abs(dayR) >= 365:
                        dayR = dayR + 365
                        yrsA = yrsA-1
                    
                    dpmB = [d for d in dpm]
                    dpmB[1] = dpmB[1]
                    monR = 13
                    dayR = dayR
                    while dayR <= 0:
                        monR = monR-1
                        dayR = dpmB[monR-1] + dayR
        
            ### Deal with BC/AD ###
            if time1[0] < 0 and yrsA >= 0: # Going from BC to AD
                yrsR = yrsA + 1
            elif time1[0] > 0 and yrsA <= 0: # Going from AD to BC
                yrsR = yrsA - 1
            else:
                yrsR = yrsA
        
        return [yrsR,monR,dayR,hrsR,minR,secR]
    
'''###########################
Ring Kernel Creation
###########################'''
def ringKernel(ri,ro,d=0):
    '''Given two radii in numpy array cells, this function will calculate a 
    numpy array of 1 and nans where 1 is the cells whose centroids are more than 
    ri units away from the center centroid but no more than ro units away. The 
    result is a field of nans with a ring of 1s.
    
    ri = inner radius in numpy array cells (integer or float)
    ro = outer radius in numpy array cells (integer or float)
    d = if d==0, then function returns an array of 1s and nans
        if d==1, then function returns an array of distances from
        center (in array cells)
    
    '''
    # Create a numpy array of 1s:
    k = int(ro*2+1)
    ringMask=np.ones((k,k))
    
    # If returing 1s and nans:
    if d == 0:
        for row in range(0,k):
            for col in range(0,k):
                d = ((row-ro)**2 + (col-ro)**2)**0.5
                if d > ro or d <= ri:
                    ringMask[row,col]=np.NaN
        return ringMask
    
    # If returning distances:
    if d == 1:
        ringDist = np.zeros((k,k))
        for row in range(0,k):
            for col in range(0,k):
                ringDist[row,col] = ((row-ro)**2 + (col-ro)**2)**0.5
                if ringDist[row,col] > ro or ringDist[row,col] <= ri:
                    ringMask[row,col]=np.NaN
        ringDist = ringMask*ringDist
        return ringDist

'''###########################
Circle Kernel Creation
###########################'''
def circleKernel(r):
    '''Given the radius in numpy array cells, this function will calculate a 
    numpy array of 1 and nans where 1 is the cells whose centroids are less than 
    the radius away from the center centroid.
    
    r = radius in numpy array cells (integer or float)
    '''
    # Create a numpy array of 1s:
    rc = int(np.ceil(r))
    k = rc*2+1
    circleMask=np.ones((k,k))
    for row in range(0,k):
        for col in range(0,k):
            d = ((row-rc)**2 + (col-rc)**2)**0.5
            if d > r:
                circleMask[row,col]=np.NaN
    return circleMask

'''###########################
Calculate a Smoothed Density Field
###########################'''
def smoothField(var,kSize,nanedge=0):
    '''Uses a rectangular kernel to smooth the input numpy array.  Dimensions
    of the kernel are set by kSize.
    
    var = a numpy array of values to be smoothed
    kSize = a tuple or list of the half-height and half-width of the kernel or,
        if the kernel is a square, this may be a single number repsresenting half-width
    nanedge = binary that indicates treatment of edges; if 0, nans are ignored in calculations, 
        if 1, the smoothed value will be a nan if any nans are observed in the kernel
    '''
    
    # Create kernel
    if (isinstance(kSize,tuple) == 0) and (isinstance(kSize,list) == 0):
        krow = range(-kSize,kSize+1)
        kcol = range(-kSize,kSize+1)
    else:
        krow = range(-kSize[0],kSize[0]+1)
        kcol = range(-kSize[1],kSize[1]+1)
    
    # Store main dimensions
    nrow = var.shape[0]
    ncol = var.shape[1]
    
    # Create list of arrays for smoothing
    varFields = []
    for r in krow:
        for c in kcol:
            varFields.append( np.hstack(( np.zeros((nrow+kSize*2, kSize-c))*np.nan , \
            np.vstack(( np.zeros((kSize-r,ncol))*np.nan, var ,np.zeros((kSize+r,ncol))*np.nan )), \
            np.zeros((nrow+kSize*2, kSize+c))*np.nan )) )
    
    # Smooth arrays
    if nanedge == 1:
        smoothedField = meanArrays(varFields,dtype="float")
    else:
        smoothedField = meanArrays_nan(varFields,dtype="float")
    
    # Return smooothed array in original dimensions
    return smoothedField[kSize:nrow+kSize,kSize:ncol+kSize]

def filterField(var,kSize,nanedge=0,style="mean"):
    '''Uses a rectangular kernel to filter/smooth the input numpy array.  
    Dimensions of the kernel are set by kSize.
    
    var = a numpy array of values to be smoothed
    kSize = a tuple or list of the half-height and half-width of the kernel or,
        if the kernel is a square, this may be a single number repsresenting half-width
    nanedge = binary that indicates treatment of edges; if 0, nans are ignored in calculations, 
        if 1, the smoothed value will be a nan if any nans are observed in the kernel
    style = the style of filter, can be mean (or average), median, maximum (or max), 
    minimum (or min), sum (or total), or standard deviation (std). Note that there is currently no
    "ignore nan" option for standard deviation
    '''
    # Create kernel
    if (isinstance(kSize,tuple) == 0) and (isinstance(kSize,list) == 0): # Squares
        krow = range(-kSize,kSize+1)
        kcol = range(-kSize,kSize+1)
    else: # Rectangles
        krow = range(-kSize[0],kSize[0]+1)
        kcol = range(-kSize[1],kSize[1]+1)
    
    # Store main dimensions
    nrow = var.shape[0]
    ncol = var.shape[1]
    
    # Create list of arrays for smoothing
    varFields = []
    for r in krow:
        for c in kcol:
            varFields.append( np.hstack(( np.zeros((nrow+kSize*2, kSize-c))*np.nan , \
            np.vstack(( np.zeros((kSize-r,ncol))*np.nan, var ,np.zeros((kSize+r,ncol))*np.nan )), \
            np.zeros((nrow+kSize*2, kSize+c))*np.nan )) )
    
    # Smooth arrays
    if nanedge == 1:
        if (style.lower() == "mean") or (style.lower() == "average"):
            smoothedField = np.apply_along_axis(np.mean,0,varFields,dtype="float")
        elif (style.lower() == "sum") or (style.lower() == "total"):
            smoothedField = np.apply_along_axis(np.sum,0,varFields,dtype="float")
        elif (style.lower() == "median"):
            smoothedField = np.apply_along_axis(np.median,0,varFields,dtype="float")
        elif (style.lower() == "max") or (style.lower() == "maximum"):
            smoothedField = np.apply_along_axis(np.max,0,varFields,dtype="float")
        elif (style.lower() == "min") or (style.lower() == "minimum"):
            smoothedField = np.apply_along_axis(np.min,0,varFields,dtype="float")
        elif style.lower() in ["sd","std","stdev","stddev","standard deviation"]:
            smoothedField = np.apply_along_axis(np.std,0,varFields,dtype="float")
        else:
            raise Exception("Invalid style passed to filterField function. Valid styles include mean, average, median, minimum, min, maximum, max, sum, total, standard deviation, sd, std, stdec, and stddev")
    else:
        if (style.lower() == "mean") or (style.lower() == "average"):
            smoothedField = np.apply_along_axis(np.nanmean,0,varFields,dtype="float")
        elif (style.lower() == "sum") or (style.lower() == "total"):
            smoothedField = np.apply_along_axis(np.nansum,0,varFields,dtype="float")
        elif (style.lower() == "median"):
            smoothedField = np.apply_along_axis(np.nanmedian,0,varFields,dtype="float")
        elif (style.lower() == "max") or (style.lower() == "maximum"):
            smoothedField = np.apply_along_axis(np.nanmax,0,varFields,dtype="float")
        elif (style.lower() == "min") or (style.lower() == "minimum"):
            smoothedField = np.apply_along_axis(np.nanmin,0,varFields,dtype="float")
        elif style.lower() in ["sd","std","stdev","stddev","standard deviation"]:
            smoothedField = np.apply_along_axis(np.std,0,varFields,dtype="float")
            raise Warning("No 'ignore nans' option for standard deviation; nans propagated.")
        else:
            raise Exception("Invalid style passed to filterField function. Valid styles include mean, average, median, minimum, min, maximum, max, sum, total, standard deviation, sd, std, stdec, and stddev")
    
    # Return smooothed array in original dimensions
    return smoothedField[kSize:nrow+kSize,kSize:ncol+kSize]

'''###########################
Calculate Mean Gradient from Center Based on a Kernel
###########################'''
def kernelGradient(field,kernel,location,cellsize):
    '''Given a field and a tuple or list of (row,col), this function
    calculates the difference between the location and each gridcell in a
    subset that matches the kernel size. It then calculates the gradient using 
    the cellsize and calculates the mean for all gricells in the kernel.
    
    field = a numpy array of floats or integers
    kernel = a numpy array (with smaller dimensions than the field) that 
        contains distances from the center of the kernel and nans for gridcells
        that are not of interest
    location = a tuple or list in the format (row,col) in array coordinates
    cellsize = the real cell size of each gridcell (e.g. m, mi, etc.)
    
    Returns a mean gradient in field units per cellsize*(kernel half-width)
    '''
    # Define row, col, and radius of kernel (radius = half-width)
    row, col = location[0], location[1]
    radius = int(kernel.shape[1]-1)/2
    
    # Define the starting and stopping rows and cols:
    rowS = row-radius
    rowE = row+radius
    colS = col-radius
    colE = col+radius
    if rowS < 0:
        rowS = 0
    if rowE > field.shape[0]:
        rowE = field.shape[0]
    if colS < 0:
        colS = 0
    if colE > field.shape[1]:
        colE = field.shape[1]
    
    # Take a subset to match the kernel size
    subset = field[rowS:rowE+1,colS:colE+1]
    
    # Add nans to fill out subset if not already of equal size with kernel
    if kernel.shape[0] > subset.shape[0]: # If there aren't enough rows to match kernel
        nanrows = np.empty( ( (kernel.shape[0]-subset.shape[0]),subset.shape[1] ) )*np.nan
        if rowS == 0: # Add on top if the first row is row 0
            subset = np.vstack( (nanrows,subset) )
        else: # Add to bottom otherwise
            subset = np.vstack( (subset,nanrows) )
    if kernel.shape[1] > subset.shape[1]: # If there aren't enough columns to match kernel
        nancols = np.empty( (subset.shape[0], (kernel.shape[1]-subset.shape[1]) ) )*np.nan
        if colS == 0: # Add to left if first col is col 0
            subset = np.hstack( (nancols,subset) )
        else: # Add to right otherwise
            subset = np.hstack( (subset,nancols) )
    
    # Apply kernel and calculate difference between each cell and the center
    subset_ring = (subset - field[row,col])/(cellsize*kernel)
    
    # Find the mean value (excluding nans)
    mean = np.nansum(subset_ring) / ( np.sum( np.isfinite(subset_ring) ) )
    
    return mean

'''###########################
Calculate Slope of a Field
###########################'''
def slope(field):
    '''Given a field, this function calculates the slope (the field's derivative
    over two dimensions). This gradient is calculated using a Sobel operator,
    so the edges are returned as NaN. It returns both the final slope and the 
    slope components in the x and y directions (and in that order).
    
    field = a numpy array of values over which to take the slope.
    '''
    # Introduce empty arrays:
    slopeX = np.zeros(field.shape, dtype=float)
    slopeY = np.zeros(field.shape, dtype=float)
    slope = np.zeros(field.shape, dtype=float)
    
    # First Gradient
    for row in range(field.shape[0]):
        for col in range(field.shape[1]):
            try:
                slopeX[row,col] = ((field[row-1,col+1]+2.*field[row,col+1]+field[row+1,col+1])\
                - (field[row-1,col-1]+2.*field[row,col-1]+field[row+1,col-1])) / (8.)
            except:
                slopeX[row,col] = np.nan
            try:
                slopeY[row,col] = -((field[row+1,col-1]+2.*field[row+1,col]+field[row+1,col+1])\
                - (field[row-1,col-1]+2.*field[row-1,col]+field[row-1,col+1])) / (8.)
            except:
                slopeY[row,col] = np.nan
    
    # Final Calculation      
    slope = (slopeX**2. + slopeY**2.)**0.5
    
    return slope, slopeX, slopeY

'''###########################
Calculate Sobel Laplacian of Field
###########################'''
def laplacian(field):
    '''Given a field, this function calculates the Laplacian (the field's 
    second derivative over two dimensions). The gradient is calculated using a 
    Sobel operator, so edge effects do exist. Following the method of Murray 
    and Simmonds (1991), the second derivative is first calculated for the x 
    and y orthognoal components individally and then combined.
    
    field = a numpy array of values over which to take the Laplacian
    '''
    # Introduce empty arrays:
    slopeX = np.zeros(field.shape, dtype=float)
    slopeY = np.zeros(field.shape, dtype=float)
        
    laplacX = np.zeros(field.shape, dtype=float)
    laplacY = np.zeros(field.shape, dtype=float)
    laplac = np.zeros(field.shape, dtype=float)
    
    # First Gradient
    for row in range(field.shape[0]):
        for col in range(field.shape[1]):
            try:
                slopeX[row,col] = ((field[row-1,col+1]+2.*field[row,col+1]+field[row+1,col+1])\
                - (field[row-1,col-1]+2.*field[row,col-1]+field[row+1,col-1])) / (8.)
            except:
                slopeX[row,col] = np.nan
            try:
                slopeY[row,col] = -((field[row+1,col-1]+2.*field[row+1,col]+field[row+1,col+1])\
                - (field[row-1,col-1]+2.*field[row-1,col]+field[row-1,col+1])) / (8.)
            except:
                slopeY[row,col] = np.nan
            
    # Second Gradient
    for row in range(field.shape[0]):
        for col in range(field.shape[1]):
            try:
                laplacX[row,col] = ((slopeX[row-1,col+1]+2.*slopeX[row,col+1]+slopeX[row+1,col+1])\
                - (slopeX[row-1,col-1]+2.*slopeX[row,col-1]+slopeX[row+1,col-1])) / (8.)
            except:
                laplacX[row,col] = np.nan
            try:
                laplacY[row,col] = -((slopeY[row+1,col-1]+2.*slopeY[row+1,col]+slopeY[row+1,col+1])\
                - (slopeY[row-1,col-1]+2.*slopeY[row-1,col]+slopeY[row-1,col+1])) / (8.)
            except:
                laplacY[row,col] = np.nan
    
    #laplac = (laplacX**2. + laplacY**2.)**0.5
    laplac = laplacX + laplacY

    return laplac

'''###########################
Detect Minima/Maxima of a Surface
###########################'''
def detectMinima(field,kSize=1,nanthreshold=0.5):
    '''Identifies local minima in a numpy array (surface) by searching within a 
    square kernel (the neighborhood) for each grid cell. Ignores nans by default.
    
    field = a numpy array that represents some surface of interest (numpy array)
    kSize = given a center grid cell, how many many gridcells should be
        searched in each direction? The default of 1 gives a 3*3 kernel, 2 
        gives 5*5, and so on.
    nanthreshold = maximum ratio of nan cells to total cells around the center
        cell for each minimum test. 0 means that no cell with a nan neighbor 
        can be considered a minimum. 0.5 means that less than half of the 
        neighbors can be nans for a cell to be considered a minimum. 1 means 
        that a cell will be considered a minimum if all cells around it are nan.
    '''
    field_min = np.zeros_like(field) # 1s for minima    
    
    for row in range(kSize,field.shape[0]-kSize):
        for col in range(kSize,field.shape[1]-kSize):
            # Slice the surface using the kernel
            Slice = field[row-kSize:row+kSize+1,col-kSize:col+kSize+1]
            # If less than half of the values around the center of the slice are nans
            nanfraction = np.sum(np.isnan(Slice)) / float((Slice.shape[0]*Slice.shape[1])-1)
            if nanthreshold > nanfraction:
                # Find the minimum non-nan value
                if Slice[kSize,kSize] == np.nanmin(Slice): # If the center of the slice is also the minimum
                    field_min[row,col] = 1 # Assign a 1 to the minimum field for that location
    
    return field_min

def detectMaxima(field,kSize=1,nanthreshold=0.5):
    '''Identifies local maxima in a numpy array (surface) by searching within a 
    square kernel (the neighborhood) for each gridcell. Ignores nans by default.
    
    field = a numpy array that represents some surface of interest (numpy array)
    kSize = given a center gridcell, how many many gridcells should be
        searched in each direction? The default of 1 gives a 3*3 kernel, 2 
        gives 5*5, and so on.
    nanthreshold = maximum ratio of nan cells to total cells around the center
        cell for each maximum test. 0 means that no cell with a nan neighbor 
        can be considered a maximum. 0.5 means that less than half of the 
        neighbors can be nans for a cell to be considered a maximum. 1 means 
        that a cell will be considered a maximum if all cells around it are nan.
    '''
    field_max = np.zeros_like(field) # 1s for maxima    
    
    for row in range(kSize,field.shape[0]-kSize):
        for col in range(kSize,field.shape[1]-kSize):
            # Slice the surface using the kernel
            Slice = field[row-kSize:row+kSize+1,col-kSize:col+kSize+1]
            # If less than half of the values aroudn the center of the slice are nans
            nanfraction = np.sum(np.isnan(Slice)) / float((Slice.shape[0]*Slice.shape[1])-1)
            if nanthreshold > nanfraction:
                # Find the maximum non-nan value
                if Slice[kSize,kSize] == np.nanmax(Slice): # If the center of the slice is also the maximum
                    field_max[row,col] = 1 # Assign a 1 to the maximum field for that location
    
    return field_max

'''###########################
Find Special Types of Minima of a Surface
###########################'''
def findCenters(field, kSize=1, d_slp=0, d_dist=0, cellsize=0, nanthreshold=0.5):
    '''This function identifies minima in a field and then eliminates minima
    that do not satisfy a gradient parameter. By default, the function will 
    ONLY find minima.
    
    field = the numpy array that you want to find the minima of.
    kSize = the number of rings out from a gridcell that should be considered
        when determining whether it is a minimum; 1 by default\n
    
    d_slp and d_dist = the SLP and distance that determine the minimum pressure
        gradient allowed for a minimum to be considered a cyclone center. By 
        default they are left at 0, so no gradients will be considered.
    cellsize = the cellsize of the numpy array. Only necessary if calculating 
        gradients, so the default is 0.\n
    
    ignorenans = if 1 (the default), the function uses np.nanmin(), which 
        ignores all nans. If changed to 0, then no gridcell can be selected as 
        a minimum if it has a nan in its neighborhood. 
    '''
    # STEP 1.1. Define gradient limit
    d_grad = float(d_slp)/d_dist
    
    # STEP 1.2. Identify Minima:
    fieldMinima = detectMinima(field,kSize,nanthreshold=nanthreshold).astype(np.int)
    rowMin, colMin = np.where(fieldMinima == 1)
    
    # STEP 1.3. Discard Weak Minima:        
    if d_slp == 0:
        fieldCenters = fieldMinima
    
    else:
        sysMin = fieldMinima.copy() # make a mutable copy of the minima locations
        # Create a kernel that will make a 1-unit thick ring around which
        ### to calculate the gradient
        radius = int(np.ceil(d_dist/cellsize)) # define outer radius of kernel
        kernel = ringKernel(radius-1,radius,d=1)
        
        for sm in range(np.sum(fieldMinima)): # For each minimum...
            # Calculate gradient:
            mean_gradient = kernelGradient(field,kernel,(rowMin[sm],colMin[sm]),cellsize)
            # Test for pressure gradient:
            if (mean_gradient < d_grad):
                sysMin[rowMin[sm],colMin[sm]] = 0
        
        # Save centers
        fieldCenters = sysMin
    
    return fieldCenters

'''###########################
Identify the Areas Associated with Special Minima of a Surface
###########################'''
def findAreas(field,fieldCenters,centers,contint,mcctol,mccdist,cellsize,maxes=0):
    if isinstance(maxes, int) == 1:
        field_max = np.zeros_like(field)
    else:
        field_max = maxes
    
    # Prepare preliminary outputs:
    cycField = np.zeros_like(field) # set up empty system field
    fieldCenters2 = fieldCenters.copy() # set up a field to modify center identification
    # Make cyclone objects for all centers
    cols, rows, ids, vals, types = [], [], [], [], []
    cyclones = copy.deepcopy(centers)
    
    # And helper lists for area detection:
    for center in centers:
            vals.append(center.p_cent)
            rows.append(center.y)
            cols.append(center.x)
            ids.append(center.id)
            types.append(0)
    
    while len([t for t in types if t == 0]) > 0:
        # Identify the center ID as the index of the lowest possible value
        ### that hasn't already been assigned to a cyclone:
        cid = np.array(ids)[np.where((np.array(types) == 0) & (np.array(vals) == np.min(np.array(vals)[np.where(np.array(types) == 0)[0]])))][0]
        
        candMin = fieldCenters.copy() # make a mutable copy of the minima locations
        
        nMins = 1 # set the flag to 1
        nMaxs = 0 # set flag to 0
        cCI = vals[cid] # set the initial contouring value
        
        while nMins == 1 and nMaxs == 0 and cCI < np.nanmax(field): # keep increasing interval as long as only one minimum is detected
            #Increase contour interval
            cCI = cCI + contint
            
            # Define contiguous areas
            fieldCI = np.where(field < cCI, 1, 0)
            areas, nA = scipy.ndimage.measurements.label(fieldCI)
            
            # Test how many minima are within the area associated with the minimum of interest
            areaTest = np.where((areas == areas[rows[cid],cols[cid]]) & (candMin == 1), 1, 0) # Limit to minima within that area
            nMins = np.sum(areaTest) # Count the number of minima identified
            
            # Test how many maxima are within the area associated with the minimum of interest
            maxTest = np.where((areas == areas[rows[cid],cols[cid]]) & (field_max == 1), 1, 0) # Limit to maxima within that area           
            nMaxs = np.sum(maxTest) # Count the number of maxima identified
            
            if nMins > 1: # If there's more than one minimum...
                # Take a subset of the ids that matches those in areaTest:
                rowTest, colTest = np.where(areaTest == 1)
                locSub = [(rowTest[ls],colTest[ls]) for ls in range(len(rowTest))]
                idsSub = [i for i in ids if ((rows[i],cols[i]) in locSub) and (i != cid)]
                
                # Make another subset to limit to possible secondary centers
                ### Based on a maximum distance and whether the center has already been assigned to a cyclone
                distTest = [( ((rows[cid]-rows[i])**2 + (cols[cid]-cols[i])**2)**0.5 )*cellsize > mccdist for i in idsSub]
                typeTest = [types[i] != 0 for i in idsSub]
                
                # If all minima both fit the distance parameter and are still unassigned...
                if sum(distTest) == 0 and sum(typeTest) == 0:
                    ################################
                    # Experiment - Add Secondaries #
                    ################################
                    # Test what would happen if those candidates were secondaries to center c  
                    # Remove secondaries as minima:
                    candMin2 = candMin.copy()
                    for i in idsSub:
                        candMin2[rows[i],cols[i]] = 0
                    
                    nMins2 = 1 # set the flag to 1
                    nMaxs2 = 0 # set the flag to 0
                    cCI2 = vals[cid] # set the initial contouring value
                    
                    while nMins2 == 1 and nMaxs2 == 0 and cCI2 < np.nanmax(field): # keep increasing interval as long as only one minimum is detected
                        #Increase contour interval
                        cCI2 = cCI2 + contint
                        
                        # Define contiguous areas
                        fieldCI2 = np.where(field < cCI2, 1, 0)
                        areas2, nA2 = scipy.ndimage.measurements.label(fieldCI2)
                        
                        # Test how many minima are within the area associated with the minimum of interest
                        # Limit to minima within that area
                        areaTest2 = np.where((areas2 == areas2[rows[cid],cols[cid]]) & (candMin2 == 1), 1, 0)
                        nMins2 = np.sum(areaTest2) # Count the number of minima identified
                        
                        # Test how many maxima are within the area associated with the minimum of interest
                        # Limit to maxima within that area
                        maxTest2 = np.where((areas2 == areas2[rows[cid],cols[cid]]) & (field_max == 1), 1, 0)           
                        nMaxs2 = np.sum(maxTest2) # Count the number of maxima identified
                    
                    #################################
                    # Once the with secondaries loop breaks, compare the number of contours used in each version:
                    numC1 = (cCI - vals[cid])/contint # The number of contour intervals WITHOUT secondaries
                    numC2 = (cCI2 - vals[cid])/contint # The number of contour intervals WITH secondaries
                    
                    # If including secondaries substantially increases the number of contours involved...
                    if (numC1 / numC2) < mcctol:
                        cCI = cCI2 # Keep the new contour level
                        for i in idsSub: # And add each of the other minima as secondaries
                            cyclones[i].add_parent(rows[cid],cols[cid],cid)
                            fieldCenters2[rows[i],cols[i]] = 2
                            cyclones[cid].secondary.append(centers[i].id)
                            cyclones[i].type = 2
                            types[i] = 2 # And change the type to secondary so it will no longer be considered
        
        
        # Go back one step on the contiguous areas:
        fieldF = np.where(field < (cCI-contint), 1, 0)
        areasF, nAF = scipy.ndimage.measurements.label(fieldF)
        
        # And identify the area associated with the minimum of interest:
        area = np.where((areasF == areasF[rows[cid],cols[cid]]) & (areasF != 0),1,0)
        if np.nansum(area) == 0: # If there's no area,
            area[rows[cid],cols[cid]] = 1 # Give it an area of 1 to match the location of the center
        
        cycField = cycField + area # And update the system field
        
        # Then assign characteristics to the cyclone:
        cyclones[cid].type = 1
        cyclones[cid].p_edge = cCI-contint
        cyclones[cid].area = np.nansum(area)
        
        # Also assign those characteristics to its secondaries:
        for cid_s in cyclones[cid].secondary:
            cyclones[cid_s].p_edge = cCI-contint
            cyclones[cid_s].area = np.nansum(area)
        
        # When complete, change type of cyclone
        types[cid] = 1
#        print "ID: " + str(cid) + ", Row: " + str(rows[cid]) + ", Col: " + str(cols[cid]) + \
#                ", Area:" + str(np.nansum(area))

    return cycField, fieldCenters2, cyclones

##### VERSION 2
def findAreas2(field,fieldCenters,centers,contint,mcctol,mccdist,cellsize,maxes=0):
    if isinstance(maxes, int) == 1:
        field_max = np.zeros_like(field)
    else:
        field_max = maxes
    
    # Prepare preliminary outputs:
    cycField = np.zeros_like(field) # set up empty system field
    fieldCenters2 = fieldCenters.copy() # set up a field to modify center identification
    # Make cyclone objects for all centers
    cols, rows, vals = [], [], []
    cyclones = copy.deepcopy(centers)
    types = np.zeros_like(np.array(centers))
    
    # And helper lists for area detection:
    for center in centers:
            vals.append(center.p_cent)
            rows.append(center.y)
            cols.append(center.x)
    
    # Create list of ids and of ids that have not been assigned yet
    ids = np.arange(len(vals))
    candids = np.where(types == 0)[0]
    
    #####################
    # Start Area Loop
    #####################
    while len([t for t in types if t == 0]) > 0:
        # Identify the center ID as the index of the lowest possible value
        ## that hasn't already been assigned to a cyclone:
        cid = ids[np.where((types == 0) & (np.array(vals) == np.min(np.array(vals)[np.where(types == 0)[0]])))][0]
                        
        nMins = 0 # set the flag to 1
        nMaxs = 0 # set flag to 0
        cCI = vals[cid] # set the initial contouring value
        
        # Identify the number of minima  within the mccdist
        distTest = [( ((rows[cid]-rows[i])**2 + (cols[cid]-cols[i])**2)**0.5 )*cellsize > mccdist for i in candids]
        ncands = len(distTest) - np.sum(distTest) # Number of centers w/n the mccdist
        icands = candids[np.where(np.array(distTest) == 0)] # IDs for centers w/n mccdist
        
        #########################
        # If No MCC Is Possible #
        #########################
        # If there's no other minima w/n the mccdist, use the simple method
        if ncands == 1:
            while nMins == 0 and nMaxs == 0 and cCI < np.nanmax(field): # keep increasing interval as long as only one minimum is detected
                #Increase contour interval
                cCI = cCI + contint
                
                # Define contiguous areas
                fieldCI = np.where(field < cCI, 1, 0)
                areas, nA = scipy.ndimage.measurements.label(fieldCI)
                
                # Test how many minima are within the area associated with the minimum of interest
                areaTest = np.where((areas == areas[rows[cid],cols[cid]]) & (fieldCenters != 0), 1, 0) # Limit to minima within that area
                nMins = np.sum(areaTest)-1 # Count the number of minima identified (besides the minimum of focus)
                
                # Test how many maxima are within the area associated with the minimum of interest
                maxTest = np.where((areas == areas[rows[cid],cols[cid]]) & (field_max == 1), 1, 0) # Limit to maxima within that area           
                nMaxs = np.sum(maxTest) # Count the number of maxima identified
            
            # Re-adjust the highest contour interval
            cCI = cCI - contint
        
        ######################
        # If MCC Is Possible #
        ######################
        else:
            cCIs, aids = [], []
            while nMins == 0 and nMaxs == 0 and cCI < np.nanmax(field): # keep increasing interval as long as only one minimum is detected
                #Increase contour interval
                cCI = cCI + contint
                
                # Define contiguous areas
                fieldCI = np.where(field < cCI, 1, 0)
                areas, nA = scipy.ndimage.measurements.label(fieldCI)
                
                # Test how many minima are within the area associated with the minimum of interest
                areaTest = np.where((areas == areas[rows[cid],cols[cid]]) & (fieldCenters != 0)) # Limit to minima within that area
                
                # Test how many maxima are within the area associated with the minimum of interest
                maxTest = np.where((areas == areas[rows[cid],cols[cid]]) & (field_max == 1), 1, 0) # Limit to maxima within that area           
                nMaxs = np.sum(maxTest) # Count the number of maxima identified
                
                # Record the area and the ids of the minima encircled for each contint
                locSub = [(areaTest[0][ls],areaTest[1][ls]) for ls in range(areaTest[0].shape[0])]
                idsSub = [i for i in ids if ((rows[i],cols[i]) in locSub) & (i != cid)]                
                cCIs.append(cCI)
                aids.append(idsSub)
                # Record the number of minima w/n the area that are outside the mccdist
                nMins = len(aids[-1]) - np.sum([i in icands for i in aids[-1]])
                
            # If there are possible secondaries within mccdist, evaluate MCC possibilities
            # Also, only check if its possible for the number of contour intervals to exceed mcctol
            if (len(aids) > 1) and ( len(aids[-1]) > 1 or (len(aids[-1]) == 1 and nMins == 0) ):
                # For each minimum in the last aids position before breaking, make a contour interval test,
                aids, cCIs = aids[:-1], cCIs[:-1]
                ## Starting with the last center to be added (fewest instances w/n the area)
                nCIs = [np.sum([i in ii for ii in aids]) for i in aids[-1]]
                breaker = 0
                
                while breaker == 0 and len(nCIs) > 0:
                    ai = aids[-1][np.where(np.array(nCIs) == np.min(nCIs))[0][0]] # Find id
                    ai0 = np.where(np.array([ai in i for i in aids]) == 1)[0][0] # First contour WITH secondary
                    
                    numC1 = (cCIs[ai0] - contint - vals[cid]) # The number of contour intervals WITHOUT secondaries
                    numC2 = (cCIs[-1] - vals[cid]) # The number of contour intervals WITH secondaries
                    
                    # If including secondaries substantially increases the number of contours involved...
                    if (numC1 / numC2) < mcctol:
                        for i in aids[-1]: # Add all of the other minima at this level as secondaries
                            cyclones[i].add_parent(rows[cid],cols[cid],cid)
                            fieldCenters2[rows[i],cols[i]] = 2
                            cyclones[cid].secondary.append(centers[i].id)
                            cyclones[i].type = 2
                            types[i] = 2 # And change the type to secondary so it will no longer be considered
                        
                        # Force the loop to end; all remaining minima must also be secondaries
                        breaker = 1
                        
                    else:
                        cCIs = cCIs[:ai0]
                        aids = aids[:ai0]
                        nCIs = [np.sum([i in ii for ii in aids]) for i in aids[-1]]
                
                # Once secondaries are accoutned for, re-establish the highest contour
                cCI = cCIs[-1]
            
            # Otherwise, ignore such possibilities
            else:
                cCI = cCI - contint
        
        #########################
        # Final Area Assignment #
        #########################
        # Assign final contiguous areas:
        fieldF = np.where(field < (cCI), 1, 0)
        areasF, nAF = scipy.ndimage.measurements.label(fieldF)
        
        # And identify the area associated with the minimum of interest:
        area = np.where((areasF == areasF[rows[cid],cols[cid]]) & (areasF != 0),1,0)
        if np.nansum(area) == 0: # If there's no area already,
            area[rows[cid],cols[cid]] = 1 # Give it an area of 1 to match the location of the center
        
        cycField = cycField + area # And update the system field
        
        # Then assign characteristics to the cyclone:
        cyclones[cid].type = 1
        cyclones[cid].p_edge = cCI
        cyclones[cid].area = np.nansum(area)
        
        # Also assign those characteristics to its secondaries:
        for cid_s in cyclones[cid].secondary:
            cyclones[cid_s].p_edge = cCI
            cyclones[cid_s].area = np.nansum(area)
        
        # When complete, change type of cyclone and re-set secondary candidates
        types[cid] = 1
        candids = np.where(types == 0)[0]
        #print "ID: " + str(cid) + ", Row: " + str(rows[cid]) + ", Col: " + str(cols[cid]) + \
        #        ", Area:" + str(np.nansum(area))

    return cycField, fieldCenters2, cyclones

'''###########################
Calculate Cyclone Associated Precipitation
###########################'''
def findCAP(cycfield,plsc,ptot,pMin=0.375,r=250000,cellsize=100000):
    '''Calculates the cyclone-associated precipitation for each cyclone in
    a cyclone field object for a particular time. Input precipitation fields 
    must have the same projection, grid cell size, and numebr of rows/columns
    as the cyclone field.\n
    
    Required inputs:\n
    cycfield = a cyclone field object with cyclone centers and areas already 
        calculated (using the findCenters and findAreas functions).
    plsc = large-scale precipitation field
    ptot = total precipitation field
    pMin = the minimum amount of large-scale precipitation required for 
        defining contiguous precipitation areas (in mm per day)
    r = radius defining  minimum area around which to search for precipitation
        (this is in addition to any area defined as part of the cyclone by the
        algorithm) -- must be same units as grid cell size (in m)
    cellsize = the grid cell size for all input arrays (in m).
    
    Returns a field of CAP, but also updates the precipitation values for each
    cyclone center in the cyclone field object.
    '''
    #############
    # PREP WORK #
    #############
    rc = np.ceil(r/cellsize*0.5)
    rc2 = np.ceil(r/cellsize)
    kern = np.where(np.isnan(circleKernel(r/cellsize)) == 1, 0, 1)
    
    # Eliminate Non-Finite values
    ptot, plsc = np.where(np.isfinite(ptot) == 1,ptot,0), np.where(np.isfinite(plsc) == 1,plsc,0)
    
    # Add edges to the rasters so that kernel never folds over
    cR, cC = cycfield.fieldAreas.shape[0], cycfield.fieldAreas.shape[1]
        
    plsc = np.hstack(( np.zeros( (cR+rc*2,rc) ) , \
    np.vstack(( np.zeros( (rc,cC) ), plsc ,np.zeros( (rc,cC) ) )), \
    np.zeros( (cR+rc*2,rc) ) ))
    
    ptot = np.hstack(( np.zeros( (cR+rc*2,rc) ) , \
    np.vstack(( np.zeros( (rc,cC) ), ptot ,np.zeros( (rc,cC) ) )), \
    np.zeros( (cR+rc*2,rc) ) ))
    
    fieldAreas = np.hstack(( np.zeros( (cR+rc*2,rc) ) , \
    np.vstack(( np.zeros( (rc,cC) ), cycfield.fieldAreas ,np.zeros( (rc,cC) ) )), \
    np.zeros( (cR+rc*2,rc) ) ))
    
    # Identify large-scale precipitation regions
    pMasked = np.where(plsc >= pMin, 1, 0)
    pAreas, nP = scipy.ndimage.measurements.label(pMasked)
    cAreas, nC = scipy.ndimage.measurements.label(fieldAreas)
    aIDs = [c.areaID for c in cycfield.cyclones]
    
    # Identify cyclone ids
    ids = np.array(range(len(cycfield.centerCount())))
    ids1 = ids[np.where(np.array(cycfield.centerType()) == 1 )]
    ids2 = ids[np.where(np.array(cycfield.centerType()) == 2 )]
    
    # Create empty lists/arrays
    cInt = [[] for p in range(nP+1)]# To store the cyc ID for each precip region
    cPrecip = np.zeros((len(ids))) # To store the total precip for each cyclone center
    cPrecipArea = np.zeros((len(ids))) # To store the precip area for each cyclone center
    
    ######################
    # FIND INTERSECTIONS #
    ######################
    for i in ids1: # For each PRIMARY center,
        # Identify corresponding area
        c = cycfield.cyclones[i]
        cArea = np.where(cAreas == aIDs[i],1,0) # Calc'd area
        y, x = c.y+rc, c.x+rc
        cArea[y-rc2:y+rc2+1,x-rc2:x+rc2+1] = \
        np.where(cArea[y-rc2:y+rc2+1,x-rc2:x+rc2+1] + kern != 0, 1, 0) # Add a radius-based area
        for ii in c.secondary: # Add any secondary radius-based areas
            cc = cycfield.cyclones[ii]
            yy, xx = cc.y+rc, cc.x+rc
            cArea[yy-rc2:yy+rc2+1,xx-rc2:xx+rc2+1] = \
            np.where(cArea[yy-rc2:yy+rc2+1,xx-rc2:xx+rc2+1] + kern != 0, 1, 0)
        
        # Find the intersecting precip areas
        pInt = np.unique(cArea*pAreas)[np.where(np.unique(cArea*pAreas) != 0)]
        
        # Assign cyc id to intersecting precip areas
        for pid in pInt:
            cInt[pid].append(i)
    
    # Identify unique intersecting precip areas
    pList = [p for p in range(nP+1) if len(cInt[p]) != 0]
    
    ####################
    # PARTITION PRECIP #
    ####################
    # For each intersecting precip area,
    for p in pList:
        # If it only intersects 1 center...
        if len(cInt[p]) == 1:
            # Assign all TOTAL precip to the cyclone
            pArea = np.where(pAreas == p,1,0)
            cPrecip[cInt[p][0]] += np.sum(ptot*pArea)
            cPrecipArea[cInt[p][0]] += np.sum(pArea)
        
        # If more than one cyclone center intersects, 
        else:
            # Assign each grid cell individually based on closest center
            plocs = np.where(pAreas == p) # Find grid cells for area            
            
            for i in range(len(plocs[0])):
                # Find the closest area
                aInt = [aIDs[a] for a in cInt[p]]
                ai = findNearestArea((plocs[0][i],plocs[1][i]),cAreas,[aIDs[a] for a in cInt[p]])
                ci = cInt[p][np.where(np.array(aInt) == ai)[0][0]]
                
                # Assign TOTAL precip to the cyclone
                cPrecip[ci] += ptot[plocs[0][i],plocs[1][i]]
                cPrecipArea[ci] += 1
    
    ##################
    # RECORD PRECIP #
    #################
    # Final assignment of precip to primary cyclones
    for i in ids1:
        cycfield.cyclones[i].precip = cPrecip[i]
        cycfield.cyclones[i].precipArea = cPrecipArea[i]
        
    for i in ids2:
        par = cycfield.cyclones[i].parent['id']
        cycfield.cyclones[i].precip = cPrecip[par]
        cycfield.cyclones[i].precipArea = cPrecipArea[par]
        
    # Return CAP field
    cap = ptot*np.in1d(pAreas,np.array(pList)).reshape(pAreas.shape)
    return cap[rc:-rc,rc:-rc]

'''###########################
Nullify Cyclone-Related Data in a Cyclone Center Track Instance (not Track-Related Data)
###########################'''
def nullifyCycloneTrackInstance(ctrack,time,ptid):
    '''This function operates on a cyclonetrack object. Given a particular 
    time, it will turn any row with the time in the main data frame 
    (ctrack.data) into a partially nullified row. Some values are left 
    unchanged (id, mcc, time, x, y, u, and v, and the event flags). Some values
    (area, centers, radius, type) will be set to 0. The cyclone-specifc values
    are set to np.nan.
    
    The reason for having this function is that sometimes it's desirable to
    track a center's position during a merge or split, but it's not appropriate
    to assign other characteristics.
    
    ctrack = a cyclonetrack object
    time = a time step (float) corresponding to a row in ctrack (usu. in days)
    ptid = the track into which it's merging
    '''
    ctrack.data.area[ctrack.data.time == time] = 0
    ctrack.data.centers[ctrack.data.time == time] = 0
    ctrack.data.radius[ctrack.data.time == time] = 0
    ctrack.data.type[ctrack.data.time == time] = 0
    
    ctrack.data.DpDr[ctrack.data.time == time] = np.nan
    ctrack.data.DpDt[ctrack.data.time == time] = np.nan
    ctrack.data.DsqP[ctrack.data.time == time] = np.nan
    ctrack.data.depth[ctrack.data.time == time] = np.nan
    ctrack.data.p_cent[ctrack.data.time == time] = np.nan
    ctrack.data.p_edge[ctrack.data.time == time] = np.nan
    
    ctrack.data.ptid[ctrack.data.time == time] = ptid
    
    return ctrack

'''###########################
Initialize Cyclone Center Tracks
###########################'''
def startTracks(cyclones):
    '''This function initializes a set of cyclone tracks from a common start date
    when gien a list of cyclone objects. It it a necessary first step to tracking
    cyclones. Returns a list of cyclonetrack objects and an update list of cyclone
    objects with track ids.
    
    cyclones = a list of objects of the class minimum.
    '''
    ct = []
    for c,cyc in enumerate(cyclones):
        # Define Etype
        if cyc.type == 2:
            Etype = 1
            ptid = cyc.parent["id"]
        else:
            Etype = 3
            ptid = c
        
        # Create Track
        ct.append(cyclonetrack(cyc,c,Etype,ptid))
        
        # Assign IDs to cyclone objects
        cyc.tid = c
    
    return ct, cyclones

'''###########################
Track Cyclone Centers Between Two Time Steps
###########################'''
def trackCyclones(cfa,cfb,ctr,cellsize,maxspeed,red):
    # First make copies of the cyclonetrack and cyclonefield inputs to ensure that nothing is overwritten
    ct = copy.deepcopy(ctr)
    cf1 = copy.deepcopy(cfa)
    cf2 = copy.deepcopy(cfb)
    
    # Calculate the maximum number of cells a cyclone can move
    time1 = cf1.time
    time2 = cf2.time
    tmstep = (time2 - time1)*24 # Assumes time is in days
    maxdist = maxspeed*1000/cellsize*tmstep # Assumes speed in km/h and cellsize is m
    
    # Create helper lists:
    y2s = cf2.y()
    x2s = cf2.x()
    mc2 = [[] for i in y2s] # stores ids of cf1 centers that map to each cf2 center
    sc2 = [[] for i in y2s] # stores ids of cf1 centers that were within distance to map but chose a closer cf2 center
    
    lsp1s = [] # stores the lifespan of the track of each center in cf1
    p1s = cf1.p_cent()
    
    ################################
    # PART 1. MAIN CENTER TRACKING #
    ################################
    # Loop through each center in cf1 to find matches in cf2
    for c,cyc in enumerate(cf1.cyclones):
        cyct = ct[cyc.tid] # Link the cyclone instance to its track
        lsp1s.append(cyct.lifespan()) # store the lifespan up to now
        
        # Create a first guess for the next location of the cyclone:
        if len(cyct.data) == 1: # If cf1 represents the genesis event, first guess is no movement
            yq = cyc.y
            xq = cyc.x
        elif len(cyct.data) > 1: # If the cyclone has moved in the past, a linear projection of its movement is the best guess
            yq = cyc.y + float(red*cyct.data[cyct.data.time == cyc.time].Dy)
            xq = cyc.x + float(red*cyct.data[cyct.data.time == cyc.time].Dx)
        
        # Test every point in cf2 to see if it's within distance d of both (yq,xq) AND (y,x)
        pdqs = [((y2s[p] - yq)**2 + (x2s[p] - xq)**2)**0.5 for p in range(len(y2s))]
        pds = [((y2s[p] - cyc.y)**2 + (x2s[p] - cyc.x)**2)**0.5 for p in range(len(y2s))]
        
        pds_n = sum([((pds[p] <= maxdist) and (pdqs[p] <= maxdist)) for p in range(len(pdqs))])
        
        ##############################
        # PART 1.1. CENTER TRACKING #
        # If one or more points fit the criterion, select the nearest neighbor to the projection
        if pds_n > 0:
            # Take the one closest to the projected point (yq xq)
            c2 = np.where(np.array(pdqs) == min(pdqs))[0][0]
            cyct.addInstance(cf2.cyclones[c2])
            cf2.cyclones[c2].tid = cyc.tid # link the two cyclone centers with a track id
            mc2[c2].append(c) # append cf1 id to the merge list
            
            # Remove that cf2 cyclone from consideration
            pds[c2] = maxdist+1
            pds_n = pds_n - 1
            
            # Add the cf1 center to the splits list for the remaining cf2 centers that fit the dist criteria
            while pds_n > 0:
                s2 = np.where(np.array(pds) == min(pds))[0][0]
                sc2[s2].append(c)
                pds[s2] = maxdist+1
                pds_n = pds_n -1
        
        ##########################
        # PART 1.2. CENTER LYSIS #
        # If no point fits the criterion, then the cyclone experienced lysis
        elif pds_n == 0:
            # Add a nullified instance for time2 at the same location as time 1
            cyc_copy = copy.deepcopy(cyc)
            cyc_copy.time = time2
            cyct.addInstance(cyc_copy)
            cyct = nullifyCycloneTrackInstance(cyct,time2,cyc.tid)
            # Add a lysis event
            cyct.addEvent(cyc,time2,"ly",3)
    
    ##################################
    # PART 2. CENTER MERGES & SPLITS #
    ##################################
    # Check for center merges (mc) and center splits (sc)
    for id2 in range(len(mc2)):
        # If there is only one entry in the merge list, then it's a simple tracking; nothing else required
        
        ##########################
        # PART 2.1. CENTER MERGE #
        # If multiple cyclones from cf1 match the same cyclone from cf2 -> CENTER MERGE
        if len(mc2[id2]) > 1:
            ### DETERMINE PRIMARY TRACK ###
            lsp1_mc2 = [lsp1s[i] for i in mc2[id2]]
            p1s_mc2 = [p1s[i] for i in mc2[id2]]
            tid_mc2 = [cf1.cyclones[i].tid for i in mc2[id2]]
            
            # Which has the longer lifetime?
            if len(np.where(np.array(lsp1_mc2) == max(lsp1_mc2))[0]) == 1: # if one center has a longer lifespan than the others 
                id1 = mc2[id2][np.where(np.array(lsp1_mc2) == max(lsp1_mc2))[0][0]] # find id of the max cf1 lifespan
            # Which center is deepest?
            else: # if multiple centers have the same lifespan, take the deepest
                id1 = mc2[id2][np.where(np.array(p1s_mc2) == min(p1s_mc2))[0][0]] # find id of the max cf1 lifespan
                # Note that if two cyclones have the same depth and lifespan, the first by id is automatically taken.
            
            # Check if ptid of id1 center is the tid of another merge candidate
            ptid1 = int(ct[cf1.cyclones[id1].tid].data.ptid[ct[cf1.cyclones[id1].tid].data.time == cf1.time])
            ptid_test = [ptid1 == i and i != cf1.cyclones[id1].tid for i in tid_mc2]
            if sum(ptid_test) > 0: # if the ptid is the tid of one of the other candidates, merge into the parent instead
                id1 = int(ct[ptid1].data.id[ct[ptid1].data.time == time1])
            
            # Assign the primary track tid
            cf2.cyclones[id2].tid = cf1.cyclones[id1].tid
            
            ### ASSIGN MERGE EVENTS ###
            for c1ida in mc2[id2]:
                for c1idb in mc2[id2]:
                    if c1ida != c1idb: # Add a merge for all pairs (order matters) except for a track with itself
                        
                        # If the two centers shared the same parent in time 1 (they were an mcc), it's just a center merge
                        if cf1.cyclones[c1ida].parent["id"] == cf1.cyclones[c1idb].parent["id"]:
                            ct[cf1.cyclones[c1ida].tid].addEvent(cf2.cyclones[id2],time2,"mg",1,otid=cf1.cyclones[c1idb].tid)
                            
                            if c1ida != id1: # for non-primary track(s)
                                # Add a lysis event
                                ct[cf1.cyclones[c1ida].tid].addEvent(cf2.cyclones[id2],time2,"ly",1)
                                # Set stats to zero for current time since I really just want the (y,x) location
                                ct[cf1.cyclones[c1ida].tid] = nullifyCycloneTrackInstance(ct[cf1.cyclones[c1ida].tid],time2,cf1.cyclones[id1].tid)
                        
                        # If they had different parents, it's both a center merge and an area merge
                        else:
                            ct[cf1.cyclones[c1ida].tid].addEvent(cf2.cyclones[id2],time2,"mg",3,otid=cf1.cyclones[c1idb].tid)
                            
                            if c1ida != id1: # for non-primary track(s)
                                # Add a lysis event
                                ct[cf1.cyclones[c1ida].tid].addEvent(cf2.cyclones[id2],time2,"ly",3)
                                # Set stats to zero for current time since I really just want the (y,x) location
                                ct[cf1.cyclones[c1ida].tid] = nullifyCycloneTrackInstance(ct[cf1.cyclones[c1ida].tid],time2,cf1.cyclones[id1].tid)
        
        ###########################
        # PART 2.2. CENTER SPLITS #
        # If no cyclones from cf1 match a particular cf2 cyclone, it's either a center split or a pure genesis event
        elif len(mc2[id2]) == 0 and len(sc2[id2]) > 0: # if there's one or more centers that could have tracked there -> SPLIT
            ### DETERMINE SOURCE CENTER ###
            # Make the split point the closest center of the candidate(s)
            dist_sc2 = [((cf1.cyclones[i].y - cf2.cyclones[id2].y)**2 + (cf1.cyclones[i].x - cf2.cyclones[id2].x)**2)**0.5 for i in sc2[id2]]
            id1= sc2[id2][np.where(np.array(dist_sc2) == min(dist_sc2))[0][0]]
            
            # Start the new track a time step earlier
            cf2.cyclones[id2].tid = len(ct) # Assign a new track id to the cf2 center
            ct.append(cyclonetrack(cf1.cyclones[id1],tid=cf2.cyclones[id2].tid)) # Make a new track with that id
            # Set stats to zero since I really just want the (y,x) location
            ct[cf2.cyclones[id2].tid] = nullifyCycloneTrackInstance(ct[cf2.cyclones[id2].tid],time1,cf1.cyclones[id1].tid)
            
            # Add an instance for the current time step (cf2)
            ct[cf2.cyclones[id2].tid].addInstance(cf2.cyclones[id2])
            # Adjust when the genesis is recorded
            ct[cf2.cyclones[id2].tid].events.time = time2
            ct[cf2.cyclones[id2].tid].data.Ege[ct[cf2.cyclones[id2].tid].data.time == time1] = 0
            ct[cf2.cyclones[id2].tid].data.Ege[ct[cf2.cyclones[id2].tid].data.time == time2] = 3
            
            ### ASSIGN SPLIT EVENTS ###
            # Find the id of the other branch of the split in the time of cf2
            id2_1 = int(ct[cf1.cyclones[id1].tid].data.id[ct[cf1.cyclones[id1].tid].data.time == time2])
            
            # Add a split event to the primary track and the new track
            if cf2.cyclones[id2_1].parent["id"] == cf2.cyclones[id2].parent["id"]:
                # If the two centers have the same parent in time 2, it was just a center split
                ct[cf1.cyclones[id1].tid].addEvent(cf2.cyclones[id2_1],time2,"sp",1,otid=cf2.cyclones[id2].tid)
                ct[cf2.cyclones[id2].tid].addEvent(cf2.cyclones[id2],time2,"sp",1,otid=cf1.cyclones[id1].tid)
                ct[cf2.cyclones[id2].tid].data.Ege[ct[cf2.cyclones[id2].tid].data.time == time2] = 1
                ct[cf2.cyclones[id2].tid].events.Etype[(ct[cf2.cyclones[id2].tid].events.time == time2) & \
                    (ct[cf2.cyclones[id2].tid].events.event == "ge")] = 1
            else:
                # If they don't, then it was also an area split
                ct[cf1.cyclones[id1].tid].addEvent(cf2.cyclones[id2_1],time2,"sp",3,otid=cf2.cyclones[id2].tid)
                ct[cf2.cyclones[id2].tid].addEvent(cf2.cyclones[id2],time2,"sp",3,otid=cf1.cyclones[id1].tid)
                # Amend the parent track id to be it's own track now that it's split
                ct[cf2.cyclones[id2].tid].data.ptid[ct[cf2.cyclones[id2].tid].data.time == time2] = cf2.cyclones[id2].tid
                ct[cf2.cyclones[id2].tid].ptid = cf2.cyclones[id2].tid
                # Note that this might be overwritten yet again if this center has an area merge with another center
                
        ############################
        # PART 2.3. CENTER GENESIS #
        elif len(mc2[id2]) == 0 and len(sc2[id2]) == 0: # if there's no center that could have tracked here -> GENESIS
            cf2.cyclones[id2].tid = len(ct) # Assign the track id to the cf2 center
            
            if cf2.cyclones[id2].centerCount() == 1: # If it's a scc, it's both an area and center genesis
                ct.append(cyclonetrack(cf2.cyclones[id2],cf2.cyclones[id2].tid,3,cf2.cyclones[id2].tid)) # Make a new track
            
            else: # If it's a mcc, things are more complicated
                # Find center ids of mcc centers
                mcc_ids = cf2.cyclones[cf2.cyclones[id2].parent["id"]].secondary
                mcc_ids.append(cf2.cyclones[id2].parent["id"])
                # Find which have prior tracks
                prior = [( (len(mc2[mccid]) > 0) or (len(sc2[mccid]) > 0) ) for mccid in mcc_ids]
                
                if (sum(prior) > 0) or (cf2.cyclones[id2].type == 2): 
                    # If it's a secondary center or if one of the centers in the mcc already has a track, 
                    ### then it's only a center genesis
                    
                    ct.append(cyclonetrack(cf2.cyclones[id2],cf2.cyclones[id2].tid,1)) # Make a new track
                else:
                    # If all centers in the mcc are new and this is the primary center, 
                    ### then it's both center and area genesis
                    ct.append(cyclonetrack(cf2.cyclones[id2],cf2.cyclones[id2].tid,3,cf2.cyclones[id2].tid)) # Make a new track
    
    #######################################################
    # PART 3. AREA-ONLY SPLITS & MERGES and SPECIAL LYSIS #
    #######################################################
    ########################
    # PART 3.1. AREA MERGE #
    for cy2 in cf2.cyclones:
        # If cy2 is part of an mcc
        if int(ct[cy2.tid].data.centers[ct[cy2.tid].data.time == time2]) != 1:
            # Find the id of cy2 in time of cf1:
            if len(ct[cy2.tid].data[(ct[cy2.tid].data.time == time1) & (ct[cy2.tid].data.type != 0)]) == 0:
                continue # gensis event, no center to compare
            else:
                cy2_id1 = int(ct[cy2.tid].data.id[ct[cy2.tid].data.time == time1])
                
                # Find track ids for each center in the mcc that isn't cy2
                ma_tids = [cy.tid for cy in cf2.cyclones if ((cy.parent["id"] == cy2.parent["id"]) and (cy.id != cy2.id))]
                
                for ti in  ma_tids: # for each track id
                    if len(ct[ti].data.id[ct[ti].data.time == time1]) == 0:
                        continue # genesis event, no center to compare
                    else: 
                        ma_id1 = int(ct[ti].data.id[ct[ti].data.time == time1] )# Find the center id for cf1
                        ma_parent_id = cf1.cyclones[ma_id1].parent["id"] # Find parent id at time of cf1
                        if ma_parent_id != cf1.cyclones[cy2_id1].parent["id"]: # If the two are not the same
                            # Assign an area merge to cy2's track
                            ct[cy2.tid].addEvent(cy2,time2,"mg",2,otid=ti)
    
    for cy1 in cf1.cyclones: 
        # If cy1 was part of an mcc
        if int(ct[cy1.tid].data.centers[ct[cy1.tid].data.time == time1]) != 1:
            # What was the primary track in time 1?
            ptid1 = int(ct[cy1.tid].data.ptid[ct[cy1.tid].data.time == time1])
            
            ###########################
            # PART 3.2. SPECIAL LYSIS #
            if len(ct[cy1.tid].data[(ct[cy1.tid].data.time == time2) & (ct[cy1.tid].data.type != 0)]) == 0:
                # If cy1 was the system center...
                if cy1.tid == ptid1:
                    # Did any other centers survive?
                    mcc_ids = copy.deepcopy(cf1.cyclones[cy1.parent["id"]].secondary)
                    mcc_ids.append(cy1.parent["id"])
                    survtest = [len(ct[cf1.cyclones[i].tid].data[(ct[cf1.cyclones[i].tid].data.time == time2) \
                        & (ct[cf1.cyclones[i].tid].data.type != 0)]) > 0 for i in mcc_ids]
                    
                    # If another center did survive, then there is cyclone re-genesis
                    if sum(survtest) > 0:
                        # Find the deepest surviving center
                        mcc_ids_surv = [mcc_ids[i] for i in range(len(mcc_ids)) if survtest[i] == 1]
                        mcc_ids_surv_p = [float(ct[cf1.cyclones[i].tid].data.p_cent[ct[cf1.cyclones[i].tid].data.time == time2]) \
                                            for i in mcc_ids_surv]
                        reid1 = mcc_ids_surv[np.where(np.array(mcc_ids_surv_p) == min(mcc_ids_surv_p))[0][0]]
                        # Find the cf2 version of that center
                        reid2 = int(ct[cf1.cyclones[reid1].tid].data.id[ct[cf1.cyclones[reid1].tid].data.time == time2])
                        # Assign it an area regenesis event & make it the primary track
                        ct[cf2.cyclones[reid2].tid].addEvent(cf2.cyclones[reid2],time2,"ge",2)
                        ct[cf2.cyclones[reid2].tid].addEvent(cf2.cyclones[reid2],time2,"rg",2,otid=cy1.tid)
                        ct[cf2.cyclones[reid2].tid].data.ptid[ct[cf2.cyclones[reid2].tid].data.time == time2] = cf2.cyclones[reid2].tid
                        ct[cf2.cyclones[reid2].tid].ptid = cf2.cyclones[reid2].tid
                    # If no other centers survived, then it's just a normal type 3 lysis --> no change
                    else:
                        continue
                
                else: # cy1 was NOT the system center...
                    # Then it's just a secondary lysis event --> change event type to 1
                    ct[cy1.tid].data.Ely[ct[cy1.tid].data.time == time2] = 1
                    ct[cy1.tid].events.Etype[(ct[cy1.tid].events.time == time2) & (ct[cy1.tid].events.event == "ly")] = 1
            
            ########################
            # PART 3.3. AREA SPLIT #
            else: 
                # Otherwise, find the id of cy1 in time of cf2:
                c1_id2 = int(ct[cy1.tid].data.id[ct[cy1.tid].data.time == time2])
                
                # Find track ids for each center in the mcc that isn't cy1
                sa_tids = [cy.tid for cy in cf1.cyclones if ((cy.parent["id"] == cy1.parent["id"]) and (cy.id != cy1.id))]
                
                for ti in  sa_tids: # for each track id
                    if len(ct[ti].data.id[(ct[ti].data.time == time2) & (ct[ti].data.type != 0)]) == 0:
                        continue # This was a lysis event, no center to compare
                    
                    else: 
                        sa_id2 = int(ct[ti].data.id[ct[ti].data.time == time2] )# Find the center id for cf2
                        sa_parent_id = cf2.cyclones[sa_id2].parent["id"] # Find parent id at time of cf2
                        if sa_parent_id != cf2.cyclones[c1_id2].parent["id"]: # If the two are not the same
                            # Assign an area split to cy1's track
                            ct[cy1.tid].addEvent(cf2.cyclones[c1_id2],time2,"sp",2,otid=ti)
                            if ptid1 != cy1.tid: # If cy1 was NOT the system center
                                # Then change the track id to its own now that it has split
                                ct[cy1.tid].data.ptid[ct[cy1.tid].data.time == time2] = cy1.tid
                                ct[cy1.tid].ptid = cy1.tid
                                # And add an area genesis event
                                ct[cy1.tid].addEvent(cf2.cyclones[c1_id2],time2,"ge",2)
    
    ##########################################################
    # PART 4. UPDATE ptid OF MULTI-CENTER CYCLONES IN TIME 2 #  
    ##########################################################    
    for cy2 in cf2.cyclones:
        # Identify MCCs by the primary center
        if int(ct[cy2.tid].data.centers[ct[cy2.tid].data.time == time2]) > 1:
            # For each mcc, identify the cycs and their tids for each center
            mcy2s = [cy for cy in cf2.cyclones if cy.parent["id"] == cy2.parent["id"]]
            mtids = [cy.tid for cy in cf2.cyclones if cy.parent["id"] == cy2.parent["id"]]
            
            # Grab the lifespan for each track and the central pressure at time 2:
            mp2s = [cy.p_cent for cy in mcy2s]
            mlsps = [ct[ti].lifespan() for ti in mtids]
            
            # Which tracks also existed in cf1? (excludes split genesis markers)
            pr_mtids = [ti for ti in mtids if len(ct[ti].data.type[ct[ti].data.time == time1]) > 0 \
                    and int(ct[ti].data.type[ct[ti].data.time == time1]) > 0]
            
            # If none of the tracks existed in cf1 time, 
            if len(pr_mtids) == 0:
                # then choose the deepest in cf2 time as ptid
                ptid2 = mtids[np.where(np.array(mp2s) == min(mp2s))[0][0]]
            
            # If only one track existed in cf1 time,
            elif len(pr_mtids) == 1:
                # Use it's tid as the ptid for everything
                ptid2 = int(ct[pr_mtids[0]].tid)
            
            # If more than one track existed in cf1 time,
            else:
                # identify which track is longest
                if len(np.where(np.array(mlsps) == max(mlsps))[0]) == 1: # if one center has a longer lifespan than the others 
                    tid2 = mtids[np.where(np.array(mlsps) == max(mlsps))[0][0]] # find tid of the max in lifespan
                # Which center is lowest pressure?
                else: # if multiple centers have the same lifespan, take the lowest pressure
                    mp2s_pr = [float(ct[ti].data.p_cent[ct[ti].data.time == time2]) for ti in pr_mtids]
                    tid2 = mtids[np.where(np.array(mp2s) == min(mp2s_pr))[0][0]] # find id of the max cf2 lifespan
                # Note that if two cyclones have the same depth and lifespan, the first by id is automatically taken.
                try:
                    ptid2 = int(ct[tid2].data.ptid[ct[tid2].data.time == time2]) # identify its ptid as ptid for system
                except: 
                    ptid2 = tid2 # if it has no ptid, then assign its tid as ptid
            
            # Loop through all centers in the mcc
            for mtid in mtids:
                # Assign the ptid to all centers in the mcc
                ct[mtid].data.ptid[ct[mtid].data.time == time2] = ptid2
                ct[mtid].ptid = ptid2
                
                # Add area lysis events to any non-ptid tracks that experienced an area merge
                if ct[mtid].tid != ptid2 and int(ct[mtid].data.Emg[ct[mtid].data.time == time2]) == 2:
                    ct[mtid].addEvent(cf2.cyclones[int(ct[mtid].data.id[ct[mtid].data.time == time2])],time2,"ly",2,otid=ptid2)
        
        # For center-only lysis events, set the final ptid to match the last observed ptid
        if int(ct[cy2.tid].data.Ely[ct[cy2.tid].data.time == time2]) == 1:
            ct[cy2.tid].data.ptid[ct[cy2.tid].data.time == time2] = int(ct[cy2.tid].data.ptid[ct[cy2.tid].data.time == time1])
            ct[cy2.tid].ptid = int(ct[cy2.tid].data.ptid[ct[cy2.tid].data.time == time1])
    
    if len(cf2.tid()) != len(list(set(cf2.tid()))):
        raise Exception("Number of centers in cf2 does not match the number of \
        tracks assigned. Multiple centers may have been assigned to the same track.")
    
    return ct, cf2 # Return an updated cyclonetrack object (corresponds to ctr)
    ### and cyclonefield object for time 2 (corresponds to cfb)

'''###########################
Split Tracks into Active and Inactive
###########################'''
def splitActiveTracks(ct,cf):
    '''Given a list of cyclone tracks, this function creates two new lists: one
    with all inactive tracks (tracks that have already experienced lysis) and
    one with all active tracks (tracks that have not experienced lysis). Next,
    the track ids (and parent track ids) are reset for all active tracks and 
    the related cyclone field.
    
    ct = list of cyclone tracks
    cf = cyclone field object
    
    Returns: ([active tracks], [inactive tracks])
    (The cyclone field object is mutable and automatically edited.)
    '''
    ct_inactive, ct_active, tid_active = [], [], [] # Create empty lists
    # Sort tracks into active and inactive
    for track in ct:
        if track != 0 and ( (1 in list(track.data.Ely)) or (3 in list(track.data.Ely)) ):
            ct_inactive.append(track)
        else:
            ct_active.append(track)
            tid_active.append(track.tid)
    
    # Reformat tids for active tracks
    tid_activeA = np.array(tid_active)
    for tr in range(len(ct_active)):
        ct_active[tr].tid = tr
        ct_active[tr].ftid = tid_active[tr]
        if ct_active[tr].ptid in tid_active:
            ct_active[tr].ptid = int(np.where(tid_activeA == ct_active[tr].ptid)[0][0])
    
    for cyctr in cf.cyclones:
        if cyctr.tid in tid_active:
            cyctr.tid = int(np.where(tid_activeA == cyctr.tid)[0][0])
        else:
            cyctr.tid = np.nan # These are inactive tracks, so they don't matter anymore.
    
    return ct_active, ct_inactive

'''###########################
Realign Track IDs for Cyclone Tracks and Cyclone Field
###########################'''
def realignPriorTID(ct,cf1):
    '''This is a very specific function used to realign the track ids for a 
    list of active cyclone tracks and a cyclone field object. The cyclone field 
    object must correspond to the final time step recorded in the track 
    objects. Additionally, the cyclone field object must have the same number 
    of centers as the track list has tracks.
    
    Inputs:
    ct = a list of cyclone track objects
    cf1 = a cyclone field object corresponding to the final time recorded in ct
    
    Output:
    no return, but the tids in cf1 are modified.
    '''
    
    if len(ct) != len(cf1.cyclones):
        raise Exception("Number of tracks in ct doesn't equal the number of centers in cf1.")
    if (cf1.time != list(ct[0].data.time)[-1]) or (cf1.time != list(ct[-1].data.time)[-1]):
        raise Exception("The time for cf1 is not the final time recorded for ct.")
    else:
        for tid in range(len(ct)): # For each track
            cid = int(ct[tid].data.id[ct[tid].data.time == cf1.time]) # Find the center id for the final time step
            cf1.cyclones[cid].tid = tid # Reset the tid for the corresponding center in cf1

'''###########################
Convert Cyclone Center Tracks to Cyclone System Tracks
###########################'''
def cTrack2sTrack(ct,cs0=[],dateref=[1900,1,1,0,0,0],rg=0):
    '''Cyclone tracking using the trackCyclones function is performed on 
    cyclone centers, including secondary centers.  But since the primary
    center of a cyclone at timestep 1 might not share the same track as the 
    primary center of the same cyclone at timestep 2, it may sometimes be 
    desirable to follow only that primary center and so receive a system-based
    view of tracks. This function performs such a conversion post-tracking,
    ensuring that the track of a system always follows the primary center of
    that system. All other tracks are then built around that idea.
    
    ct = a list of cyclonetrack objects
    cs0 = a list of systemtrack objects from the prior month (defaults to an
    empty list, meaning there is no prior month)
    dateref = the reference date to use for determining the month
    rg = boolean; if set to 1, then a system track will be extended if one of 
    the secondary centers continues (a regenesis event); if set to 0, the re-
    genesis will be ignored and the surviving secondary will be treated as a 
    new system. Defaults to 0.
    
    Two lists of cyclone system objects are returned: one for the current month 
    (ct -> cs) and an updated one for the prior month (cs0 -> cs0)
    '''
    # Define month
    mt = timeAdd(dateref,[0,0,list(ct[0].data.time)[-1],0,0,0],lys=1)
    mt[2], mt[3], mt[4], mt[5] = 1, 0, 0, 0
    days = daysBetweenDates(dateref,mt,lys=1)
    
    cs = []
    
    # STEP 1: LIMIT TO PTID (primary tracks)
    for t in ct: # Loop through each original tracks
        ptidtest = [t.tid != p for p in t.data.ptid[(t.data.type != 0) & (t.data.time >= days)]] + \
            [t.ftid != p for p in t.data.ptid[(t.data.type != 0) & (t.data.time < days)]] # Current Month \ Prior Month
        if sum(ptidtest) == 0: # If ptid always equals tid
            cs.append(systemtrack(t.data,t.events,t.tid,len(cs),t.ftid)) # Append to system track list
        
        # If ptid is never equal to tid, the track is always secondary, so ignore it
        elif sum(ptidtest) < len(t.data[t.data.type != 0]): # Otherwise...
            # Start empty data frames for data and events:
            data = pd.DataFrame()
            events = pd.DataFrame()
            # Observe each time...
            for r in t.data.time:
                # If the track is indepedent at this time step:
                if ( (t.tid == int(t.data.ptid[t.data.time == r])) or \
                    (t.ftid == int(t.data.ptid[t.data.time == r])) ) and \
                    int(t.data.type[t.data.time == r]) != 0:
                    # Append the row to the open system
                    data = data.append(t.data[t.data.time == r])
                    events = events.append(t.events[t.events.time == r])
                
                elif len(data) > 0:
                    # Append the row to the open system
                    data = data.append(t.data[t.data.time == r])
                    events = events.append(t.events[t.events.time == r])
                    # Close the system by adding it to the cs list
                    cs.append(systemtrack(data,events,t.tid,len(cs),t.ftid))
                    # Nullify the final step 
                    nullifyCycloneTrackInstance(cs[-1],r,data.ptid[data.time == r])
                    # Create a new open system:
                    data = pd.DataFrame()
                    events = pd.DataFrame()
                    
                else: # Do nothing otherwise
                    continue
            
            # After last is reached, end the open system if it has any rows
            if len(data) > 0:
                # Add any lysis events if they exist
                events = events.append(t.events[t.events.time == r+(t.data.time[1] - t.data.time[0])])
                # Append to cs list
                cs.append(systemtrack(data,events,t.tid,len(cs),t.ftid))
    
    # STEP 2: COMBINE REGENESIS CASES  
    cs = np.array(cs)
    if rg == 1:
        sys_tids = np.array([ccc.tid for ccc in cs])
        
        # CASE 1: Dead Track in Prior Month, Regenerated Track in This Month       
        # Identify the track id of regenerated tracks that died in prior month
        rg_otids, rg_tids, dels = [], [], []
        for t in cs:
            rgs = np.sum(t.events.event[t.events.time < days] == "rg")
            if rgs > 0:
                rg_tids.append(t.tid)
                rg_otids.append(int(t.events[t.events.event == "rg"].otid))
        
        otids = np.array([aa.tid for aa in cs0])
        for o in range(len(rg_otids)): # For each dead track
            # Note the position of the dead track object
            dels.append(np.where(otids == rg_otids[o])[0][-1])
            
            # Extract the dead track objects
            tDead = cs0[dels[o]] # Def of regenesis requires that primary track has experience type 3 lysis
            
            # Extract the regenerated track object
            sid_cands = np.where(sys_tids == rg_tids[o])[0] # Candidate track objects
            sid_rgcode = np.array([ np.sum(list(cs[sidc].data.Erg == 2)) > 0 and \
                (list(cs[sidc].data.time[cs[sidc].data.Erg == 2])[0] == list(tDead.data.time)[-1]) \
                 for sidc in sid_cands ]) # Does this track have a regeneration?
            sid = sid_cands[np.where(sid_rgcode == 1)[0][0]] # sid of the regenerated track
            tRegen = cs[sid]
            
            # Splice together with the regenerated track
            cs[sid].data = tDead.data[:-1].append(tRegen.data[tRegen.data.time >= list(tDead.data.time)[-1]],ignore_index=1)
            cs[sid].events = tDead.events[:-1].append(tRegen.events[(tRegen.events.time >= list(tDead.data.time)[-1])],ignore_index=1)
            cs[sid].data.Ely[cs[sid].data.Erg > 0] = 0
            cs[sid].data.Ege[cs[sid].data.Erg > 0] = 0
        
        # CLEAN UP
        # Remove the dead tracks from the current month
        cs0 = np.delete(cs0,dels)
        
        # CASE 2: Dead Track and Regenerated Track in Same Month
        # Identify the track id of regenerated tracks that died this month        
        rg_otids, rg_tids, dels = [], [], []
        for t in cs:
            rgs = np.sum(t.events.event[t.events.time >= days] == "rg")
            if rgs > 0:
                rg_tids.append(t.tid)
                rg_otids.append(int(t.events[t.events.event == "rg"].otid))
        
        for o in range(len(rg_otids)): # For each dead track
            # Note the position of the dead track object
            dels.append(np.where(sys_tids == rg_otids[o])[0][-1])
            
            # Extract the dead track object
            tDead = cs[dels[o]] # Def of regenesis requires that primary track has experienced type 3 lysis
            
            # Extract the regenerated track object
            sid_cands = np.where(sys_tids == rg_tids[o])[0] # Candidate track objects
            # Does this track have a regeneration? And does it begin when the dead track ends?
            sid_rgcode = np.array([("rg" in list(cs[sidc].events.event) ) and \
                ( list(cs[sidc].data.time)[0] == list(tDead.events.time[tDead.events.event == "ly"])[-1] ) \
                for sidc in sid_cands])
            sid = sid_cands[np.where(sid_rgcode == 1)[0][0]] # sid of the regenerated track
            tRegen = cs[sid]
            
            # Splice together with the regenerated track
            cs[sid].data = tDead.data[:-1].append(tRegen.data,ignore_index=1)
            cs[sid].events = tDead.events[:-1].append(tRegen.events[(tRegen.events.event != "ge") | \
                (tRegen.events.time > list(tRegen.events.time)[0])],ignore_index=1)
            cs[sid].data.Ely[cs[sid].data.Erg > 0] = 0
            cs[sid].data.Ege[cs[sid].data.Erg > 0] = 0
        
        # CLEAN UP
        # Remove the dead tracks from the current month
        cs = np.delete(cs,dels)
        
        # Re-format SIDs
        for c in range(len(cs)):
            cs[c].sid = c
    
    return list(cs), list(cs0)

'''###########################
Write a Numpy Array to File Using a Gdal Object as Reference
###########################'''
def writeNumpy_gdalObj(npArrays,outName,gdalObj,dtype=gdal.GDT_Byte):
    '''Write a numpy array or list of arrays to a raster file using a gdal object for geographic information.
    If a list of arrays is provided, each array will be a band in the output.
    
    npArrays = The numpy array or list of arrays to write to disk.  All arrays must have the same dimensions.\n
    outName = The name of the output (string)\n
    gdalObj = An object of osgeo.gdal.Dataset class\n
    dtype = The data type for each cell; 8-bit by default (0 to 255)
    '''
    # If a single array, convert to 1-element list:
    if str(type(npArrays)) != "<type 'list'>":
        npArrays = [npArrays]
        
    # Convert any non-finite values to -99:
    for i in range(len(npArrays)): # for each band...
        npArrays[i] = np.where(np.isfinite(npArrays[i]) == 0, -99, npArrays[i])
    
    # Create and register driver:
    driver = gdalObj.GetDriver()
    driver.Register()
    
    # Create file:
    outFile = driver.Create(outName,npArrays[0].shape[1],npArrays[0].shape[0],len(npArrays),dtype) # Create file
    for i in range(len(npArrays)): # for each band...
        outFile.GetRasterBand(i+1).WriteArray(npArrays[i],0,0) # Write array to file
        outFile.GetRasterBand(i+1).ComputeStatistics(False) # Compute stats for display purposes
    outFile.SetGeoTransform(gdalObj.GetGeoTransform()) # Set geotransform (those six needed values)
    outFile.SetProjection(gdalObj.GetProjection())  # Set projection
    
    outFile = None
    
    return

'''###########################
Pickle and Unpickle Data
###########################'''
def pickle(obj,filepath,protocol=-1):
    '''Pickles an object using the cPickle module, using the highest protocol
    by default.
    
    obj = The object to be pickled
    filepath = The path and file name for the output
    protocol = The pickling protocol to use; highest by default
    '''
    output = open(filepath, 'wb')
    cPickle.dump(obj,output,protocol=protocol)
    output.close()
    
    return

def unpickle(filepath):
    '''Unpickles a file with extension .pkl using the cPickle module and stores
    it in an object.
    
    filepath = The path and file name for the input to be unpickled
    '''
    input = open(filepath, 'rb')
    obj = cPickle.load(input)
    input.close()
    
    return obj

'''###########################
Calculate the Mean Array of a Set of Arrays
###########################'''
def meanArrays(arrays,dtype=float):
    '''Given a list of numpy arrays of the same dimensions, this function will 
    perform raster addition on the entire set and then divide by the number of arrays.
    Returns a numpy array with the same dimensions as the inputs and a data type 
    determined by the user.
    
    arrays = A list or tuple of numpy arrays
    dtype = The desired output data type for array elements (defaults to python float)
    '''
    sums = np.zeros(arrays[0].shape, dtype=dtype)
    n = len(arrays)
    for i in range(n):
        sums = sums + arrays[i]
    mean = sums/n
    
    return mean

def meanArrays_nan(arrays,dtype=float):
    '''Given a list of numpy arrays of the same dimensions, this function will 
    perform raster addition (skipping nan values) on the entire set and then 
    divide by the number of (non-nan) arrays. Returns a numpy array with the 
    same dimensions as the inputs and a data type determined by the user.
    
    arrays = A list or tuple of numpy arrays
    dtype = The desired output data type for array elements (defaults to python float)
    '''
    sums = np.zeros(arrays[0].shape, dtype=dtype) # sum array
    n = np.zeros(arrays[0].shape, dtype=dtype) # number of non-nans array
    for i in range(len(arrays)):
        add_sums = np.where(np.isnan(arrays[i]) == 1,0,arrays[i]) # Turn nans into 0s
        add_n = np.where(np.isnan(arrays[i]) == 1,0,1) # Turns nans into 0s, all else into 1s
        sums = sums + add_sums # Add away!
        n = n + add_n # Add away!
    
    mean = sums/n # Calculate mean on gridcell-by-gridcell basis.
    
    return mean

'''###########################
Calculate the Sum Array of a Set of Arrays
###########################'''
def addArrays_nan(arrays,dtype=float):
    '''Given a list of numpy arrays of the same dimensions, this function will 
    perform raster addition (skipping nan values) on the entire set and then 
    divide by the number of (non-nan) arrays.Returns a numpy array with the 
    same dimensions as the inputs and a data type determined by the user.
    
    arrays = A list or tuple of numpy arrays
    dtype = The desired output data type for array elements (defaults to python float)
    '''
    sums = np.zeros(arrays[0].shape, dtype=dtype) # sum array
    nans = np.zeros(arrays[0].shape, dtype=dtype)
    n = len(arrays)
    for i in range(n):
        add_sums = np.where(np.isnan(arrays[i]) == 1,0,arrays[i]) # Turn nans into 0s
        sums = sums + add_sums # Add away!
        
        add_nans = np.where(np.isnan(arrays[i]) == 1,1,0) # Turns nans into 1s, all else into 0s
        nans = nans + add_nans # The higher the number, the more nans that location saw
    
    # Quality Control -- if all values were nans, make it a nan, not 0!
    nans = np.where(nans == n,np.nan,0)
    sums = sums + nans
    
    return sums 

'''###########################
Calculate the Circular Mean Array of a Set of Arrays
###########################'''
def meanCircular(a,amin,amax,favorMax=1):
    '''Given a list of values (a), this function appends one member at a time. 
    Weights are assigned for each appending step based on how many members 
    have already been appended. If the resultant mean crosses the min/max 
    boundary at any point, it is redefined as is the min and max are the same 
    point.
    
    a = a list of values
    amin = the minimum on the circle
    amax = the maximum on the circle
    favorMax = if 1 (default), the maximum is always returned (never the minimum);
        if 0, the minimum is always returned (never the maximum)
    '''
    circum = amax-amin # calculate range (circumference) of the circle
    # Sort from largest to smallest
    aa = copy.deepcopy(a)
    aa.sort()
    aa.reverse()
    
    for i,v in enumerate(aa): # for each value in a
        v = float(v)
        if i == 0: # if it's the first value, value = mean
            mean = v
        else: # otherwise...
            arc = mean-v # calculate arc length that doesn't cross min/max
            if arc/circum < 0.5: # if that arc length is less than half the total circumference
                mean = mean*i/(i+1) + v/(i+1) # then weight everything like normal; making the mean smaller
            else: # otherwise, the influence will pull the mean toward min/max
                mean = mean*i/(i+1) + (v+circum)/(i+1) # making the mean larger
            
            # After nudging the mean, check to see if it crossed the min/max line
            ## Adjust value if necessary
            if mean < amin:
                mean = amax-amin+mean
            elif mean > amax:
                mean = amin-amax+mean
    
    # Check for minimum/maximum and replace if necessary
    if mean == amin and favorMax == 1:
        mean = amax
    
    elif mean == amax and favorMax == 0:
        mean = amin
    
    return mean

def meanArraysCircular_nan(arrays,amin,amax,favorMax=1,dtype=float):
    '''Given a list of arrays, this function applies a circular mean function
    to each array location across all arrays. Returns an array with the same
    dimensions as the inputs (all of which must have the same dimensions).
    NaNs are eliminated from consideration.
    
    a = a list of values
    amin = the minimum on the circle
    amax = the maximum on the circle
    favorMax = if 1 (default), the maximum is always returned (never the minimum);
        if 0, the minimum is walawyas returned (never the maximum)
    arrays = A list or tuple of numpy arrays
    dtype = The desired output data type for array elements (defaults to python float)
    '''
    means = np.zeros(arrays[0].shape, dtype=dtype)
    
    # Loop through each location
    for r in range(arrays[0].shape[0]):
        for c in range(arrays[0].shape[1]):
            # Collect values across all arrays
            #print str(r) + " of " + str(arrays[0].shape[0]) + ", " + str(c) + " of " + str(arrays[0].shape[1])
            a = []
            for i in range(len(arrays)):
                if np.isnan(arrays[i][r,c]) == 0: # Only if a non-nan value
                    a.append(arrays[i][r,c])
                
                # Calculate circular mean
                if len(a) > 0:
                    means[r,c] = meanCircular(a,amin,amax,favorMax=favorMax)
                else:
                    means[r,c] = np.nan
    
    return means

'''###########################
Aggregate the Event Frequency for a Month of Cyclone Tracks
###########################'''
def aggregateEvents(tracks,typ,days,shape):
    '''Aggregates cyclone events (geneis, lysis, splitting, and merging) for 
    a given month and returns a list of 4 numpy arrays in the order
    [gen, lys, spl, mrg] recording the event frequency
    
    tracks = a list or tuple of tracks in the order [trs,trs0,trs2], where 
    trs = the current month, trs0 = the previous month, and trs2 = the active
        tracks remaining at the end of the current month
    typ = "cyclone" or "system"
    days = the time in days since a common reference date for 0000 UTC on the 
        1st day of the current month
    shape = a tuple of (r,c) where r and c are the number of rows and columns,
        respectively, for the output
    '''
    fields = [np.zeros(shape),np.zeros(shape),np.zeros(shape),np.zeros(shape)]
    
    if typ.lower() == "cyclone":
        excludeType = 2
    else:
        excludeType = 1
    
    # Limit events to only those tracks that satisfy above criteria
    tids = [tr.tid for tr in tracks[0]]
    ftids = [tr.ftid for tr in tracks[0]]
    tids0 = [tr.tid for tr in tracks[1]]
    ftids2 = [tr.ftid for tr in tracks[2]]
    
    for tr in tracks[0]: # For each track
        # Record first and last instance as genesis and lysis, respectively
        fields[0][int(tr.data.y.irow(0)),int(tr.data.x.irow(0))] =  fields[0][int(tr.data.y.irow(0)),int(tr.data.x.irow(0))] + 1
        fields[1][int(list(tr.data.y)[-1]),int(list(tr.data.x)[-1])] =  fields[1][int(list(tr.data.y)[-1]),int(list(tr.data.x)[-1])] + 1
        
        for e in range(len(tr.events)): # Check the stats for each event
            if tr.events.Etype.irow(e) != excludeType: # Area-only or Point-only events may not be of interest
                # For splits, merges, and re-genesis, only record the event if the
                ## interacting track also satisfies the lifespan/track length criteria
                # If the event time occurs during the month of interest...
                # Check if the otid track exists in either this month or the next month:
                if tr.events.time.irow(e) >= days and ( (tr.events.otid.irow(e) in tids) or (tr.events.otid.irow(e) in ftids2) ):
                    # And if so, record the event type
                    if tr.events.event.irow(e) == "mg":
                        fields[3][tr.events.y.irow(e),tr.events.x.irow(e)] =  fields[3][tr.events.y.irow(e),tr.events.x.irow(e)] + 1
                    elif tr.events.event.irow(e) == "sp":
                        fields[2][tr.events.y.irow(e),tr.events.x.irow(e)] =  fields[2][tr.events.y.irow(e),tr.events.x.irow(e)] + 1
                # If the event time occurs during the previous month...
                # Check if the otid track exists in either this month or the previous month:
                elif tr.events.time.irow(e) < days and ( (tr.events.otid.irow(e) in tids0) or (tr.events.otid.irow(e) in ftids) ):
                    # And if so, record the event type
                    if tr.events.event.irow(e) == "mg":
                        fields[3][tr.events.y.irow(e),tr.events.x.irow(e)] =  fields[3][tr.events.y.irow(e),tr.events.x.irow(e)] + 1
                    elif tr.events.event.irow(e) == "sp":
                        fields[2][tr.events.y.irow(e),tr.events.x.irow(e)] =  fields[2][tr.events.y.irow(e),tr.events.x.irow(e)] + 1
    
    return fields

'''###########################
Aggregate Track-wise Stats for a Month of Cyclone Tracks
###########################'''
def aggregateTrackWiseStats(trs,date,shape):
    '''Aggregates cyclone stats that have a single value for each track:
    genesis and lysis time, max propagation speed, max deepening rate, 
    max depth, min central pressure, max laplacian of central pressure,
    lifespan, track length, average area, and whether its a MCC. Returns a 
    list containing a) a pandas dataframe of stats for the month and b) five 
    numpy arrays for the frequency of the extremes at each location in the
    order: max propagation speed, max deepening rate, max depth, 
    min central pressure, max laplacian of central pressure
    
    trs = List of cyclone track objects for current month
    date = A date in the format [Y,M,D] or [Y,M,D,H,M,S]
    shape = a tuple of (r,c) where r and c are the number of rows and columns,
        respectively, for the output
    '''
    # Prep inputs
    maxuv_field, maxdpdt_field, maxdep_field, minp_field, maxdsqp_field = \
    np.zeros(shape),np.zeros(shape),np.zeros(shape),np.zeros(shape),np.zeros(shape)
    
    statsPDF = pd.DataFrame(columns=["year","month","timeBegin","timeEnd",\
    "maxUV","maxDpDt","maxDep","maxDsqP","minP","lifespan","trlength",\
    "avgArea","MCC","CAP"])

    # Look at each track and aggregate stats
    for tr in trs:
        # Collect Track-Wise Stats
        trmaxuv = tr.maxUV()
        for i in range(len(trmaxuv[1])):
            maxuv_field[trmaxuv[2][i],trmaxuv[3][i]] = maxuv_field[trmaxuv[2][i],trmaxuv[3][i]] + 1
        trmaxdpdt = tr.maxDpDt()
        for i in range(len(trmaxdpdt[1])):
            maxdpdt_field[trmaxdpdt[2][i],trmaxdpdt[3][i]] = maxdpdt_field[trmaxdpdt[2][i],trmaxdpdt[3][i]] + 1
        trmaxdsqp = tr.maxDsqP()
        for i in range(len(trmaxdsqp[1])):
            maxdsqp_field[trmaxdsqp[2][i],trmaxdsqp[3][i]] = maxdsqp_field[trmaxdsqp[2][i],trmaxdsqp[3][i]] + 1
        trmaxdep = tr.maxDepth()
        for i in range(len(trmaxdep[1])):
            maxdep_field[trmaxdep[2][i],trmaxdep[3][i]] = maxdep_field[trmaxdep[2][i],trmaxdep[3][i]] + 1
        trminp = tr.minP()
        for i in range(len(trminp[1])):
            minp_field[trminp[2][i],trminp[3][i]] = minp_field[trminp[2][i],trminp[3][i]] + 1
        
        # Store Stats in DF
        row = pd.DataFrame([dict(maxUV=trmaxuv[0],maxDpDt=trmaxdpdt[0],maxDsqP=trmaxdsqp[0],maxDep=trmaxdep[0],\
            minP=trminp[0],lifespan=tr.lifespan(),trlength=tr.trackLength(),avgArea=tr.avgArea(),MCC=tr.mcc(),\
            timeBegin=np.min(list(tr.data.time)),timeEnd=np.max(list(tr.data.time)),\
            CAP=np.nansum(list(tr.data.precip[tr.data.type != 0])),\
            year=date[0],month=date[1]), ])
        statsPDF = statsPDF.append(row, ignore_index=True)
    
    return [statsPDF, maxuv_field, maxdpdt_field, maxdep_field, minp_field, maxdsqp_field]

'''###########################
Aggregate Point-wise Stats for a Month of Cyclone Tracks
###########################'''
def aggregatePointWiseStats(trs,n,shape):
    '''Aggregates cyclone counts, track density, and a host of other Eulerian
    measures of cyclone characteristics. Returns a list of numpy arrays in the
    following order: track density, cyclone center frequnecy, cyclone center 
    frequency for centers with valid pressure, and multi-center cyclone 
    frequnecy; the average propagation speed, propogation direction, radius, 
    area, depth, depth/radius, deepening rate, central pressure, and laplacian 
    of central pressure.
     
    trs = List of cyclone track objects for current month
    n = The number of time slices considered in the creation of trs (usually the 
        number of days in the given month times 24 hours divided by the time interval in hours)
    shape = A tuple of (r,c) where r and c are the number of rows and columns,
        respectively, for the output
    '''
    # Ensure that n is a float
    n = float(n)
    
    # Create empty fields
    sys_field, trk_field, countU_field, countA_field, countP_field = \
    np.zeros(shape), np.zeros(shape), np.zeros(shape), np.zeros(shape), np.zeros(shape)
    
    pcent_field, dpdt_field, dpdr_field, dsqp_field, depth_field = \
    np.zeros(shape), np.zeros(shape), np.zeros(shape), np.zeros(shape), np.zeros(shape)
    
    uvDi_fields, uvAb_field, radius_field, area_field, mcc_field = \
    [], np.zeros(shape), np.zeros(shape), np.zeros(shape), np.zeros(shape)
    
    for tr in trs:
        uvDi_field = np.zeros(shape)
        
        # Count Point-Wise Stats
        trk_tracker = np.zeros(shape) # This array tracks whether the track has been counted yet in each grid cell
        for i in list(tr.data.index)[:-1]:
            # Existance of System/Track
            sys_field[tr.data.y[i],tr.data.x[i]] = sys_field[tr.data.y[i],tr.data.x[i]] + 1
            if trk_tracker[tr.data.y[i],tr.data.x[i]] == 0: # Only count in trk_field if it hasn't yet been counted there!
                trk_field[tr.data.y[i],tr.data.x[i]] = trk_field[tr.data.y[i],tr.data.x[i]] + 1
                trk_tracker[tr.data.y[i],tr.data.x[i]] = trk_tracker[tr.data.y[i],tr.data.x[i]] + 1
            # Special Cases:
            if i > 0:
                countU_field[tr.data.y[i],tr.data.x[i]] = countU_field[tr.data.y[i],tr.data.x[i]] + 1
            if tr.data.radius[i] != 0:
                countA_field[tr.data.y[i],tr.data.x[i]] = countA_field[tr.data.y[i],tr.data.x[i]] + 1
            if np.isnan(tr.data.p_cent[i]) != 1:
                countP_field[tr.data.y[i],tr.data.x[i]] = countP_field[tr.data.y[i],tr.data.x[i]] + 1
            
            # Other Eulerian Measures
            pcent_field[tr.data.y[i],tr.data.x[i]] = pcent_field[tr.data.y[i],tr.data.x[i]] + float(np.where(np.isnan(tr.data.p_cent[i]) == 1,0,tr.data.p_cent[i]))
            dpdt_field[tr.data.y[i],tr.data.x[i]] = dpdt_field[tr.data.y[i],tr.data.x[i]] + float(np.where(np.isnan(tr.data.DpDt[i]) == 1,0,tr.data.DpDt[i]))
            dpdr_field[tr.data.y[i],tr.data.x[i]] = dpdr_field[tr.data.y[i],tr.data.x[i]] + float(np.where(np.isnan(tr.data.DpDr[i]) == 1,0,tr.data.DpDr[i]))
            dsqp_field[tr.data.y[i],tr.data.x[i]] = dsqp_field[tr.data.y[i],tr.data.x[i]] + float(np.where(np.isnan(tr.data.DsqP[i]) == 1,0,tr.data.DsqP[i]))
            depth_field[tr.data.y[i],tr.data.x[i]] = depth_field[tr.data.y[i],tr.data.x[i]] + float(np.where(np.isnan(tr.data.depth[i]) == 1,0,tr.data.depth[i]))
            uvAb_field[tr.data.y[i],tr.data.x[i]] = uvAb_field[tr.data.y[i],tr.data.x[i]] + float(np.where(np.isnan(tr.data.uv[i]) == 1,0,tr.data.uv[i]))
            uvDi_field[tr.data.y[i],tr.data.x[i]] = uvDi_field[tr.data.y[i],tr.data.x[i]] + vectorDirectionFrom(tr.data.u[i],tr.data.v[i])
            radius_field[tr.data.y[i],tr.data.x[i]] = radius_field[tr.data.y[i],tr.data.x[i]] + float(np.where(np.isnan(tr.data.radius[i]) == 1,0,tr.data.radius[i]))
            area_field[tr.data.y[i],tr.data.x[i]] = area_field[tr.data.y[i],tr.data.x[i]] + float(np.where(np.isnan(tr.data.area[i]) == 1,0,tr.data.area[i]))
            mcc_field[tr.data.y[i],tr.data.x[i]] = mcc_field[tr.data.y[i],tr.data.x[i]] + float(np.where(float(tr.data.centers[i]) > 1,1,0))
        
        uvDi_fields.append(np.where(uvDi_field == 0,np.nan,uvDi_field))
        
    ### AVERAGES AND DENSITIES ###
    uvDi_fieldAvg = meanArraysCircular_nan(uvDi_fields,0,360)
    
    pcent_fieldAvg = pcent_field/countP_field
    dpdt_fieldAvg = dpdt_field/countU_field
    dpdr_fieldAvg = dpdr_field/countA_field
    dsqp_fieldAvg = dsqp_field/countP_field
    depth_fieldAvg = depth_field/countA_field
    uvAb_fieldAvg = uvAb_field/countU_field
    radius_fieldAvg = radius_field/countP_field
    area_fieldAvg = area_field/countP_field
    
    return [trk_field, sys_field/n, countP_field/n, countU_field/n, countA_field/n, mcc_field/n, \
        uvAb_fieldAvg, uvDi_fieldAvg, radius_fieldAvg, area_fieldAvg, depth_fieldAvg, \
        dpdr_fieldAvg, dpdt_fieldAvg, pcent_fieldAvg, dsqp_fieldAvg]

'''###########################
Aggregate Fields the Exist for Each Time Step in a Month of Cyclone Tracking
###########################'''
def aggregateTimeStepFields(inpath,trs,mt,timestep,dateref=[1900,1,1,],lys=1):
    '''Aggregates fields that exist for each time step in a month of cyclone
    tracking data. Returns a list of numpy arrays.
    
    inpath = a path to the directory for the cyclone detection/tracking output
    (should end with the folder containing an "AreaFields" folder)
    mt = month time in format [Y,M,1] or [Y,M,1,0,0,0]
    timestep = timestep in format [Y,M,D] or [Y,M,D,H,M,S]
    lys = 1 for Gregorian calendar, 0 for 365-day calendar
    '''
    # Supports
    monthstep = [0,1,0,0,0,0]
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    mons = ["01","02","03","04","05","06","07","08","09","10","11","12"]
    days = ["01","02","03","04","05","06","07","08","09","10","11","12","13",\
        "14","15","16","17","18","19","20","21","22","23","24","25","26","27",\
        "28","29","30","31"]
    hours = ["0000","0100","0200","0300","0400","0500","0600","0700","0800",\
        "0900","1000","1100","1200","1300","1400","1500","1600","1700","1800",\
        "1900","2000","2100","2200","2300"]
    
    # Start timers
    t = mt
    tcount = 0
    
    # Create an empty array to start
    date = str(t[0])+mons[t[1]-1]+days[t[2]-1]+"_"+hours[t[3]]
    cf = unpickle(inpath[:-7]+"/CycloneFields/"+str(t[0])+"/"+months[t[1]-1]+"/CF"+date+".pkl")
    fieldAreas = 0*cf.fieldAreas
    
    while t != timeAdd(mt,monthstep,lys):
        date = str(t[0])+mons[t[1]-1]+days[t[2]-1]+"_"+hours[t[3]]
        
        # Load Cyclone Field for this time step
        cf = unpickle(inpath[:-7]+"/CycloneFields/"+str(t[0])+"/"+months[t[1]-1]+"/CF"+date+".pkl")
        cAreas, nC = scipy.ndimage.measurements.label(cf.fieldAreas)
        
        # For each track...
        for tr in trs:
            d = daysBetweenDates(dateref,t)
            try:
                x = int(list(tr.data.x[(tr.data.time == d) & (tr.data.type != 0)])[0])
                y = int(list(tr.data.y[(tr.data.time == d) & (tr.data.type != 0)])[0])
                
                # Add the area for this time step
                fieldAreas = fieldAreas + np.where(cAreas == cAreas[y,x], 1, 0)
            
            except:
                continue
        
        # Increment time step
        tcount = tcount+1
        t = timeAdd(t,timestep,lys)
    return [fieldAreas/tcount]

'''###########################
Calculate the Distance Between Two (Lat,Long) Locations
###########################'''
def haversine(lats1, lats2, longs1, longs2, units="meters"):
    '''This function uses the haversine formula to calculate the distance between
    two points on the Earth's surface when given the latitude and longitude (in decimal 
    degrees). It returns a distance in the units specified (default is meters). If 
    concerned with motion, (lats1,longs1) is the initial position and (lats2,longs2)
    is the final position.
    '''
    import numpy as np
    
    # Convert to radians:
    lat1, lat2 = lats1*np.pi/180, lats2*np.pi/180
    long1, long2 = longs1*np.pi/180, longs2*np.pi/180
    
    # Perform distance calculation:
    R = 6371000
    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((long2-long1)/2)**2
    c = 2*np.arctan2(np.sqrt(a),np.sqrt(1-a))
    d = R*c
    
    # Conversions:
    km = ["km","Km","KM","kms","Kms","KMS","kilometer","kilometre","Kilometer",\
    "Kilometre","kilometers","kilometres","Kilometers","Kilometres"]
    ft = ["ft","FT","feet","Feet"]
    mi = ["mi","MI","miles","Miles"]
    
    if units in ft:
        d = d*3.28084
    elif units in km:
        d = d/1000
    elif units in mi:
        d = d*0.000621371
    
    return d

'''###########################
Calculate the Direction a Vector is Coming From
###########################'''
def vectorDirectionFrom(u,v,deg=1):
    '''This function calculates the direction a vector is coming from when given
    a u and v component. By default, it will return a value in degrees. Set
    deg = 0 to get radians instead.
    
    Returns a value in the range (0,360], with a 0 indicating no movement.
    '''
    # Take the 180 degree arctangent
    ### Rotate results counter-clockwise by 90 degrees
    uvDi = 0.5*np.pi - np.arctan2(-v,-u) # Use negatives because it's FROM
    
    # If you get a negative value (or 0), add 2 pi to make it positive
    if uvDi <= 0:
        uvDi = uvDi+2*np.pi
    
    # Set to 0 if no motion occurred
    if u == 0 and v == 0:
        uvDi = 0
    
    # If the answer needs to be in degrees, convert
    if deg == 1:
        uvDi = 180*uvDi/np.pi
    
    return uvDi

'''###########################
Calculate a Longitudinal Angle along a Certain Distance of a Parallel
###########################'''
def dist2long(d, lat1, lat2, units="km", r=6371.):
    '''This function converts from standard distance units to a longitudinal 
    angle  on a sphere when given two latitudes (in degrees) and the radius of 
    the sphere using the haversine formula. By default, the distance is assumed
    to be in kiolmeters and the radius is assumed to be 6371 km (i.e., the 
    sphere is Earth). Returns the longitudinal angle in degrees.'''
    import numpy as np
    
    # Convert latitudes to radians:
    lat1, lat2 = lat1*np.pi/180, lat2*np.pi/180
    
    # Other Conversions:
    km = ["km","kms","kilometer","kilometre","kilometers","kilometres"]
    m = ["m","ms","meter","meters","metres","metres"]
    ft = ["ft","feet"]
    mi = ["mi","miles","mile","mi"]
    nm = ["nm","nautical mile","nms","nautical miles","mile nautical",\
        "miles nautical","mile (nautical)","miles (nautical)"]
    
    if units.lower() in km:
        d = d
    elif units.lower() in ft:
        d = d/3280.84
    elif units.lower() in mi:
        d = d/0.621371
    elif units.lower() in nm:
        d = d/0.5399568
    elif units.lower() in m:
        d = d/1000
    
    # Main calculation
    dlat = lat2 - lat1
    c = d/r
    dlon = 2*np.arcsin(( ( np.sin(c/2)**2 - np.sin(dlat/2)**2 ) / ( np.cos(lat1)*np.cos(lat2) ) )**0.5)*180.0/np.pi
    
    #dlon = 2*np.arcsin( np.sin(c/2) / np.cos(lat1) ) # for constant latitude
    
    return dlon

'''#########################
Compare Tracks from Different Datasets
#########################'''
def comparetracks(trs1,trs2,trs2b,date1,refdate=[1900,1,1,0,0,0],minmatch=0.6,maxsep=500):
    '''This function performs a track-matching comparison between two different
    sets of tracks. The tracks being compared should be from the same month and 
    have the same temporal resolution. They should differ based on input data,
    spatial resolution, or detection/tracking parameters. The function returns
    a pandas dataframe with a row for each cyclone track in first dataset. If 
    one exists, the cyclone track in the second dataset that best matches is 
    compared by the separation distance and intensity differences (central
    pressure, its Laplacian, area, and depth).\n
    
    trs1 = A list of cyclone track objects from the first version
    trs2 = A list of cyclone track objects from the second version; must be for
        the same month as trs1
    trs2b = A list of cylone track objects from the second vesion; must be for
        one month prior to trs1
    date1 = A date in the format [YYYY,MM,1,0,0,0], corresponds to trs1
    refdate = The reference date used during cyclone tracking in the format
        [YYYY,MM,DD,HH,MM,SS], by default [1900,1,1,0,0,0]
    minmatch = The minimum ratio of matched times to total times for any two
        cyclone tracks to be considered a matching pair... using the equation
        2*N(A & B) / (N(A) + N(B)) >= minmatch, where N is the number of 
        observations times, and A and B are the tracks being compared
    maxsep = The maximum allowed separation between a matching pair of tracks;
        separation is calculated as the average distance between the tracks 
        during matching observation times using the Haversine formula
    '''
    pdf = pd.DataFrame(columns=["Year1","Month1","sid1","Num_Matches","Year2","Month2","sid2","Dist","pcentDiff","areaDiff","depthDiff","dsqpDiff"])
    refday = daysBetweenDates(refdate,date1)
    timeb = timeAdd(date1,[0,-1,0])
        
    # For each track in version 1, find all of the version 2 tracks that overlap at least *minmatch* (e.g. 60%) of the obs times
    for i1 in range(len(trs1)):
        # Extract the observation times for the version 1 track
        times1 = np.array(trs1[i1].data.time[trs1[i1].data.type != 0])
        lats1 = np.array(trs1[i1].data.lat[trs1[i1].data.type != 0])
        lons1 = np.array(trs1[i1].data.long[trs1[i1].data.type != 0])
        
        ids2, ids2b = [], [] # Lists in which to store possible matches from version 2
        avgdist2, avgdist2b = [], [] # Lits in which to store the average distances between cyclone tracks
        for i2 in range(len(trs2)):
            # Extract the observation times for the version 2 track
            times2 = np.array(trs2[i2].data.time[trs2[i2].data.type != 0])
            # Assess the fraction of matching observations
            matchfrac = 2*np.sum([t in times2 for t in times1]) / float(len(times1) + len(times2))
            # If that's satisfied, calculate the mean separation for matching observation times
            if matchfrac >= minmatch:
                timesm = [t for t in times1 if t in times2] # Extract matched times
                lats2 = np.array(trs2[i2].data.lat[trs2[i2].data.type != 0])
                lons2 = np.array(trs2[i2].data.long[trs2[i2].data.type != 0])
                
                # Calculate the mean separation between tracks
                avgdist2.append( np.mean( [haversine(lats1[np.where(times1 == tm)][0],lats2[np.where(times2 == tm)][0],\
                    lons1[np.where(times1 == tm)][0],lons2[np.where(times2 == tm)][0],units='km') for tm in timesm] ) )
                
                # And store the track id for the version 2 cyclone
                ids2.append(i2)
        
        # If the version 1 track also existed last month, check last month's version 2 tracks, too... 
        if times1[0] < refday:
            for i2b in range(len(trs2b)):
                # Extract the observation times for the version 2b track
                times2b = np.array(trs2b[i2b].data.time[trs2b[i2b].data.type != 0])
                # Assess the fraction of matching observations
                matchfrac = 2*np.sum([t in times2b for t in times1]) / float(len(times1) + len(times2b))
                if matchfrac >= minmatch:
                    timesmb = [t for t in times1 if t in times2b] # Extract matched times
                    lats2b = np.array(trs2b[i2b].data.lat[trs2b[i2b].data.type != 0])
                    lons2b = np.array(trs2b[i2b].data.long[trs2b[i2b].data.type != 0])
                    
                    # Calculate the mean separation between tracks
                    avgdist2b.append( np.mean( [haversine(lats1[np.where(times1 == tmb)][0],lats2b[np.where(times2b == tmb)][0],\
                        lons1[np.where(times1 == tmb)][0],lons2b[np.where(times2b == tmb)][0],units='km') for tmb in timesmb] ) )
                    
                    # And store the track id for the version 2b cyclone
                    ids2b.append(i2b)
            
            # Identify how many possible matches are satisfy the maximum average separation distance
            nummatch = np.where(np.array(avgdist2+avgdist2b) < maxsep)[0].shape[0]
            
            # Determine which version 2(b) track has the shortest average separation
            if nummatch == 0: # If there's no match...
                pdf = pdf.append(pd.DataFrame([{"Year1":date1[0],"Month1":date1[1],"sid1":trs1[i1].sid,"Num_Matches":nummatch,"Year2":np.nan,"Month2":np.nan,\
                    "sid2":np.nan,"Dist":np.nan,"pcentDiff":np.nan,"areaDiff":np.nan,"depthDiff":np.nan,"dsqpDiff":np.nan},]))
            
            elif np.min(avgdist2+[np.inf]) > np.min(avgdist2b+[np.inf]): # If the best match is from previous month...
                im = ids2b[np.where(avgdist2b == np.min(avgdist2b))[0][0]]
                timesmb = [t for t in times1 if t in np.array(trs2b[im].data.time[trs2b[im].data.type != 0])] # Extract matched times
                
                # Find average intensity differences
                areaDiff = np.mean([int(trs1[i1].data.area[trs1[i1].data.time == tmb]) - int(trs2b[im].data.area[trs2b[im].data.time == tmb]) for tmb in timesmb])
                pcentDiff = np.mean([int(trs1[i1].data.p_cent[trs1[i1].data.time == tmb]) - int(trs2b[im].data.p_cent[trs2b[im].data.time == tmb]) for tmb in timesmb])
                dsqpDiff = np.nanmean([float(trs1[i1].data.DsqP[trs1[i1].data.time == tmb]) - float(trs2b[im].data.DsqP[trs2b[im].data.time == tmb]) for tmb in timesmb])
                depthDiff = np.mean([int(trs1[i1].data.depth[trs1[i1].data.time == tmb]) - int(trs2b[im].data.depth[trs2b[im].data.time == tmb]) for tmb in timesmb])
                
                pdf = pdf.append(pd.DataFrame([{"Year1":date1[0],"Month1":date1[1],"sid1":trs1[i1].sid,"Num_Matches":nummatch,"Year2":timeb[0],"Month2":timeb[1],"sid2":trs2b[im].sid,\
                    "Dist":np.min(avgdist2b),"pcentDiff":pcentDiff,"areaDiff":areaDiff,"depthDiff":depthDiff,"dsqpDiff":dsqpDiff},]))
            
            else: # If the best match is from current month...
                im = ids2[np.where(avgdist2 == np.min(avgdist2))[0][0]]
                timesm = [t for t in times1 if t in np.array(trs2[im].data.time[trs2[im].data.type != 0])] # Extract matched times
                
                # Find average intensity differences
                areaDiff = np.mean([int(trs1[i1].data.area[trs1[i1].data.time == tm]) - int(trs2[im].data.area[trs2[im].data.time == tm]) for tm in timesm])
                pcentDiff = np.mean([int(trs1[i1].data.p_cent[trs1[i1].data.time == tm]) - int(trs2[im].data.p_cent[trs2[im].data.time == tm]) for tm in timesm])
                dsqpDiff = np.nanmean([float(trs1[i1].data.DsqP[trs1[i1].data.time == tm]) - float(trs2[im].data.DsqP[trs2[im].data.time == tm]) for tm in timesm])
                depthDiff = np.mean([int(trs1[i1].data.depth[trs1[i1].data.time == tm]) - int(trs2[im].data.depth[trs2[im].data.time == tm]) for tm in timesm])
                
                pdf = pdf.append(pd.DataFrame([{"Year1":date1[0],"Month1":date1[1],"sid1":trs1[i1].sid,"Num_Matches":nummatch,"Year2":date1[0],"Month2":date1[1],"sid2":trs2[im].sid,\
                    "Dist":np.min(avgdist2),"pcentDiff":pcentDiff,"areaDiff":areaDiff,"depthDiff":depthDiff,"dsqpDiff":dsqpDiff},]))
        
        # If the version 1 track only existed in the current month...
        else:
            # Identify how many possible matches are satisfy the maximum average separation distance
            nummatch = np.where(np.array(avgdist2) < maxsep)[0].shape[0]
            
            # Determine which version 2 track has the shortest average separation
            if nummatch == 0: # If there's no match...
                pdf = pdf.append(pd.DataFrame([{"Year1":date1[0],"Month1":date1[1],"sid1":trs1[i1].sid,"Num_Matches":nummatch,"Year2":np.nan,"Month2":np.nan,\
                    "sid2":np.nan,"Dist":np.nan,"pcentDiff":np.nan,"areaDiff":np.nan,"depthDiff":np.nan,"dsqpDiff":np.nan},]))
            
            else: # If the best match is from current month...
                im = ids2[np.where(avgdist2 == np.min(avgdist2))[0][0]]
                timesm = [t for t in times1 if t in np.array(trs2[im].data.time[trs2[im].data.type != 0])] # Extract matched times
                
                # Find average intensity differences
                areaDiff = np.mean([int(trs1[i1].data.area[trs1[i1].data.time == tm]) - int(trs2[im].data.area[trs2[im].data.time == tm]) for tm in timesm])
                pcentDiff = np.mean([int(trs1[i1].data.p_cent[trs1[i1].data.time == tm]) - int(trs2[im].data.p_cent[trs2[im].data.time == tm]) for tm in timesm])
                dsqpDiff = np.nanmean([float(trs1[i1].data.DsqP[trs1[i1].data.time == tm]) - float(trs2[im].data.DsqP[trs2[im].data.time == tm]) for tm in timesm])
                depthDiff = np.mean([int(trs1[i1].data.depth[trs1[i1].data.time == tm]) - int(trs2[im].data.depth[trs2[im].data.time == tm]) for tm in timesm])
                
                pdf = pdf.append(pd.DataFrame([{"Year1":date1[0],"Month1":date1[1],"sid1":trs1[i1].sid,"Num_Matches":nummatch,"Year2":date1[0],"Month2":date1[1],"sid2":trs2[im].sid,\
                    "Dist":np.min(avgdist2),"pcentDiff":pcentDiff,"areaDiff":areaDiff,"depthDiff":depthDiff,"dsqpDiff":dsqpDiff},]))
    return pdf