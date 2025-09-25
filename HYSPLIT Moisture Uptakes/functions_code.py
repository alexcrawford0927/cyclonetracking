# ====== RUNNING HYSPLIT USING PYTHON AND BATCH ======

# ~ import ~
# === Standard Library ===
import glob
import os

# === Third-Party Packages ===
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import netCDF4 as nc
from geopy.distance import geodesic
from pyproj import Transformer
import math as math
from matplotlib import cm, colors
import cartopy.crs as ccrs
from scipy.ndimage import uniform_filter
import cartopy.feature as cfeature
import matplotlib.patheffects as pe

# uploading the arctic mask 
# note that you arctic mask may not be saved in the same place 
f = nc.Dataset('D:/CycloneTracking/13_2E5R/BBox10/AggregationSystem/EASE2_N0_25km_GenesisRegions.nc')
CAO2 = np.flipud(f.variables['CAO2'][:])

x_mask = f.variables['x'][:]
y_mask = f.variables['y'][:] 

x_grid, y_grid = np.meshgrid(x_mask, y_mask)
transformer = Transformer.from_crs(6931, 4326) #3031 is the antarctic polar projection, 4326 is the standard code for lat and lon
lat_mask, lon_mask = transformer.transform(x_grid, y_grid) # lat and lon produced from transformer
lat_mask, lon_mask = np.flipud(lat_mask), np.flipud(lon_mask)


# ==== Saving a specific storm in a CSV ====
folderpath = "D:/CycloneTracking/13_2E5R/BBox10/AggregationSystem/CSVSystem"

def save_storm_CSV(SID, year, month, folderpath = folderpath):
    '''
    Saves a particular storm into a CSV in the same folder as the code. The columns saved are 
    [SID, year, month, day, hour, lat, lon, x, y] in that order.  
    
    Inputs:
        - SID, year, month = [int, int, int]; the storm's identifiers
        - folderpath = [str]; where the data on all the storms is stored
        
    Creates:
        - A CSV with columns [SID, year, month, day, hour, lat, lon, x, y] pertaining to the selected storm
        
    Notes:
        - Your storm data may be stored in a different folder path. Adjust folderpath to match the "base" directory you 
          need -- this is the directory that doesn't require any specification of year or month. Adjust specifications
          where storm_path is defined to open the desired CSV.
            - may have to adjust usecols or the delimiter to match the CSV you are using.
        - if there is additional data that you want to save, change usecols to match your desired variable. Change the 
          header names accordingly. 
        - If the directory/naming system is change, please change it in all future function.         
    '''
    # turn the month into a string (4 becomes "04", for example)
    month = str(month).zfill(2)
    
    # open the larger storm data file
    storm_path = f"{folderpath}/{year}/System13_2E5R_BBox10_{year}{month}.csv"
    data_withNaN = np.genfromtxt(storm_path, skip_header = 1, usecols = [0, 1, 2, 3, 4, 6, 7, 8, 9], delimiter = ",")
    data = data_withNaN[~np.isnan(data_withNaN).any(axis=1)]
      
    # find where the storm of interest is (with the SID)
    storm = data[data[:, 0] == SID]
    header = "SID, year, month, day, hour, lat, lon, x, y"
    
    # Save in CSV
    output_file = fr"storm{SID}_{year}{month}.csv"
    np.savetxt(output_file, storm, header = header, delimiter=",", fmt="%.6f")
    print(f"Saved file {output_file}")
    
# ==== Finding and Recording Precipitating Parcels ====
# == Adjusting nc files ==
def adjust_ncfile(filepath, year, month, dayInterval):
    '''
    Processes an ERA5 NetCDF file by combining cloud liquid water content (clwc) and cloud ice water content (ciwc) 
    into a single variable called cloud water content (cwc). The original clwc and ciwc variables are then removed
    to reduce file size. The resulting file is saved to a designated directory named after the storm's ID and year.

    Inputs:
        - filepath [str]: Path to the raw ERA5 NetCDF file containing 'clwc' and 'ciwc'.
        - year, month = [int, int]; storm identifiers
        - dayInterval [tuple(int, int)]: Tuple specifying the start and end days the file covers.

    Output:
        - A new NetCDF file with only the merged 'cwc' variable, saved in a folder 
          named 'ERA5_<year>' with a filename indicating the year, month, and day range.

    Example:
        >>> adjust_ncfile("S0921818122000194.nc", 2003, 2, (2, 6))
        Saved file: D:/ERA5_2003/HumidityCWC_200302_day2to6.nc  
    '''

    ds = xr.open_dataset(filepath)
    print(ds)
    month = str(month).zfill(2)
    
    ds['cwc'] = ds['ciwc'] + ds['clwc']
    print('Merged columns')
    
    ds = ds.drop_vars(['ciwc', 'clwc'])
    print('Deleted columns')
    
    output_dir = f"D:/ERA5_{year}/"
    os.makedirs(output_dir, exist_ok=True)

    output = f"{output_dir}HumidityCWC_{year}{month}_day{dayInterval[0]}to{dayInterval[1]}.nc"
    ds.to_netcdf(output)
        
    print("Saved file:", output)
    
# == Accessing nc files ==
def get_dataset_for_time(nc_paths, var_type, time_val):
    """
    Opens the corresponding dataset for variable var_type and time time_val. 
    
    Inputs:
        - nc_paths = [dict]; A dictionary with keys of var_types and values that are a list of tuples that are formatted
                            (filename, start_day, end_day)
        - var_type = [str]; the variable key that you need to access
        - time_val = [numpy datetime64]; the time that you need the nc file for
        
    Returns:
        - The xarray dataset for a given variable type and time_val.
        
    Raises:
        - FileNotFoundError if the file isn't found (incorrect file name/path, no corresponding day)
        
    Example:
        >>> time = np.datetime64("2003-01-29T14:00")
        >>> ds = get_dataset_for_time(nc_paths_2003, "precip_SST_SLP", time)
        
        [Opens D:\ERA5_2003\precip_SST_SLP_200302_day27to30.nc]
        
    """
    day = time_val.item().day # access the day in the time val 
    
    for path, start, end in nc_paths.get(var_type, []):
        if start <= day <= end: # if within the day interval
            return xr.open_dataset(path, engine = 'h5netcdf', cache = False)
        
    # if no appropriate day interval/file found
    raise FileNotFoundError(f"No {var_type} file found for {time_val} in nc_paths[{var_type}]")


    
# this is an example of what nc_paths_2003 looks like. This matches the setup in the file structure
# nc_paths_2003['precip_SST_SLP'] is a list of tuples with structure (filepath, start day, end day).
        
nc_paths_2003 = {
    "HumidityCWC": [(r"D:\ERA5_2003\HumidityCWC_200301_day27to31.nc", 27, 31),
                    (r"D:\ERA5_2003\HumidityCWC_200302_day1to6.nc", 1, 6)],
    
    "precip_SST_SLP": [(r"D:\ERA5_2003/precip_SST_SLP_200302_day1to6.nc", 1, 6),
                       (r"D:\ERA5_2003/precip_SST_SLP_200302_day27to30.nc", 27, 31)]
                }


# == Generating points within a radius ==
def points_within_radius(lat0, lon0, radius_km = 500):
    """
    Generate a list of latitude-longitude points within a specified radius of a central location. This function creates a 
    grid of latitude and longitude points around a central point (`lat0`, `lon0`) and returns only those that fall
    within a specified geodesic radius.

    Inputs:
        - lat0, lon0 = [float, float]; Latitude and longitude of the central point in degrees.
        - radius_km = [float]; Radius in kilometers to define the search area.

    Returns:
        - latlon_within_radius = [list of tupl]; A list of (lat, lon) tuples representing points within the specified radius.
        
    Notes:
        - If you want to reduce the resolution, manually increase lat_step and lon_step. This increases the grid size. 
          Conversely, increase the resolution by decreasing lat_step and lon_step. This will make the function take longer.
    """
    lat_step=0.5
    lon_step=1.0

    # Max degrees of lat/lon to cover the radius
    max_lat_delta = radius_km / 111  # Approx 111 km per 1° latitude
    max_lon_delta = radius_km / (111 * np.cos(np.radians(lat0)))  # varies with latitude

    # create latitude and longitude values over the bounding box
    lat_range = np.arange(lat0 - max_lat_delta, lat0 + max_lat_delta + lat_step, lat_step)
    lon_range = np.arange(lon0 - max_lon_delta, lon0 + max_lon_delta + lon_step, lon_step)

    # normalize longitude values to the [-180, 180] range to handle wraparound
    lon_range = ((lon_range + 180) % 360) - 180
    lat_range = lat_range[lat_range < 90]
    lat_range = lat_range[lat_range > -90]

    latlon_within_radius = []

    # Loop through each point in the lat/lon grid and compute distance to center
    for lat in lat_range:
        for lon in lon_range:
            distance = geodesic((lat0, lon0), (lat, lon)).km
            if distance <= radius_km:
                latlon_within_radius.append((lat, lon)) # add if point within radius

    return latlon_within_radius

# == Generating CONTROL files ==
def generate_control_files(precipParcels, SID, YEAR, MONTH, level, metdata_number, run_hours = -120):
    """
    Generate HYSPLIT CONTROL files for a given set of precipitating air parcel trajectories. It handles both single 
    and dual meteorological file cases depending on `metdata_number`, and writes the relevant control and 
    output file paths to log files for reference and future use. 

    Inputs:
        - precipParcels = [list of lists]; Each sublist contains 
            [year, month, day, hour, latitude, longitude, height] for an air parcel.
        - SID, YEAR, MONTH = [int, int, int]; storm identifiers
        - metdata_number = [int]; Number of meteorological files (1 or 2).
        - met_dir = [str]: Directory containing the meteorological data files. Defaults to "hysplit_data/metdata/" inside
                         current directory.
        - run_hours = [int]; Number of hours to run the trajectory (negative for backward runs). Defaults to -120 hours.

    Creates:
        - A CONTROL file for each parcel in the storm
        - Log files: control_names_{YEAR}{MONTH}.txt, traj_files_{YEAR}{MONTH}.txt

    Raises:
        - ValueError: If `metdata_number` is not 1 or 2.

    Notes:
        - If `metdata_number == 2`, the function attempts to find the previous month's met file.
        - Longitude wraparound is not handled here — assumed to be within valid range.
        - The file paths for storm_dir and met_dir must be the FULL path, ending with a slash. CHANGE THEM to match your case.
        
    """
    MONTH_str = str(MONTH).zfill(2)
    for i, row in enumerate(precipParcels):
        y, m, d, h, lat, lon, height = row
        y, m, d, h = str(y), str(int(m)).zfill(2), str(int(d)).zfill(2), str(int(h)).zfill(2)

        storm_dir = f"C:/Users/project/hysplit_info/storm{SID}_{YEAR}{MONTH_str}_1200km/height{height}/"
        control_filename = storm_dir + f"CONTROL_{y}{m}day{d}_hour{h}_{i}"
        traj_filename = f"tdump_{y}{m}day{d}_hour{h}_{i}"
        met_file_1 = f"RP{YEAR}{MONTH_str}.gbl"
        output_dir = storm_dir + "output/"
        met_dir = "C:/Users/project/hysplit_info/metdata/"
        
        os.makedirs(storm_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        if metdata_number == 1:
            with open(control_filename, "w") as f:
                f.write(f"{str(y)[2:]} {m} {d} {h}\n")
                f.write("1\n")
                f.write(f"{lat} {lon} {height}\n")
                f.write(f"{run_hours}\n")
                f.write("0\n")
                f.write("10000.0\n")
                f.write(f"{metdata_number}\n")
                f.write(f"{met_dir}\n")
                f.write(f"{met_file_1}\n")
                f.write(f"{output_dir}\n")
                f.write(f"{traj_filename}")

                with open(storm_dir+f"control_names_{YEAR}{MONTH_str}.txt", "a") as fs:
                    fs.write(f"{control_filename}\n")

                with open(storm_dir+f"traj_files_{YEAR}{MONTH_str}.txt", "a") as fss:
                    fss.write(f"{output_dir}{traj_filename}\n")

        elif metdata_number == 2:
            prev_month = int(MONTH) - 1 # find the previous month 
            prev_year = YEAR
            if prev_month < 1: # if the storm was in january, the previous month must be in the previous year's december
                prev_month = 12
                prev_year = YEAR - 1
                
            prev_month = str(prev_month).zfill(2)
            met_file_2 = f"RP{prev_year}{prev_month}.gbl"

            with open(control_filename, "w") as f:
                f.write(f"{y[2:]} {m} {d} {h}\n")
                f.write("1\n")
                f.write(f"{lat} {lon} {height}\n")
                f.write(f"{run_hours}\n")
                f.write("0\n")
                f.write("10000.0\n")
                f.write("2\n")
                f.write(f"{met_dir}\n")
                f.write(f"{met_file_1}\n")
                f.write(f"{met_dir}\n")
                f.write(f"{met_file_2}\n")
                f.write(f"{output_dir}\n")
                f.write(f"{traj_filename}")

                with open(storm_dir+f"control_names_{YEAR}{MONTH_str}.txt", "a") as fs:
                    fs.write(f"{control_filename}\n")

                with open(storm_dir+f"traj_files_{YEAR}{MONTH_str}.txt", "a") as fss:
                    fss.write(f"{output_dir}{traj_filename}\n")
                    
        else:
            raise ValueError("metdata_number not valid")
            
    with open(fr"{storm_dir}ASCDATA.CFG", "w") as f:
        f.write("-90.0   -180.0  lat/lon of lower left corner\n"+
                "1.0     1.0     lat/lon spacing in degrees\n" +   
                "180     360     lat/lon number of data points\n" 
                "2               default land use category\n"     
                "0.2             default roughness length (m)\n"  
                r"'\bdyfiles\'   directory of files")  

    print(f"Done storm {SID} ({YEAR}/{MONTH}), level {level}m")
    
# == Finding precipitating parcels ==
def find_precipitating_parcels(SID, YEAR, MONTH, level, nc_paths, metdata_number, radius_km = 500, run_hours = -120):
    '''
    Finds precipitating parcels for a storm's lat/lon points that are in the Arctic. It creates and saves a control file 
    that can be run in hysplit. It handles cases where the storm's back trajectory will enter the previous month. A 
    precipitating parcel is within radius_km of the main storm track and fits these criteria:
        - total precipitation is larger than 0.1 mm/h
        - cloud water content is larger than 10 mg/h
        - specific humidity has decreased by 0.1 g/kg in the last hour
    
    For each point where the storm center is within the Arctic, the function `points_within_radius` finds all lat/lon values
    that are within radius_km of the storm. Using `get_dataset_for_time`, the function finds the value of the total 
    precipitation, cloud water content and specific humidity. If the point fits all the criteria necessary, the point's
    time, location, and height are saved to a list. Finally, all the points are passed through `generate_control_files`.

    Inputs:
        - SID, YEAR, MONTH = [int, int, int]; specifications of the storm
        - level = [int]; height above ground level (in meters) being inspected
        - nc_paths = [dict]; A dictionary with keys of var_types and values that are a list of tuples that are formatted
                                (filename, start_day, end_day)
        - metdata_number = [int]; the number of meterological files will be needed
                         -> metdata_number = 1, if the storm starts more than run_hours after the start of the month 
        - radius_km = [int]; radius (in kilometers) to define the search area for precipitating parcels
        - run_hours = [int]; how long you want to run the hysplit back trajectory
        - met_dir = [str]; where the meteorlogical data is stored
    
    Creates:
        - A CONTROL file for each parcel in the storm
        - Log files: control_names_{YEAR}{MONTH}.txt, traj_files_{YEAR}{MONTH}.txt
    
    Raises:
        - ValueError, if no precipitating parcels are found. 
        - NameError, if the level does not have an pressure in hPa
        
    Notes:
        - The function will allow precipitating parcels that are outside the Arctic if the storm center is within the Arctic
        
    '''        
    # find correspond pressure (hPa)
    if level == 1500:
        pressure = 850
        
    elif level == 2250:
        pressure = 770
        
    elif level == 2625:
        pressure = 735
        
    elif level == 3000:
        pressure = 700
        
    elif level == 5500:
        pressure = 500
        
    else:
        raise NameError("Invalid level, confirm that it exists within the list")
    
    print(f"Storm {SID} - {YEAR}/{MONTH}, level {level}")
    # upload storm track
    MONTH_str = str(MONTH).zfill(2)
    storm_track = np.genfromtxt(f'storm{SID}_{YEAR}{MONTH_str}.csv', skip_header = 1, delimiter = ',')
    x, y = storm_track[:, 7].astype('int64'), storm_track[:, 8].astype('int64')

    # find where the storm is within the arctic 
    in_mask = np.array([(CAO2[y[j], x[j]] == 1) for j in range(len(x))]).astype('bool')
    arctic_storm_track = storm_track[in_mask]
    
    print(f"{len(arctic_storm_track)} points to consider")
    
    precipParcels = []
    for r, row in enumerate(arctic_storm_track):
        if r % 10 == 0:
            print(f"On point {r}. {len(precipParcels)} found")
        _, y, m, d, h, lat_storm, lon_storm, _, _ = row
        y, m, d, h = str(int(y)), str(int(m)).zfill(2), str(int(d)).zfill(2), str(int(h)).zfill(2)
        time_val = np.datetime64(f'{y}-{m}-{d}T{h}:00')

        # find points within radius
        radius_points = points_within_radius(lat_storm, lon_storm, radius_km) # results in a list of points in the radius
        
        for pt in radius_points:
            lat, lon = pt
            
            # find the total precipitation for that lat/lon/time
            with get_dataset_for_time(nc_paths, 'precip_SST_SLP', time_val) as ds_tp:
                time_ind = np.argmin(np.abs(ds_tp.valid_time.values - time_val))
                lat_ind = np.argmin(np.abs(ds_tp.latitude.values - lat))
                lon_ind = np.argmin(np.abs(ds_tp.longitude.values - lon))

                tp = ds_tp['tp'].isel(valid_time=time_ind, latitude=lat_ind, longitude=lon_ind).values.item() #in meters

            if tp * 1000 > 0.1:
                
                # find the cloud water content for that lat/lon/time
                with get_dataset_for_time(nc_paths, "HumidityCWC", time_val) as ds:
                    press_ind = np.argmin(np.abs(ds.pressure_level.values - pressure))
                    time_ind = np.argmin(np.abs(ds.valid_time.values - time_val))
                    lat_ind = np.argmin(np.abs(ds.latitude.values - lat))
                    lon_ind = np.argmin(np.abs(ds.longitude.values - lon))

                    cwc = ds['cwc'].isel(valid_time=time_ind, pressure_level = press_ind, 
                                         latitude=lat_ind, longitude=lon_ind).values.item() # in kg/kg
                    if cwc * 1e6 > 10:
                        
                        # finding the specific humidity for that lat/lon/time, reusing the indices above 
                        q = ds['q'].isel(valid_time=time_ind, pressure_level = press_ind, 
                                         latitude=lat_ind, longitude=lon_ind).values.item()

                        # finding the specific humidity for the previous hour at that lat/lon point
                        time_prevHour_val = (time_val - np.datetime64(1, "h")).astype("datetime64")
                        with get_dataset_for_time(nc_paths, "HumidityCWC", time_prevHour_val) as ds_prev:
                            time_prevHour_ind = np.argmin(np.abs(ds_prev.valid_time.values - time_prevHour_val))
                            q_prevHour = ds_prev['q'].isel(valid_time=time_prevHour_ind, pressure_level = press_ind, 
                                         latitude=lat_ind, longitude=lon_ind).values.item()
                            
                        
                        if (q_prevHour - q) * 1000 >= 0.1:
                            # append with the columns [year, month, day, hour, lat, lon, level (m)]
                            precipParcels.append([y, m, d, h, lat, lon, level])
    
    print(f"{len(precipParcels)} found")
    
    try:
        generate_control_files(precipParcels, SID, YEAR, MONTH, level, metdata_number, run_hours)
        return precipParcels
        
    except:
        return precipParcels
    
    
    
# ==== Running HYSPLIT with batch files ====
# see manual

# ==== Loading Trajectory Data ====
def save_traj_npz(traj_data, SID, year, month, level):
    """
    Saves HYSPLIT tdump trajectory data to a compressed .npz file with labeled columns. Each trajectory is assigned a
    numeric ID (`traj_id`) to enable separation and filtering when working with the stored data. The output .npz file 
    includes both the trajectory data and column labels as metadata. The traj_id is the same as the number after the final
    underscore in the control/tdump filename, for example, CONTROL_200302day02_hour09_0 will have traj_id = 0. 

    Parameters:
        - storm_data = [list]; a list of arrays, where each array is a HYSPLIT trajectory
        - SID, year, month = [int, int, int]; storm identifiers
        - level = [int]; height above ground level (in meters) being inspected
    
    Creates:
        - A .npz file with the columns [traj_id, year, month, day, hour, lat, lon, height, pressure] where each
            trajectory has a different traj_id. This file is saved as 
            ``traj_data_storm{SID}_{year}{month}/traj_storm{SID}_height{level}m.npz``
            
    """
    print("Saving files...")
    month = str(month).zfill(2)
    
    # creating directory to hold the npz file (if it doesn't already exist)
    output_dir = f"traj_data_storm{SID}_{year}{month}"
    os.makedirs(output_dir, exist_ok=True)

    all_data = []
    
    # for each trajectory, we add a column of the traj_id 
    for traj_id, traj_arr in enumerate(traj_data):
        # creating a column full of the appropriate id number
        id_column = np.full((traj_arr.shape[0], 1), traj_id, dtype=int)
        # attach to the trajectory information
        arr_with_id = np.hstack((id_column, traj_arr))
        all_data.append(arr_with_id)
    
    # define the column names and filename
    combined_data = np.vstack(all_data)
    column_names = ['traj_id', 'year', 'month', 'day', 'hour', 'lat', 'lon', 'height', 'pressure']
    out_path = os.path.join(output_dir, f"traj_storm{SID}_height{level}m.npz")

    # save file
    np.savez(out_path, data = combined_data, column_names = column_names, 
             SID = SID, year = year, month = month, level=level)

    print(f"Saved: {out_path}")


def load_tdump_files(SID, year, month, level, metdata_number, save_as_npz = True):
    '''
    Loads the trajectory information from the tdump files generated by HYSPLIT. The function accesses the text file holding
    all the tdump filenames, opens each one, and parses through each line to save the generated trajectory. If desired,
    the function will call `save_traj_npz` to save the trajectory information in an .npz file. 
    
    Inputs:
        - SID, year, month = [int, int, int]; storm identifiers
        - level = [int]; height above ground level (in meters) being inspected
        - metdata_number = [int]; the number of meterological files will be needed
                            - metdata_number = 1, if the storm starts more than run_hours after the start of the month 
        - save_as_npz = [bool]; if the user wants to save the data as a npz file
        
    Returns:
        - A list of arrays, where each array is a HYSPLIT trajectory   

    Notes:
        - If you changed the naming convention of the tdump files or directory, please change it here accordingly.
    '''    
    # each meteorological file has a line in the tdump files, so we need to filter out those extra lines and only save
    # the trajectory data
    row_with_trajinfo = metdata_number + 3

    print("Loading data....")
    month = str(month).zfill(2)
    # file that holds all the tdump names
    traj_filenames = f"storm{SID}_{year}{month}/height{level}/traj_files_{year}{month}.txt"
    
    level_trajectory = []
    # opening the text file that holds all the tdump names
    with open(f"hysplit_info/{traj_filenames}") as traj_names: 
        for traj_file in traj_names:
            
            # skip if it's an empty line
            if traj_file[:-1] == " " or traj_file[:-1] == "":
                continue
                
            trajectory_points = []
            # open the tdump file
            with open(traj_file[:-1]) as file:
                # for each line in tdump file (that holds trajectory data)
                for i, line in enumerate(file):
                    if i > row_with_trajinfo: 
                        # saving as [year, month, day, hour, lat, lon, height(m-AGL), pressure]
                        points = [line[16:19], line[22:26], line[28:30], line[34:36], line[59:65], line[66:74], 
                                  line[77:83], line[84:-1]]
                        
                        # the year in the tdump file is 1 or 2 digits (3 for 2003, 95 for 1995)
                        # Pad the tdump value with zeros (3 → "03", 95 → "95") and combine it with the input year, 
                        # giving the full year
                        traj_year_suffix = int(points[0])
                        # if necessary, it goes into the previous century
                        if traj_year_suffix > int(str(year)[2:]):
                            year_val = str(int(str(year)[:2]) - 1) + str(traj_year_suffix).zfill(2)
                        else:
                            year_val = str(year)[:2] + str(traj_year_suffix).zfill(2)
                        
                        # recombine with the points to have a 4-digit year in the first column
                        points = [year_val] + points[1:]
                
                        trajectory_points.append(points)
            
            # turn into array
            trajectory_points = np.array(trajectory_points).reshape(len(trajectory_points), len(points)).astype('float')
            # append each array into the overall storage list
            level_trajectory.append(trajectory_points)
    
    if save_as_npz:
        # save in npz file
        save_traj_npz(level_trajectory, SID, year, month, level)
        
    return level_trajectory


# ==== Finding moisture uptake parcels ====
def diagnose_moisture_uptakes(SID, YEAR, MONTH, level, nc_paths, rows_per_file=5000):
    """
    Diagnose moisture uptake parcels along HYSPLIT trajectories. A moisture uptake parcel has an hourly increase in 
    specific humidity greater than 0.025 g/kgh. After uploading the npz file with the trajectories, it goes through each
    point to find the change in specific humidity in the past hour. All uptake events are saved in an array and a series of 
    CSV files. 
    
    Inputs:
        - SID, YEAR, MONTH = [int, int, int]; storm identifiers
        - level = [int]; height above ground level (in meters) being inspected
        - nc_paths = [dict]; A dictionary with keys of var_types and values that are a list of tuples that are formatted
                                (filename, start_day, end_day)
        - rows_per_file = [int]; max number of rows per output CSV
        
    Returns:
        - uptake_event = [arr]; array of uptake events with shape (N, 8) for N uptake events found. The columns are 
                        [year, month, day, hour, lat, lon, height(m-AGL), pressure]. 
                        - if no uptake events are found, the data from the npz file is returned 
    
    Creates:
        CSV files holding the uptake events that were found. 
        
    Raises:
        - FileNotFoundError when `get_dataset_for_time` can't find a file for a given time, it will skip that time and move 
                            on. This maybe because there is an insufficient amount of data, the days are mislabelled, 
                            or there is an incorrectly written filepath.
        
    Notes:
        - This function uses `get_dataset_for_time` to find the values. 
        - Function used the npz files created from the back trajectories in `save_traj_npz`. If another naming/directory
          setup was used in creating these files, please adjust function accordingly.  
        - The moisture uptake CSV files can be accessed using `load_all_moisture_csvs`.
        
    """
    MONTH = str(MONTH).zfill(2)
    
    # === Loading the trajectory data ===
    print(f"Storm {SID}: {YEAR}/{MONTH} -- level {level}")
    #print("Opening trajectory data file")
    traj_path = f"traj_data_storm{SID}_{YEAR}{MONTH}/traj_storm{SID}_1200km_height{level}m.npz"
    datafile = np.load(traj_path)
    alldata = datafile["data"] # [traj_id, year, month, day, hour, lat, lon, height, pressure]

    if alldata.size == 0:
        print("No data found.")
        return []

    # Separating the trajectories using the ID
    ID = alldata[:, 0]
    _, indices = np.unique(ID, return_index=True)
    indices.sort()
    # Save in data array without the ID column (no longer useful after separating data)
    data = np.split(alldata[:, 1:], indices[1:]) 
    print(f"Found {len(data)} trajectories") 
    
    #data has columns [year, month, day, hour, lat, lon, height, pressure]

    uptake_event = []
    
    # === find uptake events ===
    for i, traj in enumerate(data):
        progress_interval = 100 if len(data) > 500 else 50
        if i % progress_interval == 0:
            print(f"Processing trajectory {i}, uptake events found: {len(uptake_event)}")

        for row in traj:
            year_val, month_val_int, day_val_int, hour_val_int, lat_val, lon_val, height_val, press_val = row
            # pad the day, hour, month values with zeros (4 → "04", 12 → "12")
            day_val = str(int(day_val_int)).zfill(2)
            hour_val = str(int(hour_val_int)).zfill(2)
            month_val = str(int(month_val_int)).zfill(2)

            # the current row's time
            time_val = np.datetime64(f'{year_val}-{month_val}-{day_val}T{hour_val}:00')
            
            # Get the previous hour time
            prevHour_val = (time_val - np.timedelta64(1, 'h'))

            try:
                # open the nc files for both time values, they may not be the same nc file
                with get_dataset_for_time(nc_paths, 'HumidityCWC', time_val) as ds, \
                     get_dataset_for_time(nc_paths, 'HumidityCWC', prevHour_val) as ds_prev:
                    
                    # Find indices of the time value, lat, lon, and pressure
                    time_ind = np.argmin(np.abs(ds.valid_time.values - time_val))
                    time_prev_ind = np.argmin(np.abs(ds_prev.valid_time.values - prevHour_val))
                    press_ind = np.argmin(np.abs(ds.pressure_level.values - press_val))
                    lat_ind = np.argmin(np.abs(ds.latitude.values - lat_val))
                    lon_ind = np.argmin(np.abs(ds.longitude.values - lon_val))

                    specific_humidity = ds['q'].isel(valid_time=time_ind, pressure_level=press_ind, 
                                     latitude=lat_ind, longitude=lon_ind).values.item()  # in kg/kg

                    specific_humidity_previous = ds_prev['q'].isel(valid_time=time_prev_ind, pressure_level=press_ind, 
                                     latitude=lat_ind, longitude=lon_ind).values.item()  # in kg/kg

                    if (specific_humidity - specific_humidity_previous) * 1000 >= 0.025: # check criteria for uptake
                        # save with the full year
                        uptake_info = [int(year_val), int(month_val_int), int(day_val_int), int(hour_val_int), 
                                       float(lat_val), float(lon_val), float(height_val), float(press_val)]
                        uptake_event.append(uptake_info)
            
            # skip time stamp if get_dataset_for_time can't find a time value
            except (FileNotFoundError, KeyError, IndexError) as e:
                print(f"Skipping point due to error: {e}")
                continue
    
    # if no uptake_events found
    if not uptake_event:
        print("No uptake events found. Exiting.")
        return data # returns data loaded from npz file
    
    uptake_event = np.array(uptake_event)
    print(f"Found {len(uptake_event)} moisture uptake events.")

    # Save results 
    # save in same directory as npz files
    output_dir = f"traj_data_storm{SID}_{YEAR}{MONTH}"
    os.makedirs(output_dir, exist_ok=True)
    file_base = f"moisture_uptake_storm{SID}_height{level}m"
    nrows = uptake_event.shape[0]
    # calculate number of files to be created
    # (if rows_per_file = 5000 and there are 7500 uptake events, there will be 2 files)
    num_chunks = (nrows + rows_per_file - 1) // rows_per_file
    header = "year, month, day, hour, lat, lon, height, pressure"

    print(f"Saving {num_chunks} CSV files...")
    for i in range(num_chunks):
        # define which rows to save
        start = i * rows_per_file
        end = min(start + rows_per_file, nrows)
        event_chunk = uptake_event[start:end]
        # define filename: will be differentiated by using part_ (starting from 1)
        filename = os.path.join(output_dir, f'{file_base}_part{i+1}.csv')
        np.savetxt(filename, event_chunk, header = header, delimiter=',', 
                   fmt=['%d','%d','%d','%d','%.6f','%.6f','%.2f','%.2f'])
        print(f"Saved {filename} with shape {event_chunk.shape}")
    
    return uptake_event

# Accessing the moisture uptake files
def load_all_moisture_csvs(SID, YEAR, MONTH, level):
    """
    Load all the moisture uptake CSVs associated with a particular storm and level. 
    
    Inputs:
        - SID, YEAR, MONTH = [int, int, int]; storm identifiers
        - level = [int]; height above ground level (in meters) being inspected
    
    Returns: 
        - Array of moisture uptakes with columns [year, month, day, hour, lat, lon, height, pressure]
    
    Raises:
        - FileNotFoundError if there are no files with that naming structure in the directory
        
    Notes:
        - If the naming/directory structure of the CSVs was changed at some point, please adjust here
        
    """
    MONTH = str(MONTH).zfill(2)
    
    # access files in folder
    folder_path = f"traj_data_storm{SID}_{YEAR}{MONTH}"
    pattern = os.path.join(
        folder_path, f"moisture_uptake_storm{SID}_height{level}m_part*.csv")
    files = sorted(glob.glob(pattern))
    
    # if no files are found
    if not files:
        raise FileNotFoundError("No moisture uptake CSV files found in {folder_path}")

    print(f"Loading {len(files)} moisture uptake files...")
    # put into a array
    arrays = [np.genfromtxt(f, delimiter=',', skip_header = 1) for f in files]
    return np.vstack(arrays)  


# ==== Plotting ====
# == Draw a circle on a plot ==
def draw_circle(center_lon, center_lat, radius_km):
    '''
    To plot a circle around the center lon and lat with a specified radius. 
    
    Inputs:
    - center_lon, center_lat = [float, float]; where you want the circle to be centered
    - radius_km = [float]; radius of the circle in km
    
    Outputs:
    - circle_lons, circle_lats = [list, list]; a list of 
    ''''
    radius = radius_km*1000
    geod = Geod(ellps="WGS84")  # Use the WGS84 Earth model

    # Azimuths from 0 to 360 degrees
    azimuths = np.linspace(0, 360, 72)

    # Repeat the center point into arrays that match azimuths
    lons = np.full(azimuths.shape, center_lon)
    lats = np.full(azimuths.shape, center_lat)
    dists = np.full(azimuths.shape, radius)
    
    # Compute the circle (500 km radius)
    circle_lons, circle_lats, _ = geod.fwd(lons, lats, azimuths, dists)
    
    return circle_lons, circle_lats 

# == Simple storm track plot ==
# change to fit your storm
SID, YEAR, MONTH = _, _, _
MONTH = str(MONTH).zfill(2)

# upload storm information
lats, lons, x, y = np.genfromtxt(fr"storm{SID}_{YEAR}{MONTH}.csv", usecols = (5, 6, 7, 8), delimiter = ',').T
x, y = x.astype(int), y.astype(int)

# define the arctic mask
inMask = np.array([CAO2[y[j], x[j]] == 1 for j in range(len(x))])

# set up figure
fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection=ccrs.NorthPolarStereo())
ax.set_extent([-180, 180, 25, 90], crs=ccrs.PlateCarree())

# Map features
ax.gridlines(draw_labels=False, linestyle='dotted')
ax.add_feature(cfeature.COASTLINE, edgecolor='lightgrey', alpha=0.5)
ax.add_feature(cfeature.LAND, facecolor='lightgrey', alpha=.7)

# plot mask and storm
ax.pcolormesh(lon_mask, lat_mask, CAO2, transform=ccrs.PlateCarree(), cmap='Blues', alpha=0.2)
ax.scatter(lons[inMask], lats[inMask], transform=ccrs.Geodetic(), linewidths=.25, color='tab:red')
ax.scatter(lons[~inMask][1:], lats[~inMask][1:], transform=ccrs.Geodetic(), linewidths=.25, color='black')
ax.scatter(lons[0], lats[0], marker='X', transform=ccrs.Geodetic(), linewidths=1, color='tab:purple')

plt.show()

# == Figure 3 in Papritz ==
# these are the colourmaps used in the Papritz paper
cols_uptakes = ["#FFFFFF", "#56cefb", "#00c5ff", "#00a2f7", "#1a69e4", "#ffa0b6", "#ff67a3", "#ff309b", "#cb0081", "#8e007a"]
cols_precip = ["#ffffff", "#b4ffcb", "#00be9f", "#0097b6", "#0071b4", "#00199a", "#7a285a"]

moisture_uptake_cmap = colors.LinearSegmentedColormap.from_list("moisture_uptake_colourmap", cols_uptakes)
precip_cmap = colors.LinearSegmentedColormap.from_list("precip_cmap", cols_precip)

def moisture_uptake_plot_separate(SID, YEAR, MONTH, nc_paths, levels):
    """
    Plot gridded moisture uptake footprint and storm diagnostics as three timestamps (t = -24h, 0h, 24h) where 0h is when
    the storm enters the Arctic. 
    
    The function will first upload all of the moisture uptakes associated with the level(s) you specify. Then, it creates
    a grid with dimensions (day/hour, lats, lons) and will grid the moisture uptakes. After smoothing using a Gaussian mean
    filter, the function will reshape the grid from 31^2 km^2 to 10^5 km^2 (matching the Papritz paper). Finally, it will
    plot the grids. 
    
    Inputs:
    - SID, YEAR, MONTH = [int, int, int]; storm identifiers
    - levels = [int]; height above ground level (in meters) being inspected
    - nc_paths = [dict]; A dictionary with keys of var_types and values that are a list of tuples that are formatted
                            (filename, start_day, end_day)
                            
    Creates:
    - a 1x3 plot of moisture uptakes at the timestamps (-24h, 0h, 24h)
    
    Notes:
    - This function uses the helper functions load_moisture_csvs and get_dataset_for_time
    
    """
    MONTH = str(MONTH).zfill(2)
    if isinstance(levels, int):
        levels = [levels]
        
    # Load storm path
    storm_path = f"storm{SID}_{YEAR}{MONTH}.csv"
    _, STORM_YEAR, STORM_MONTH, STORM_DAY, STORM_HOUR, _, _, STORM_X, STORM_Y = np.genfromtxt(storm_path, delimiter=',', 
                                                                                              dtype = int, skip_header = 1).T
    STORM_LAT, STORM_LON = np.genfromtxt(storm_path, delimiter = ',', usecols = (5, 6), skip_header = 1).T

    IN_MASK = np.array([(CAO2[STORM_Y[ind], STORM_X[ind]] == 1) for ind in range(len(STORM_X))]).astype('bool')
    
    # finding the timestamps
    idx_0h = np.argmax(IN_MASK)
    # since the time is recorded in 3 hour increments, ±24h is ±8 indices ahead 
    idx_fwd24h = idx_0h + 8
    idx_back24h = idx_0h - 8
    study_idx = [idx_fwd24h, idx_0h, idx_back24h]
   
    # A list of the (day, hour) combination of the 3 timestamps we're studying
    day_hour_combo = [(STORM_DAY[i], STORM_HOUR[i]) for i in study_idx]
    
    # upload the moisture uptake data
    data = []
    for level in levels:
        print(f"== Level {level} ==")
        level_data = load_all_moisture_csvs(SID, YEAR, MONTH, level)
        data.append(level_data[:, 2:]) # only saving the [day, hour, lat, lon, height, pressure] columns
        
    data = np.concatenate(data)
    
    # set up grids
    lats_range = np.arange(-90, 90, 0.5)
    lons_range = np.arange(-180, 180, 1)
    lon_grid, lat_grid = np.meshgrid(lons_range, lats_range)
    lat_size, lon_size = len(lats_range), len(lons_range)

    grid_ERA5 = np.zeros((3, lat_size, lon_size))
    smoothed_grid_ERA5 = np.zeros_like(grid_ERA5)
    norm_OGgrid_ERA5 = np.zeros_like(grid_ERA5)

    # Populate grid counts
    for row in data:  
        day, hour, lat_raw, lon_raw = int(row[0]), int(row[1]), row[2], row[3]
        lat, lon = math.floor(lat_raw * 4)/4, math.floor(lon_raw*4)/4
        # find the index location of the moisture uptake
        lat_idx = np.argmin(np.abs(lats_range - lat))
        lon_idx = np.argmin(np.abs(lons_range - lon))
        
        if (0 <= lat_idx < lat_size) and (0 <= lon_idx < lon_size) and (day, hour) in day_hour_combo:
            # find what the timestamp of the moisture uptake is
            day_idx = day_hour_combo.index((day, hour))
            grid_ERA5[day_idx, lat_idx, lon_idx] += 1
            
            
    # normalize each gridded count by the total counts at that day/hour 
    for d in range(len(grid_ERA5)):
        if np.sum(grid_ERA5[d]) == 0:
            norm_OGgrid_ERA5[d] = 0
        else:
            norm_OGgrid_ERA5[d] = (grid_ERA5[d] / np.sum(grid_ERA5[d]))*100 

    # apply mean filter
    for d in range(len(grid_ERA5)):
        # the size denotes what range to smooth over, here it is over 5deg grids
        smoothed_grid_ERA5[d] = gaussian_filter(norm_OGgrid_ERA5[d], sigma=5, mode='grid-wrap')

    # regridding from ERA5 31x31 km^2 resolution to target 10^5 km^2 resolution
    ERA5_res = 31**2
    target_res = 10**5
    norm_grid_ERA5 = smoothed_grid_ERA5 * target_res/ERA5_res

    plot_idx = np.concatenate((study_idx, study_idx))
    plot_day_hour = np.concatenate((day_hour_combo, day_hour_combo))
    pos_label = ["-24h", "0h", "24h", "-24h", "0h", "24h"]

    # define the plot
    fig, axs = plt.subplots(2, 3, figsize=(15, 10), subplot_kw={'projection': ccrs.NorthPolarStereo()}, layout='tight')
    axs = axs.flatten()
    fig.suptitle(f"Storm {SID}: ({YEAR}/{MONTH})")

    # === Per-hour plotting loop ===
    for i, (idx, day_hour, ax) in enumerate(zip(plot_idx, plot_day_hour, axs)):
        year, month = int(STORM_YEAR[idx]), int(STORM_MONTH[idx])
        day, hour = day_hour

        # check if the grid is empty for that time
        if max(norm_grid_ERA5[i%3].flatten()) < 1e-3:
            print(f"Skipping day {day}, hour {hour} because max is {max(norm_grid_ERA5[i%3].flatten())}")
            continue
        
        # find the circle around the storm center (500km radius)
        circle_lons, circle_lats = draw_circle(STORM_LON[idx], STORM_LAT[idx], 500) 
        
        # access the sea surface temperature (sst) and mean sea level pressure (msl) for the timestamp
        time_val = np.datetime64(f"{year}-{str(month).zfill(2)}-{str(day).zfill(2)}T{str(hour).zfill(2)}:00")
        ds_precip_sst_msl = get_dataset_for_time(nc_paths, "precip_SST_SLP", time_val)
        time_ind = np.argmin(np.abs(ds_precip_sst_msl.valid_time.values - time_val))
        sst = ds_precip_sst_msl.sst.values[time_ind]
        msl = ds_precip_sst_msl.msl.values[time_ind] * 0.01
        
        # find the plotting levels for the sst and msl
        levels_sst = np.arange(np.nanmin(sst), np.nanmax(sst), 3)
        levels_msl = np.arange(np.min(msl), np.max(msl), 5)
        ds_lon_grid, ds_lat_grid = np.meshgrid(ds_precip_sst_msl.longitude.values, ds_precip_sst_msl.latitude.values)

        # create figure labels
        if i < 3:
            ax.set_title(f"Day {day} / Hour {hour}\n t = {pos_label[i]}")
        else:
            ax.set_title(f"t = {pos_label[i]}")
    
        # add basic features (coastlines, gridlines, storm track, storm center, circle around center, extent)
        ax.add_feature(cfeature.COASTLINE, edgecolor='black', alpha = 0.6, zorder = 2)
        ax.gridlines(draw_labels=False, linestyle='dotted') 
        ax.plot(STORM_LON, STORM_LAT, color = 'white', lw = 1.5, zorder = 3,
                path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()], transform = ccrs.Geodetic())            
        ax.scatter(STORM_LON[idx], STORM_LAT[idx], color = 'black', lw = 1, zorder = 5, marker = 'x', 
           transform = ccrs.Geodetic(), s = 75)
        ax.scatter(STORM_LON[idx], STORM_LAT[idx], facecolor = 'white', edgecolor = 'black', marker = "o",
           transform = ccrs.Geodetic(), zorder = 4, lw = 1, s = 150)
        ax.plot(circle_lons, circle_lats, transform=ccrs.Geodetic(), color='black', linestyle='-', linewidth=1, zorder = 3)
        ax.set_extent([-180, 180, 25.0, 90.0], crs=ccrs.PlateCarree())
    
        # in the top row, the plot has the precipitation and slp
        if i < 3:
            precip_levels = np.arange(0.0, 5.1, 0.25)
            # if the precipitation is saved in a separate nc file, access it here using this code:
            #ds_precip = get_dataset_for_time(nc_paths, "precip", time_val)
            # assuming that your precipitation and SST/SLP nc file follows the same grid/time limitations, just replace
            # ds_precip_sst_msl.tp.values[time_ind]*1000 below with ds_precip.tp.values[time_ind]*1000            
            cn0 = ax.contourf(ds_lon_grid, ds_lat_grid, ds_precip_sst_msl.tp.values[time_ind]*1000, levels = precip_levels,
                                  cmap = precip_cmap, transform = ccrs.PlateCarree(), extend = 'both', zorder=1)
            fig.colorbar(cn0, orientation = 'horizontal', ax = ax, shrink = 0.8,
                         label = "precipitation (mm/h)")
            vc0 = ax.contour(ds_lon_grid, ds_lat_grid, ds_precip_sst_msl.msl.values[time_ind]*0.01, 
                                 levels = levels_msl, colors=['lightgrey'], linewidths = 0.75, transform=ccrs.PlateCarree(), zorder=1)        
        
        # in the bottom row, the plot has the moisture uptakes and sst
        else:
            cn1 = ax.contourf(lon_grid, lat_grid, norm_grid_ERA5[i%3], transform=ccrs.PlateCarree(), 
                                  cmap = moisture_uptake_cmap, extend = 'both', zorder = 1)
            fig.colorbar(cn1, orientation = "horizontal", ax = ax, shrink = 0.8,
                         label = r"[% moisture uptake $(10^{5} km^{2})^{-1}$]")

            vc1 = ax.contour(ds_lon_grid, ds_lat_grid, ds_precip_sst_msl.sst.values[time_ind], 
                                 levels = levels_sst, colors=['lightgrey'], linewidths = 0.75, transform=ccrs.PlateCarree(), zorder = 1)

    plt.show()
    

# these are the colourmaps used in the Papritz paper
cols_uptakes = ["#FFFFFF", "#56cefb", "#00c5ff", "#00a2f7", "#1a69e4", "#ffa0b6", "#ff67a3", "#ff309b", "#cb0081", "#8e007a"]
moisture_uptake_cmap = colors.LinearSegmentedColormap.from_list("moisture_uptake_colourmap", cols_uptakes)

def moisture_uptake_plot_together(SID, YEAR, MONTH, levels):
    """
    Plots gridded moisture uptakes for all times for a particular time at specified levels. 
    
    The function will first upload all of the moisture uptakes associated with the level(s) you specify. Then, it creates
    a grid with dimensions (lats, lons) and will grid the moisture uptakes. After smoothing using a Gaussian mean
    filter, the function will reshape the grid from 31^2 km^2 to 10^5 km^2 (matching the Papritz paper). Finally, it will
    plot the grid. 
    
    Inputs:
    - SID, YEAR, MONTH = [int, int, int]; storm identifiers
    - levels = [int]; height above ground level (in meters) being inspected

    Creates:
    - a 1x1 plot of gridded moisture uptakes
    
    Notes:
    - This function uses the helper 'get_moisture_uptake_csvs'
    
    """
    if isinstance(levels, int):
        levels = [levels]
        
    MONTH = str(MONTH).zfill(2)
        
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection=ccrs.NorthPolarStereo())
    ax.set_extent([-180, 180, 25, 90], crs=ccrs.PlateCarree())
    
    # Add title, coasts, and grids
    ax.set_title(f"Storm {SID}: {YEAR}/{MONTH}", fontsize = 21)
    ax.add_feature(cfeature.COASTLINE, edgecolor='black', alpha=0.6, zorder = 2)
    ax.gridlines(draw_labels=False, linestyle='dotted') 

    # Load storm path
    STORM_YEAR, STORM_MONTH, STORM_DAY, STORM_HOUR, STORM_LAT, STORM_LON = np.genfromtxt(f"storm{SID}_{YEAR}{MONTH}.csv",
                                                                delimiter=',', skip_header = 1, usecols = range(1, 7)).T
    
    # plot storm track
    ax.plot(STORM_LON, STORM_LAT, color = 'white', lw = 2, zorder = 3,
            path_effects=[pe.Stroke(linewidth=4, foreground='black'), pe.Normal()], transform = ccrs.Geodetic())            
    
    data = []
    # Load moisture uptake data
    for level in levels:
        print(f"== Level {level} ==")
        level_data = load_all_moisture_csvs(SID, YEAR, MONTH, level)
        data.append(level_data[:, 2:]) # only saving the [day, hour, lat, lon, height, pressure] columns
        
    data = np.concatenate(data)
    
    # define the grid
    lats_range = np.arange(-90, 90, 0.5)
    lons_range = np.arange(-180, 180, 1)
    lon_grid, lat_grid = np.meshgrid(lons_range, lats_range)
    lat_size, lon_size = len(lats_range), len(lons_range)

    grid_ERA5 = np.zeros((lat_size, lon_size))

    # Populate grid counts
    for row in data:
        lat_raw, lon_raw = row[2], row[3]
        lat, lon = math.floor(lat_raw * 4)/4, math.floor(lon_raw*4)/4
        # find lat/lon indices
        lat_idx = np.argmin(np.abs(lats_range - lat))
        lon_idx = np.argmin(np.abs(lons_range - lon))
        grid_ERA5[lat_idx, lon_idx] += 1

    # Normalize
    norm_OGgrid_ERA5 = grid_ERA5 / np.sum(grid_ERA5) * 100

    # Filter
    smoothed_grid_ERA5 = gaussian_filter(norm_OGgrid_ERA5, sigma=5, mode='grid-wrap')

    # regridding from ERA5 31x31 km^2 resolution to target 10^5 km^2 resolution
    ERA5_res = 31**2
    target_res = 10**5
    norm_grid_ERA5 = smoothed_grid_ERA5 * target_res/ERA5_res
    
    # plot moisture uptakes
    cn1 = ax.contourf(lon_grid, lat_grid, norm_grid_ERA5, transform=ccrs.PlateCarree(), cmap = moisture_uptake_cmap, 
                      extend = 'both', zorder = 1)
    fig.colorbar(cn1, orientation = "horizontal", ax = ax, shrink = 0.75,
                 label = fr"[% moisture uptake $(10^{5} km^{2})^{{-1}}$]")
                
    plt.tight_layout()
    plt.show()

# == Figure 4 in Papritz ==
def storm_trajectory_plot_together(SID, YEAR, MONTH, nc_paths, levels, choosing_val = 4):
    '''
    Plots storm trajectories and changes in specific humidity (Δq) over 12 hours for random points along each trajectory
    for a given storm event.

    Inputs:
    - SID, YEAR, MONTH = [int, int, int]; storm identifiers
    - nc_paths = [dict]; A dictionary with keys of var_types and values that are a list of tuples that are formatted
                            (filename, start_day, end_day)
    - levels = [int]; height above ground level (in meters) being inspected
    - choosing_val = [int]; Sampling rate for the parcel trajectories. Every `choosing_val`th trajectory will be plotted.
        Default is 4.

    Notes:
    ------
    - Requires storm track CSV and corresponding trajectory `.npz` files.
    - Assumes existence of a helper function `get_dataset_for_time(nc_paths, var_name, time)` 
      to load NetCDF datasets.   
    '''
    if isinstance(levels, int):
        levels = [levels]
        
    MONTH = str(MONTH).zfill(2)
    delta_q_arr = []
    
    # define figure
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection=ccrs.NorthPolarStereo())
    ax.set_extent([-180, 180, 25, 90], crs=ccrs.PlateCarree())
    
    # Add title, coasts, and grids
    ax.set_title(f"Storm {SID}: {YEAR}/{MONTH}", fontsize = 21)
    ax.add_feature(cfeature.COASTLINE, edgecolor='black', alpha=0.6, zorder = 2)
    ax.gridlines(draw_labels=False, linestyle='dotted') 
    
    # finding the main storm trajectory and location 
    STORM_YEAR, STORM_MONTH, STORM_DAY, STORM_HOUR, STORM_LAT, STORM_LON = np.genfromtxt(f"storm{SID}_{YEAR}{MONTH}.csv",
                                                                    delimiter=',', skip_header = 1, usecols = range(1, 7)).T
    # Convert date/time data to integers
    STORM_YEAR, STORM_MONTH, STORM_DAY, STORM_HOUR = map(lambda x: x.astype('int64'),
        [STORM_YEAR, STORM_MONTH, STORM_DAY, STORM_HOUR])
    
    # plot storm track
    ax.plot(STORM_LON, STORM_LAT, color = 'white', lw = 1.5, zorder = 4,
                path_effects=[pe.Stroke(linewidth=5, foreground='black'), pe.Normal()], 
                transform = ccrs.Geodetic())
        
    data_list = []
    # == getting the trajectories from the precipitating parcels ==  
    for level in levels:
        traj_path = f"traj_data_storm{SID}_{YEAR}{MONTH}/traj_storm{SID}_height{level}m.npz"
        
        # load trajectory data
        datafile = np.load(traj_path)
        alldata = datafile["data"]                   
        if alldata.size == 0:
            print("No data found.")
        
        # separate trajectories by ID
        ID = alldata[:, 0]
        _, indices = np.unique(ID, return_index=True)
        indices.sort()
        level_data = np.split(alldata[:, 1:], indices[1:])  # Skip ID column 
        
        #columns: [year, month, day, hour, lat, lon, height, pressure]
        for traj_num, traj in enumerate(level_data):
            if traj_num % choosing_val == 0:
                # Plot trajectory
                ax.plot(traj[:, 5], traj[:, 4], transform=ccrs.Geodetic(), color='tab:blue', alpha=0.5, lw=0.25)

                # Choose a random point on the trajectory
                random_idx = random.randint(0, len(traj) - 1)
                year_val, month_val, day_val, hour_val, lat_val, lon_val, _, press_val = traj[random_idx]
                
                # Format datetime values
                year_str = str(int(year_val))
                month_str = str(int(month_val)).zfill(2)
                day = str(int(day_val)).zfill(2)
                hour = str(int(hour_val)).zfill(2)
                time_val = np.datetime64(f"{year_str}-{month_str}-{day}T{hour}:00")
                time_minus12h_val = time_val - np.timedelta64(12, "h")

                # Load current and previous (12-hour earlier) humidity datasets
                with get_dataset_for_time(nc_paths, 'HumidityCWC', time_val) as ds, \
                     get_dataset_for_time(nc_paths, 'HumidityCWC', time_minus12_val) as ds_prev:

                    # Find nearest indices for time, pressure, lat, lon
                    time_ind = np.argmin(np.abs(ds.valid_time.values - time_val))
                    time_prev_ind = np.argmin(np.abs(ds_prev.valid_time.values - time_minus12h_val))
                    press_ind = np.argmin(np.abs(ds.pressure_level.values - press_val))
                    lat_ind = np.argmin(np.abs(ds.latitude.values - lat_val))
                    lon_ind = np.argmin(np.abs(ds.longitude.values - lon_val))

                    # Retrieve specific humidity at both times
                    q = ds['q'].isel(valid_time=time_ind, pressure_level=press_ind,
                                     latitude=lat_ind, longitude=lon_ind).values.item()
                    q_prev = ds_prev['q'].isel(valid_time=time_prev_ind, pressure_level=press_ind,
                                               latitude=lat_ind, longitude=lon_ind).values.item()

                    # Store the change in specific humidity (Δq in g/kg)
                    delta_q = (q - q_prev) * 1000
                    delta_q_arr.append([lon_val, lat_val, delta_q])


    delta_q_arr = np.array(delta_q_arr)
    print(f"{len(delta_q_arr)} trajectory found")  
    
    try:
        # define a diverging colourmap
        divnorm = colors.TwoSlopeNorm(vmin=min(delta_q_arr[:, 2]), vcenter=0, vmax=max(delta_q_arr[:, 2]))
        # plot the delta q values
        sc = ax.scatter(delta_q_arr[:, 0], delta_q_arr[:, 1], c = delta_q_arr[:, 2], cmap='seismic', 
                            transform=ccrs.Geodetic(), linewidths=0.25, edgecolor='black', zorder = 3, norm = divnorm)
        
    except: # if the function can't create a diverging colourbar
        # plot the delta q values
        sc = ax.scatter(delta_q_arr[:, 0], delta_q_arr[:, 1], c = delta_q_arr[:, 2], cmap='seismic', 
                            transform=ccrs.Geodetic(), linewidths=0.25, edgecolor='black', zorder = 3)
        
    cbar = plt.colorbar(sc, orientation = "horizontal", ax = ax, shrink = 1, extend = 'both')
    cbar.set_label("Δq [g/kg]", fontsize = 14)     
    
    plt.tight_layout()
    plt.show()
    
def storm_trajectory_plot_together(SID, YEAR, MONTH, nc_paths, levels, choosing_val = 4):
    '''
    Plots storm trajectories and changes in specific humidity (Δq) over 12 hours for random points along each trajectory
    for a given storm event.

    Inputs:
    - SID, YEAR, MONTH = [int, int, int]; storm identifiers
    - nc_paths = [dict]; A dictionary with keys of var_types and values that are a list of tuples that are formatted
                            (filename, start_day, end_day)
    - levels = [int]; height above ground level (in meters) being inspected
    - choosing_val = [int]; Sampling rate for the parcel trajectories. Every `choosing_val`th trajectory will be plotted.
        Default is 4.

    Notes:
    ------
    - Requires storm track CSV and corresponding trajectory `.npz` files.
    - Assumes existence of a helper function `get_dataset_for_time(nc_paths, var_name, time)` 
      to load NetCDF datasets.   
    '''
    if isinstance(levels, int):
        levels = [levels]
        
    MONTH = str(MONTH).zfill(2)
    delta_q_arr = []
    
    # define figure
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection=ccrs.NorthPolarStereo())
    ax.set_extent([-180, 180, 25, 90], crs=ccrs.PlateCarree())
    
    # Add title, coasts, and grids
    ax.set_title(f"Storm {SID}: {YEAR}/{MONTH}", fontsize = 21)
    ax.add_feature(cfeature.COASTLINE, edgecolor='black', alpha=0.6, zorder = 2)
    ax.gridlines(draw_labels=False, linestyle='dotted') 
    
    # finding the main storm trajectory and location 
    STORM_YEAR, STORM_MONTH, STORM_DAY, STORM_HOUR, STORM_LAT, STORM_LON = np.genfromtxt(f"storm{SID}_{YEAR}{MONTH}.csv",
                                                                    delimiter=',', skip_header = 1, usecols = range(1, 7)).T
    # Convert date/time data to integers
    STORM_YEAR, STORM_MONTH, STORM_DAY, STORM_HOUR = map(lambda x: x.astype('int64'),
        [STORM_YEAR, STORM_MONTH, STORM_DAY, STORM_HOUR])
    
    # plot storm track
    ax.plot(STORM_LON, STORM_LAT, color = 'white', lw = 1.5, zorder = 4,
                path_effects=[pe.Stroke(linewidth=5, foreground='black'), pe.Normal()], 
                transform = ccrs.Geodetic())
        
    data_list = []
    # == getting the trajectories from the precipitating parcels ==  
    for level in levels:
        traj_path = f"traj_data_storm{SID}_{YEAR}{MONTH}/traj_storm{SID}_height{level}m.npz"
        
        # load trajectory data
        datafile = np.load(traj_path)
        alldata = datafile["data"]                   
        if alldata.size == 0:
            print("No data found.")
        
        # separate trajectories by ID
        ID = alldata[:, 0]
        _, indices = np.unique(ID, return_index=True)
        indices.sort()
        level_data = np.split(alldata[:, 1:], indices[1:])  # Skip ID column 
        
        #columns: [year, month, day, hour, lat, lon, height, pressure]
        for traj_num, traj in enumerate(level_data):
            if traj_num % choosing_val == 0:
                # Plot trajectory
                ax.plot(traj[:, 5], traj[:, 4], transform=ccrs.Geodetic(), color='tab:blue', alpha=0.5, lw=0.25)

                # Choose a random point on the trajectory
                random_idx = random.randint(0, len(traj) - 1)
                year_val, month_val, day_val, hour_val, lat_val, lon_val, _, press_val = traj[random_idx]
                
                # Format datetime values
                year_str = str(int(year_val))
                month_str = str(int(month_val)).zfill(2)
                day = str(int(day_val)).zfill(2)
                hour = str(int(hour_val)).zfill(2)
                time_val = np.datetime64(f"{year_str}-{month_str}-{day}T{hour}:00")
                time_minus12_val = (time_val - np.timedelta64(12, "h"))

                # Load current and previous (12-hour earlier) humidity datasets
                with get_dataset_for_time(nc_paths, 'HumidityCWC', time_val) as ds, \
                     get_dataset_for_time(nc_paths, 'HumidityCWC', time_minus12_val) as ds_prev:

                    # Find nearest indices for time, pressure, lat, lon
                    time_ind = np.argmin(np.abs(ds.valid_time.values - time_val))
                    time_prev_ind = np.argmin(np.abs(ds_prev.valid_time.values - time_minus12_val))
                    press_ind = np.argmin(np.abs(ds.pressure_level.values - press_val))
                    lat_ind = np.argmin(np.abs(ds.latitude.values - lat_val))
                    lon_ind = np.argmin(np.abs(ds.longitude.values - lon_val))

                    # Retrieve specific humidity at both times
                    q = ds['q'].isel(valid_time=time_ind, pressure_level=press_ind,
                                     latitude=lat_ind, longitude=lon_ind).values.item()
                    q_prev = ds_prev['q'].isel(valid_time=time_prev_ind, pressure_level=press_ind,
                                               latitude=lat_ind, longitude=lon_ind).values.item()

                    # Store the change in specific humidity (Δq in g/kg)
                    delta_q = (q - q_prev) * 1000
                    delta_q_arr.append([lon_val, lat_val, delta_q])


    delta_q_arr = np.array(delta_q_arr)
    print(f"{len(delta_q_arr)} trajectory found")  
    
    try:
        # define a diverging colourmap
        divnorm = colors.TwoSlopeNorm(vmin=min(delta_q_arr[:, 2]), vcenter=0, vmax=max(delta_q_arr[:, 2]))
        # plot the delta q values
        sc = ax.scatter(delta_q_arr[:, 0], delta_q_arr[:, 1], c = delta_q_arr[:, 2], cmap='seismic', 
                            transform=ccrs.Geodetic(), linewidths=0.25, edgecolor='black', zorder = 3, norm = divnorm)
        
    except:
        # plot the delta q values
        sc = ax.scatter(delta_q_arr[:, 0], delta_q_arr[:, 1], c = delta_q_arr[:, 2], cmap='seismic', 
                            transform=ccrs.Geodetic(), linewidths=0.25, edgecolor='black', zorder = 3)
        
    cbar = plt.colorbar(sc, orientation = "horizontal", ax = ax, shrink = 1, extend = 'both')
    cbar.set_label("Δq [g/kg]", fontsize = 14)     
    
    plt.tight_layout()
    plt.show()