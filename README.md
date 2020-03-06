# cyclonetracking
Lagrangian cyclone tracking algorithm developed while at the National Snow and Ice Data Center
Inputs include several raster data sets: sea level pressure fields, a digital elevation model, arrays of latitude, longitude, and x and y distances across the input grid, and (optionally) total and large-scale precipitation. A suite of parameters set by the user are also needed, and the algorithm assumes that the user has already a) set up some output directories and b) regridded all inputs to an equal-area grid.  Polar grids (e.g., the EASE2 grid) are ideal. An example script (C2E_Reprojection3_SLP.py) is included in Version 11.1 Scripts that shows how ERA-Interim data were converted.

Please email Alex Crawford at acrawford0927 -at- gmail.com with questions.  If you use this algorithm, please a) let me know about the project and b) cite the paper that introduced the algortithm:

If using Version 10.3 (2016):
Crawford AD & MC Serreze (2016). Does the Arctic frontal zone impact summer cyclone activity in the Arctic Ocean? Journal of Climate, 29, 4977-4993, doi:10.1175/JCLI-D-15-0755.s1.

If using the latest update that yielded Version 11.1 (2020):
Crawford, A.D., K.E. Alley, A.M. Cooke, and M.C. Serreze, 2020: Synoptic Climatology of Rain-on-Snow Events in Alaska. Mon. Wea. Rev., 148, 1275–1295, https://doi.org/10.1175/MWR-D-19-0311.1 
