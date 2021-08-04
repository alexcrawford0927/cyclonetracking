# cyclonetracking
Lagrangian cyclone tracking algorithm developed while at the National Snow and Ice Data Center
Inputs include several raster data sets: sea level pressure fields, a digital elevation model, arrays of latitude, longitude, and x and y distances across the input grid. A suite of parameters set by the user are also needed, and the algorithm assumes that the user has already a) set up some output directories and b) regridded all inputs to an equal-area grid.  Polar grids (e.g., the EASE2 grid) are ideal. An example script is included in Version 12 Scripts that shows how ERA5 data were converted.

Please email Alex Crawford at acrawford0927 -at- gmail.com with questions.  If you use this algorithm, please a) let me know about the project and b) cite the paper that introduced the algortithm:

If using Version 10.x (2016):
Crawford A.D. & M.C. Serreze (2016). Does the Arctic frontal zone impact summer cyclone activity in the Arctic Ocean? Journal of Climate, 29, 4977-4993, doi:10.1175/JCLI-D-15-0755.s1.

If using Version 12.x (2021):
Crawford, A. D., Schreiber, E. A. P., Sommer, N., Serreze, M. C., Stroeve, J. C., & Barber, D. G. (2021). Sensitivity of Northern Hemisphere Cyclone Detection and Tracking Results to Fine Spatial and Temporal Resolution using ERA5. Monthly Weather Review, 149, 2581-2589. https://doi.org/10.1175/mwr-d-20-0417.1

