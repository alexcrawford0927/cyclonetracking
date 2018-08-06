# cyclonetracking
Lagrangian cyclone tracking algorithm developed while at the National Snow and Ice Data Center
Input include several raster data sets: sea level pressure fields, a digital elevation model, arrays of latitude and longitude, and (optionally) total and large-scale precipitation. A suite of parameters set by the user are also needed, and the algorithm assumes that the user has already a) set up the output directories and b) regridded all inputs to an equal-area grid.  Polar grids (e.g., the EASE2 grid) are ideal.

Please email Alex Crawford at acrawford0927 -at- gmail.com with questions.  If you use this algorithm, please a) let me know about the project and b) cite the paper that introduced the algortithm:
Crawford AD & MC Serreze (2016). Does the Arctic frontal zone impact summer cyclone activity in the Arctic Ocean? Journal of Climate, 29, 4977-4993, doi:10.1175/JCLI-D-15-0755.s1.
