# cyclonetracking
The CEOS/NSIDC Extratroipcal Cyclone Tracking (CNECT) algorithm identifies cyclones as closed low pressure centers satisfying a minimum intensity threshold, then relates cyclones in successive observation times to generate tracks that record the entire life history of individual cyclone centers. The algorithm also keeps track of various intensity measures, propagation characteristics, and interactions between cyclone centers. Those interactions can be used by a follow-on code to identify multi-center storm systems, which is especially helpful when working with high-resolution data.

Inputs include several raster data sets: sea level pressure fields, a digital elevation model, arrays of latitude, longitude, and x and y distances across the input grid. A suite of parameters set by the user are also needed, and unless working exclusively at latitudes lower than about 50°N, it is advised that input SLP fields be re-gridded to an equal-area polar projection (e.g., the EASE2 grid). An example script is included in Version 12 Scripts that shows how ERA5 data were converted.

Note that if you just want to work with output from this algorithm from the historical record and don't want to run the code yourself, results using ERA5 as the input are available at https://canwin-datahub.ad.umanitoba.ca/data/project/cnect and updated periodically.

Please email Alex Crawford at alex.crawford -at- umanitoba.ca with questions.  If you use this algorithm, please a) let me know about the project and b) cite the paper that introduced the algortithm:

If using Version 10.x (2016):
Crawford A.D. & M.C. Serreze (2016). Does the Arctic frontal zone impact summer cyclone activity in the Arctic Ocean? Journal of Climate, 29, 4977-4993, doi:10.1175/JCLI-D-15-0755.s1.

If using Version 12.x or 13.x (2021, 2023):
Crawford, A. D., Schreiber, E. A. P., Sommer, N., Serreze, M. C., Stroeve, J. C., & Barber, D. G. (2021). Sensitivity of Northern Hemisphere Cyclone Detection and Tracking Results to Fine Spatial and Temporal Resolution using ERA5. Monthly Weather Review, 149, 2581-2589. https://doi.org/10.1175/mwr-d-20-0417.1

