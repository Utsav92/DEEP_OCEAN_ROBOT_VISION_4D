# Licence and data-use notes

* **Terrain data** - NOAA National Centers for Environmental Information (NCEI) Continuously Updated Digital Elevation
  Model (CUDEM) 1/9 arc-second tiles `ncei19_n37x00_w122x00_2023v1` and `ncei19_n37x00_w122x25_2023v1` (and neighbours),
  fetched through the NCEI *DEM global mosaic* ImageServer. NOAA data are US Government works distributed under the NCEI
  open-data policy (https://www.ncei.noaa.gov/about/data-policy): no use restrictions, attribution to NOAA NCEI requested.
  The per-tile metadata URL (see `output/diagnostics/dataset_manifest.json`) should be checked before redistribution of
  the raw tiles. The vertical datum of the CUDEM tiles is NAVD88; elevations are used as supplied.
* **Everything else is synthetic.** The multibeam, forward-looking sonar, DVL, AHRS, pressure data, the navigation error
  and the four inserted seabed objects are produced by `simulation/` and are labelled `SIMULATED` in every output.
  They are not measurements and carry no third-party licence.
* **GEBCO** (https://www.gebco.net/) data may be used with attribution "GEBCO Compilation Group (year) GEBCO Grid"; the
  loader supports its NetCDF layout but no GEBCO file was used for the results in this repository.
* **MB-System** (GPL) and **GTSAM** (BSD) are not bundled and not required.
* TouchDesigner: the project was built and tested with a non-commercial licence (1280 px TOP cap).
* Code in this repository: MIT.
