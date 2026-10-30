# Example data

Paths below are relative to this directory. Original filenames and contents are retained.

| File | Role | Size (bytes) |
| --- | --- | ---: |
| `env_var/pixel_coords/riv_pix_CH_example.csv` | Example river-pixel coordinates | 2,296,310 |
| `env_var/hydro_map/topo_alti3d_dem_mean_flow_direct.tif` | D8 flow directions for catchment calculation | 17,802,106 |
| `env_var/hydro_map/river_pixel_template_fish_sr.tif` | River-pixel mask; scripts select values > 0 | 8,465,168 |

The flow-direction raster is accompanied by `.tfw`, `.tif.aux.xml`, `.tif.xml`, `.tif.vat.dbf`, and `.tif.vat.cpg` files. Keep them together pending final packaging review.

## Coordinate table

The table contains 35,128 records, with unique, nonblank `SiteID` values. Columns are an unnamed exported index, `ID`, `SiteID`, `CHX_ori`, `CHY_ori`, `CHX_calib`, `CHY_calib`, `catch_id`, and `modified`. No blank fields were found in the supplied table. Treat identifiers as strings; the unnamed index and ID are not interchangeable with row position.

Scripts use the calibrated coordinates for catchment outlets and compare original and calibrated coordinates for shifts. The manuscript specifies CH1903+ / LV95 coordinates and 25 m cells. The supplied world file independently records 25 m horizontal and −25 m vertical spacing. Embedded raster CRS, alignment, NoData values, and pixel validity still require GDAL-based validation before computation. The meanings of `catch_id` and all `modified` codes should be confirmed by the data producer.

The example is smaller than the manuscript's 351,289 sampled river pixels. Do not infer the full sample size from the script slice endpoint of 351292.

## Biodiversity data source

The biodiversity data used in this study come from SDMapCH v1.3, described in:

Adde, A., et al. (2025). SDMapCH: a Comprehensive database of >7,500 modelled species habitat suitability maps for Switzerland. *Scientific Data*, **12**, 1752. [Read the article and its data-access information](https://www.nature.com/articles/s41597-025-06037-x).

The manuscript identifies Swiss ALTI3D as the elevation source. Third-party datasets retain their respective licenses and attribution requirements.
