# **Project: Spatial Associations of Aquatic and Terrestrial Biodiversity across Landscape Scales**

**Repository**: Speed2Zero_Terrestrial_Aquatic_Biodiv

**Code author and contact**: Heng Zhang — heng.zhang@eawag.ch

**Date**: 10/30/2026

This repository contains Python, R, and CUDA scripts associated with the SPEED2ZERO study of terrestrial–aquatic biodiversity associations in Switzerland.

# **Abstract**

Terrestrial and aquatic ecosystems are tightly connected, creating spatial patterns of biodiversity that reflect land-water associations. Understanding the scale and strength of spatial associations between aquatic and terrestrial biodiversity is of both fundamental ecological interest and relevance to land management. Yet, it remains unclear whether and how far these associations extend across broad landscape scales. We used modeled habitat-suitability maps for about 7,500 species covering both terrestrial and aquatic realms in Switzerland to systematically assess how predicted species richness in the two realms is spatially associated, including both overall predicted biodiversity patterns and differences among specific taxonomic groups. By developing a spatially explicit model, we identified consistent terrestrial-aquatic spatial associations between overall terrestrial and aquatic biodiversity at the landscape scale, extending up to 50–219 km upstream, with their strength varying among taxonomic groups. We detected strong associations between terrestrial biodiversity signals and aquatic vascular plants and aquatic mammals, yet weaker associations between terrestrial species richness and aquatic mayflies, stoneflies, and caddisflies. Such cross-ecosystem spatial associations were also evident between particular terrestrial-aquatic group pairs, such as amphibians and reptiles, which have similar habitats, and dragonflies and birds, which may be functionally related. Our findings suggest that biodiversity on land and in water is interconnected at landscape scales, even within complex landscape mosaics. This highlights the need to give aquatic and terrestrial biodiversity equal attention in conservation planning and management.

# **Platform and Packages**

The project uses Python and CUDA for catchment calculation, Python for distance-weighted richness and spatial autocorrelation calculations, and R for statistical analysis and figures. An Anaconda-managed Python environment is recommended. Script 01 requires an NVIDIA GPU, a compatible CUDA toolkit (including the compiler), and PyCUDA. The other numbered scripts use CPU computation.

Python packages imported by the supplied code:

_numpy, pandas, GDAL (osgeo), Pillow (PIL), PyCUDA, SciPy, scikit-learn, statsmodels, imbalanced-learn, scikit-gstat (skgstat)_

R packages loaded by the supplied scripts:

_ggplot2, Rmisc, spam, fields, sp, gstat_

See [DEPENDENCIES.md](docs/DEPENDENCIES.md) for package-to-script mappings. Package versions and a working CUDA configuration have not yet been validated for this release.

# **Code and Data**

The main scripts are in `code/`; shared Python functions and CUDA kernels are in `code/src/`. The intended processing order is 01 → 02 → 03, with a separate spatial autocorrelation branch 04 → 05. Set the local paths and analysis settings in each script before use.

**01_calc_catchment_HS.py**: Calculates upstream catchments and flow-distance rasters for river-pixel coordinates using a D8 flow-direction raster. NVIDIA GPU required. The supplied input is derived from Swiss ALTI3D, despite the historical HydroSHEDS name in the script.

**02_calc_C_r_dist_matrix.py**: Calculates upstream terrestrial richness contributions across 59 effective flow distances (0.1–400 km), using catchments and species-richness rasters. CPU computation.

**03_est_terra_aqua_biodiv_effect_overall.R**: Fits overall terrestrial–aquatic associations across distances using 100 subsets and within-subset permutations for a null comparison; plots adjusted R² curves.

**04_calc_spatial_autocorrelation.py**: Samples pixel pairs, calculates empirical semivariograms, and fits an exponential variogram with SciPy curve fitting.

**05_plot_spatial_variogram.R**: Plots the empirical and fitted variograms using script 04 outputs.

The `data/` folder contains 35,128 example river-pixel coordinates, a flow-direction raster with sidecars, and a river-pixel mask. See the [data inventory](data/README.md).

The biodiversity data used in this study come from **SDMapCH v1.3**, described by [Adde et al. (2025)](https://www.nature.com/articles/s41597-025-06037-x). Please refer to that article for the dataset description and data-access information.

# **Reference**

*Spatial associations of aquatic and terrestrial biodiversity across landscape scales.* Manuscript, 2026.

Adde, A., et al. (2025). SDMapCH: a Comprehensive database of >7,500 modelled species habitat suitability maps for Switzerland. *Scientific Data*, **12**, 1752. [https://doi.org/10.1038/s41597-025-06037-x](https://doi.org/10.1038/s41597-025-06037-x).

This study is part of the SPEED2ZERO Joint Initiative, supported by the ETH Board under the Joint Initiatives scheme, as acknowledged in the supplied manuscript.

# **License**

The code is released under the [MIT License](LICENSE). Third-party datasets retain their respective licenses and attribution requirements.
