# Dependencies

This is an inventory from source imports, not a tested environment specification. No software was installed and no package versions are claimed as validated.

| Component | Dependencies |
| --- | --- |
| Script 01 | NumPy, pandas, GDAL, Pillow, PyCUDA; NVIDIA CUDA toolkit, driver, GPU |
| Script 02 | NumPy, pandas, GDAL, Pillow, SciPy, scikit-learn, statsmodels, imbalanced-learn |
| Script 03 | R, ggplot2, Rmisc |
| Script 04 | NumPy, pandas, GDAL, Pillow, SciPy, scikit-gstat |
| Script 05 | R, spam, fields, sp, gstat, ggplot2 |

Some packages are required by unconditional helper-module imports even when the numbered script does not call the relevant functions. In particular, script 02 imports scikit-learn, statsmodels, and imbalanced-learn through `model_funcs_terra_aqua.py`.

Install into a dedicated Anaconda environment on the intended computing machine, then verify imports, GDAL raster access, R packages, and a small CUDA compilation before running analyses. A validated environment file should be added after that check. NumPy version compatibility deserves testing: legacy helpers use `np.bool8`, and script 04 converts one-element arrays to scalar integers.

No CPU fallback for catchment tracing is provided. Do not assume that a macOS laptop can execute the CUDA step. CPU scripts can be assessed independently once their data and paths are prepared.
