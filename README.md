# Figures: Multi-model Projection of Hydropower potential in a Changing Climate: A case study of Kainji Dam, Nigeria

Figures, captions, derived data and plotting code for Chapter 4 (Results) and the Appendix of the MSc dissertation by Olumide Emmanuel Awomewe (Department of Space Science and Applied Physics, University of Zimbabwe).

The study asks how precipitation, river flow and hydropower potential at Kainji Dam (Niger River, Nigeria; 760 MW) may change under SSP2-4.5 and SSP5-8.5.
The chain is: precipitation (10 NEX-GDDP-CMIP6 models) -> river flow (HMF-WA, 5 GCMs) -> turbine discharge -> power, P = eta rho g Q H
(efficiency 0.87, net head 33-42 m, mean 37.5 m, baseline discharge 739.51 m3/s, baseline potential 236.7 MW).

## What is here

| Item | Content |
|---|---|
| `captions.csv` | stand-alone caption, data files and script for every figure |
| `code/` | plotting scripts (provenance; they need the raw gridded data and path edits, see below) |

## Figures

| Figure | File | Content |
|---|---|---|
| 4.1 | `fig4_01_seasonal_precipitation_climatology.png` | Seasonal precipitation climatology |
| 4.2 | `fig4_02_annual_precipitation_timeseries.png` | Annual precipitation time series |
| 4.3 | `fig4_03_spi12_timeseries.png` | SPI-12 drought index |
| 4.4 | `fig4_04_annual_mean_precipitation_maps_chirps_cmip6.png` | Annual mean precipitation: CHIRPS vs CMIP6 |
| 4.5 | `fig4_05_monthly_climatology.png` | Mean monthly precipitation |
| 4.6 | `fig4_06_taylor_diagram.png` | Taylor diagram |
| 4.7 | `fig4_07a-d_*.png` | Projected precipitation change maps |
| 4.8 | `fig4_08_precipitation_change_timeseries.png` | Annual precipitation change, 1981-2100 |
| 4.9 | `fig4_09_hmfwa_vs_observed_seasonal_flow.png` | HMF-WA vs observed seasonal flow |
| 4.10 | `fig4_10_seasonal_flow_and_precipitation_future.png` | Future seasonal cycle of flow and precipitation |
| 4.11 | `fig4_11_hmfwa_mean_flow_map.png` | HMF-WA historical mean flow map |
| 4.12 | `fig4_12_hmfwa_kainji_annual_flow.png` | HMF-WA annual flow at Kainji, 1980-2014 |
| 4.13 | `fig4_13_flow_change_map_ssp245.png` | Flow change map, SSP2-4.5 |
| 4.14 | `fig4_14_flow_change_map_ssp585.png` | Flow change map, SSP5-8.5 |
| 4.15 | `fig4_15_turbine_discharge_by_gcm.png` | Projected turbine discharge by GCM |
| 4.16 | `fig4_16_turbine_discharge_trajectory.png` | Projected turbine discharge, 2015-2100 |
| 4.17 | `fig4_17_hydropower_potential_surplus.png` | Projected hydropower potential and surplus |
| 4.18 | `fig4_18_scenario_matrix.png` | Scenario matrix of hydropower potential |
| A.1 | `figA1_seasonal_precipitation_bias.png` | Seasonal precipitation: CHIRPS, model ensemble and bias |
| A.2 | `figA2_precipitation_change_by_model.png` | Projected precipitation change by model and scenario |
| A.3 | `figA3_hmfwa_annual_flow_1950_2100.png` | HMF-WA annual flow, 1950-2100 |
| A.4 | `figA4_turbine_discharge_by_driver.png` | Projected turbine discharge by driver |
| A.5 | `figA5_hydropower_potential_by_driver.png` | Projected hydropower potential by driver |
| A.6 | `figA6_flow_change_vs_precipitation_change.png` | Flow change against precipitation change |

## Conventions

- Baseline 1984-2014 (precipitation evaluation 1981-2014). Mid-century = 2030-2060; late-century = 2070-2100.
- Scenarios: SSP2-4.5 and SSP5-8.5. Ensemble mean = mean of the models (10 for precipitation, 5 for river flow).
- Precipitation in mm/month, river flow and discharge in m3/s, power in MW, percentage changes relative to the 1984-2014 baseline of the same model.
- Period labels in the data files are `mid_century` and `late_century` (the plotting scripts use the codes `near`/`far` internally).

## Data sources (not redistributed in full)

- CHIRPS v2.0 precipitation (Funk et al., 2015), regional monthly means only.
- NEX-GDDP-CMIP6 (NASA Earth Exchange), regional monthly means only.
- HMF-WA river-flow simulations (Rameshwaran et al., 2022), five GCMs, time series at the Kainji cell and four further river cells (up1, up2, down1, down2) only.
- Observed Kainji turbine discharge (monthly means, 1970-2010; mean 739.51 m3/s), as cited in the dissertation (Ladokun et al.).
The gridded inputs used for the maps (Figures 4.1, 4.4, 4.7, 4.11, 4.13, 4.14, A.1) are not included because of their size; obtain them from the original providers. Please check and cite the original data licences when reusing the data files.

## Reproducing the figures

Scripts in `code/` were run with Python 3 (numpy, pandas, xarray, geopandas, cartopy, matplotlib, scipy). Edit `<PROJECT_ROOT>` in `config.py` (and the scripts) to your local folder
and supply the gridded inputs; figures that use only the CSV files in `data/` (4.5, 4.8-4.10, 4.12, 4.15-4.17, A.2-A.6) can be redrawn from them.

## How to cite

Awomewe, Olumide Emmanuel (2026). Figures and derived data for "Multi-model Projection of Hydropower Potential in a Changing Climate: A case study of Kainji Dam, Nigeria". Zenodo. [DOI to be inserted after publication]. Licence: CC BY 4.0.
