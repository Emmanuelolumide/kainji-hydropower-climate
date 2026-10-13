"""Shared settings for the revised (Kainji-upstream-mask) analysis."""
A = "<PROJECT_ROOT>"
R = A + "/revised"
MASK_SHP = A + "/shp/kainji_upstream_49.shp"          # 49 HydroBASINS polygons that drain to Kainji
CHIRPS = A + "/data/chirps/chirps_wa_mon.nc"
HIST_DIR = A + "/data/models/hist"
PROJ_DIR = A + "/data/models/proj"
HMF_DIR = A + "/data/River flow cmip data"
KAINJI = (9.867, 4.55)        # lat, lon used in Chapter 3
HMF_CELL = (9.85, 4.65)       # HMF-WA 0.1-degree cell that contains Kainji Dam
# sub-regions = lon/lat boxes intersected with the mask (lon_min, lon_max, lat_min, lat_max)
REGIONS = {
    "Whole":  None,
    "Upper":  (-12.0, -2.0, 8.0, 13.0),     # Upper Niger (Guinea Highlands)
    "Middle": (-4.0, 4.0, 13.0, 18.0),      # Middle Niger (Sahel zone)
    "Lower":  (2.0, 8.0, 9.87, 13.0),       # dam-proximal zone (replaces the old 7-10N box, 98% non-draining)
}
LABELS = {"Whole": "Kainji Contributing Catchment", "Upper": "Upper Niger\n(Guinea Highlands)",
          "Middle": "Middle Niger\n(Sahel Zone)", "Lower": "Lower Niger\n(Near Kainji)"}
BASE = (1984, 2014); NEAR = (2030, 2060); FAR = (2070, 2100)
EVAL = (1981, 2014)
DRY = [11, 12, 1, 2, 3]; WET = [4, 5, 6, 7, 8, 9, 10]
MODELS = {   # model -> (hist file, ssp245 file, ssp585 file)
    "ACCESS-CM2":     ("Access-cm2.nc",   "ACCESS-CM2.nc",             "ACCESS-CM2.nc"),
    "ACCESS-ESM1-5":  ("Access-esm1.nc",  "ACCESS-ESM1-5.nc",          "ACCESS-ESM1-5.nc"),
    "CanESM5":        ("Canesm5.nc",      "CanESM5.nc",                "CanESM5.nc"),
    "EC-Earth3":      ("Ec-earth3.nc",    "EC-Earth3_pr_ssp245.nc",    "EC-Earth3_pr_ssp585.nc"),
    "INM-CM4-8":      ("Inm-cm4.nc",      "INM-CM4-8_pr_ssp245.nc",    "INM-CM4-8_pr_ssp585.nc"),
    "INM-CM5-0":      ("Inm-cm5.nc",      "INM-CM5-0_pr_ssp245.nc",    "INM-CM5-0_pr_ssp585.nc"),
    "IPSL-CM6A-LR":   ("Ipsl-cm6a-lr.nc", "IPSL-CM6A-LR_pr_ssp245.nc", "IPSL-CM6A-LR_pr_ssp585.nc"),
    "KIOST-ESM":      ("Kiost.nc",        "KIOST-ESM_pr_ssp245.nc",    "KIOST-ESM_pr_ssp585.nc"),
    "NorESM2-MM":     ("Noresm2.nc",      "NorESM2-MM_pr_ssp245.nc",   "NorESM2-MM_pr_ssp585.nc"),
    "MPI-ESM1-2-LR":  ("Mpi-esm1.nc",     "MPI-ESM1-2-LR_pr_ssp245.nc","MPI-ESM1-2-LR_pr_ssp585.nc"),
}
