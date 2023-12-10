import iris


nwp_path = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/compare_holo_nwp_vs_art/'
file_nwp_nd_c = nwp_path + 'test_nwp.nc'

leeds_path = '/home/mhaghigh/leeds_holo/ADVANCE/Volc/'
file_leeds_nd_c = leeds_path + 'cdnc_vert_ave_wtd_by_lwp_alt_cs159_NAtl_hilat_km2p5_ra3_p3_casim_ukca_20140901_024.nc'

cubes = iris.load_cube(file_nwp_nd_c)
print(cubes)
leeds_nd_c = iris.load_cube(file_leeds_nd_c)

regular_leeds_cdnc = leeds_nd_c.regrid(cubes, iris.analysis.Linear())

