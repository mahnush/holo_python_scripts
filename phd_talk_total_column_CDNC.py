import matplotlib.pyplot as plt
import postpro_func
import numpy as np
import matplotlib
import numpy.ma as ma

"""
this script compute the tqc (model LWP) for inside and outside of plume in 3 different models
in this part the path of data for ICON-NWP, ICON-ART and UM models are given
"""

# ICON-NWP
# control run
input_path = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/' \
             'compare_holo_nwp_vs_art/files_used_inscripts/diff_timestep/'
#ICON-NWP
# control run
holo_con_nwp_1sep = input_path + '1sep_con_nwp.nc'
holo_con_nwp_2sep = input_path + '2sep_con_nwp.nc'
holo_con_nwp_3sep = input_path + '3sep_con_nwp.nc'
holo_con_nwp_4sep = input_path + '4sep_con_nwp.nc'
holo_con_nwp_5sep = input_path + '5sep_con_nwp.nc'
holo_con_nwp_6sep = input_path + '6sep_con_nwp.nc'
holo_con_nwp_7sep = input_path + '7sep_con_nwp.nc'
#perturbed run
holo_per_nwp_1sep = input_path +'1sep_per_nwp.nc'
holo_per_nwp_2sep = input_path +'2sep_per_nwp.nc'
holo_per_nwp_3sep = input_path +'3sep_per_nwp.nc'
holo_per_nwp_4sep = input_path +'4sep_per_nwp.nc'
holo_per_nwp_5sep = input_path +'5sep_per_nwp.nc'
holo_per_nwp_6sep = input_path +'6sep_per_nwp.nc'
holo_per_nwp_7sep = input_path +'7sep_per_nwp.nc'

#ICON-ART
input_path_art = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/compare_holo_nwp_vs_art/' \
                 'files_used_inscripts/art_mistral/'
#control run
holo_con_art_1sep = input_path_art +'1sep_con_art.nc'
holo_con_art_2sep = input_path_art +'2sep_con_art.nc'
holo_con_art_3sep = input_path_art + '3sep_con_art.nc'
holo_con_art_4sep = input_path_art +'4sep_con_art.nc'
holo_con_art_5sep = input_path_art +'5sep_con_art.nc'
holo_con_art_6sep = input_path_art +'6sep_con_art.nc'
holo_con_art_7sep = input_path_art +'6sep_con_art.nc'
#perturbed run
holo_per_art_1sep = input_path_art +'1sep_per_art.nc'
holo_per_art_2sep = input_path_art +'2sep_per_art.nc'
holo_per_art_3sep = input_path_art +'3sep_per_art.nc'
holo_per_art_4sep = input_path_art +'4sep_per_art.nc'
holo_per_art_5sep = input_path_art +'5sep_per_art.nc'
holo_per_art_6sep = input_path_art +'6sep_per_art.nc'
holo_per_art_7sep = input_path_art +'6sep_per_art.nc'

# OMPS_DATA
omps_path_1sep = '/home/mhaghigh/nc_file_in_outplume/so2_Iceland_1sep_0.7res.nc'
omps_path_2sep = '/home/mhaghigh/nc_file_in_outplume/so2_Iceland_2sep_0.7res.nc'
omps_path_3sep = '/home/mhaghigh/nc_file_in_outplume/so2_Iceland_3sep_0.7res.nc'
omps_path_4sep = '/home/mhaghigh/nc_file_in_outplume/so2_Iceland_4sep_0.7res.nc'
omps_path_5sep = '/home/mhaghigh/nc_file_in_outplume/so2_Iceland_5sep_0.7res.nc'
omps_path_6sep = '/home/mhaghigh/nc_file_in_outplume/so2_Iceland_7sep_0.7res.nc'
omps_path_7sep = '/home/mhaghigh/nc_file_in_outplume/so2_Iceland_7sep_0.7res.nc'

# MODIS DATA
modis_path_1sep = '/home/mhaghigh/nc_file_in_outplume/geo_modis_data_1d.nc'
modis_path_2sep = '/home/mhaghigh/nc_file_in_outplume/geo_modis_data_2d.nc'
modis_path_3sep = '/home/mhaghigh/nc_file_in_outplume/geo_modis_data_3d.nc'
modis_path_4sep = '/home/mhaghigh/nc_file_in_outplume/geo_modis_data_4d.nc'
modis_path_5sep = '/home/mhaghigh/nc_file_in_outplume/geo_modis_data_5d.nc'
modis_path_6sep = '/home/mhaghigh/nc_file_in_outplume/geo_modis_data_6d.nc'
modis_path_7sep = '/home/mhaghigh/nc_file_in_outplume/geo_modis_data_7d.nc'

# Leeds DATA
# control
holo_con_leeds_1sep = '/home/mhaghigh/leeds_holo/ADVANCE/NoVolc/regridded_data/1sep_con_all_data_final.nc'
holo_con_leeds_2sep = '/home/mhaghigh/leeds_holo/ADVANCE/NoVolc/regridded_data/2sep_con_all_data_final.nc'
holo_con_leeds_3sep = '/home/mhaghigh/leeds_holo/ADVANCE/NoVolc/regridded_data/3sep_con_all_data_final.nc'
holo_con_leeds_4sep = '/home/mhaghigh/leeds_holo/ADVANCE/NoVolc/regridded_data/4sep_con_all_data_final.nc'
holo_con_leeds_5sep = '/home/mhaghigh/leeds_holo/ADVANCE/NoVolc/regridded_data/5sep_con_all_data_final.nc'
holo_con_leeds_6sep = '/home/mhaghigh/leeds_holo/ADVANCE/NoVolc/regridded_data/6sep_con_all_data_final.nc'
holo_con_leeds_7sep = '/home/mhaghigh/leeds_holo/ADVANCE/NoVolc/regridded_data/7sep_con_all_data_final.nc'
# perturbed
holo_per_leeds_1sep = '/home/mhaghigh/leeds_holo/ADVANCE/Volc/regridded_data/1sep_per_all_data_final.nc'
holo_per_leeds_2sep = '/home/mhaghigh/leeds_holo/ADVANCE/Volc/regridded_data/2sep_per_all_data_final.nc'
holo_per_leeds_3sep = '/home/mhaghigh/leeds_holo/ADVANCE/Volc/regridded_data/3sep_per_all_data_final.nc'
holo_per_leeds_4sep = '/home/mhaghigh/leeds_holo/ADVANCE/Volc/regridded_data/4sep_per_all_data_final.nc'
holo_per_leeds_5sep = '/home/mhaghigh/leeds_holo/ADVANCE/Volc/regridded_data/5sep_per_all_data_final.nc'
holo_per_leeds_6sep = '/home/mhaghigh/leeds_holo/ADVANCE/Volc/regridded_data/6sep_per_all_data_final.nc'
holo_per_leeds_7sep = '/home/mhaghigh/leeds_holo/ADVANCE/Volc/regridded_data/7sep_per_all_data_final.nc'

"""
in this part the name of variables in the netcdf file is given 
"""
var_name_model = ['qnc', 'tqc', 'so4_at', 'z_mc', 'geopt', 'rho', 'so2']
var_name_modis = ['nd', 'lwp', 'tau', 're', 'cf']
var_names_leeds = ['column_cloud_number', 'load_cdnc', 'load_so2', 'lwp']

"""
in this part arrays consist of netcdf names are given 
"""
model_con_nwp = [holo_con_nwp_1sep, holo_con_nwp_2sep, holo_con_nwp_3sep, holo_con_nwp_4sep, holo_con_nwp_5sep
                 , holo_con_nwp_6sep, holo_con_nwp_7sep]
model_per_nwp = [holo_per_nwp_1sep, holo_per_nwp_2sep, holo_per_nwp_3sep, holo_per_nwp_4sep, holo_per_nwp_5sep
                 , holo_per_nwp_6sep, holo_per_nwp_7sep]
model_con_art = [holo_con_art_1sep, holo_con_art_2sep, holo_con_art_3sep, holo_con_art_4sep, holo_con_art_5sep,
                  holo_con_art_6sep, holo_con_art_7sep]
model_per_art = [holo_per_art_1sep, holo_per_art_2sep, holo_per_art_3sep, holo_per_art_4sep, holo_per_art_5sep,
                 holo_per_art_6sep, holo_per_art_7sep]
model_con_leeds = [holo_con_leeds_1sep, holo_con_leeds_2sep, holo_con_leeds_3sep, holo_con_leeds_4sep,
                   holo_con_leeds_5sep, holo_con_leeds_6sep, holo_con_leeds_7sep]
model_per_leeds = [holo_per_leeds_1sep, holo_per_leeds_2sep, holo_per_leeds_3sep, holo_per_leeds_4sep,
                   holo_per_leeds_5sep,  holo_per_leeds_6sep, holo_per_leeds_7sep]
omps_data = [omps_path_1sep, omps_path_2sep, omps_path_3sep, omps_path_4sep, omps_path_5sep, omps_path_6sep,
             omps_path_7sep]
modis_data = [modis_path_1sep, modis_path_2sep, modis_path_3sep, modis_path_4sep, modis_path_5sep,
              modis_path_6sep, modis_path_7sep]
"""
in this part the functions are given
"""


def vertical_integrate_qnc_nwp(i):
    from scipy.integrate import trapz
    g = 9.8
    #create levels:
    var_lev_con = postpro_func.read_nc(model_con_nwp[i], var_name_model[4])[:, :, 200:801]
    var_lev_con = var_lev_con / g
    #lev_ocean = np.zeros(np.shape(var_lev_con))
    lat_fine = postpro_func.read_nc(model_con_nwp[i], 'lat')
    lon_fine = postpro_func.read_nc(model_con_nwp[i], 'lon')[200:801]
    mask_land = np.zeros((75, 301, 601))
    mask_land = ma.masked_where(mask_land == 0., mask_land)
    for k in range(75):
        mask_land[k, :, :] = postpro_func.mask_land(var_lev_con[k, :, :], lat_fine, lon_fine)
    var_lev_ocean = np.ma.mean(mask_land, axis = (1, 2))
    var_lev_ocean = np.round(var_lev_ocean)
    #read-in qnc
    var_qnc_con = postpro_func.read_nc(model_con_nwp[i], var_name_model[0])[:, :, 200:801]
    var_density_con = postpro_func.read_nc(model_con_nwp[i], var_name_model[5])[:, :, 200:801]
    var_qnc_con = var_qnc_con[::-1, :, :]
    var_density_con = var_density_con[::-1, :, :]
    var_qnc_den_con = var_qnc_con * var_density_con
    var_lev_ocean = var_lev_ocean[::-1]
    integrated_var_con = trapz(var_qnc_den_con, x = var_lev_ocean, axis = 0)
    integrated_var_con = integrated_var_con * 1e-4
    # per
    var_qnc_per = postpro_func.read_nc(model_per_nwp[i], var_name_model[0])[:, :, 200:801]
    var_density_per = postpro_func.read_nc(model_per_nwp[i], var_name_model[5])[:, :, 200:801]
    var_qnc_per = var_qnc_per[::-1, :, :]
    var_density_per = var_density_per[::-1, :, :]
    var_qnc_den_per = var_qnc_per * var_density_per
    integrated_var_per = trapz(var_qnc_den_per, x = var_lev_ocean, axis = 0)
    integrated_var_per = integrated_var_per * 1e-4
    return integrated_var_con, integrated_var_per


def vertical_integrate_qnc_art(i):
    from scipy.integrate import trapz
    #compute levels:
    var_lev_con = postpro_func.read_nc(model_con_art[i], var_name_model[3])[:, :, :]
    lat_fine = postpro_func.read_nc(model_con_art[i], 'lat')
    lon_fine = postpro_func.read_nc(model_con_art[i], 'lon')
    mask_land = np.zeros((50, 301, 601))
    mask_land = ma.masked_where(mask_land == 0., mask_land)
    for k in range(50):
        mask_land[k, :, :] = postpro_func.mask_land(var_lev_con[k, :, :], lat_fine, lon_fine)
    var_lev_ocean = np.ma.mean(mask_land, axis = (1, 2))
    var_lev_ocean = np.round(var_lev_ocean)
    var_lev_ocean = var_lev_ocean[::-1]
    #read in qnc control
    var_qnc_con = postpro_func.read_nc(model_con_art[i], var_name_model[0])[:, :, :]
    var_density_con = postpro_func.read_nc(model_con_art[i], var_name_model[5])[:, :, :]
    var_qnc_con = var_qnc_con[::-1, :, :]
    var_density_con = var_density_con[::-1, :, :]
    var_qnc_den_con = var_qnc_con * var_density_con
    integrated_var_con = trapz(var_qnc_den_con, x = var_lev_ocean, axis = 0)
    integrated_var_con = integrated_var_con * 1e-4
    # perturbed
    var_qnc_per = postpro_func.read_nc(model_per_art[i], var_name_model[0])[:, :, :]
    #var_qnc_per = ma.masked_where(var_qnc_per_read >= 2e9, var_qnc_per_read)
    var_density_per = postpro_func.read_nc(model_per_art[i], var_name_model[5])[:, :, :]
    var_qnc_per = var_qnc_per[::-1, :, :]
    var_density_per = var_density_per[::-1, :, :]
    var_qnc_den_per = var_qnc_per * var_density_per
    integrated_var_per = trapz(var_qnc_den_per, x = var_lev_ocean, axis = 0)
    integrated_var_per = integrated_var_per * 1e-4
    return integrated_var_con, integrated_var_per


def vertical_integrate_so4_art(i):
    var_lev_con = postpro_func.read_nc(model_con_art[i], var_name_model[3])[:, :, :]
    lat_fine = postpro_func.read_nc(model_con_art[i], 'lat')
    lon_fine = postpro_func.read_nc(model_con_art[i], 'lon')
    mask_land = np.zeros((50, 301, 601))
    mask_land = ma.masked_where(mask_land == 0., mask_land)
    for k in range(50):
        mask_land[k, :, :] = postpro_func.mask_land(var_lev_con[k, :, :], lat_fine, lon_fine)
    var_lev_ocean = np.ma.mean(mask_land, axis = (1, 2))
    var_lev_ocean = np.round(var_lev_ocean)
    var_lev_ocean = var_lev_ocean[::-1]
    from scipy.integrate import trapz
    var_so4 = postpro_func.read_nc(model_per_art[i], var_name = var_name_model[2])[:, :, :]
    var_density = postpro_func.read_nc(model_per_art[i], var_name = var_name_model[5])[:, :, :]
    var_so4 = var_so4[::-1, :, :]
    var_density = var_density[::-1, :, :]
    var_so4_den = var_so4 * var_density
    integrated_var = trapz(var_so4_den, x = var_lev_ocean, axis = 0)
    return integrated_var

def vertical_integrate_so2_art(i):
    var_lev_con = postpro_func.read_nc(model_con_art[i], var_name_model[3])[:, :, :]
    lat_fine = postpro_func.read_nc(model_con_art[i], 'lat')
    lon_fine = postpro_func.read_nc(model_con_art[i], 'lon')
    mask_land = np.zeros((50, 301, 601))
    mask_land = ma.masked_where(mask_land == 0., mask_land)
    for k in range(50):
        mask_land[k, :, :] = postpro_func.mask_land(var_lev_con[k, :, :], lat_fine, lon_fine)
    var_lev_ocean = np.ma.mean(mask_land, axis = (1, 2))
    var_lev_ocean = np.round(var_lev_ocean)
    var_lev_ocean = var_lev_ocean[::-1]
    from scipy.integrate import trapz
    var_so2 = postpro_func.read_nc(model_per_art[i], var_name = var_name_model[6])[:, :, :]
    var_density = postpro_func.read_nc(model_per_art[i], var_name = var_name_model[5])[:, :, :]
    var_so2 = var_so2[::-1, :, :]
    var_density = var_density[::-1, :, :]
    const = 1000 / (28.98 * 0.00044615)
    var_so2_den = var_so2 * var_density*const
    integrated_var = trapz(var_so2_den, x = var_lev_ocean, axis = 0)
    return integrated_var

def get_in_out_nwp(i):
    import scipy.interpolate as sci
    lat_fine = postpro_func.read_nc(model_con_nwp[i], 'lat')
    lon_fine = postpro_func.read_nc(model_con_nwp[i], 'lon')[200:801]
    control_var_lando = vertical_integrate_qnc_nwp(i)[0]
    perturbed_var_lando = vertical_integrate_qnc_nwp(i)[1]
    control_var = postpro_func.mask_land(control_var_lando, lat_fine, lon_fine)
    perturbed_var = postpro_func.mask_land(perturbed_var_lando, lat_fine, lon_fine)
    so2 = postpro_func.read_nc(omps_data[i], 'so2_TRL')[0:43, 29:115]
    lat_so2 = postpro_func.read_nc(omps_data[i], 'lat')
    lon_so2 = postpro_func.read_nc(omps_data[i], 'lon')
    lon_coarse = lon_so2[0, 29:115]
    lat_coarse = lat_so2[0:43, 0]
    # print(lon_coarse)
    # print(lat_coarse)
    so2_mask = np.ma.filled(so2, fill_value = 0)
    f = sci.RectBivariateSpline(lat_coarse, lon_coarse, so2_mask)
    scale_interp = f(lat_fine, lon_fine)
    so2_treshold = 1.0
    con_inside = control_var[scale_interp > so2_treshold]
    con_outside = control_var[scale_interp < so2_treshold]
    per_inside = perturbed_var[scale_interp > so2_treshold]
    per_outside = perturbed_var[scale_interp < so2_treshold]
    con_inside_mask = ma.masked_where(con_inside <= 0, con_inside)
    # con_inside_mask = ma.masked_where(con_inside_mask_1 >= 2000, con_inside_mask_1)
    con_inside_mask = con_inside_mask.compressed()
    con_outside_mask = ma.masked_where(con_outside <= 0., con_outside)
    con_outside_mask = con_outside_mask.compressed()
    per_inside_mask = ma.masked_where(per_inside <= 0, per_inside)
    # per_inside_mask = ma.masked_where(per_inside_mask_1 >= 2000, per_inside_mask_1)
    per_inside_mask = per_inside_mask.compressed()
    per_outside_mask = ma.masked_where(per_outside <= 0., per_outside)
    per_outside_mask = per_outside_mask.compressed()

    return con_inside_mask, con_outside_mask, per_inside_mask, per_outside_mask


def get_in_out_art(i):
    lat_fine = postpro_func.read_nc(model_con_art[i], 'lat')
    lon_fine = postpro_func.read_nc(model_con_art[i], 'lon')
    control_var_lando = vertical_integrate_qnc_art(i)[0]
    perturbed_var_lando = vertical_integrate_qnc_art(i)[1]
    var_so4_lando = vertical_integrate_so4_art(i)
    control_var = postpro_func.mask_land(control_var_lando, lat_fine, lon_fine)
    perturbed_var = postpro_func.mask_land(perturbed_var_lando, lat_fine, lon_fine)
    so4_mask = postpro_func.mask_land(var_so4_lando, lat_fine, lon_fine)
    so4_threshold = 1e2
    con_inside = control_var[so4_mask > so4_threshold]
    con_outside = control_var[so4_mask < so4_threshold]
    per_inside = perturbed_var[so4_mask > so4_threshold]
    per_outside = perturbed_var[so4_mask < so4_threshold]
    con_inside_mask_1 = ma.masked_where(con_inside <= 0., con_inside)
    con_inside_mask = ma.masked_where(con_inside_mask_1 > 5e8, con_inside_mask_1)
    con_inside_mask = con_inside_mask.compressed()
    con_outside_mask = ma.masked_where(con_outside <= 0., con_outside)
    con_outside_mask = con_outside_mask.compressed()
    per_inside_mask = ma.masked_where(per_inside <= 0., per_inside)
    per_inside_mask = per_inside_mask.compressed()
    per_outside_mask = ma.masked_where(per_outside <= 0, per_outside)
    per_outside_mask = per_outside_mask.compressed()
    return con_inside_mask, con_outside_mask, per_inside_mask, per_outside_mask

def get_in_out_art_so2(i):
    lat_fine = postpro_func.read_nc(model_con_art[i], 'lat')
    lon_fine = postpro_func.read_nc(model_con_art[i], 'lon')
    control_var_lando = vertical_integrate_qnc_art(i)[0]
    perturbed_var_lando = vertical_integrate_qnc_art(i)[1]
    var_so2_lando = vertical_integrate_so2_art(i)
    control_var = postpro_func.mask_land(control_var_lando, lat_fine, lon_fine)
    perturbed_var = postpro_func.mask_land(perturbed_var_lando, lat_fine, lon_fine)
    so2_mask = postpro_func.mask_land(var_so2_lando, lat_fine, lon_fine)
    so2_threshold = 0.01
    con_inside = control_var[so2_mask > so2_threshold]
    con_outside = control_var[so2_mask < so2_threshold]
    per_inside = perturbed_var[so2_mask > so2_threshold]
    per_outside = perturbed_var[so2_mask < so2_threshold]
    con_inside_mask = ma.masked_where(con_inside <= 0., con_inside)
    con_inside_mask = con_inside_mask.compressed()
    con_outside_mask = ma.masked_where(con_outside <= 0., con_outside)
    con_outside_mask = con_outside_mask.compressed()
    per_inside_mask = ma.masked_where(per_inside <= 0., per_inside)
    per_inside_mask = per_inside_mask.compressed()
    per_outside_mask = ma.masked_where(per_outside <= 0, per_outside)
    per_outside_mask = per_outside_mask.compressed()
    return con_inside_mask, con_outside_mask, per_inside_mask, per_outside_mask


def get_in_out_leeds(i):
    control_var_lando_t = postpro_func.read_nc(model_con_leeds[i], var_name = var_names_leeds[1])[0:1, :, 1600:4601]*1e-4
    perturbed_var_lando_t = postpro_func.read_nc(model_per_leeds[i], var_name = var_names_leeds[1])[0:1, :, 1600:4601]*1e-4
    var_so2_lando_t = postpro_func.read_nc(model_per_leeds[i], var_name = var_names_leeds[2])[0:1, :, 1600:4601]
    control_var_lando = np.ma.mean(control_var_lando_t, axis = 0)
    perturbed_var_lando = np.ma.mean(perturbed_var_lando_t, axis = 0)
    var_so2_lando = np.ma.mean(var_so2_lando_t, axis = 0)
    so2_threshold = 3e-5
    lat_fine = postpro_func.read_nc(model_per_leeds[i], 'lat')
    lon_fine = postpro_func.read_nc(model_per_leeds[i], 'lon')[1600:4601]
    control_var = control_var_lando
    perturbed_var = perturbed_var_lando
    so2_mask = postpro_func.mask_land(var_so2_lando, lat_fine, lon_fine)
    con_inside = control_var[so2_mask > so2_threshold]
    con_outside = control_var[so2_mask < so2_threshold]
    per_inside = perturbed_var[so2_mask > so2_threshold]
    per_outside = perturbed_var[so2_mask < so2_threshold]
    con_inside_mask = ma.masked_where(con_inside <= 0., con_inside)
    con_inside_mask = con_inside_mask.compressed()
    con_outside_mask = ma.masked_where(con_outside <= 0., con_outside)
    con_outside_mask = con_outside_mask.compressed()
    per_inside_mask = ma.masked_where(per_inside <= 0., per_inside)
    per_inside_mask = per_inside_mask.compressed()
    per_outside_mask = ma.masked_where(per_outside <= 0., per_outside)
    per_outside_mask = per_outside_mask.compressed()
    return con_inside_mask, con_outside_mask, per_inside_mask, per_outside_mask


def weight(var):
    weight = np.zeros_like(var) + 1. / (var.size)
    return weight


def lable_hist(var):
    median = str(np.median(var))
    mean = str("{:.2e}".format(round(np.mean(var))))
    median = str("{:.2e}".format(round(np.median(var))))
    std = str(np.std(var))
    lable = '(' + 'mean = ' + mean + ' median = ' + median + ')'
    return lable


def pdf_var_prepare(axs0, var, range, color, label, titel, x_axis_label='', y_axis_label=''):
    font_legend = 20
    font_tick = 25
    numbin = 100
    line_width = 2
    #axs0.hist(var, bins = numbin, range = range, weights = weight(var), histtype = 'step',
    #          linewidth = line_width, color = color, label = label + lable_hist(var), log = True)
    axs0.hist(var, bins = numbin, range = range, weights = weight(var), histtype = 'step',
              linewidth = line_width, color = color, label = label, log = True)
    axs0.legend(loc = 'upper right', fontsize = font_legend, frameon = True)
    axs0.set_xlabel(x_axis_label, fontsize = font_tick)
    axs0.tick_params(axis = 'x', labelsize = font_tick)  # to Set Matplotlib Tick Labels Font Size
    axs0.tick_params(axis = 'y', labelsize = font_tick)
    ticks = np.arange(0, 0.18, 0.02)
    # axs0.set_yticks(ticks)
    axs0.set_ylabel(y_axis_label, fontsize = font_tick)
    axs0.set_title(titel, fontsize = font_tick)
    axs0.grid(True)
    return


numdays = 6

allvar_in_per_nwp = []
allvar_out_per_nwp = []
allvar_in_con_nwp = []
allvar_out_con_nwp = []

allvar_in_per_art = []
allvar_out_per_art = []
allvar_in_con_art = []
allvar_out_con_art = []

allvar_in_per_art_so2 = []
allvar_out_per_art_so2 = []
allvar_in_con_art_so2 = []
allvar_out_con_art_so2 = []

allvar_in_per_um = []
allvar_out_per_um = []
allvar_in_con_um = []
allvar_out_con_um = []

allvar_in_modis = []
allvar_out_modis = []
for i in range(numdays):
    print(i)
    con_inside_nwp, con_outside_nwp, per_inside_nwp, per_outside_nwp = get_in_out_nwp(i)
    allvar_in_per_nwp = np.concatenate((allvar_in_per_nwp, per_inside_nwp), axis = 0)
    allvar_out_per_nwp = np.concatenate((allvar_out_per_nwp, per_outside_nwp), axis = 0)
    allvar_in_con_nwp = np.concatenate((allvar_in_con_nwp, con_inside_nwp), axis = 0)
    allvar_out_con_nwp = np.concatenate((allvar_out_con_nwp, con_outside_nwp), axis = 0)

    con_inside_art, con_outside_art, per_inside_art, per_outside_art = get_in_out_art(i)
    allvar_in_per_art = np.concatenate((allvar_in_per_art, per_inside_art), axis = 0)
    allvar_out_per_art = np.concatenate((allvar_out_per_art, per_outside_art), axis = 0)
    allvar_in_con_art = np.concatenate((allvar_in_con_art, con_inside_art), axis = 0)
    allvar_out_con_art = np.concatenate((allvar_out_con_art, con_outside_art), axis = 0)

    con_inside_art_so2, con_outside_art_so2, per_inside_art_so2,\
    per_outside_art_so2 = get_in_out_art_so2(i)
    allvar_in_per_art_so2 = np.concatenate((allvar_in_per_art_so2,
                                            per_inside_art_so2), axis=0)
    allvar_out_per_art_so2 = np.concatenate((allvar_out_per_art_so2,
                                             per_outside_art_so2), axis=0)
    allvar_in_con_art_so2 = np.concatenate((allvar_in_con_art_so2,
                                            con_inside_art_so2), axis=0)
    allvar_out_con_art_so2 = np.concatenate((allvar_out_con_art_so2,
                                             con_outside_art_so2), axis=0)

    con_inside_um, con_outside_um, per_inside_um, per_outside_um = get_in_out_leeds(i)
    allvar_in_per_um = np.concatenate((allvar_in_per_um, per_inside_um), axis = 0)
    allvar_out_per_um = np.concatenate((allvar_out_per_um, per_outside_um), axis = 0)
    allvar_in_con_um = np.concatenate((allvar_in_con_um, con_inside_um), axis = 0)
    allvar_out_con_um = np.concatenate((allvar_out_con_um, con_outside_um), axis = 0)

titel_kind = ['(NWP)', '(ART)', '(Unified Model)', '(NWP-COSP)']
run_type = ['No-Volcano', 'Volcano', 'MODIS']
x_name = 'total column $ \mathrm{N_d}$ ($\mathrm{cm^{-2}}$)'
y_name = 'Relative Frequency'

range_nwp = (0, 3e8)
range_art = (0, 3e8)
range_um = (0, 3e8)

fig, ((axs0, axs1), (axs2, axs3), (axs4, axs5)) = \
    plt.subplots(3, 2, figsize = (16, 15), sharey = True)
font_tick = 25
pdf_var_prepare(axs0, var = allvar_in_con_nwp, range = range_nwp, color = 'blue', label = run_type[0],
                titel = 'Inside_Plume ' + titel_kind[0])
pdf_var_prepare(axs0, var = allvar_in_per_nwp, range = range_nwp, color = 'red', label = run_type[1],
                titel = 'Inside_Plume ' + titel_kind[0], y_axis_label = y_name)
axs0.annotate("(a)", xy=(0.07, 0.93), xycoords="axes fraction", fontsize= font_tick)

pdf_var_prepare(axs1, var = allvar_out_con_nwp, range = range_nwp, color = 'blue', label = run_type[0],
                titel = 'Outside_Plume ' + titel_kind[0])
pdf_var_prepare(axs1, var = allvar_out_per_nwp, range = range_nwp, color = 'red', label = run_type[1],
                titel = 'Outside_Plume ' + titel_kind[0])
axs1.annotate("(b)", xy=(0.07, 0.93), xycoords="axes fraction", fontsize= font_tick)
pdf_var_prepare(axs2, var = allvar_in_con_art_so2, range = range_art, color = 'blue', label = run_type[0],
                titel = 'Inside_Plume ' + titel_kind[1])
pdf_var_prepare(axs2, var = allvar_in_per_art_so2, range = range_art, color = 'red', label = run_type[1],
                titel = 'Inside_Plume ' + titel_kind[1], y_axis_label = y_name)
axs2.annotate("(c)", xy=(0.07, 0.93), xycoords="axes fraction", fontsize= font_tick)
pdf_var_prepare(axs3, var = allvar_out_con_art_so2, range = range_nwp, color = 'blue', label = run_type[0],
                titel = 'Outside_Plume ' + titel_kind[1])
pdf_var_prepare(axs3, var = allvar_out_per_art_so2, range = range_nwp, color = 'red', label = run_type[1],
                titel = 'Outside_Plume ' + titel_kind[1])
axs3.annotate("(d)", xy=(0.07, 0.93), xycoords="axes fraction", fontsize= font_tick)
pdf_var_prepare(axs4, var = allvar_in_con_um, range = range_um, color = 'blue', label = run_type[0],
               titel = 'Inside_Plume ' + titel_kind[2])
pdf_var_prepare(axs4, var = allvar_in_per_um, range = range_um, color = 'red', label = run_type[1],
               titel = 'Inside_Plume ' + titel_kind[2], y_axis_label = y_name, x_axis_label = x_name)
axs4.annotate("(e)", xy=(0.07, 0.93), xycoords="axes fraction", fontsize= font_tick)
pdf_var_prepare(axs5, var = allvar_out_con_um, range = range_um, color = 'blue', label = run_type[0],
               titel = 'Outside_Plume ' + titel_kind[2])
pdf_var_prepare(axs5, var = allvar_out_per_um, range = range_um, color = 'red', label = run_type[1],
               titel = 'Outside_Plume ' + titel_kind[2], x_axis_label = x_name)
axs5.annotate("(f)", xy=(0.07, 0.93), xycoords="axes fraction", fontsize= font_tick)
#plt.tight_layout()
#plt.savefig('./manuscript_figures/qnc_total_column_nwp_art_leeds_with_legends.png')
#plt.savefig('./manuscript_figures/qnc_total_column_nwp_art_leeds_with_legends.pdf')
#plt.show()

"""
in this part, I want to add the part which total nd is computed in modis and also COSP in ICON-NWP model
"""

input_path = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/compare_holo_nwp_vs_art/' \
             'files_used_inscripts/diff_timestep/holo_cosp_runs_nwp_10kmremap/'
#ICON-NWP
# control run
holo_cosp_con_1sep = input_path + '1_sep_control.nc'
holo_cosp_con_2sep = input_path + '2_sep_control.nc'
holo_cosp_con_3sep = input_path + '3_sep_control.nc'
holo_cosp_con_4sep = input_path + '4_sep_control.nc'
holo_cosp_con_5sep = input_path + '5_sep_control.nc'
holo_cosp_con_6sep = input_path + '6_sep_control.nc'
holo_cosp_con_7sep = input_path + '7_sep_control.nc'
#perturbed run
holo_cosp_per_1sep = input_path +'1_sep_per.nc'
holo_cosp_per_2sep = input_path +'2_sep_per.nc'
holo_cosp_per_3sep = input_path +'3_sep_per.nc'
holo_cosp_per_4sep = input_path +'4_sep_per.nc'
holo_cosp_per_5sep = input_path +'5_sep_per.nc'
holo_cosp_per_6sep = input_path +'6_sep_per.nc'
holo_cosp_per_7sep = input_path +'7_sep_per.nc'

# OMPS_DATA
omps_path_1sep = '/home/mhaghigh/nc_file_in_outplume/so2_Iceland_1sep_0.7res.nc'
omps_path_2sep = '/home/mhaghigh/nc_file_in_outplume/so2_Iceland_2sep_0.7res.nc'
omps_path_3sep = '/home/mhaghigh/nc_file_in_outplume/so2_Iceland_3sep_0.7res.nc'
omps_path_4sep = '/home/mhaghigh/nc_file_in_outplume/so2_Iceland_4sep_0.7res.nc'
omps_path_5sep = '/home/mhaghigh/nc_file_in_outplume/so2_Iceland_5sep_0.7res.nc'
omps_path_6sep = '/home/mhaghigh/nc_file_in_outplume/so2_Iceland_7sep_0.7res.nc'
omps_path_7sep = '/home/mhaghigh/nc_file_in_outplume/so2_Iceland_7sep_0.7res.nc'

# MODIS DATA
modis_path_1sep = '/home/mhaghigh/nc_file_in_outplume/geo_modis_data_1d.nc'
modis_path_2sep = '/home/mhaghigh/nc_file_in_outplume/geo_modis_data_2d.nc'
modis_path_3sep = '/home/mhaghigh/nc_file_in_outplume/geo_modis_data_3d.nc'
modis_path_4sep = '/home/mhaghigh/nc_file_in_outplume/geo_modis_data_4d.nc'
modis_path_5sep = '/home/mhaghigh/nc_file_in_outplume/geo_modis_data_5d.nc'
modis_path_6sep = '/home/mhaghigh/nc_file_in_outplume/geo_modis_data_6d.nc'
modis_path_7sep = '/home/mhaghigh/nc_file_in_outplume/geo_modis_data_7d.nc'
model_con_nwp = [holo_cosp_con_1sep, holo_cosp_con_2sep, holo_cosp_con_3sep,
                  holo_cosp_con_4sep, holo_cosp_con_5sep, holo_cosp_con_6sep, holo_cosp_con_7sep]
model_per_nwp = [holo_cosp_per_1sep, holo_cosp_per_2sep, holo_cosp_per_3sep,
                 holo_cosp_per_4sep, holo_cosp_per_5sep, holo_cosp_per_6sep, holo_cosp_per_7sep]
omps_data = [omps_path_1sep, omps_path_2sep, omps_path_3sep, omps_path_4sep, omps_path_5sep,
             omps_path_6sep, omps_path_7sep]

modis_data = [modis_path_1sep, modis_path_2sep, modis_path_3sep, modis_path_4sep, modis_path_5sep,
              modis_path_6sep, modis_path_7sep]

var_name_modis = ['nd', 'lwp', 'tau', 're', 'cf', 'nd_c']

def compute_total_column_CDNC_cosp(i):
    re_con = postpro_func.read_nc(model_con_nwp[i], var_name = 're_dw')[:, 200:801]
    tau_con = postpro_func.read_nc(model_con_nwp[i], var_name = 'tau_dw')[:, 200:801]
    re_per = postpro_func.read_nc(model_per_nwp[i], var_name = 're_dw')[:, 200:801]
    tau_per = postpro_func.read_nc(model_per_nwp[i], var_name = 'tau_dw')[:, 200:801]
    beta = 0.30
    nd_c_con = (beta * (tau_con) * ((re_con * 1e-6) ** -2))*1e-4
    nd_c_per = (beta * (tau_per) * ((re_per * 1e-6) ** -2))*1e-4
    return nd_c_con, nd_c_per

def get_in_out_nwp_cosp(i):
    import scipy.interpolate as sci
    # lat_fine = np.arange(50, 80.02, 0.02)
    # lon_fine = np.arange(-40, 20.02, 0.02)
    lat_fine = np.arange(50, 80.1, 0.1)
    lon_fine = np.arange(-40, 20.1, 0.1)
    var_con, var_per = compute_total_column_CDNC_cosp(i)
    var_ocean = postpro_func.mask_land(var_con, lat_fine, lon_fine)
    so2 = postpro_func.read_nc(omps_data[i], 'so2_TRL')[0:43, 29:115]
    lat_so2 = postpro_func.read_nc(omps_data[i], 'lat')
    lon_so2 = postpro_func.read_nc(omps_data[i], 'lon')
    lon_coarse = lon_so2[0, 29:115]
    lat_coarse = lat_so2[0:43, 0]
    so2_mask = np.ma.filled(so2, fill_value=0)
    f = sci.RectBivariateSpline(lat_coarse, lon_coarse, so2_mask)
    scale_interp = f(lat_fine, lon_fine)
    so2_threshold = 1.0
    var_inside = var_ocean[scale_interp > so2_threshold]
    var_outside = var_ocean[scale_interp < so2_threshold]
    var_inside_mask = ma.masked_where(var_inside <= 0, var_inside)
    #var_inside_mask_1 = ma.masked_where(var_inside_mask > 6e7, var_inside_mask)
    #var_inside_mask_test = var_inside_mask_1.compressed()
    var_inside_mask = var_inside_mask.compressed()
    var_outside_mask = ma.masked_where(var_outside <= 0, var_outside)
    var_outside_mask = var_outside_mask.compressed()
    #per
    var_ocean_per = postpro_func.mask_land(var_per, lat_fine, lon_fine)
    var_inside_per = var_ocean_per[scale_interp > 1.0]
    var_outside_per = var_ocean_per[scale_interp < 1.0]
    var_inside_mask_per = ma.masked_where(var_inside_per <= 0, var_inside_per)
    #var_inside_mask_1_per = ma.masked_where(var_inside_mask_per > 6e7, var_inside_mask_per)
    #var_inside_mask_test_per = var_inside_mask_1_per.compressed()
    var_inside_mask_per = var_inside_mask_per.compressed()
    var_outside_mask_per = ma.masked_where(var_outside_per <= 0, var_outside_per)
    var_outside_mask_per = var_outside_mask_per.compressed()
    return var_inside_mask, var_outside_mask, var_inside_mask_per, var_outside_mask_per


def compute_total_column_CDNC_modis(i):
    re = postpro_func.read_nc(modis_data[i], var_name = 're')
    tau = postpro_func.read_nc(modis_data[i], var_name = 'tau')
    nd_c = 0.32 * (tau) * ((re * 1e-6) ** -2)
    return nd_c


def get_in_out_modis(i):
    import scipy.interpolate as sci
    lat_fine_modis = np.arange(50, 80.02, 0.02)
    lon_fine_modis = np.arange(-40, 20.02, 0.02)
    modis_var_lando = compute_total_column_CDNC_modis(i)
    modis_var_lando = np.transpose(modis_var_lando)[:, 1000:4001] * 1e-4
    modis_var = postpro_func.mask_land(modis_var_lando, lat_fine_modis, lon_fine_modis)
    so2 = postpro_func.read_nc(omps_data[i], 'so2_TRL')[0:43, 29:115]
    lat_so2 = postpro_func.read_nc(omps_data[i], 'lat')
    lon_so2 = postpro_func.read_nc(omps_data[i], 'lon')
    lon_coarse = lon_so2[0, 29:115]
    lat_coarse = lat_so2[0:43, 0]
    so2_mask = np.ma.filled(so2, fill_value=0)
    f = sci.RectBivariateSpline(lat_coarse, lon_coarse, so2_mask)
    scale_interp = f(lat_fine_modis, lon_fine_modis)
    modis_inside = modis_var[scale_interp > 1.0]
    modis_outside = modis_var[scale_interp < 1.0]
    modis_inside_mask = ma.masked_where(modis_inside <= 0, modis_inside)
    #modis_inside_mask_1 = ma.masked_where(modis_inside_mask > 1e8, modis_inside_mask)
    modis_inside_mask = modis_inside_mask.compressed()
    modis_outside_mask = ma.masked_where(modis_outside <= 0, modis_outside)
    modis_outside_mask = modis_outside_mask.compressed()
    return modis_inside_mask, modis_outside_mask

numdays = 6

allvar_in_per_cosp = []
allvar_out_per_cosp = []
allvar_in_con_cosp = []
allvar_out_con_cosp = []

allvar_in_modis = []
allvar_out_modis = []

for i in range(numdays):
    print(i)
    con_inside_cosp, con_outside_cosp, per_inside_cosp, per_outside_cosp = get_in_out_nwp_cosp(i)
    allvar_in_per_cosp = np.concatenate((allvar_in_per_cosp, per_inside_cosp), axis = 0)
    allvar_out_per_cosp = np.concatenate((allvar_out_per_cosp, per_outside_cosp), axis = 0)
    allvar_in_con_cosp = np.concatenate((allvar_in_con_cosp, con_inside_cosp), axis = 0)
    allvar_out_con_cosp = np.concatenate((allvar_out_con_cosp, con_outside_cosp), axis = 0)

    in_modis, out_modis = get_in_out_modis(i)
    allvar_in_modis = np.concatenate((allvar_in_modis, in_modis), axis=0)
    allvar_out_modis = np.concatenate((allvar_out_modis, out_modis), axis=0)


pdf_var_prepare(axs0, var = allvar_in_modis, range = range_nwp, color = 'black', label = run_type[2],
                titel = 'Inside_Plume ' + titel_kind[0], y_axis_label = y_name)
pdf_var_prepare(axs1, var = allvar_out_modis, range = range_nwp, color = 'black', label = run_type[2],
                titel = 'Outside_Plume ' + titel_kind[0])

pdf_var_prepare(axs2, var = allvar_in_modis, range = range_nwp, color = 'black', label = run_type[2],
                titel = 'Inside_Plume ' + titel_kind[1], y_axis_label = y_name)
pdf_var_prepare(axs3, var = allvar_out_modis, range = range_nwp, color = 'black', label = run_type[2],
                titel = 'Outside_Plume ' + titel_kind[1])

pdf_var_prepare(axs4, var = allvar_in_modis, range = range_nwp, color = 'black', label = run_type[2],
                titel = 'Inside_Plume ' + titel_kind[2], y_axis_label = y_name)
pdf_var_prepare(axs5, var = allvar_out_modis, range = range_nwp, color = 'black', label = run_type[2],
                titel = 'Outside_Plume ' + titel_kind[2])

plt.tight_layout()
#plt.savefig('./manuscript_figures/qnc_total_column_nwp_artmistral_'
#            'leeds_cosp_modis_no_legends_6days.png')
#plt.savefig('./manuscript_figures/qnc_total_column_nwp_artmistral_'
#            'leeds_cosp_modis_no_legends_6days.pdf')
plt.show()

