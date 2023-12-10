import matplotlib.pyplot as plt
import postpro_func
import numpy as np
import numpy.ma as ma

#compute the CDNC burden from COSP:

# holo_cosp_con_1sep = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/' \
#                      'fix_cosp_runs/fix_cosp_control/1_sep_mean.nc'
# holo_cosp_con_2sep = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/' \
#                      'fix_cosp_runs/fix_cosp_control/2_sep_mean.nc'
# holo_cosp_con_3sep = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/' \
#                      'fix_cosp_runs/fix_cosp_control/3_sep_mean.nc'
# holo_cosp_con_4sep = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/' \
#                      'fix_cosp_runs/fix_cosp_control/4_sep_mean.nc'
# holo_cosp_con_5sep = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/' \
#                      'fix_cosp_runs/fix_cosp_control/5_sep_mean.nc'
# #per
# holo_cosp_per_1sep = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/' \
#                       'fix_cosp_runs/fix_cosp_perturbed/per_1_sep_mean.nc'
# holo_cosp_per_2sep = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/' \
#                       'fix_cosp_runs/fix_cosp_perturbed/per_2_sep_mean.nc'
# holo_cosp_per_3sep = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/' \
#                       'fix_cosp_runs/fix_cosp_perturbed/per_3_sep_mean.nc'
# holo_cosp_per_4sep = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/' \
#                       'fix_cosp_runs/fix_cosp_perturbed/per_4_sep_mean.nc'
# holo_cosp_per_5sep = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/' \
#                       'fix_cosp_runs/fix_cosp_perturbed/per_5_sep_mean.nc'


input_path = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/compare_holo_nwp_vs_art/' \
             'files_used_inscripts/diff_timestep/holo_cosp_runs_nwp_10kmremap/'
#ICON-NWP
# control run
holo_cosp_con_1sep = input_path + '1_sep_control.nc'
holo_cosp_con_2sep = input_path + '2_sep_control.nc'
holo_cosp_con_3sep = input_path + '3_sep_control.nc'
holo_cosp_con_4sep = input_path + '4_sep_control.nc'
holo_cosp_con_5sep = input_path + '5_sep_control.nc'
#perturbed run
holo_cosp_per_1sep = input_path +'1_sep_per.nc'
holo_cosp_per_2sep = input_path +'2_sep_per.nc'
holo_cosp_per_3sep = input_path +'3_sep_per.nc'
holo_cosp_per_4sep = input_path +'4_sep_per.nc'
holo_cosp_per_5sep = input_path +'5_sep_per.nc'
# OMPS_DATA
omps_path_1sep = '/home/mhaghigh/nc_file_in_outplume/so2_Iceland_1sep_0.7res.nc'
omps_path_2sep = '/home/mhaghigh/nc_file_in_outplume/so2_Iceland_2sep_0.7res.nc'
omps_path_3sep = '/home/mhaghigh/nc_file_in_outplume/so2_Iceland_3sep_0.7res.nc'
omps_path_4sep = '/home/mhaghigh/nc_file_in_outplume/so2_Iceland_4sep_0.7res.nc'
omps_path_5sep = '/home/mhaghigh/nc_file_in_outplume/so2_Iceland_5sep_0.7res.nc'

# MODIS DATA
modis_path_1sep = '/home/mhaghigh/nc_file_in_outplume/geo_modis_data_1d.nc'
modis_path_2sep = '/home/mhaghigh/nc_file_in_outplume/geo_modis_data_2d.nc'
modis_path_3sep = '/home/mhaghigh/nc_file_in_outplume/geo_modis_data_3d.nc'
modis_path_4sep = '/home/mhaghigh/nc_file_in_outplume/geo_modis_data_4d.nc'
modis_path_5sep = '/home/mhaghigh/nc_file_in_outplume/geo_modis_data_5d.nc'

model_con_nwp = [holo_cosp_con_1sep, holo_cosp_con_2sep, holo_cosp_con_3sep,
                  holo_cosp_con_4sep, holo_cosp_con_5sep]
model_per_nwp = [holo_cosp_per_1sep, holo_cosp_per_2sep, holo_cosp_per_3sep,
                 holo_cosp_per_4sep, holo_cosp_per_5sep]
omps_data = [omps_path_1sep, omps_path_2sep, omps_path_3sep, omps_path_4sep, omps_path_5sep]

modis_data = [modis_path_1sep, modis_path_2sep, modis_path_3sep, modis_path_4sep, modis_path_5sep]

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
    var_inside = var_ocean[scale_interp > 1.0]
    var_outside = var_ocean[scale_interp < 1.0]
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

def weight(var):
    weight = np.zeros_like(var) + 1. / (var.size)
    return weight

def lable_hist(var):
    median = str(np.median(var))
    mean = str("{:.2e}".format(round(np.mean(var))))
    median = str("{:.2e}".format(round(np.median(var))))
    std = str(np.std(var))
    lable = '('+'mean = ' + mean + ' median = ' + median +')'
    return lable

def pdf_var_prepare(axs0, var, range, color, label, titel, x_axis_label = '', y_axis_label = ''):
    font_legend = 15
    font_tick = 20
    numbin = 100
    line_width = 2
    axs0.hist(var,  bins=numbin, range = range, weights=weight(var),  histtype='step',
         linewidth=line_width, color= color, label= label+ lable_hist(var) , log = True)

    axs0.legend(loc = 'upper right', fontsize = font_legend, frameon = True)
    axs0.set_xlabel(x_axis_label, fontsize= font_tick)
    axs0.tick_params(axis = 'x', labelsize = font_tick)  # to Set Matplotlib Tick Labels Font Size
    axs0.tick_params(axis = 'y', labelsize = font_tick)
    ticks = np.arange(0, 0.18, 0.02)
    #axs0.set_yticks(ticks)
    axs0.set_ylabel(y_axis_label, fontsize = font_tick)
    axs0.set_title(titel, fontsize= font_tick)
    axs0.grid(True)
    return

titel_kind = ['', 'NWP', 'ART']
run_type = ['No-Volcano', 'Volcano',  'MODIS']
x_name = 'total column $ \mathrm{N_d}$ ($\mathrm{cm^{-2}}$)'
y_name = 'Relative Frequency'

numdays = 5

allvar_in_per_nwp = []
allvar_out_per_nwp = []
allvar_in_con_nwp = []
allvar_out_con_nwp = []

allvar_in_modis = []
allvar_out_modis = []

for i in range(numdays):
    print(i)
    con_inside_nwp, con_outside_nwp, per_inside_nwp, per_outside_nwp = get_in_out_nwp_cosp(i)
    allvar_in_per_nwp = np.concatenate((allvar_in_per_nwp, per_inside_nwp), axis = 0)
    allvar_out_per_nwp = np.concatenate((allvar_out_per_nwp, per_outside_nwp), axis = 0)
    allvar_in_con_nwp = np.concatenate((allvar_in_con_nwp, con_inside_nwp), axis = 0)
    allvar_out_con_nwp = np.concatenate((allvar_out_con_nwp, con_outside_nwp), axis = 0)

    in_modis, out_modis = get_in_out_modis(i)
    allvar_in_modis = np.concatenate((allvar_in_modis, in_modis), axis=0)
    allvar_out_modis = np.concatenate((allvar_out_modis, out_modis), axis=0)

range_nwp = (0, 3e8)
#range_art = (0, 5e7)
fig, ((axs0, axs1)) = plt.subplots(1, 2, figsize = (15, 11))
pdf_var_prepare(axs0, var = allvar_in_modis, range = range_nwp, color = 'black', label = run_type[2],
                titel = 'Inside_Plume ' + titel_kind[0])
pdf_var_prepare(axs0, var = allvar_in_con_nwp, range = range_nwp, color = 'blue', label = run_type[0],
                titel = 'Inside_Plume ' + titel_kind[0])
pdf_var_prepare(axs0, var = allvar_in_per_nwp, range = range_nwp, color = 'red', label = run_type[1],
                titel = 'Inside_Plume ' + titel_kind[0], x_axis_label = x_name, y_axis_label = y_name)


pdf_var_prepare(axs1, var = allvar_out_modis, range = range_nwp, color = 'black', label = run_type[2],
                titel = 'Outside_Plume ' + titel_kind[0])
pdf_var_prepare(axs1, var = allvar_out_con_nwp, range = range_nwp, color = 'blue', label = run_type[0],
                titel = 'Outside_Plume ' + titel_kind[0])
pdf_var_prepare(axs1, var = allvar_out_per_nwp, range = range_nwp, color = 'red', label = run_type[1],
                titel = 'Outside_Plume ' + titel_kind[0], x_axis_label = x_name, y_axis_label = y_name)

plt.savefig('./manuscript_figures/nd_burden_cosp_modis_all_days_no_legend.png')
plt.savefig('./manuscript_figures/nd_burden_cosp_modis_all_days_no_legend.pdf')

plt.show()