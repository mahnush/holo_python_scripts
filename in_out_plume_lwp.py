import matplotlib.pyplot as plt
import postpro_func
import numpy as np
import numpy.ma as ma

path_c = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/fix_cosp_runs/fix_cosp_control/'
path_p = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/fix_cosp_runs/fix_cosp_perturbed/'

omps_path_1sep = '/home/mhaghigh/nc_file_in_outplume/so2_Iceland_1sep_0.7res.nc'
omps_path_2sep = '/home/mhaghigh/nc_file_in_outplume/so2_Iceland_2sep_0.7res.nc'
omps_path_3sep = '/home/mhaghigh/nc_file_in_outplume/so2_Iceland_3sep_0.7res.nc'
omps_path_4sep = '/home/mhaghigh/nc_file_in_outplume/so2_Iceland_4sep_0.7res.nc'
omps_path_5sep = '/home/mhaghigh/nc_file_in_outplume/so2_Iceland_5sep_0.7res.nc'

modis_path_1sep = '/home/mhaghigh/nc_file_in_outplume/geo_modis_data_1d.nc'
modis_path_2sep = '/home/mhaghigh/nc_file_in_outplume/geo_modis_data_2d.nc'
modis_path_3sep = '/home/mhaghigh/nc_file_in_outplume/geo_modis_data_3d.nc'
modis_path_4sep = '/home/mhaghigh/nc_file_in_outplume/geo_modis_data_4d.nc'
modis_path_5sep = '/home/mhaghigh/nc_file_in_outplume/geo_modis_data_5d.nc'

icon_output_names_1sep = '1_sep_mean.nc'
icon_output_names_2sep = '2_sep_mean.nc'
icon_output_names_3sep = '3_sep_mean.nc'
icon_output_names_4sep = '4_sep_mean.nc'
icon_output_names_5sep = '5_sep_mean.nc'
icon_output_names_1sep_per = 'per_1_sep_mean.nc'
icon_output_names_2sep_per = 'per_2_sep_mean.nc'
icon_output_names_3sep_per = 'per_3_sep_mean.nc'
icon_output_names_4sep_per = 'per_4_sep_mean.nc'
icon_output_names_5sep_per = 'per_5_sep_mean.nc'
model_con_1sep = path_c + icon_output_names_1sep
model_con_2sep = path_c + icon_output_names_2sep
model_con_3sep = path_c + icon_output_names_3sep
model_con_4sep = path_c + icon_output_names_4sep
model_con_5sep = path_c + icon_output_names_5sep

model_per_1sep = path_p + icon_output_names_1sep_per
model_per_2sep = path_p + icon_output_names_2sep_per
model_per_3sep = path_p + icon_output_names_3sep_per
model_per_4sep = path_p + icon_output_names_4sep_per
model_per_5sep = path_p + icon_output_names_5sep_per
omps_data = [omps_path_1sep, omps_path_2sep, omps_path_3sep, omps_path_4sep, omps_path_5sep]
model_per = [model_per_1sep, model_per_2sep, model_per_3sep, model_per_4sep, model_per_5sep]
model_con = [model_con_1sep, model_con_2sep, model_con_3sep, model_con_4sep, model_con_5sep]
modis_data = [modis_path_1sep, modis_path_2sep, modis_path_3sep, modis_path_4sep, modis_path_5sep]

var_name_model = ['nd_dw', 'lwp_dw', 'tau_dw', 're_dw']
var_name_modis = ['nd', 'lwp', 'tau', 're', 'cf']
numdays = 5
#fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(30, 20))
#i =0
#for ax in axes.flatten():
def get_in_out(i):
    import scipy.interpolate as sci
    lat_fine = postpro_func.read_nc(model_con[i], 'lat')
    lon_fine = postpro_func.read_nc(model_con[i], 'lon')
    control_var_lando = postpro_func.read_nc(model_con[i], var_name_model[1])[:, :]*1000
    perturbed_var_lando = postpro_func.read_nc(model_per[i], var_name_model[1])[ :, :]*1000
    control_var = postpro_func.mask_land(control_var_lando, lat_fine, lon_fine)
    perturbed_var = postpro_func.mask_land(perturbed_var_lando, lat_fine, lon_fine)
    modis_var_lando = postpro_func.read_nc(modis_data[i], var_name_modis[1])
    modis_var_lando = np.transpose(modis_var_lando)
    modis_var = postpro_func.mask_land(modis_var_lando, lat_fine, lon_fine)
    so2 = postpro_func.read_nc(omps_data[i], 'so2_TRL')[0:43, 0:115]
    lat_so2 = postpro_func.read_nc(omps_data[i], 'lat')
    lon_so2 = postpro_func.read_nc(omps_data[i], 'lon')
    lon_coarse = lon_so2[0, 0:115]
    lat_coarse = lat_so2[0:43, 0]
    so2_mask = np.ma.filled(so2, fill_value=0)
    f = sci.RectBivariateSpline(lat_coarse, lon_coarse, so2_mask)
    scale_interp = f(lat_fine, lon_fine)
    con_inside = control_var[scale_interp > 1.0]
    con_outside = control_var[scale_interp < 1.0]
    per_inside = perturbed_var[scale_interp > 1.0]
    per_outside = perturbed_var[scale_interp < 1.0]
    modis_inside = modis_var[scale_interp > 1.0]
    modis_outside = modis_var[scale_interp < 1.0]
    con_inside_mask = ma.masked_where(con_inside <= 0, con_inside)
    #con_inside_mask = ma.masked_where(con_inside_mask_1 >= 2000, con_inside_mask_1)
    con_inside_mask = con_inside_mask.compressed()
    con_outside_mask = ma.masked_where(con_outside <= 0., con_outside)
    con_outside_mask = con_outside_mask.compressed()
    per_inside_mask = ma.masked_where(per_inside <= 0, per_inside)
    #per_inside_mask = ma.masked_where(per_inside_mask_1 >= 2000, per_inside_mask_1)
    per_inside_mask = per_inside_mask.compressed()
    per_outside_mask = ma.masked_where(per_outside <= 0., per_outside)
    per_outside_mask = per_outside_mask.compressed()
    modis_inside_mask_1 = ma.masked_where(modis_inside <= 0, modis_inside)
    modis_inside_mask = ma.masked_where(modis_inside_mask_1 >= 1500, modis_inside_mask_1)
    modis_inside_mask = modis_inside_mask.compressed()
    modis_outside_mask = ma.masked_where(modis_outside <= 0., modis_outside)
    modis_outside_mask = modis_outside_mask.compressed()
    return con_inside_mask, con_outside_mask, per_inside_mask, per_outside_mask, modis_inside_mask, modis_outside_mask

allvar_in_per = []
allvar_out_per = []
allvar_in_con = []
allvar_out_con = []
allvar_in_modis = []
allvar_out_modis = []
for i in range(numdays):
    con_inside, con_outside, per_inside, per_outside, modis_inside, modis_outside = get_in_out(i)
    allvar_in_per = np.concatenate((allvar_in_per, per_inside), axis=0)
    allvar_out_per = np.concatenate((allvar_out_per, per_outside), axis=0)
    allvar_in_con = np.concatenate((allvar_in_con, con_inside), axis=0)
    allvar_out_con = np.concatenate((allvar_out_con, con_outside), axis=0)
    allvar_in_modis = np.concatenate((allvar_in_modis, modis_inside), axis=0)
    allvar_out_modis = np.concatenate((allvar_out_modis, modis_outside), axis=0)


def weight(var):
    #weight = (1 + np.zeros(len(var))) / len(var)
    weight = np.zeros_like(var) + 1. / (var.size)
    return weight


def lable_hist(var):
    median = str(round(np.median(var)))
    mean = str(round(np.mean(var)))
    std = str(np.std(var))
    lable = '('+'median = ' + median + ' ,' + 'mean = ' + mean +')'
    return lable


fig, (axs0, axs1) = plt.subplots(1, 2, figsize = (12, 6))
numbin = np.arange(.0001,1000,10)
font_legend = 10
font_lable = 15
line_width = 2
font_tick = 13
#name = '$ \mathrm{N_d}$ ($\mathrm{cm^{-3}}$)'
name =  '$ \mathrm{LWP}$ ($\mathrm{g m^{-2}}$)'
axs0.hist(allvar_in_per, bins=numbin, weights=weight(allvar_in_per) , histtype='step',
          linewidth=line_width, color='red', label='Volcano '+lable_hist(allvar_in_per))

axs0.hist(allvar_in_con,  bins=numbin, weights=weight(allvar_in_con),  histtype='step',
          linewidth=line_width, color='blue', label='No-Volcano '+lable_hist(allvar_in_con))
axs0.hist(allvar_in_modis, bins=numbin, weights=weight(allvar_in_modis),  histtype='step',
         linewidth=line_width, color='black',  label='MODIS '+lable_hist(allvar_in_modis))
axs0.legend(loc='upper right', fontsize=font_legend, frameon=True)
ticks = np.arange(0, 0.35, 0.05)
#ticks_x = np.arange(0,1200,200)
#axs0.set_yticks(ticks)
axs0.tick_params(axis='x', labelsize=font_tick)  # to Set Matplotlib Tick Labels Font Size
axs0.tick_params(axis='y', labelsize=font_tick)
axs0.set_xlabel(name, fontsize=font_lable)
axs0.set_ylabel('Relative Frequency', fontsize=font_lable)
axs1.hist(allvar_out_per, bins=numbin, weights=weight(allvar_out_per),  histtype='step',
          linewidth=line_width, color = 'red', label='Volcano '+lable_hist(allvar_out_per))
axs1.hist(allvar_out_con, bins=numbin, weights=weight(allvar_out_con),  histtype='step',
          linewidth=line_width, color = 'blue', label='No-Volcano '+lable_hist(allvar_out_con))
axs1.hist(allvar_out_modis, bins=numbin, weights=weight(allvar_out_modis), histtype='step',
          linewidth=line_width, color = 'black', label='MODIS ' + lable_hist(allvar_out_modis))

axs1.legend(loc='upper right', fontsize=font_legend, frameon=True)
#axs1.set_yticks(ticks)
#axs0.set_xticks(ticks_x)
#axs1.set_xticks(ticks_x)
axs1.tick_params(axis='x', labelsize=font_tick)  # to Set Matplotlib Tick Labels Font Size
axs1.tick_params(axis='y', labelsize=font_tick)
#axs1.yticks(" ")
#axs1.set_yticklabels([])
axs1.set_xlabel(name, fontsize=font_lable)
#axs1.set_ylabel('probability density function', fontsize=font_lable)

axs0.set_title('Inside Plume', fontsize= font_lable)
axs1.set_title('Outside Plume', fontsize=font_lable)
#axs0.annotate('(a)', xy=(-30, 0.079), size=font_lable)
#axs1.annotate('(b)', xy=(-30, 0.0785), size=font_lable)
plt.tight_layout()
axs0.grid(True)
axs1.grid(True)
plt.savefig('lwp.png')
plt.savefig('lwp.pdf')
plt.show()