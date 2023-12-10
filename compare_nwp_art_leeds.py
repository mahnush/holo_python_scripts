"""
this is the block which I want to include the results from the model in Leeds
"""
import matplotlib.pyplot as plt
import postpro_func
import numpy as np
import numpy.ma as ma

leeds_no_vol_path = '/home/mhaghigh/leeds_holo/ADVANCE/NoVolc/regridded_data/'
leeds_vol_path = '/home/mhaghigh/leeds_holo/ADVANCE/Volc/regridded_data/'

con_2sep = '2sep_con_all_data.nc'
con_3sep = '3sep_con_all_data.nc'
per_2sep = '2sep_per_all_data.nc'
per_3sep = '3sep_per_all_data.nc'

var_names_leeds = ['cdnc_vert_ave_wtd_by_lwp', 'load_so2', 'lwp']

cdcn_2sep_con = postpro_func.read_nc(leeds_no_vol_path + con_2sep, var_name = var_names_leeds[0])
lwp_2sep_con = postpro_func.read_nc(leeds_no_vol_path + con_2sep, var_name = var_names_leeds[2])*1e3
so2_2sep_con = postpro_func.read_nc(leeds_no_vol_path+con_2sep, var_name = var_names_leeds[1])

cdcn_2sep_con_mean = np.ma.mean(cdcn_2sep_con, axis = 0)
lwp_2sep_con_mean = np.ma.mean(lwp_2sep_con, axis = 0)
so2_2sep_con_mean = np.ma.mean(so2_2sep_con, axis = 0)

cdcn_2sep_per = postpro_func.read_nc(leeds_vol_path + per_2sep, var_name = var_names_leeds[0])
lwp_2sep_per = postpro_func.read_nc(leeds_vol_path + per_2sep, var_name = var_names_leeds[2])*1e3
so2_2sep_per = postpro_func.read_nc(leeds_vol_path + per_2sep, var_name = var_names_leeds[1])

cdcn_2sep_per_mean = np.ma.mean(cdcn_2sep_per, axis = 0)
lwp_2sep_per_mean = np.ma.mean(lwp_2sep_per, axis = 0)
so2_2sep_per_mean = np.ma.mean(so2_2sep_per, axis = 0)

cdcn_3sep_con = postpro_func.read_nc(leeds_no_vol_path + con_3sep, var_name = var_names_leeds[0])
lwp_3sep_con = postpro_func.read_nc(leeds_no_vol_path + con_3sep, var_name = var_names_leeds[2])*1e3
so2_3sep_con = postpro_func.read_nc(leeds_no_vol_path+con_3sep, var_name = var_names_leeds[1])

cdcn_3sep_con_mean = np.ma.mean(cdcn_3sep_con, axis = 0)
lwp_3sep_con_mean = np.ma.mean(lwp_3sep_con, axis = 0)
so2_3sep_con_mean = np.ma.mean(so2_3sep_con, axis = 0)

cdcn_3sep_per = postpro_func.read_nc(leeds_vol_path + per_3sep, var_name = var_names_leeds[0])
lwp_3sep_per = postpro_func.read_nc(leeds_vol_path + per_3sep, var_name = var_names_leeds[2])*1e3
so2_3sep_per = postpro_func.read_nc(leeds_vol_path + per_3sep, var_name = var_names_leeds[2])

cdcn_3sep_per_mean = np.ma.mean(cdcn_3sep_per, axis = 0)
lwp_3sep_per_mean = np.ma.mean(lwp_3sep_per, axis = 0)
so2_3sep_per_mean = np.ma.mean(so2_3sep_per, axis = 0)

lat_fine = postpro_func.read_nc(leeds_no_vol_path + con_2sep, 'lat')
lon_fine = postpro_func.read_nc(leeds_no_vol_path + con_2sep, 'lon')

lon_fine = lon_fine[1600:4601]
def get_in_out_art( var_con, var_per, var_so2):
    control_var_lando = var_con[:, 1600:4601]
    perturbed_var_lando = var_per[:, 1600:4601]
    var_so2_lando = var_so2[:, 1600:4601]
    so2_threshold = 5e-5
    control_var = postpro_func.mask_land(control_var_lando, lat_fine, lon_fine)
    perturbed_var = postpro_func.mask_land(perturbed_var_lando, lat_fine, lon_fine)
    so2_mask = postpro_func.mask_land(var_so2_lando, lat_fine, lon_fine)
    con_inside = control_var[so2_mask > so2_threshold]
    con_outside = control_var[so2_mask < so2_threshold]
    per_inside = perturbed_var[so2_mask > so2_threshold]
    per_outside = perturbed_var[so2_mask < so2_threshold]
    con_inside_mask = ma.masked_where(con_inside <= 5., con_inside)
    con_inside_mask = con_inside_mask.compressed()
    con_outside_mask = ma.masked_where(con_outside <= 5., con_outside)
    con_outside_mask = con_outside_mask.compressed()
    per_inside_mask = ma.masked_where(per_inside <= 5., per_inside)
    per_inside_mask = per_inside_mask.compressed()
    per_outside_mask = ma.masked_where(per_outside <= 5., per_outside)
    per_outside_mask = per_outside_mask.compressed()
    return con_inside_mask, con_outside_mask, per_inside_mask, per_outside_mask

con_inside_2sep, con_outside_2sep, per_inside_2sep, per_outside_2sep = \
    get_in_out_art(lwp_2sep_con_mean, lwp_2sep_per_mean, so2_2sep_per_mean)


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


x_name = 'total column $ \mathrm{N_d}$ ($\mathrm{cm^{-2}}$)'
y_name = 'Relative Frequency'

def pdf_var_prepare(axs0, var, range, color, label, titel, x_axis_label = '', y_axis_label = ''):
    font_legend = 15
    font_tick = 20
    numbin = 100
    line_width = 2
    axs0.hist(var,  bins=numbin, range = range, weights=weight(var),  histtype='step',
         linewidth=line_width, color= color, label= label + lable_hist(var))#, density = True)#, log = True)

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

titel_kind = ['Leeds', 'ART']
run_type = ['no-vol', 'vol',  'MODIS']
range_nwp = (0, 1000)
range_art = (0, 1e8)

#fig, ((axs0, axs1), (axs2, axs3)) = plt.subplots(2, 2, figsize = (16, 16))
fig, ((axs0, axs1)) = plt.subplots(1, 2, figsize = (16, 16))
pdf_var_prepare(axs0, var = con_inside_2sep, range = range_nwp, color = 'blue', label = run_type[0],
                titel = 'Inside_Plume ' + titel_kind[0])
pdf_var_prepare(axs0, var = per_inside_2sep, range = range_nwp, color = 'red', label = run_type[1],
                titel = 'Inside_Plume ' + titel_kind[0], y_axis_label = y_name)

pdf_var_prepare(axs1, var = con_outside_2sep, range = range_nwp, color = 'blue', label = run_type[0],
                titel = 'Outside_Plume ' + titel_kind[0])
pdf_var_prepare(axs1, var = per_outside_2sep, range = range_nwp, color = 'red', label = run_type[1],
                titel = 'Outside_Plume ' + titel_kind[0])

plt.show()