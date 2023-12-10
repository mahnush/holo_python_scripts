import matplotlib.pyplot as plt
import postpro_func
import numpy as np
import matplotlib
import numpy.ma as ma


hole_onetenth = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/holo_auto_runs/2_sep_con_auto_onetenth.nc'
holo_ten = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/holo_auto_runs/2_sep_con_auto_ten.nc'
holo_control = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/holo_auto_runs/2_sep_con_control.nc'

per_onetenth = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/holo_auto_runs/2_sep_per_auto_onetenth.nc'
per_holo_ten = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/holo_auto_runs/2_sep_per_auto_ten.nc'
per_holo_control = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/holo_auto_runs/2_sep_per_control.nc'

hole_onetenth_3sep = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/holo_auto_runs/3_sep_con_auto_onetenth.nc'
holo_ten_3sep = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/holo_auto_runs/3_sep_con_auto_ten.nc'
holo_control_3sep = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/holo_auto_runs/3_sep_con_control.nc'

per_onetenth_3sep = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/holo_auto_runs/3_sep_per_auto_onetenth.nc'
per_holo_ten_3sep = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/holo_auto_runs/3_sep_per_auto_ten.nc'
per_holo_control_3sep = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/holo_auto_runs/3_sep_per_control.nc'


model_con = [holo_control, holo_ten, hole_onetenth, holo_control_3sep, holo_ten_3sep, hole_onetenth_3sep]
model_per = [per_holo_control, per_holo_ten, per_onetenth, per_holo_control_3sep, per_holo_ten_3sep, per_onetenth_3sep]
omps_path_2sep = '/home/mhaghigh/nc_file_in_outplume/so2_Iceland_2sep_0.7res.nc'
modis_path_2sep = '/home/mhaghigh/nc_file_in_outplume/geo_modis_data_2d.nc'
omps_path_3sep = '/home/mhaghigh/nc_file_in_outplume/so2_Iceland_3sep_0.7res.nc'
modis_path_3sep = '/home/mhaghigh/nc_file_in_outplume/geo_modis_data_3d.nc'

var_name_model = ['nd_dw', 'lwp_dw', 'tau_dw', 're_dw']
var_name_modis = ['nd', 'lwp', 'tau', 're', 'cf']
modis_data = [modis_path_2sep, modis_path_2sep, modis_path_2sep, modis_path_3sep, modis_path_3sep, modis_path_3sep]
omps_data = [omps_path_2sep, omps_path_2sep, omps_path_2sep, omps_path_3sep, omps_path_3sep, omps_path_3sep]

def get_in_out(i):
    import scipy.interpolate as sci
    lat_fine = postpro_func.read_nc(model_con[i], 'lat')
    lon_fine = postpro_func.read_nc(model_con[i], 'lon')
    control_var_lando = postpro_func.read_nc(model_con[i], var_name_model[1])[:, :]*1e3
    perturbed_var_lando = postpro_func.read_nc(model_per[i], var_name_model[1])[ :, :]*1e3
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



def weight(var):
    weight = np.zeros_like(var) + 1. / (var.size)
    return weight


def lable_hist(var):
    median = str(np.median(var))
    mean = str((round(np.mean(var))))
    std = str(np.std(var))
    #lable = '('+'mean = ' + mean +')'
    lable = ' (' + mean + ' $\mathrm{g m^{-2}}$' + ')'
    return lable


x_name =  '$ \mathrm{LWP}$ ($\mathrm{g m^{-2}}$)'
y_name = 'Relative Frequency'

def pdf_var_prepare(axs0, var, color, label, titel, x_axis_label = '', y_axis_label = ''):
    font_legend = 15
    font_tick = 20
    numbin = np.arange(0, 1000, 10)

    line_width = 2
    axs0.hist(var,  bins=numbin, weights=weight(var),  histtype='step',
         linewidth=line_width, color= color, label= label + lable_hist(var), log = True)
    #axs0.hist(var,  bins=numbin, weights=weight(var), histtype='step',
    #       linewidth=line_width, color= color, label= label , log = True)
    axs0.legend(loc = 'upper right', fontsize = font_legend, frameon = True)
    axs0.set_xlabel(x_axis_label, fontsize= font_tick)
    axs0.tick_params(axis = 'x', labelsize = font_tick)  # to Set Matplotlib Tick Labels Font Size
    axs0.tick_params(axis = 'y', labelsize = font_tick)
    ticks = np.arange(0, 0.18, 0.02)
    ticks_x = np.arange(0, 1200, 200)
    #axs0.set_yticks(ticks)
    axs0.set_xticks(ticks_x)
    axs0.set_ylabel(y_axis_label, fontsize = font_tick)
    axs0.set_title(titel, fontsize= font_tick)
    axs0.grid(True)
    return

control_default_in, control_default_out, per_default_in, per_default_out, modis_in, modis_out = get_in_out(0)
control_ten_in, control_ten_out, per_ten_in, per_ten_out, modis_in, modis_out = get_in_out(1)
control_onetenth_in, control_onetenth_out, per_onetenth_in, per_onetenth_out, modis_in, modis_out = get_in_out(2)

control_default_in_3sep, control_default_out_3sep, per_default_in_3sep, per_default_out_3sep, \
modis_in_3sep, modis_out_3sep = get_in_out(3)
control_ten_in_3sep, control_ten_out_3sep, per_ten_in_3sep, per_ten_out_3sep, modis_in_3sep,\
modis_out_3sep = get_in_out(4)
control_onetenth_in_3sep, control_onetenth_out_3sep, per_onetenth_in_3sep,\
per_onetenth_out_3sep, modis_in_3sep, modis_out_3sep = get_in_out(5)

control_default_in_all = np.concatenate((control_default_in_3sep, control_default_in), axis = 0)
control_default_out_all = np.concatenate((control_onetenth_out_3sep, control_default_out), axis = 0)
per_default_in_all = np.concatenate((per_default_in_3sep, per_default_in),  axis = 0)
per_default_out_all = np.concatenate((per_default_out_3sep, per_default_out),  axis = 0)
modis_in_all = np.concatenate((modis_in, modis_in_3sep),  axis = 0)
modis_out_all = np.concatenate((modis_out, modis_out_3sep),  axis = 0)
control_ten_in_all = np.concatenate((control_ten_in, control_ten_in_3sep),  axis = 0)
control_ten_out_all = np.concatenate((control_ten_out_3sep, control_ten_out),  axis = 0)
per_ten_in_all = np.concatenate((per_ten_in_3sep, per_ten_in),  axis = 0)
per_ten_out_all = np.concatenate((per_ten_out, per_ten_out_3sep), axis = 0)

control_onetenth_in_all = np.concatenate((control_onetenth_in, control_onetenth_in_3sep), axis = 0)
control_onetenth_out_all = np.concatenate((control_onetenth_out, control_onetenth_out_3sep), axis = 0)
per_onetenth_in_all = np.concatenate((per_onetenth_in_3sep, per_onetenth_in), axis = 0)
per_onetenth_out_all = np.concatenate((per_onetenth_out, per_onetenth_out_3sep), axis = 0)

titel_kind = ['(default ac rate)', '(10 times ac rate)', '(0.1 times ac rate)']
run_type = ['No-Volcano', 'Volcano',  'MODIS']

fig, ((axs0, axs1), (axs2, axs3), (axs4, axs5)) = plt.subplots(3, 2, figsize = (10, 15))
pdf_var_prepare(axs0, var = control_default_in, color = 'blue', label = run_type[0] ,
                titel = 'Inside_Plume ' + titel_kind[0])
pdf_var_prepare(axs0, var = per_default_in, color = 'red', label = run_type[1],
                titel = 'Inside_Plume ' + titel_kind[0])
pdf_var_prepare(axs0, var = modis_in, color = 'black', label = run_type[2],
                titel = 'Inside_Plume ' + titel_kind[0], y_axis_label = y_name)
pdf_var_prepare(axs1, var = control_default_out, color = 'blue', label = run_type[0],
                titel = 'Outside_Plume ' + titel_kind[0])
pdf_var_prepare(axs1, var = per_default_out, color = 'red', label = run_type[1],
                titel = 'Outside_Plume ' + titel_kind[0])
pdf_var_prepare(axs1, var = modis_out, color = 'black', label = run_type[2],
                titel = 'Outside_Plume ' + titel_kind[0])


pdf_var_prepare(axs2, var = control_ten_in, color = 'blue', label = run_type[0],
                titel = 'Inside_Plume ' + titel_kind[1])
pdf_var_prepare(axs2, var = per_ten_in, color = 'red', label = run_type[1],
                titel = 'Inside_Plume ' + titel_kind[1])
pdf_var_prepare(axs2, var = modis_in, color = 'black', label = run_type[2],
                titel = 'Inside_Plume ' + titel_kind[1], y_axis_label = y_name)
pdf_var_prepare(axs3, var = control_ten_out, color = 'blue', label = run_type[0],
                titel = 'Outside_Plume ' + titel_kind[1])
pdf_var_prepare(axs3, var = per_ten_out, color = 'red', label = run_type[1],
                titel = 'Outside_Plume ' + titel_kind[1])
pdf_var_prepare(axs3, var = modis_out, color = 'black', label = run_type[2] ,
                titel = 'Outside_Plume ' + titel_kind[1])


pdf_var_prepare(axs4, var = control_onetenth_in, color = 'blue', label = run_type[0],
                titel = 'Inside_Plume ' + titel_kind[2], y_axis_label = y_name)
pdf_var_prepare(axs4, var = per_onetenth_in, color = 'red', label = run_type[1],
                titel = 'Inside_Plume ' + titel_kind[2])
pdf_var_prepare(axs4, var = modis_in, color = 'black', label = run_type[2],
                titel = 'Inside_Plume ' + titel_kind[2], x_axis_label = x_name, y_axis_label = y_name)
pdf_var_prepare(axs5, var = control_onetenth_out, color = 'blue', label = run_type[0],
                titel = 'Outside_Plume ' + titel_kind[2])
pdf_var_prepare(axs5, var = per_onetenth_out, color = 'red', label = run_type[1],
                titel = 'Outside_Plume ' + titel_kind[2])
pdf_var_prepare(axs5, var = modis_out, color = 'black', label = run_type[2],
                titel = 'Outside_Plume ' + titel_kind[2], x_axis_label = x_name)
fig.suptitle('2 Sep', fontsize=20)
#plt.tight_layout()
#plt.savefig('auto_conversion_test_2sep.png')
#plt.savefig('auto_conversion_test_2sep.pdf')
#plt.show()

fig, ((axs0, axs1), (axs2, axs3), (axs4, axs5)) = plt.subplots(3, 2, figsize = (10, 15))
pdf_var_prepare(axs0, var = control_default_in_3sep, color = 'blue', label = run_type[0] ,
                titel = 'Inside_Plume ' + titel_kind[0])
pdf_var_prepare(axs0, var = per_default_in_3sep, color = 'red', label = run_type[1],
                titel = 'Inside_Plume ' + titel_kind[0])
pdf_var_prepare(axs0, var = modis_in_3sep, color = 'black', label = run_type[2],
                titel = 'Inside_Plume ' + titel_kind[0])
pdf_var_prepare(axs1, var = control_default_out_3sep, color = 'blue', label = run_type[0],
                titel = 'Outside_Plume ' + titel_kind[0])
pdf_var_prepare(axs1, var = per_default_out_3sep, color = 'red', label = run_type[1],
                titel = 'Outside_Plume ' + titel_kind[0])
pdf_var_prepare(axs1, var = modis_out_3sep, color = 'black', label = run_type[2],
                titel = 'Outside_Plume ' + titel_kind[0])


pdf_var_prepare(axs2, var = control_ten_in_3sep, color = 'blue', label = run_type[0],
                titel = 'Inside_Plume ' + titel_kind[1])
pdf_var_prepare(axs2, var = per_ten_in_3sep, color = 'red', label = run_type[1],
                titel = 'Inside_Plume ' + titel_kind[1])
pdf_var_prepare(axs2, var = modis_in_3sep, color = 'black', label = run_type[2],
                titel = 'Inside_Plume ' + titel_kind[1])
pdf_var_prepare(axs3, var = control_ten_out_3sep, color = 'blue', label = run_type[0],
                titel = 'Outside_Plume ' + titel_kind[1])
pdf_var_prepare(axs3, var = per_ten_out_3sep, color = 'red', label = run_type[1],
                titel = 'Outside_Plume ' + titel_kind[1])
pdf_var_prepare(axs3, var = modis_out_3sep, color = 'black', label = run_type[2] ,
                titel = 'Outside_Plume ' + titel_kind[1])


pdf_var_prepare(axs4, var = control_onetenth_in_3sep, color = 'blue', label = run_type[0],
                titel = 'Inside_Plume ' + titel_kind[2])
pdf_var_prepare(axs4, var = per_onetenth_in_3sep, color = 'red', label = run_type[1],
                titel = 'Inside_Plume ' + titel_kind[2])
pdf_var_prepare(axs4, var = modis_in_3sep, color = 'black', label = run_type[2],
                titel = 'Inside_Plume ' + titel_kind[2])
pdf_var_prepare(axs5, var = control_onetenth_out_3sep, color = 'blue', label = run_type[0],
                titel = 'Outside_Plume ' + titel_kind[2])
pdf_var_prepare(axs5, var = per_onetenth_out_3sep, color = 'red', label = run_type[1],
                titel = 'Outside_Plume ' + titel_kind[2])
pdf_var_prepare(axs5, var = modis_out_3sep, color = 'black', label = run_type[2],
                titel = 'Outside_Plume ' + titel_kind[2], x_axis_label = x_name, y_axis_label = y_name)
fig.suptitle('3 Sep', fontsize=20)
#plt.tight_layout()
#plt.savefig('auto_conversion_test_3sep.png')
#plt.savefig('auto_conversion_test_3sep.pdf')
#plt.show()
font_tick = 20
fig, ((axs0, axs1), (axs2, axs3), (axs4, axs5)) = plt.subplots(3, 2, figsize = (11, 15), sharey = True)
pdf_var_prepare(axs0, var = control_default_in_all, color = 'blue', label = run_type[0] ,
                titel = 'Inside_Plume ' + titel_kind[0])
pdf_var_prepare(axs0, var = per_default_in_all, color = 'red', label = run_type[1],
                titel = 'Inside_Plume ' + titel_kind[0])
pdf_var_prepare(axs0, var = modis_in_all, color = 'black', label = run_type[2],
                titel = 'Inside_Plume ' + titel_kind[0], y_axis_label = y_name)
axs0.annotate("(a)", xy=(0.05, 0.95), xycoords="axes fraction", fontsize= font_tick)

pdf_var_prepare(axs1, var = control_default_out_3sep, color = 'blue', label = run_type[0],
                titel = 'Outside_Plume ' + titel_kind[0])
pdf_var_prepare(axs1, var = per_default_out_3sep, color = 'red', label = run_type[1],
                titel = 'Outside_Plume ' + titel_kind[0])
pdf_var_prepare(axs1, var = modis_out_all, color = 'black', label = run_type[2],
                titel = 'Outside_Plume ' + titel_kind[0])
axs1.annotate("(b)", xy=(0.05, 0.95), xycoords="axes fraction", fontsize= font_tick)

pdf_var_prepare(axs2, var = control_ten_in_all, color = 'blue', label = run_type[0],
                titel = 'Inside_Plume ' + titel_kind[1])
pdf_var_prepare(axs2, var = per_ten_in_all, color = 'red', label = run_type[1],
                titel = 'Inside_Plume ' + titel_kind[1])
pdf_var_prepare(axs2, var = modis_in_all, color = 'black', label = run_type[2],
                titel = 'Inside_Plume ' + titel_kind[1], y_axis_label = y_name)
axs2.annotate("(c)", xy=(0.05, 0.95), xycoords="axes fraction", fontsize= font_tick)

pdf_var_prepare(axs3, var = control_ten_out_all, color = 'blue', label = run_type[0],
                titel = 'Outside_Plume ' + titel_kind[1])
pdf_var_prepare(axs3, var = per_ten_out_all, color = 'red', label = run_type[1],
                titel = 'Outside_Plume ' + titel_kind[1])
pdf_var_prepare(axs3, var = modis_out_all, color = 'black', label = run_type[2],
                titel = 'Outside_Plume ' + titel_kind[1])
axs3.annotate("(d)", xy=(0.05, 0.95), xycoords="axes fraction", fontsize= font_tick)

pdf_var_prepare(axs4, var = control_onetenth_in_all, color = 'blue', label = run_type[0],
                titel = 'Inside_Plume ' + titel_kind[2])
pdf_var_prepare(axs4, var = per_onetenth_in_all, color = 'red', label = run_type[1],
                titel = 'Inside_Plume ' + titel_kind[2])
pdf_var_prepare(axs4, var = modis_in_all, color = 'black', label = run_type[2],
                titel = 'Inside_Plume ' + titel_kind[2], x_axis_label = x_name, y_axis_label = y_name)
axs4.annotate("(e)", xy=(0.05, 0.95), xycoords="axes fraction", fontsize= font_tick)

pdf_var_prepare(axs5, var = control_onetenth_out_all, color = 'blue', label = run_type[0],
                titel = 'Outside_Plume ' + titel_kind[2])
pdf_var_prepare(axs5, var = per_onetenth_out_all, color = 'red', label = run_type[1],
                titel = 'Outside_Plume ' + titel_kind[2])
pdf_var_prepare(axs5, var = modis_out_all, color = 'black', label = run_type[2],
                titel = 'Outside_Plume ' + titel_kind[2], x_axis_label = x_name)
axs5.annotate("(f)", xy=(0.05, 0.95), xycoords="axes fraction", fontsize= font_tick)

#fig.suptitle('2 and 3 Sep', fontsize=16)
plt.tight_layout()
#plt.savefig('./manuscript_figures/sensitivity_ac_rate_LWP_no_legend.png')
#plt.savefig('./manuscript_figures/sensitivity_ac_rate_LWP_no_legend.pdf')
plt.show()