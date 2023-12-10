import matplotlib.pyplot as plt
import postpro_func
import numpy as np
import matplotlib
import numpy.ma as ma


hole_onetenth = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/holo_auto_runs/' \
                'albedo_2_sep_con_onetenth_tqc.nc'
holo_ten = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/holo_auto_runs/' \
           'albedo_2_sep_con_ten_tqc.nc'
holo_control = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/holo_auto_runs/' \
               'albedo_2_sep_con_tqc.nc'

per_onetenth = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/holo_auto_runs/' \
               'albedo_2_sep_per_onetenth_tqc.nc'
per_holo_ten = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/holo_auto_runs/' \
               'albedo_2_sep_per_ten_tqc.nc'
per_holo_control = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/holo_auto_runs/' \
                   'albedo_2_sep_per_tqc.nc'

hole_onetenth_3sep = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/holo_auto_runs/' \
                     'albedo_3_sep_con_onetenth_tqc.nc'
holo_ten_3sep = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/holo_auto_runs/' \
                'albedo_3_sep_con_ten_tqc.nc'
holo_control_3sep = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/holo_auto_runs' \
                    '/albedo_3_sep_con_tqc.nc'

per_onetenth_3sep = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/holo_auto_runs/' \
                    'albedo_3_sep_per_onetenth_tqc.nc'
per_holo_ten_3sep = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/holo_auto_runs/' \
                    'albedo_3_sep_per_ten_tqc.nc'
per_holo_control_3sep = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/holo_auto_runs' \
                        '/albedo_3_sep_per_tqc.nc'


model_con = [holo_control, holo_ten, hole_onetenth, holo_control_3sep, holo_ten_3sep, hole_onetenth_3sep]
model_per = [per_holo_control, per_holo_ten, per_onetenth, per_holo_control_3sep, per_holo_ten_3sep, per_onetenth_3sep]
omps_path_2sep = '/home/mhaghigh/nc_file_in_outplume/so2_Iceland_2sep_0.7res.nc'
#modis_path_2sep = '/home/mhaghigh/nc_file_in_outplume/geo_modis_data_2d.nc'
omps_path_3sep = '/home/mhaghigh/nc_file_in_outplume/so2_Iceland_3sep_0.7res.nc'
#modis_path_3sep = '/home/mhaghigh/nc_file_in_outplume/geo_modis_data_3d.nc'

var_name_model = ['albedo_TOA']
var_name_modis = ['nd', 'lwp', 'tau', 're', 'cf']
#modis_data = [modis_path_2sep, modis_path_2sep, modis_path_2sep, modis_path_3sep, modis_path_3sep, modis_path_3sep]
omps_data = [omps_path_2sep, omps_path_2sep, omps_path_2sep, omps_path_3sep, omps_path_3sep, omps_path_3sep]

def get_in_out(i):
    import scipy.interpolate as sci
    lat_fine = postpro_func.read_nc(model_con[i], 'lat')
    lon_fine = postpro_func.read_nc(model_con[i], 'lon')
    control_var_lando = postpro_func.read_nc(model_con[i], var_name_model[0])[:, :]
    perturbed_var_lando = postpro_func.read_nc(model_per[i], var_name_model[0])[:, :]
    control_var = postpro_func.mask_land(control_var_lando, lat_fine, lon_fine)
    perturbed_var = postpro_func.mask_land(perturbed_var_lando, lat_fine, lon_fine)
    #modis_var_lando = postpro_func.read_nc(modis_data[i], var_name_modis[1])
    #modis_var_lando = np.transpose(modis_var_lando)
    #modis_var = postpro_func.mask_land(modis_var_lando, lat_fine, lon_fine)
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
    #modis_inside = modis_var[scale_interp > 1.0]
    #modis_outside = modis_var[scale_interp < 1.0]
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
    #modis_inside_mask_1 = ma.masked_where(modis_inside <= 0, modis_inside)
    #modis_inside_mask = ma.masked_where(modis_inside_mask_1 >= 1500, modis_inside_mask_1)
    #modis_inside_mask = modis_inside_mask.compressed()
    #modis_outside_mask = ma.masked_where(modis_outside <= 0., modis_outside)
    #modis_outside_mask = modis_outside_mask.compressed()
    return con_inside_mask, con_outside_mask, per_inside_mask, per_outside_mask#, modis_inside_mask, modis_outside_mask



def weight(var):
    weight = np.zeros_like(var) + 1. / (var.size)
    return weight


def lable_hist(var):
    median = str(np.median(var))
    mean = str((round(np.ma.mean(var), 3)))
    #median = str((round(np.ma.median(var), 3)))
    #mean = str((np.mean(var)))
    std = str(np.std(var))
    lable = '('+'mean = ' + mean +')'
    return lable


x_name = 'TOA albedo'
y_name = 'Relative Frequency'

def pdf_var_prepare(axs0, var, color, label, titel, x_axis_label = '', y_axis_label = ''):
    font_legend = 15
    font_tick = 15
    numbin = np.arange(0, 1.01, 0.01)

    line_width = 2
    axs0.hist(var,  bins=numbin, weights=weight(var),  histtype='step',
         linewidth=line_width, color= color, label= label + lable_hist(var))#, log = True)

    axs0.legend(loc = 'upper right', fontsize = font_legend, frameon = True)
    axs0.set_xlabel(x_axis_label, fontsize= font_tick)
    axs0.tick_params(axis = 'x', labelsize = font_tick)  # to Set Matplotlib Tick Labels Font Size
    axs0.tick_params(axis = 'y', labelsize = font_tick)
    ticks = np.arange(0, 0.05, 0.005)
    axs0.set_yticks(ticks)
    axs0.set_ylabel(y_axis_label, fontsize = font_tick)
    axs0.set_title(titel, fontsize= font_tick)
    axs0.grid(True)
    return

control_default_in, control_default_out, per_default_in, per_default_out = get_in_out(0)
control_ten_in, control_ten_out, per_ten_in, per_ten_out = get_in_out(1)
control_onetenth_in, control_onetenth_out, per_onetenth_in, per_onetenth_out = get_in_out(2)

control_default_in_3sep, control_default_out_3sep, per_default_in_3sep, per_default_out_3sep = get_in_out(3)
control_ten_in_3sep, control_ten_out_3sep, per_ten_in_3sep, per_ten_out_3sep = get_in_out(4)
control_onetenth_in_3sep, control_onetenth_out_3sep, per_onetenth_in_3sep,\
per_onetenth_out_3sep = get_in_out(5)

control_default_in_all = np.concatenate((control_default_in_3sep, control_default_in), axis = 0)
control_default_out_all = np.concatenate((control_onetenth_out_3sep, control_default_out), axis = 0)
per_default_in_all = np.concatenate((per_default_in_3sep, per_default_in),  axis = 0)
per_default_out_all = np.concatenate((per_default_out_3sep, per_default_out),  axis = 0)
#modis_in_all = np.concatenate((modis_in, modis_in_3sep),  axis = 0)
#modis_out_all = np.concatenate((modis_out, modis_out_3sep),  axis = 0)
control_ten_in_all = np.concatenate((control_ten_in, control_ten_in_3sep),  axis = 0)
control_ten_out_all = np.concatenate((control_ten_out_3sep, control_ten_out),  axis = 0)
per_ten_in_all = np.concatenate((per_ten_in_3sep, per_ten_in),  axis = 0)
per_ten_out_all = np.concatenate((per_ten_out, per_ten_out_3sep), axis = 0)

control_onetenth_in_all = np.concatenate((control_onetenth_in, control_onetenth_in_3sep), axis = 0)
control_onetenth_out_all = np.concatenate((control_onetenth_out, control_onetenth_out_3sep), axis = 0)
per_onetenth_in_all = np.concatenate((per_onetenth_in_3sep, per_onetenth_in), axis = 0)
per_onetenth_out_all = np.concatenate((per_onetenth_out, per_onetenth_out_3sep), axis = 0)

titel_kind = [' 1 time', '10 times', '0.1 times ', 'modis']
run_type = ['no-volacano', 'volcano',  'MODIS']

fig, ((axs0, axs1), (axs2, axs3), (axs4, axs5)) = plt.subplots(3, 2, figsize = (15, 20))
pdf_var_prepare(axs0, var = control_default_in, color = 'blue', label = run_type[0] ,
                titel = 'Inside_Plume ' + titel_kind[0])
pdf_var_prepare(axs0, var = per_default_in, color = 'red', label = run_type[1],
                titel = 'Inside_Plume ' + titel_kind[0], y_axis_label = y_name)
#pdf_var_prepare(axs0, var = modis_in, color = 'black', label = run_type[2],
#                titel = 'Inside_Plume ' + titel_kind[0], y_axis_label = y_name)
pdf_var_prepare(axs1, var = control_default_out, color = 'blue', label = run_type[0],
                titel = 'Outside_Plume ' + titel_kind[0])
pdf_var_prepare(axs1, var = per_default_out, color = 'red', label = run_type[1],
                titel = 'Outside_Plume ' + titel_kind[0])
#pdf_var_prepare(axs1, var = modis_out, color = 'black', label = run_type[2],
#                titel = 'Outside_Plume ' + titel_kind[0])


pdf_var_prepare(axs2, var = control_ten_in, color = 'blue', label = run_type[0],
                titel = 'Inside_Plume ' + titel_kind[1])
pdf_var_prepare(axs2, var = per_ten_in, color = 'red', label = run_type[1],
                titel = 'Inside_Plume ' + titel_kind[1], y_axis_label = y_name)
#pdf_var_prepare(axs2, var = modis_in, color = 'black', label = run_type[2],
#                titel = 'Inside_Plume ' + titel_kind[1], y_axis_label = y_name)
pdf_var_prepare(axs3, var = control_ten_out, color = 'blue', label = run_type[0],
                titel = 'Outside_Plume ' + titel_kind[1])
pdf_var_prepare(axs3, var = per_ten_out, color = 'red', label = run_type[1],
                titel = 'Outside_Plume ' + titel_kind[1])
#pdf_var_prepare(axs3, var = modis_out, color = 'black', label = run_type[2] ,
 #               titel = 'Outside_Plume ' + titel_kind[1])


pdf_var_prepare(axs4, var = control_onetenth_in, color = 'blue', label = run_type[0],
                titel = 'Inside_Plume ' + titel_kind[2], y_axis_label = y_name)
pdf_var_prepare(axs4, var = per_onetenth_in, color = 'red', label = run_type[1],
                titel = 'Inside_Plume ' + titel_kind[2], x_axis_label = x_name, y_axis_label = y_name)
#pdf_var_prepare(axs4, var = modis_in, color = 'black', label = run_type[2],
#                titel = 'Inside_Plume ' + titel_kind[2], x_axis_label = x_name, y_axis_label = y_name)
pdf_var_prepare(axs5, var = control_onetenth_out, color = 'blue', label = run_type[0],
                titel = 'Outside_Plume ' + titel_kind[2])
pdf_var_prepare(axs5, var = per_onetenth_out, color = 'red', label = run_type[1],
                titel = 'Outside_Plume ' + titel_kind[2], x_axis_label = x_name)
#pdf_var_prepare(axs5, var = modis_out, color = 'black', label = run_type[2],
#                titel = 'Outside_Plume ' + titel_kind[2], x_axis_label = x_name)
fig.suptitle('2 Sep', fontsize=20)
#plt.tight_layout()
#plt.savefig('albedo_auto_conversion_test_2sep.png')
#plt.savefig('albedo_auto_conversion_test_2sep.pdf')
#plt.show()

fig, ((axs0, axs1), (axs2, axs3), (axs4, axs5)) = plt.subplots(3, 2, figsize = (15, 17))
pdf_var_prepare(axs0, var = control_default_in_3sep, color = 'blue', label = run_type[0] ,
                titel = 'Inside_Plume ' + titel_kind[0])
pdf_var_prepare(axs0, var = per_default_in_3sep, color = 'red', label = run_type[1],
                titel = 'Inside_Plume ' + titel_kind[0], y_axis_label = y_name)
#pdf_var_prepare(axs0, var = modis_in_3sep, color = 'black', label = run_type[2],
#                titel = 'Inside_Plume ' + titel_kind[0])
pdf_var_prepare(axs1, var = control_default_out_3sep, color = 'blue', label = run_type[0],
                titel = 'Outside_Plume ' + titel_kind[0])
pdf_var_prepare(axs1, var = per_default_out_3sep, color = 'red', label = run_type[1],
                titel = 'Outside_Plume ' + titel_kind[0], y_axis_label = y_name)
#pdf_var_prepare(axs1, var = modis_out_3sep, color = 'black', label = run_type[2],
#                titel = 'Outside_Plume ' + titel_kind[0])


pdf_var_prepare(axs2, var = control_ten_in_3sep, color = 'blue', label = run_type[0],
                titel = 'Inside_Plume ' + titel_kind[1])
pdf_var_prepare(axs2, var = per_ten_in_3sep, color = 'red', label = run_type[1],
                titel = 'Inside_Plume ' + titel_kind[1], y_axis_label = y_name)
#pdf_var_prepare(axs2, var = modis_in_3sep, color = 'black', label = run_type[2],
#                titel = 'Inside_Plume ' + titel_kind[1])
pdf_var_prepare(axs3, var = control_ten_out_3sep, color = 'blue', label = run_type[0],
                titel = 'Outside_Plume ' + titel_kind[1])
pdf_var_prepare(axs3, var = per_ten_out_3sep, color = 'red', label = run_type[1],
                titel = 'Outside_Plume ' + titel_kind[1])
#pdf_var_prepare(axs3, var = modis_out_3sep, color = 'black', label = run_type[2] ,
#                titel = 'Outside_Plume ' + titel_kind[1])


pdf_var_prepare(axs4, var = control_onetenth_in_3sep, color = 'blue', label = run_type[0],
                titel = 'Inside_Plume ' + titel_kind[2])
pdf_var_prepare(axs4, var = per_onetenth_in_3sep, color = 'red', label = run_type[1],
                titel = 'Inside_Plume ' + titel_kind[2], x_axis_label = x_name, y_axis_label = y_name)
#pdf_var_prepare(axs4, var = modis_in_3sep, color = 'black', label = run_type[2],
#                titel = 'Inside_Plume ' + titel_kind[2])
pdf_var_prepare(axs5, var = control_onetenth_out_3sep, color = 'blue', label = run_type[0],
                titel = 'Outside_Plume ' + titel_kind[2])
pdf_var_prepare(axs5, var = per_onetenth_out_3sep, color = 'red', label = run_type[1],
                titel = 'Outside_Plume ' + titel_kind[2], x_axis_label = x_name)
#pdf_var_prepare(axs5, var = modis_out_3sep, color = 'black', label = run_type[2],
#                titel = 'Outside_Plume ' + titel_kind[2], x_axis_label = x_name, y_axis_label = y_name)
fig.suptitle('3 Sep', fontsize=20)
#plt.tight_layout()
#plt.savefig('albedo_auto_conversion_test_3sep.png')
#plt.savefig('albedo_auto_conversion_test_3sep.pdf')
#plt.show()

fig, ((axs0, axs1), (axs2, axs3), (axs4, axs5)) = plt.subplots(3, 2, figsize = (15, 17))
pdf_var_prepare(axs0, var = control_default_in_all, color = 'blue', label = run_type[0] ,
                titel = 'Inside_Plume ' + titel_kind[0])
pdf_var_prepare(axs0, var = per_default_in_all, color = 'red', label = run_type[1],
                titel = 'Inside_Plume ' + titel_kind[0], y_axis_label = y_name)
#pdf_var_prepare(axs0, var = modis_in_all, color = 'black', label = run_type[2],
#                titel = 'Inside_Plume ' + titel_kind[0], y_axis_label = y_name)
pdf_var_prepare(axs1, var = control_default_out_all, color = 'blue', label = run_type[0],
                titel = 'Outside_Plume ' + titel_kind[0])
pdf_var_prepare(axs1, var = per_default_out_all, color = 'red', label = run_type[1],
                titel = 'Outside_Plume ' + titel_kind[0])
#pdf_var_prepare(axs1, var = modis_out_all, color = 'black', label = run_type[2],
#                titel = 'Outside_Plume ' + titel_kind[0])


pdf_var_prepare(axs2, var = control_ten_in_all, color = 'blue', label = run_type[0],
                titel = 'Inside_Plume ' + titel_kind[1])
pdf_var_prepare(axs2, var = per_ten_in_all, color = 'red', label = run_type[1],
                titel = 'Inside_Plume ' + titel_kind[1], y_axis_label = y_name)
#pdf_var_prepare(axs2, var = modis_in_all, color = 'black', label = run_type[2],
#                titel = 'Inside_Plume ' + titel_kind[1], y_axis_label = y_name)
pdf_var_prepare(axs3, var = control_ten_out_all, color = 'blue', label = run_type[0],
                titel = 'Outside_Plume ' + titel_kind[1])
pdf_var_prepare(axs3, var = per_ten_out_all, color = 'red', label = run_type[1],
                titel = 'Outside_Plume ' + titel_kind[1])
#pdf_var_prepare(axs3, var = modis_out_all, color = 'black', label = run_type[2],
#                titel = 'Outside_Plume ' + titel_kind[1])


pdf_var_prepare(axs4, var = control_onetenth_in_all, color = 'blue', label = run_type[0],
                titel = 'Inside_Plume ' + titel_kind[2])
pdf_var_prepare(axs4, var = per_onetenth_in_all, color = 'red', label = run_type[1],
                titel = 'Inside_Plume ' + titel_kind[2], x_axis_label = x_name, y_axis_label = y_name)
#pdf_var_prepare(axs4, var = modis_in_all, color = 'black', label = run_type[2],
#                titel = 'Inside_Plume ' + titel_kind[2], x_axis_label = x_name, y_axis_label = y_name)
pdf_var_prepare(axs5, var = control_onetenth_out_all, color = 'blue', label = run_type[0],
                titel = 'Outside_Plume ' + titel_kind[2])
pdf_var_prepare(axs5, var = per_onetenth_out_all, color = 'red', label = run_type[1],
                titel = 'Outside_Plume ' + titel_kind[2], x_axis_label = x_name)
#pdf_var_prepare(axs5, var = modis_out_all, color = 'black', label = run_type[2],
#                titel = 'Outside_Plume ' + titel_kind[2], x_axis_label = x_name)
fig.suptitle('2 and 3 Sep', fontsize=16)
#plt.tight_layout()
plt.savefig('albedo_auto_conversion_all_tqc.png')
plt.savefig('albedo_auto_conversion_all_tqc.pdf')
#plt.show()
run_type= [' 1 time', '10 times', '0.1 times ', 'modis']
titel_kind = ['no-volacano', 'volcano',  'MODIS']
color_default = 'blue'
color_ten = 'darkviolet'
color_onetenth =  'cornflowerblue'
fig, ((axs0, axs1), (axs2, axs3)) = plt.subplots(2, 2, figsize = (15, 15))
pdf_var_prepare(axs0, var = control_default_in_all, color = color_default, label = run_type[0],
                titel = 'Inside_Plume ' + titel_kind[0])
pdf_var_prepare(axs0, var = control_ten_in_all, color = color_ten, label = run_type[1],
                titel = 'Inside_Plume ' + titel_kind[0])
pdf_var_prepare(axs0, var = control_onetenth_in_all, color = color_onetenth, label = run_type[2],
                titel = 'Inside_Plume ' + titel_kind[0], y_axis_label = y_name)

pdf_var_prepare(axs1, var = per_default_in_all, color = color_default, label = run_type[0],
                titel = 'Inside_Plume ' + titel_kind[1], y_axis_label = y_name)
pdf_var_prepare(axs1, var = per_ten_in_all, color = color_ten, label = run_type[1],
                titel = 'Inside_Plume ' + titel_kind[1], y_axis_label = y_name)
pdf_var_prepare(axs1, var = per_onetenth_in_all, color = color_onetenth, label = run_type[2],
                titel = 'Inside_Plume ' + titel_kind[1])

pdf_var_prepare(axs2, var = control_default_out_all, color = color_default, label = run_type[0],
                titel = 'Outside_Plume ' + titel_kind[0])
pdf_var_prepare(axs2, var = control_ten_out_all, color = color_ten, label = run_type[1],
                titel = 'Outside_Plume ' + titel_kind[0])
pdf_var_prepare(axs2, var = control_onetenth_out_all, color = color_onetenth, label = run_type[2],
                titel = 'Outside_Plume ' + titel_kind[0], x_axis_label = x_name, y_axis_label = y_name)

pdf_var_prepare(axs3, var = per_default_out_all, color = color_default, label = run_type[1],
                titel = 'Outside_Plume ' + titel_kind[1])
pdf_var_prepare(axs3, var = per_ten_out_all, color = color_ten, label = run_type[1],
                titel = 'Outside_Plume ' + titel_kind[1])
pdf_var_prepare(axs3, var = per_onetenth_out_all, color = color_onetenth, label = run_type[1],
                titel = 'Outside_Plume ' + titel_kind[1], x_axis_label = x_name)
plt.savefig('albedo_new_style_all_tqc_differentbar.png')
plt.savefig('albedo_new_style_all_tqc_diffrenetbar.pdf')
plt.show()


def get_in_out_diff(i):
    import scipy.interpolate as sci
    lat_fine = postpro_func.read_nc(model_con[i], 'lat')
    lon_fine = postpro_func.read_nc(model_con[i], 'lon')
    control_var_lando = postpro_func.read_nc(model_con[i], var_name_model[0])[:, :]
    perturbed_var_lando = postpro_func.read_nc(model_per[i], var_name_model[0])[:, :]
    diff_var_lando = perturbed_var_lando - control_var_lando
    diff_var = postpro_func.mask_land(diff_var_lando, lat_fine, lon_fine)
    so2 = postpro_func.read_nc(omps_data[i], 'so2_TRL')[0:43, 0:115]
    lat_so2 = postpro_func.read_nc(omps_data[i], 'lat')
    lon_so2 = postpro_func.read_nc(omps_data[i], 'lon')
    lon_coarse = lon_so2[0, 0:115]
    lat_coarse = lat_so2[0:43, 0]
    so2_mask = np.ma.filled(so2, fill_value=0)
    f = sci.RectBivariateSpline(lat_coarse, lon_coarse, so2_mask)
    scale_interp = f(lat_fine, lon_fine)
    diff_inside = diff_var[scale_interp > 1.0]
    diff_outside = diff_var[scale_interp < 1.0]
    #diff_inside_mask = ma.masked_where(diff_inside == 0., diff_inside)
    #diff_outside_mask = ma.masked_where(diff_outside == 0.,diff_outside)
    diff_inside_mask = diff_inside.compressed()
    diff_outside_mask = diff_outside.compressed()
    return diff_inside_mask, diff_outside_mask



def pdf_var_prepare_diff(axs0, var, color, label, titel, x_axis_label = '', y_axis_label = ''):
    font_legend = 15
    font_tick = 15
    numbin = np.arange(-0.4, 0.4, 0.01)

    line_width = 3
    axs0.hist(var,  bins=numbin, weights=weight(var),  histtype='step',
         linewidth=line_width, color= color, label= label + lable_hist(var))#, log = True)

    axs0.legend(loc = 'upper right', fontsize = font_legend, frameon = True)
    axs0.set_xlabel(x_axis_label, fontsize= font_tick)
    axs0.tick_params(axis = 'x', labelsize = font_tick)  # to Set Matplotlib Tick Labels Font Size
    axs0.tick_params(axis = 'y', labelsize = font_tick)
    ticks = np.arange(0, 0.12, 0.02)
    axs0.set_yticks(ticks)
    axs0.set_ylabel(y_axis_label, fontsize = font_tick)
    axs0.set_title(titel, fontsize= font_tick)
    axs0.grid(True)
    return


diff_default_in, diff_default_out = get_in_out_diff(0)
diff_ten_in, diff_ten_out = get_in_out_diff(1)
diff_onetenth_in, diff_onetenth_out = get_in_out_diff(2)

diff_default_in_3sep, diff_default_out_3sep = get_in_out_diff(3)
diff_ten_in_3sep, diff_ten_out_3sep = get_in_out_diff(4)
diff_onetenth_in_3sep, diff_onetenth_out_3sep = get_in_out_diff(5)

diff_default_in_all = np.concatenate((diff_default_in_3sep, diff_default_in), axis = 0)
diff_default_out_all = np.concatenate((diff_default_out_3sep, diff_default_out), axis = 0)

diff_ten_in_all = np.concatenate((diff_ten_in_3sep, diff_ten_in), axis = 0)
diff_ten_out_all = np.concatenate((diff_ten_out_3sep, diff_ten_out), axis = 0)

diff_onetenth_in_all = np.concatenate((diff_onetenth_in_3sep, diff_onetenth_in), axis = 0)
diff_onetenth_out_all = np.concatenate((diff_onetenth_out_3sep, diff_onetenth_out), axis = 0)

fig, ((axs0, axs1)) = plt.subplots(1, 2, figsize = (13, 11))

run_type = [' 1 time', '10 times', '0.1 times ', 'modis']
titel_kind = ['no-volacano', 'volcano',  'MODIS']

pdf_var_prepare_diff(axs0, var=diff_default_in_all, color = color_default, label = run_type[0],
                     titel = 'Inside_Plume')
pdf_var_prepare_diff(axs0, var=diff_ten_in_all, color = color_ten, label = run_type[1],
                     titel = 'Inside_Plume')
pdf_var_prepare_diff(axs0, var=diff_onetenth_in_all, color = color_onetenth, label = run_type[2],
                     titel = 'Inside_Plume', y_axis_label = y_name, x_axis_label = x_name)

pdf_var_prepare_diff(axs1, var = diff_default_out_all, color = color_default, label = run_type[0],
                     titel = 'Outside_Plume')
pdf_var_prepare_diff(axs1, var = diff_ten_out_all, color = color_ten, label = run_type[1],
                     titel = 'Outside_Plume')
pdf_var_prepare_diff(axs1, var = diff_onetenth_out_all, color = color_onetenth, label = run_type[2],
                     titel = 'Outside_Plume', x_axis_label = x_name)
fig.suptitle('volcano - no-volcano', fontsize = 15)
plt.savefig('albedo_diff.png')
plt.show()