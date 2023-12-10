import matplotlib.pyplot as plt
import postpro_func
import numpy as np
import matplotlib
import numpy.ma as ma


holo_con_nwp_2sep = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/' \
                    'compare_holo_nwp_vs_art/2sep_con_nwp_final.nc'
holo_con_nwp_3sep = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/' \
                    'compare_holo_nwp_vs_art/3sep_con_nwp_final.nc'

holo_per_nwp_2sep = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/' \
                    'compare_holo_nwp_vs_art/2sep_per_nwp_final.nc'
holo_per_nwp_3sep = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/' \
                    'compare_holo_nwp_vs_art/3sep_per_nwp_final.nc'

holo_con_art_2sep = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/' \
                    'compare_holo_nwp_vs_art/2sep_con_art_final.nc'
holo_con_art_3sep = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/' \
                    'compare_holo_nwp_vs_art/3sep_con_art_final.nc'

holo_per_art_2sep = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/' \
                    'compare_holo_nwp_vs_art/2sep_per_art_final.nc'
holo_per_art_3sep = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/' \
                    'compare_holo_nwp_vs_art/3sep_per_art_final.nc'

model_con = [holo_con_nwp_2sep, holo_con_nwp_3sep, holo_con_art_2sep, holo_con_art_3sep]
model_per = [holo_per_nwp_2sep, holo_per_nwp_3sep, holo_per_art_2sep, holo_per_art_3sep]

omps_path_2sep = '/home/mhaghigh/nc_file_in_outplume/so2_Iceland_2sep_0.7res.nc'
#modis_path_2sep = '/home/mhaghigh/nc_file_in_outplume/geo_modis_data_2d.nc'
modis_path_2sep = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/modis_holo/2sep/modis_2sep_nd_c_2km.nc'
omps_path_3sep = '/home/mhaghigh/nc_file_in_outplume/so2_Iceland_3sep_0.7res.nc'
modis_path_3sep = '/home/mhaghigh/nc_file_in_outplume/geo_modis_data_3d.nc'

var_name_model = ['qnc', 'tqc', 'so4_at', 'z_mc', 'geopt', 'rho']
var_name_modis = ['nd', 'lwp', 'tau', 're', 'cf']
omps_data = [omps_path_2sep, omps_path_3sep]
modis_data = [modis_path_2sep, modis_path_3sep]

def get_in_out_nwp(i, var_con, var_per):
    import scipy.interpolate as sci
    lat_fine = postpro_func.read_nc(model_con[i], 'lat')
    lon_fine = postpro_func.read_nc(model_con[i], 'lon')[200:801]
    control_var_lando = var_con
    perturbed_var_lando = var_per
    control_var = postpro_func.mask_land(control_var_lando, lat_fine, lon_fine)
    perturbed_var = postpro_func.mask_land(perturbed_var_lando, lat_fine, lon_fine)
    so2 = postpro_func.read_nc(omps_data[i], 'so2_TRL')[0:43, 29:115]
    lat_so2 = postpro_func.read_nc(omps_data[i], 'lat')
    lon_so2 = postpro_func.read_nc(omps_data[i], 'lon')
    lon_coarse = lon_so2[0, 29:115]
    lat_coarse = lat_so2[0:43, 0]
    #print(lon_coarse)
    #print(lat_coarse)
    so2_mask = np.ma.filled(so2, fill_value=0)
    f = sci.RectBivariateSpline(lat_coarse, lon_coarse, so2_mask)
    scale_interp = f(lat_fine, lon_fine)
    con_inside = control_var[scale_interp > 1.0]
    con_outside = control_var[scale_interp < 1.0]
    per_inside = perturbed_var[scale_interp > 1.0]
    per_outside = perturbed_var[scale_interp < 1.0]
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

    return con_inside_mask, con_outside_mask, per_inside_mask, per_outside_mask

def get_in_out_modis(i):
    import scipy.interpolate as sci
    lat_fine_modis = np.arange(50, 80.02, 0.02)
    lon_fine_modis = np.arange(-40, 20.02, 0.02)
    #lat_fine_modis = np.arange(50, 80.1, 0.1)
    #lon_fine_modis = np.arange(-40, 20.1, 0.1)
    modis_var_lando = postpro_func.read_nc(modis_data[i], var_name_modis[1])
    modis_var_lando = np.transpose(modis_var_lando)[:, 1000:4001]
    #modis_var_lando = np.transpose(modis_var_lando)[:, :]
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
    modis_inside_mask_1 = ma.masked_where(modis_inside <= 0, modis_inside)
    modis_inside_mask = ma.masked_where(modis_inside_mask_1 >= 1500, modis_inside_mask_1)
    modis_inside_mask = modis_inside_mask.compressed()
    modis_outside_mask = ma.masked_where(modis_outside <= 0., modis_outside)
    modis_outside_mask = modis_outside_mask.compressed()
    return modis_inside_mask, modis_outside_mask



tqc_con_nwp = postpro_func.read_nc(model_con[0], var_name_model[1]) [:, 200:801]*1e3
tqc_per_nwp = postpro_func.read_nc(model_per[0], var_name_model[1])[:, 200:801]*1e3
# in out vars
con_nwp_inside, con_nwp_outside, per_nwp_inside, per_nwp_outside = \
    get_in_out_nwp(i = 0, var_con = tqc_con_nwp, var_per = tqc_per_nwp)

modis_inside, modis_outside = get_in_out_modis(0)


def vertical_integrate_so4_art(file_path, var_name, lev_name):
    from scipy.integrate import trapz
    var_so4 = postpro_func.read_nc(file_path, var_name)[13:50, :, :]
    var_density = postpro_func.read_nc(file_path, var_name_model[5])[13:50, :, :]
    var_so4 = var_so4[::-1, :, :]
    var_density = var_density[::-1, :, :]
    var_so4_den = var_so4*var_density
    var_lev = postpro_func.read_nc(file_path, lev_name)[13:50, :, :]
    var_lev_lon = np.ma.mean(var_lev, axis=1)
    var_lev_1d = np.ma.mean(var_lev_lon, axis=1)
    var_lev_1d = var_lev_1d[::-1]
    integrated_var = trapz(var_so4_den, x=var_lev_1d, axis=0)
    return integrated_var

int_so4_art = vertical_integrate_so4_art(holo_per_art_2sep, var_name_model[2], var_name_model[3])

def get_in_out_art(i, var_con, var_per, var_so4):
    lat_fine = postpro_func.read_nc(model_con[i], 'lat')
    lon_fine = postpro_func.read_nc(model_con[i], 'lon')
    control_var_lando = var_con
    perturbed_var_lando = var_per
    var_so4_lando =var_so4
    control_var = postpro_func.mask_land(control_var_lando, lat_fine, lon_fine)
    perturbed_var = postpro_func.mask_land(perturbed_var_lando, lat_fine, lon_fine)
    so4_mask = postpro_func.mask_land(var_so4_lando, lat_fine, lon_fine)
    so4_threshold = 1e-3
    con_inside = control_var[so4_mask > so4_threshold ]
    con_outside = control_var[so4_mask < so4_threshold ]
    per_inside = perturbed_var[so4_mask > so4_threshold]
    per_outside = perturbed_var[so4_mask < so4_threshold]
    con_inside_mask = ma.masked_where(con_inside <= 5., con_inside)
    con_inside_mask = con_inside_mask.compressed()
    con_outside_mask = ma.masked_where(con_outside <= 5., con_outside)
    con_outside_mask = con_outside_mask.compressed()
    per_inside_mask = ma.masked_where(per_inside <= 5., per_inside)
    per_inside_mask = per_inside_mask.compressed()
    per_outside_mask = ma.masked_where(per_outside <= 5., per_outside)
    per_outside_mask = per_outside_mask.compressed()
    return con_inside_mask, con_outside_mask, per_inside_mask, per_outside_mask

tqc_con_art = postpro_func.read_nc(model_con[2], var_name_model[1]) *1e3
tqc_per_art = postpro_func.read_nc(model_per[2], var_name_model[1])*1e3

con_art_inside, con_art_outside, per_art_inside, per_art_outside \
    = get_in_out_art(i = 2, var_con = tqc_con_art, var_per = tqc_per_art, var_so4 = int_so4_art)


def weight(var):
    weight = np.zeros_like(var) + 1. / (var.size)
    return weight


def lable_hist(var):
    median = str(np.median(var))
    mean = str((round(np.mean(var))))
    median = str((round(np.median(var))))
    std = str(np.std(var))
    lable = '('+'mean = ' + mean + ' median = ' + median +')'
    return lable


x_name = '$ \mathrm{LWP}$ ($\mathrm{g m^{-2}}$)'
y_name = 'Relative Frequency'

def pdf_var_prepare(axs0, var, color, label, titel, x_axis_label = '', y_axis_label = ''):
    font_legend = 15
    font_tick = 20
    numbin = np.arange(0, 1000, 10)
    line_width = 3
    axs0.hist(var,  bins=numbin, weights=weight(var),  histtype='step',
         linewidth=line_width, color= color, label= label + lable_hist(var))#, log = True)

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
titel_kind = ['NWP', 'ART', 'Unified Model']
run_type = ['no-vol', 'vol',  'MODIS']


"""
this is the block which I want to include the results from the model in Leeds
"""

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
def get_in_out_leeds( var_con, var_per, var_so2):
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
    get_in_out_leeds(lwp_2sep_con_mean, lwp_2sep_per_mean, so2_2sep_per_mean)

fig, ((axs0, axs1), (axs2, axs3), (axs4, axs5)) = plt.subplots(3, 2, figsize = (16, 18))

pdf_var_prepare(axs0, var = modis_inside, color = 'black', label = run_type[2],
                titel = 'Inside_Plume ' + titel_kind[0])
pdf_var_prepare(axs0, var = con_nwp_inside, color = 'blue', label = run_type[0],
                titel = 'Inside_Plume ' + titel_kind[0])
pdf_var_prepare(axs0, var = per_nwp_inside, color = 'red', label = run_type[1],
                titel = 'Inside_Plume ' + titel_kind[0], y_axis_label = y_name)


pdf_var_prepare(axs1, var = modis_outside, color = 'black', label = run_type[2],
                titel = ' Outside_Plume' + titel_kind[0])
pdf_var_prepare(axs1, var = con_nwp_outside, color = 'blue', label = run_type[0],
                titel = 'Outside_Plume ' + titel_kind[0])
pdf_var_prepare(axs1, var = per_nwp_outside,  color = 'red', label = run_type[1],
                titel = 'Outside_Plume ' + titel_kind[0])

pdf_var_prepare(axs2, var = modis_inside, color = 'black', label = run_type[2],
                titel = 'Inside_Plume ' + titel_kind[1])
pdf_var_prepare(axs2, var = con_art_inside,  color = 'blue', label = run_type[0],
                titel = 'Inside_Plume ' + titel_kind[1])
pdf_var_prepare(axs2, var = per_art_inside, color = 'red', label = run_type[1],
                titel = 'Inside_Plume ' + titel_kind[1], y_axis_label = y_name)

pdf_var_prepare(axs3, var = modis_outside, color = 'black', label = run_type[2],
                titel = ' Outside_Plume' + titel_kind[1])
pdf_var_prepare(axs3, var = con_art_outside,  color = 'blue', label = run_type[0],
                titel = 'Outside_Plume ' + titel_kind[1])
pdf_var_prepare(axs3, var = per_art_outside,  color = 'red', label = run_type[1],
                titel = 'Outside_Plume ' + titel_kind[1])

pdf_var_prepare(axs4, var = modis_inside, color = 'black', label = run_type[2],
                titel = 'Inside_Plume ' + titel_kind[2])
pdf_var_prepare(axs4, var = con_inside_2sep,  color = 'blue', label = run_type[0],
                titel = 'Inside_Plume ' + titel_kind[2])
pdf_var_prepare(axs4, var = per_inside_2sep, color = 'red', label = run_type[1],
                titel = 'Inside_Plume ' + titel_kind[2], y_axis_label = y_name, x_axis_label = x_name)

pdf_var_prepare(axs5, var = modis_outside, color = 'black', label = run_type[2],
                titel = ' Outside_Plume' + titel_kind[2])
pdf_var_prepare(axs5, var = con_outside_2sep, color = 'blue', label = run_type[0],
                titel = 'Outside_Plume ' + titel_kind[2])
pdf_var_prepare(axs5, var = per_outside_2sep, color = 'red', label = run_type[1],
                titel = 'Outside_Plume ' + titel_kind[2], x_axis_label = x_name)

plt.savefig('tqc_compare_art_nwp_leeds.png')
plt.savefig('tqc_compare_art_nwp_leeds.pdf')
plt.show()


