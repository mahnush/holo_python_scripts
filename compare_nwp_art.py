import matplotlib.pyplot as plt
import postpro_func
import numpy as np
import matplotlib
import numpy.ma as ma


holo_con_nwp_2sep = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/' \
                    'compare_holo_nwp_vs_art/files_used_inscripts/diff_timestep/3sep_con_nwp.nc'
holo_con_nwp_3sep = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/' \
                    'compare_holo_nwp_vs_art/diff_timestep/3sep_con_nwp.nc'

holo_per_nwp_2sep = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/' \
                    'compare_holo_nwp_vs_art/files_used_inscripts/diff_timestep/3sep_per_nwp.nc'
holo_per_nwp_3sep = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/' \
                    'compare_holo_nwp_vs_art/diff_timestep/3sep_per_nwp.nc'

holo_con_art_2sep = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/compare_holo_nwp_vs_art' \
                    '/files_used_inscripts/art_mistral/3sep_con_art.nc'
holo_con_art_3sep = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/' \
                    'compare_holo_nwp_vs_art/diff_timestep/3sep_con_art.nc'

holo_per_art_2sep = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/' \
                    'compare_holo_nwp_vs_art/files_used_inscripts/art_mistral/3sep_per_art.nc'
holo_per_art_3sep = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/' \
                    'compare_holo_nwp_vs_art/diff_timestep/3sep_per_art.nc'

model_con = [holo_con_nwp_2sep, holo_con_nwp_3sep, holo_con_art_2sep, holo_con_art_3sep]
model_per = [holo_per_nwp_2sep, holo_per_nwp_3sep, holo_per_art_2sep, holo_per_art_3sep]

omps_path_2sep = '/home/mhaghigh/nc_file_in_outplume/so2_Iceland_2sep_0.7res.nc'
modis_path_2sep = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/modis_holo/2sep/modis_2sep_nd_c_2km.nc'
omps_path_3sep = '/home/mhaghigh/nc_file_in_outplume/so2_Iceland_3sep_0.7res.nc'
modis_path_3sep = '/home/mhaghigh/nc_file_in_outplume/geo_modis_data_3d.nc'

var_name_model = ['qnc', 'tqc', 'so4_at', 'z_mc', 'geopt', 'rho', 'so2']
var_name_modis = ['nd', 'lwp', 'tau', 're', 'cf', 'nd_c']
omps_data = [omps_path_2sep, omps_path_3sep]
modis_data = [modis_path_2sep, modis_path_3sep]

def vertical_integrate_qnc_nwp(file_path, var_name, lev_name):
    from scipy.integrate import trapz
    g = 9.8
    var_qnc = postpro_func.read_nc(file_path, var_name)[35:75, :, 200:801]
    var_density = postpro_func.read_nc(file_path, var_name_model[5])[35:75, :, 200:801]
    var_qnc = var_qnc[::-1, :, :]
    var_density = var_density[::-1, :, :]
    var_qnc_den = var_qnc * var_density
    var_lev = postpro_func.read_nc(file_path, lev_name)[35:75, :, 200:801]
    var_lev = var_lev/g
    var_lev_lon = np.ma.mean(var_lev, axis=1)
    var_lev_1d = np.ma.mean(var_lev_lon, axis=1)
    var_lev_1d = var_lev_1d[::-1]
    #print(var_lev_1d)
    #print(np.shape(var_qnc))
    # Compute total column integrated value of var_3D
    integrated_var = trapz(var_qnc_den, x=var_lev_1d, axis=0)
    #print(np.shape(integrated_var))
    integrated_var = integrated_var *1e-4
    return integrated_var

int_qnc_nwp = vertical_integrate_qnc_nwp(holo_con_nwp_2sep, var_name_model[0], var_name_model[4])
int_qnc_nwp_per = vertical_integrate_qnc_nwp(holo_per_nwp_2sep, var_name_model[0], var_name_model[4])


def vertical_integrate_qnc_art(file_path, var_name, lev_name):
    from scipy.integrate import trapz
    var_qnc = postpro_func.read_nc(file_path, var_name)[13:50, :, :]
    var_density = postpro_func.read_nc(file_path, var_name_model[5])[13:50, :, :]
    var_qnc = var_qnc[::-1, :, :]
    var_density = var_density[::-1, :, :]
    var_qnc_den = var_qnc * var_density
    var_lev = postpro_func.read_nc(file_path, lev_name)[13:50, :, :]
    var_lev_lon = np.ma.mean(var_lev, axis=1)
    var_lev_1d = np.ma.mean(var_lev_lon, axis=1)
    var_lev_1d = var_lev_1d[::-1]
    #print(var_lev_1d)
    #print(np.shape(var_qnc))
    # Compute total column integrated value of var_3D
    integrated_var = trapz(var_qnc_den, x=var_lev_1d, axis=0)
    integrated_var = integrated_var*1e-4
    #print(np.shape(integrated_var))
    return integrated_var


int_qnc_art = vertical_integrate_qnc_art(holo_con_art_2sep, var_name_model[0], var_name_model[3])
int_qnc_art_per = vertical_integrate_qnc_art(holo_per_art_2sep, var_name_model[0], var_name_model[3])


def vertical_integrate_so4_art(file_path, var_name, lev_name):
    from scipy.integrate import trapz
    var_so4 = postpro_func.read_nc(file_path, var_name)[:, :, :]
    var_density = postpro_func.read_nc(file_path, var_name_model[5])[:, :, :]
    var_so4 = var_so4[::-1, :, :]
    var_density = var_density[::-1, :, :]
    var_so4_den = var_so4*var_density
    var_lev = postpro_func.read_nc(file_path, lev_name)[:, :, :]
    var_lev_lon = np.ma.mean(var_lev, axis=1)
    var_lev_1d = np.ma.mean(var_lev_lon, axis=1)
    var_lev_1d = var_lev_1d[::-1]
    integrated_var = trapz(var_so4_den, x=var_lev_1d, axis=0)
    return integrated_var

def vertical_integrate_so2_art(file_path, var_name, lev_name):
    from scipy.integrate import trapz
    var_so2 = postpro_func.read_nc(file_path, var_name)[:, :, :]
    var_density = postpro_func.read_nc(file_path, var_name_model[5])[:, :, :]
    var_so2 = var_so2[::-1, :, :]
    var_density = var_density[::-1, :, :]
    const = 1000/(28.98*0.00044615)
    var_so2_den = var_so2*var_density*const
    var_lev = postpro_func.read_nc(file_path, lev_name)[:, :, :]
    var_lev_lon = np.ma.mean(var_lev, axis=1)
    var_lev_1d = np.ma.mean(var_lev_lon, axis=1)
    var_lev_1d = var_lev_1d[::-1]
    integrated_var = trapz(var_so2_den, x=var_lev_1d, axis=0)
    return integrated_var
int_so4_art = vertical_integrate_so4_art(holo_per_art_2sep, var_name_model[2], var_name_model[3])
int_so4_art_con = vertical_integrate_so4_art(holo_con_art_2sep, var_name_model[2], var_name_model[3])
int_so2_art_per = vertical_integrate_so2_art(holo_per_art_2sep, var_name_model[6], var_name_model[3])
# this part is just for checking the threshold and see how everything is look like.

limit = np.zeros(4)
limit[0] = 50
limit[1] = 80
limit[2] = -40
limit[3] = 20
int_so4_art_mask = np.ma.masked_where(int_so4_art <= 100, int_so4_art)
fig, (axs0) = plt.subplots(1, 1, figsize = (17, 12))
postpro_func.visulize_model_fast(axs0, var = int_so4_art_mask, varMin = np.min(int_so4_art_mask),
                                 varMax = 25000, map_limit = limit)

#int_so4_art_mask_con = np.ma.masked_where(int_so4_art_con <= 5e-6, int_so4_art_con)
#fig, (axs0) = plt.subplots(1, 1, figsize = (17, 12))
#postpro_func.visulize_model_fast(axs0, var = int_so4_art_con, varMin = np.min(int_so4_art_con),
#                                 varMax = np.max(int_so4_art_con), map_limit = limit)
int_so2_art_per_mask = np.ma.masked_where(int_so2_art_per < 0.01, int_so2_art_per)
fig, (axs0) = plt.subplots(1, 1, figsize = (17, 12))
postpro_func.visulize_model_fast(axs0, var = int_so2_art_per_mask, varMin = 0,
                                varMax = 100, map_limit = limit)

# postpro_func.visulize_model_fast(axs0, var = int_so2_art_per, varMin = np.min(int_so2_art_per),
#                                  varMax = np.max(int_so2_art_per), map_limit = limit)
#plt.show()

limit = np.zeros(4)
limit[0] = 50
limit[1] = 80
limit[2] = -40
limit[3] = 20
var_min = 0.0
var_max = 3e8
fig, (axs0, axs1) = plt.subplots(1, 2, figsize = (17, 12))
postpro_func.visulize_model_fast(axs0, var = int_qnc_art_per, varMin = var_min,
                                 varMax = var_max, map_limit = limit)
postpro_func.visulize_model_fast(axs1, var = int_qnc_art, varMin = var_min,
                                 varMax = var_max, map_limit = limit)

#fig, (axs0, axs1) = plt.subplots(1, 2, figsize = (17, 12))
#postpro_func.visulize_model_fast(axs0, var = int_qnc_nwp_per, varMin = var_min,
#                                 varMax = var_max, map_limit = limit)
#postpro_func.visulize_model_fast(axs1, var = int_qnc_nwp, varMin = var_min,
#                                 varMax = var_max, map_limit = limit)

#plt.show()

def get_in_out_nwp(i, var_con, var_per):
    import scipy.interpolate as sci
    lat_fine = postpro_func.read_nc(model_con[i], 'lat')
    lon_fine = postpro_func.read_nc(model_con[i], 'lon')[200:801]
    #lat_fine_modis = np.arange(50, 80.02, 0.02)
    #lon_fine_modis = np.arange(-40, 20.02, 0.02)
    control_var_lando = var_con
    perturbed_var_lando = var_per
    control_var = postpro_func.mask_land(control_var_lando, lat_fine, lon_fine)
    perturbed_var = postpro_func.mask_land(perturbed_var_lando, lat_fine, lon_fine)
    #modis_var_lando = postpro_func.read_nc(modis_data[i], var_name_modis[1])
    #modis_var_lando = np.transpose(modis_var_lando)[:, 1000:4001]
    #print(np.shape(modis_var_lando))
    #modis_var = postpro_func.mask_land(modis_var_lando, lat_fine_modis, lon_fine_modis)
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
    #scale_interp_modis = f(lat_fine_modis, lon_fine_modis)
    #modis_inside = modis_var[scale_interp_modis > 1.0]
    #modis_outside = modis_var[scale_interp_modis < 1.0]
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

con_nwp_inside, con_nwp_outside, per_nwp_inside, per_nwp_outside = get_in_out_nwp(0, int_qnc_nwp, int_qnc_nwp_per)

def get_in_out_art(i, var_con, var_per, var_so4):
    lat_fine = postpro_func.read_nc(model_con[i], 'lat')
    lon_fine = postpro_func.read_nc(model_con[i], 'lon')
    control_var_lando = var_con
    perturbed_var_lando = var_per
    var_so4_lando =var_so4
    control_var = postpro_func.mask_land(control_var_lando, lat_fine, lon_fine)
    perturbed_var = postpro_func.mask_land(perturbed_var_lando, lat_fine, lon_fine)
    so4_mask = postpro_func.mask_land(var_so4_lando, lat_fine, lon_fine)
    con_inside = control_var[so4_mask > 0.001]
    con_outside = control_var[so4_mask < 0.001]
    per_inside = perturbed_var[so4_mask > 0.001]
    per_outside = perturbed_var[so4_mask < 0.001]
    con_inside_mask = ma.masked_where(con_inside <= 0., con_inside)
    con_inside_mask = con_inside_mask.compressed()
    con_outside_mask = ma.masked_where(con_outside <= 0., con_outside)
    con_outside_mask = con_outside_mask.compressed()
    per_inside_mask = ma.masked_where(per_inside <= 0., per_inside)
    per_inside_mask = per_inside_mask.compressed()
    per_outside_mask = ma.masked_where(per_outside <= 0., per_outside)
    per_outside_mask = per_outside_mask.compressed()
    return con_inside_mask, con_outside_mask, per_inside_mask, per_outside_mask

con_art_inside, con_art_outside, per_art_inside, per_art_outside \
    = get_in_out_art(2, int_qnc_art, int_qnc_art_per, int_so4_art)


def get_in_out_modis(i):
    import scipy.interpolate as sci
    lat_fine_modis = np.arange(50, 80.02, 0.02)
    lon_fine_modis = np.arange(-40, 20.02, 0.02)
    modis_var_lando = postpro_func.read_nc(modis_data[i], var_name_modis[5])
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
    #modis_inside_mask = ma.masked_where(modis_inside_mask_1 >= 1500, modis_inside_mask_1)
    modis_inside_mask = modis_inside_mask.compressed()
    modis_outside_mask = ma.masked_where(modis_outside <= 0, modis_outside)
    modis_outside_mask = modis_outside_mask.compressed()
    return modis_inside_mask, modis_outside_mask

modis_inside, modis_outside = get_in_out_modis(0)

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
titel_kind = ['NWP', 'ART']
run_type = ['no-vol', 'vol',  'MODIS']
range_nwp = (0, 1e8)
range_art = (0, 1e9)
fig, ((axs0, axs1), (axs2, axs3)) = plt.subplots(2, 2, figsize = (16, 16))
#pdf_var_prepare(axs0, var = modis_inside, range = range_nwp, color = 'black', label = run_type[2],
#                titel = 'Inside_Plume ' + titel_kind[0])
pdf_var_prepare(axs0, var = con_nwp_inside, range = range_nwp, color = 'blue', label = run_type[0],
                titel = 'Inside_Plume ' + titel_kind[0])
pdf_var_prepare(axs0, var = per_nwp_inside, range = range_nwp, color = 'red', label = run_type[1],
                titel = 'Inside_Plume ' + titel_kind[0], y_axis_label = y_name)

#pdf_var_prepare(axs1, var = modis_outside, range = range_nwp, color = 'black', label = run_type[2],
#                titel = 'Outside_Plume ' + titel_kind[0])
pdf_var_prepare(axs1, var = con_nwp_outside, range = range_nwp, color = 'blue', label = run_type[0],
                titel = 'Outside_Plume ' + titel_kind[0])
pdf_var_prepare(axs1, var = per_nwp_outside, range = range_nwp, color = 'red', label = run_type[1],
                titel = 'Outside_Plume ' + titel_kind[0])

#pdf_var_prepare(axs2, var = modis_inside, range = range_nwp, color = 'black', label = run_type[2],
#                titel = 'Inside_Plume ' + titel_kind[1])
pdf_var_prepare(axs2, var = con_art_inside,  range = range_art, color = 'blue', label = run_type[0],
                titel = 'Inside_Plume ' + titel_kind[1])
pdf_var_prepare(axs2, var = per_art_inside, range = range_art, color = 'red', label = run_type[1],
                titel = 'Inside_Plume ' + titel_kind[1], y_axis_label = y_name, x_axis_label = x_name)

#pdf_var_prepare(axs3, var = modis_outside, range = range_nwp, color = 'black', label = run_type[2],
#                titel = 'Outside_Plume ' + titel_kind[1])
pdf_var_prepare(axs3, var = con_art_outside, range = range_art, color = 'blue', label = run_type[0],
                titel = 'Outside_Plume ' + titel_kind[1])
pdf_var_prepare(axs3, var = per_art_outside, range = range_art, color = 'red', label = run_type[1],
                titel = 'Outside_Plume ' + titel_kind[1], x_axis_label = x_name)

plt.savefig('qnc_compare_nwp_art.png')
plt.savefig('qnc_compare_nwp_art.pdf')
plt.show()

#compute the CDNC burden from COSP:

holo_cosp_con_2sep = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/' \
                     'fix_cosp_runs/fix_cosp_control/2_sep_mean.nc'

holo_cosp_per_2sep = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/' \
                      'fix_cosp_runs/fix_cosp_perturbed/per_2_sep_mean.nc'

re_con = postpro_func.read_nc(holo_cosp_con_2sep, var_name = 're_dw')[:, 1000:4001]
tau_con = postpro_func.read_nc(holo_cosp_con_2sep, var_name = 'tau_dw')[:, 1000:4001]

re_per = postpro_func.read_nc(holo_cosp_per_2sep, var_name = 're_dw')[:, 1000:4001]
tau_per = postpro_func.read_nc(holo_cosp_per_2sep, var_name = 'tau_dw')[:, 1000:4001]
beta = 0.30

nd_c_con = (beta * (tau_con) * ((re_con * 1e-6) ** -2))*1e-4
nd_c_per = (beta * (tau_per) * ((re_per * 1e-6) ** -2))*1e-4

def get_in_out_nwp_cosp(var, i):
    import scipy.interpolate as sci
    lat_fine = np.arange(50, 80.02, 0.02)
    lon_fine = np.arange(-40, 20.02, 0.02)
    var_ocean = postpro_func.mask_land(var, lat_fine, lon_fine)
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
    var_inside_mask_1 = ma.masked_where(var_inside_mask > 6e7, var_inside_mask)
    var_inside_mask_test = var_inside_mask_1.compressed()
    var_outside_mask = ma.masked_where(var_outside <= 0, var_outside)
    var_outside_mask = var_outside_mask.compressed()
    return var_inside_mask_test, var_outside_mask

nd_c_con_inside, nd_c_con_outside = get_in_out_nwp_cosp(nd_c_con, 0)
nd_c_per_inside, nd_c_per_outside = get_in_out_nwp_cosp(nd_c_per, 0)

range_nwp = (0, 5e7)
#range_art = (0, 5e7)
fig, ((axs0, axs1)) = plt.subplots(1, 2, figsize = (15, 11))
pdf_var_prepare(axs0, var = modis_inside, range = range_nwp, color = 'black', label = run_type[2],
                titel = 'Inside_Plume ' + titel_kind[0])
pdf_var_prepare(axs0, var = nd_c_con_inside, range = range_nwp, color = 'blue', label = run_type[0],
                titel = 'Inside_Plume ' + titel_kind[0])
pdf_var_prepare(axs0, var = nd_c_per_inside, range = range_nwp, color = 'red', label = run_type[1],
                titel = 'Inside_Plume ' + titel_kind[0], x_axis_label = x_name, y_axis_label = y_name)


pdf_var_prepare(axs1, var = modis_outside, range = range_nwp, color = 'black', label = run_type[2],
                titel = 'Outside_Plume ' + titel_kind[0])
pdf_var_prepare(axs1, var = nd_c_con_outside, range = range_nwp, color = 'blue', label = run_type[0],
                titel = 'Outside_Plume ' + titel_kind[0])
pdf_var_prepare(axs1, var = nd_c_per_outside, range = range_nwp, color = 'red', label = run_type[1],
                titel = 'Outside_Plume ' + titel_kind[0], x_axis_label = x_name, y_axis_label = y_name)

plt.savefig('nd_burden_cosp_modis.png')
plt.savefig('nd_burden_cosp_modis.pdf')
plt.show()



