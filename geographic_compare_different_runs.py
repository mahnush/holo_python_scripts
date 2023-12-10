import matplotlib.pyplot as plt
import postpro_func
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
old_version_icon_rccn_c = '/work/bb1093/b380900/work_mistral/holouraun_nccn/output_nccn_alpha/control_run_final_run/3_sep/NWP_LAM_DOM01_20140903T100000Z_0058.nc'
new_version_icon_rccn_c = '/work/bb1036/b380900/work_levante/icon_lam_lim_holo_con/3_sep/remap_final_data/' \
                          'NWP_LAM_DOM01_20140903T100000Z_0058.nc'
new_version_icon_base = '/home/b/b380900/icon_nwp_newversion/icon/experiments/icon_lam_lim_holo_base_run' \
                        '/remap_NWP_LAM_DOM01_20140903T100000Z_0058.nc'
lwp_MODIS = '/work/bb1093/b380900/work_mistral/holouraun_nccn/modis_holo/geo_modis_data_3d.nc'
var_name = 'tqc'
var_MODIS_name = 'lwp'
oldv_rccn_tqc = postpro_func.read_nc(old_version_icon_rccn_c, var_name)[0,:,:]*1000
newv_rccn_tqc = postpro_func.read_nc(new_version_icon_rccn_c, var_name)[0,:,:]*1000
newv_base_tqc = postpro_func.read_nc(new_version_icon_base, var_name)[0,:,:]*1000

MODIS_lwp = postpro_func.read_nc(lwp_MODIS, var_MODIS_name)
MODIS_lwp = np.transpose(MODIS_lwp)
MODIS_lat = np.transpose(postpro_func.read_nc(lwp_MODIS, 'lat'))
MODIS_lon = np.transpose(postpro_func.read_nc(lwp_MODIS, 'lon'))
lwp_bound = np.arange(0, 202, 2)

cbar_label = ['$\mathrm{cm^{-3}}$', '$\mathrm{g\,m^{-2}}$', '', '$\mathrm{{\mu}m}$']
titel_kind = ['old_version_rccn', 'new_version_rccn', 'new_version_base_run', 'MODIS']

fs_label = 10
fs_titel = 10

limit = np.zeros(4)
limit[0] = 50
limit[1] = 80
limit[2] = -60
limit[3] = 20

titel = '3 Sep 2014'
fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize = (17, 12))
fig.suptitle(titel, fontsize = fs_titel)
postpro_func.visulize_sat(ax0, MODIS_lwp, MODIS_lat, MODIS_lon, lwp_bound, cbar_label[1], titel_kind[3] , limit)
postpro_func.visulize_model(ax1, oldv_rccn_tqc, lwp_bound, cbar_label[1], titel_kind[0], limit)
postpro_func.visulize_model(ax2, newv_rccn_tqc, lwp_bound, cbar_label[1], titel_kind[1], limit)
postpro_func.visulize_model(ax3, newv_base_tqc, lwp_bound, cbar_label[1], titel_kind[2], limit)
#plt.show()
plt.savefig('/home/b/b380900/Pycharm_project/holo_newv_icon/lwp_3sep_geodist_compare.png')

#def weight(var):
    #weight = (1 + np.zeros(len(var))) / len(var)
#    weight = np.zeros_like(var) + 1. / (var.size)
#    return weight
#def lable_hist(var):
#    median = str(np.median(var))
#    mean = str((np.mean(var)))
#    std = str(np.std(var))
#    lable = '('+'mean = ' + mean +')'
#    return lable
#def pdf_var_prepare(var, color, label):
#    font_legend = 20
#    var_n_zero = var[var>0]
#    var_c = var_n_zero.compressed()
#    numbin = np.arange(0, 200, 1)
 #   name =  '$ \mathrm{LWP}$ ($\mathrm{g m^{-2}}$)'
  #  line_width = 2
   # axs0.hist(var_c,  bins=numbin, weights=weight(var_c),  histtype='step',
    #     linewidth=line_width, color= color, label= label)

    #axs0.legend(loc = 'upper right', fontsize = font_legend, frameon = True)
    #return
#fig, axs0 = plt.subplots(figsize = (20, 15))
#pdf_var_prepare(MODIS_lwp, 'black', titel_kind[3])
#pdf_var_prepare(var = oldv_rccn_tqc, color = 'blue', label = titel_kind[0])
#pdf_var_prepare(var = newv_rccn_tqc, color = 'red', label = titel_kind[1])
#pdf_var_prepare(var = newv_base_tqc, color = 'green', label = titel_kind[2])

plt.show()