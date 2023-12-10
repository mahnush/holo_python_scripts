
import matplotlib.pyplot as plt
import postpro_func
import numpy as np
import matplotlib
#new_version_icon_rccn_c = '/home/b/b380900/icon_nwp_newversion/icon/experiments/' \
#                          'icon_lam_lim_holo_change_vn_ncolumn/remap_selvar_NWP_LAM_DOM01_20140902T020000Z_0026.nc'
#old_version_icon_rccn_c = '/work/bb1093/b380900/work_mistral/holouraun_nccn/output_nccn_alpha/control_run_final_run/' \
#                          '2_sep/NWP_LAM_DOM01_20140902T020000Z_0026.nc'
#new_version_icon_rccn_c = '/home/b/b380900/icon_nwp_newversion/icon/experiments/icon_lam_lim_hol_con/' \
#                          'remap_unselvar_NWP_LAM_DOM01_20140901T100000Z_0010.nc'
new_version_icon_rccn_c = '/home/b/b380900/icon_nwp_newversion/icon/experiments/icon_lam_lim_hol_con/' \
                          'remap_icac_selvar_NWP_LAM_DOM01_20140901T030000Z_0003.nc'
old_version_icon_rccn_c = '/work/bb1093/b380900/work_mistral/holouraun_nccn/output_nccn_alpha/control_run_final_run/1_sep/' \
                          'NWP_LAM_DOM01_20140901T030000Z_0003.nc'
new_version_test = '/home/b/b380900/icon_nwp_newversion/icon/experiments/icon_lam_lim_hol_con/' \
                   'remap_sel_4_NWP_LAM_DOM01_20140901T030000Z_0003.nc'
var_name = 'modis_Liquid_Water_Path_Mean'
var_name_tqi = 'modis_Ice_Water_Path_Mean'
var_name_old= 'modis_Cloud_Particle_Size_Water_'
var_name_old_tqi = 'modis_Cloud_Particle_Size_Ice_Me'

oldv_rccn_tqc = postpro_func.read_nc(old_version_icon_rccn_c, var_name)[0, :, :]*1e3
newv_rccn_tqc = postpro_func.read_nc(new_version_icon_rccn_c, var_name)[0, :, :]*1e3
newv_rccn_test = postpro_func.read_nc(new_version_test, var_name)[0, :, :]*1e3
#newv_rccn_tqi = postpro_func.read_nc(new_version_icon_rccn_c, var_name_tqi)[0, :, :]*1e3

cbar_label = ['$\mathrm{cm^{-3}}$', '$\mathrm{g\,m^{-2}}$', '', '$\mathrm{{\mu}m}$']
titel_kind = ['old_v', 'new_v_rc=100','new_v_rc=4', 'old_iwp']

fs_label = 10
fs_titel = 10

limit = np.zeros(4)
limit[0] = 50
limit[1] = 80
limit[2] = -60
limit[3] = 20

titel = '1 sep'
fs_label = 10
fs_titel = 10

limit = np.zeros(4)
limit[0] = 50
limit[1] = 80
limit[2] = -60
limit[3] = 20
lwp_bound = np.arange(0, 1000, 10)

fig, ((ax0, ax1),(ax2,ax3)) = plt.subplots(2, 2, figsize = (17, 12))
#fig, ((ax0, ax1)) = plt.subplots(2, 1, figsize = (17, 12))
fig.suptitle(titel, fontsize = fs_titel)
postpro_func.visulize_model(ax0, oldv_rccn_tqc, lwp_bound, cbar_label[1], titel_kind[0], limit)
postpro_func.visulize_model(ax1, newv_rccn_tqc, lwp_bound, cbar_label[1], titel_kind[1], limit)
postpro_func.visulize_model(ax2, newv_rccn_test, lwp_bound, cbar_label[1], titel_kind[2], limit)
#postpro_func.visulize_model(ax3, oldv_rccn_tqi, lwp_bound, cbar_label[1], titel_kind[3], limit)
plt.savefig('lwp_tqc_new changes.png')
#plt.show()

def weight(var):
    weight = np.zeros_like(var) + 1. / (var.size)
    return weight
def lable_hist(var):
    median = str(np.median(var))
    mean = str((np.mean(var)))
    std = str(np.std(var))
    lable = '('+'mean = ' + mean +')'
    return lable
def pdf_var_prepare(var, color, label):
    font_legend = 20
    var_n_zero = var[var>1.0]
    print(np.min(var_n_zero))
    print(np.shape(var_n_zero))
    var_c = var_n_zero.compressed()
    print(np.shape(var_c))
    numbin = np.arange(0, 1000, 10)
    name =  '$ \mathrm{LWP}$ ($\mathrm{g m^{-2}}$)'
    line_width = 2
    axs0.hist(var_c,  bins=numbin, weights=weight(var_c),  histtype='step',
         linewidth=line_width, color= color, label= label + lable_hist(var_c))

    axs0.legend(loc = 'upper right', fontsize = font_legend, frameon = True)
    return
fig, axs0 = plt.subplots(figsize = (20, 15))
#pdf_var_prepare(MODIS_lwp, 'black', titel_kind[3])
pdf_var_prepare(var = oldv_rccn_tqc, color = 'black', label = titel_kind[0])
pdf_var_prepare(var = newv_rccn_tqc, color = 'red', label = titel_kind[1])
pdf_var_prepare(var = newv_rccn_test, color = 'blue', label = titel_kind[2])

#plt.show()
plt.savefig('lwp.png')
plt.show()