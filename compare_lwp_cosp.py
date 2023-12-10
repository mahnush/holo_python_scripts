import matplotlib.pyplot as plt
import postpro_func
import numpy as np
import matplotlib
old_version_icon_rccn_c = '/work/bb1093/b380900/experiments/icon_lam_lim_hol_con_nest/' \
                          'remap_cor_selvar_NWP_LAM_DOM01_20140901T030000Z_0003.nc'
new_version_icon_rccn_c = '/work/bb1093/b380900/experiments/icon_lam_lim_hol_con_nest/' \
                          'remap_selvar_NWP_LAM_DOM02_20140901T030000Z_0002.nc'

var_name = 'modis_Cloud_Particle_Size_Water_Mean'
var_name_lwp = 'tqc'
oldv_rccn_tqc_read = postpro_func.read_nc(old_version_icon_rccn_c, var_name)[0, :, :]*1e6
newv_rccn_tqc_read = postpro_func.read_nc(new_version_icon_rccn_c, var_name)[0, :, :]*1e6

oldv_rccn_tqc = np.ma.masked_where(oldv_rccn_tqc_read<= 0 , oldv_rccn_tqc_read)
newv_rccn_tqc = np.ma.masked_where(newv_rccn_tqc_read<= 0 , newv_rccn_tqc_read)


cbar_label = ['$\mathrm{cm^{-3}}$', '$\mathrm{g\,m^{-2}}$', '', '$\mathrm{{\mu}m}$']
titel_kind = ['2km', '1km', 'new_version_base_run', 'MODIS']

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

fig, ((ax0, ax1)) = plt.subplots(2, 1, figsize = (17, 12))
fig.suptitle(titel, fontsize = fs_titel)
postpro_func.visulize_model(ax0, oldv_rccn_tqc, lwp_bound, cbar_label[1], titel_kind[0], limit)
postpro_func.visulize_model(ax1, newv_rccn_tqc, lwp_bound, cbar_label[1], titel_kind[1], limit)
plt.savefig('lwp_tqc_nwe_old_version.png')

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
    numbin = np.arange(0, 30, 1)
    name =  '$ \mathrm{LWP}$ ($\mathrm{g m^{-2}}$)'
    line_width = 2
    axs0.hist(var_c,  bins=numbin, weights=weight(var_c),  histtype='step',
         linewidth=line_width, color= color, label= label + lable_hist(var_c))

    axs0.legend(loc = 'upper right', fontsize = font_legend, frameon = True)
    return
fig, axs0 = plt.subplots(figsize = (20, 15))
#pdf_var_prepare(MODIS_lwp, 'black', titel_kind[3])
pdf_var_prepare(var = oldv_rccn_tqc, color = 'blue', label = titel_kind[0])
pdf_var_prepare(var = newv_rccn_tqc, color = 'red', label = titel_kind[1])
#pdf_var_prepare(var = newv_base_tqc, color = 'green', label = titel_kind[2])

plt.show()