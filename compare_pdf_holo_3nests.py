import matplotlib.pyplot as plt
import postpro_func
import numpy as np
import matplotlib

dom1_holo = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/3nest_run/2sep_dom1.nc'
dom2_holo = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/3nest_run/2sep_dom2.nc'
dom3_holo = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/3nest_run/2sep_dom3.nc'
modis_holo = '/home/mhaghigh/nc_file_in_outplume/geo_modis_data_2d.nc'
per_dom1_holo = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/3nest_run/per_2sep_dom1.nc'
per_dom2_holo = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/3nest_run/per_2sep_dom2.nc'
per_dom3_holo = '/home/mhaghigh/nc_file_in_outplume/holuhraun_newv_icon/3nest_run/per_2sep_dom3.nc'
var_name = 'lwp_dw'

dom1_var = postpro_func.read_nc(dom1_holo, var_name) * 1e3
dom2_var = postpro_func.read_nc(dom2_holo, var_name) * 1e3
dom3_var = postpro_func.read_nc(dom3_holo, var_name) * 1e3

per_dom1_var = postpro_func.read_nc(per_dom1_holo, var_name) * 1e3
per_dom2_var = postpro_func.read_nc(per_dom2_holo, var_name) * 1e3
per_dom3_var = postpro_func.read_nc(per_dom3_holo, var_name) * 1e3

var_modis = postpro_func.read_nc(modis_holo, 'lwp')
var_modis = np.transpose(var_modis)
print(np.shape(var_modis))

dom1_lat = postpro_func.read_nc(dom1_holo, 'lat')
dom1_lon = postpro_func.read_nc(dom1_holo, 'lon')
dom2_lat = postpro_func.read_nc(dom2_holo, 'lat')
dom2_lon = postpro_func.read_nc(dom2_holo, 'lon')
dom3_lat = postpro_func.read_nc(dom3_holo, 'lat')
dom3_lon = postpro_func.read_nc(dom3_holo, 'lon')



def find_index(given_cor, target_cor):
    index = np.where(given_cor == target_cor)[0][0]
    return index


dom1_var_selected_domain = dom1_var[find_index(dom1_lat, 65):find_index(dom1_lat, 71),
                           find_index(dom1_lon, -20):find_index(dom1_lon, -8)]

dom2_var_selected_domain = dom2_var[find_index(dom2_lat, 65):find_index(dom2_lat, 71),
                           find_index(dom2_lon, -20):find_index(dom2_lon, -8)]

per_dom1_var_selected_domain = per_dom1_var[find_index(dom1_lat, 65):find_index(dom1_lat, 71),
                           find_index(dom1_lon, -20):find_index(dom1_lon, -8)]

per_dom2_var_selected_domain = per_dom2_var[find_index(dom2_lat, 65):find_index(dom2_lat, 71),
                           find_index(dom2_lon, -20):find_index(dom2_lon, -8)]


modis_holo_selected_domain = var_modis[find_index(dom1_lat, 65):find_index(dom1_lat, 71),
                           find_index(dom1_lon, -20):find_index(dom1_lon, -8)]


def weight(var):
    weight = np.zeros_like(var) + 1. / (var.size)
    return weight


def lable_hist(var):
    median = str(np.median(var))
    mean = str((round(np.mean(var))))
    std = str(np.std(var))
    lable = ' (' + mean + ' $\mathrm{g m^{-2}}$' + ')'
    return lable


def pdf_var_prepare(axs0, var, color, label, titel):
    font_legend = 15
    font_tick = 20
    var_n_zero = var[var>1.0]
    var_c = var_n_zero.compressed()
    numbin = np.arange(0, 1000, 10)
    name =  '$ \mathrm{LWP}$ ($\mathrm{gm^{-2}}$)'
    line_width = 2
    axs0.hist(var_c,  bins=numbin, weights=weight(var_c),  histtype='step',
         linewidth=line_width, color= color, label= label + lable_hist(var_c), log= True)

    axs0.legend(loc = 'upper right', fontsize = font_legend, frameon = True)
    axs0.set_xlabel(name, fontsize= font_tick)
    axs0.tick_params(axis = 'x', labelsize = font_tick)  # to Set Matplotlib Tick Labels Font Size
    axs0.tick_params(axis = 'y', labelsize = font_tick)
    ticks = np.arange(0, 0.12, 0.02)
    #axs0.set_yticks(ticks)
    axs0.set_title(titel, fontsize= font_tick)
    axs0.grid(True)
    return

titel_kind = ['2 km horizontal resoluosion', '1 km horizontal resoluosion',
              '0.5 km horizontal resoluosion', 'MODIS']
#fig, axs0 = plt.subplots(figsize = (15, 10))

#pdf_var_prepare(var = dom1_var_selected_domain, color = 'blue', label = titel_kind[0])
#pdf_var_prepare(var = dom2_var_selected_domain, color = 'red', label = titel_kind[1])
#pdf_var_prepare(var = dom3_var, color = 'green', label = titel_kind[2])
#pdf_var_prepare(var= modis_holo_selected_domain, color = 'black', label = titel_kind[3])
#plt.savefig('lwp_run_3nests.png')
#plt.savefig('lwp_run_3nests.pdf')
label_v = 'Volcano'
label_nv = 'No-Volcano'
label_modis = 'MODIS'
font_tick = 20
#fig, ((axs1, axs2), (axs3, axs4)) = plt.subplots(2, 2, figsize = (12, 12))
fig,((axs1, axs2, axs3)) = plt.subplots(1, 3, figsize = (18, 6), sharey=True)
pdf_var_prepare(axs1, var = dom1_var_selected_domain, color = 'blue', label = label_nv, titel = titel_kind[0])
pdf_var_prepare(axs1, var = per_dom1_var_selected_domain, color = 'red', label = label_v, titel = titel_kind[0])
pdf_var_prepare(axs1, var= modis_holo_selected_domain, color = 'black', label = label_modis, titel = titel_kind[0])
axs1.annotate("(a)", xy=(0.02, 0.95), xycoords="axes fraction", fontsize= font_tick)
axs1.set_ylabel('Relative Frequency', fontsize = font_tick)
pdf_var_prepare(axs2, var = dom2_var_selected_domain, color = 'blue', label = label_nv, titel = titel_kind[1])
pdf_var_prepare(axs2, var = per_dom2_var_selected_domain, color = 'red', label = label_v, titel= titel_kind[1])
pdf_var_prepare(axs2, var= modis_holo_selected_domain, color = 'black', label = label_modis, titel = titel_kind[1])
axs2.annotate("(b)", xy=(0.02, 0.95), xycoords="axes fraction", fontsize= font_tick)

pdf_var_prepare(axs3, var = dom3_var, color = 'blue', label = label_nv, titel = titel_kind[2])
pdf_var_prepare(axs3, var = per_dom3_var, color = 'red', label = label_v, titel = titel_kind[2])
pdf_var_prepare(axs3, var= modis_holo_selected_domain, color = 'black',  label = label_modis, titel = titel_kind[2])
axs3.annotate("(c)", xy=(0.02, 0.95), xycoords="axes fraction", fontsize= font_tick)

plt.tight_layout()

plt.savefig('./manuscript_figures/sensitivity_lwp_resolusion_manuscript_log.png')
plt.savefig('./manuscript_figures/sensitivity_lwp_resolusion_manuscript_log.pdf')

plt.show()

