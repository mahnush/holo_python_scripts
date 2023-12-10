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
var_name = 'nd_dw'

dom1_var = postpro_func.read_nc(dom1_holo, var_name)
dom2_var = postpro_func.read_nc(dom2_holo, var_name)
dom3_var = postpro_func.read_nc(dom3_holo, var_name)

per_dom1_var = postpro_func.read_nc(per_dom1_holo, var_name)
per_dom2_var = postpro_func.read_nc(per_dom2_holo, var_name)
per_dom3_var = postpro_func.read_nc(per_dom3_holo, var_name)

var_modis = postpro_func.read_nc(modis_holo, 'nd')
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

print(np.shape(dom1_var_selected_domain))
print(np.shape(dom2_var_selected_domain))
print(np.shape(modis_holo_selected_domain))


def weight(var):
    weight = np.zeros_like(var) + 1. / (var.size)
    return weight


def lable_hist(var):
    median = str(np.median(var))
    mean = str((round(np.mean(var))))
    std = str(np.std(var))
    lable = '('+'mean = ' + mean +')'
    return lable


def pdf_var_prepare(axs0, var, color, label, titel):
    font_legend = 15
    font_tick = 15
    var_n_zero = var[var>1.0]
    var_up_limit = var_n_zero[var_n_zero<1000]

    var_c = var_up_limit.compressed()
    print(np.shape(var_c))
    numbin = np.arange(0, 1000, 10)
    #name =  '$ \mathrm{LWP}$ ($\mathrm{g m^{-2}}$)'
    name = '$ \mathrm{N_d}$ ($\mathrm{cm^{-3}}$)'
    line_width = 2
    axs0.hist(var_c,  bins=numbin, weights=weight(var_c),  histtype='step',
         linewidth=line_width, color= color, label= label + lable_hist(var_c))#, log= True)

    axs0.legend(loc = 'upper right', fontsize = font_legend, frameon = True)
    axs0.set_xlabel(name, fontsize= font_tick)
    axs0.tick_params(axis = 'x', labelsize = font_tick)  # to Set Matplotlib Tick Labels Font Size
    axs0.tick_params(axis = 'y', labelsize = font_tick)
    ticks = np.arange(0, 0.11, 0.02)
    #axs0.set_yticks(ticks)
    axs0.set_ylabel('Relative Frequency', fontsize = font_tick)
    axs0.set_title(titel, fontsize= font_tick)
    axs0.grid(True)
    return

titel_kind = ['2 km resoluosion', '1 km resoluosion', '0.5 km resoluosion', 'modis']
#fig, axs0 = plt.subplots(figsize = (15, 10))

#pdf_var_prepare(var = dom1_var_selected_domain, color = 'blue', label = titel_kind[0])
#pdf_var_prepare(var = dom2_var_selected_domain, color = 'red', label = titel_kind[1])
#pdf_var_prepare(var = dom3_var, color = 'green', label = titel_kind[2])
#pdf_var_prepare(var= modis_holo_selected_domain, color = 'black', label = titel_kind[3])
#plt.savefig('lwp_run_3nests.png')
#plt.savefig('lwp_run_3nests.pdf')

fig, ((axs1, axs2), (axs3, axs4)) = plt.subplots(2, 2, figsize = (12, 12))

pdf_var_prepare(axs1, var = dom1_var_selected_domain, color = 'blue', label = 'no-volcano', titel = titel_kind[0])
pdf_var_prepare(axs1, var = per_dom1_var_selected_domain, color = 'red', label = 'volcano', titel = titel_kind[0])
pdf_var_prepare(axs1, var= modis_holo_selected_domain, color = 'black', label = 'MODIS', titel = titel_kind[0])

pdf_var_prepare(axs2, var = dom2_var_selected_domain, color = 'blue', label = 'no-volcano', titel = titel_kind[1])
pdf_var_prepare(axs2, var = per_dom2_var_selected_domain, color = 'red', label = 'volcano', titel= titel_kind[1])
pdf_var_prepare(axs2, var= modis_holo_selected_domain, color = 'black', label = 'MODIS', titel = titel_kind[1])

pdf_var_prepare(axs3, var = dom3_var, color = 'blue', label = 'no-volcano', titel = titel_kind[2])
pdf_var_prepare(axs3, var = per_dom3_var, color = 'red', label = 'volcano', titel = titel_kind[2])
pdf_var_prepare(axs3, var= modis_holo_selected_domain, color = 'black',  label = 'MODIS', titel = titel_kind[2])


fig.delaxes(axs4)
plt.tight_layout()

plt.savefig('run_nd_3nests.png')
plt.show()

