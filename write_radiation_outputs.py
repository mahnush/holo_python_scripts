import numpy as np
from netCDF4 import Dataset

ncout = Dataset(
    '/work/bb1036/b380900/work_levante/cosp_var_new_version_icon/holo_auto_runs/albedo_2_sep_per_tqc.nc',
    mode = "w", format = 'NETCDF4_CLASSIC')

ipath = '/scratch/b/b380900/work_levante/perturbed_holo_cosp_fix/'
fileList = open('/scratch/b/b380900/work_levante/control_holo_cosp_fix'
                '/file_list_2_sep.txt', 'r+')

nlon = 4001
nlat = 1501
ns = 8

sob_t = np.zeros((ns, nlat, nlon))
sod_t = np.zeros((ns, nlat, nlon))
lwp_modis = np.zeros((ns, nlat, nlon))
ih = 0

for FILE_NAME in fileList:
    FILE_NAME = FILE_NAME.strip()
    FILE_NAME = ipath + FILE_NAME
    print(FILE_NAME)
    nc_dw = Dataset(FILE_NAME, 'r')
    sob_t[ih, :, :] = nc_dw.variables['sob_t'][0, :, :]
    sod_t[ih, :, :] = nc_dw.variables['sod_t'][0, :, :]
    #lwp_modis[ih, :, :] = nc_dw.variables['modis_Liquid_Water_Path_Mean'][0, :, :]
    lwp_modis[ih, :, :] = nc_dw.variables['tqc'][0, :, :]
    ih += 1
    print(ih)
lon = nc_dw.variables['lon'][:]
lat = nc_dw.variables['lat'][:]


sw_up = sod_t - sob_t

sw_up_mask = np.ma.masked_where(lwp_modis ==0, sw_up)
albedo = sw_up_mask / sod_t

albedo_mean = np.ma.mean(albedo, axis = 0)
#albedo_mean = albedo
ncout.createDimension('lon', nlon)
ncout.createDimension('lat', nlat)
lon_o = ncout.createVariable('lon', np.float32, ('lon',))
lat_o = ncout.createVariable('lat', np.float32, ('lat',))
albedo_dw_mean_o = ncout.createVariable('albedo_TOA', np.float32, ('lat', 'lon'))


lon_o[:] = lon[:]
lat_o[:] = lat[:]
albedo_dw_mean_o[:] = albedo_mean[:]