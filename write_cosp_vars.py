import numpy as np
from netCDF4 import Dataset

ncout = Dataset(
    '/work/bb1036/b380900/work_levante/cosp_var_new_version_icon/holo_auto_runs/2_sep_per_control.nc',
    mode = "w", format = 'NETCDF4_CLASSIC')

ipath = '/scratch/b/b380900/work_levante/perturbed_holo_cosp_fix/2_sep/'
fileList = open(
    '/scratch/b/b380900/work_levante/control_holo_cosp_fix/2_sep/2_sep_file_list.txt', 'r+')

nlon = 4001
nlat = 1501
#ns = 2
#nlon = 2001
#nlat = 1201
#nlon = 2401
#nlat = 1201
ns = 16
re_dw = np.zeros((ns, nlat, nlon))
tau_dw = np.zeros((ns, nlat, nlon))
lwp_dw = np.zeros((ns, nlat, nlon))
nd_dw = np.zeros((ns, nlat, nlon))
ctt_dw = np.zeros((ns, nlat, nlon))
ih = 0

for FILE_NAME in fileList:
    FILE_NAME = FILE_NAME.strip()
    FILE_NAME = ipath + FILE_NAME
    print(FILE_NAME)
    nc_dw = Dataset(FILE_NAME, 'r')
    re_dw[ih, :, :] = nc_dw.variables['modis_Cloud_Particle_Size_Water_Mean'][0, :, :]
    tau_dw[ih, :, :] = nc_dw.variables['modis_Optical_Thickness_Water_Mean'][0, :, :]
    lwp_dw[ih, :, :] = nc_dw.variables['modis_Liquid_Water_Path_Mean'][0, :, :]
    ih += 1
    print(ih)
re_mask = np.ma.masked_where(re_dw == 0, re_dw)
nd_dw = 1.37e-5 * (tau_dw ** 0.5) * (re_mask ** -2.5) * 1e-6

#lon = nc_dw.variables['lon'][:]
#lat = nc_dw.variables['lat'][*-:]
#ncout.createDimension('lon', nlon)
#ncout.createDimension('lat', nlat)
#ncout.createDimension('time', ns)
#lon_o = ncout.createVariable('lon', np.float32, ('lon',))
#lat_o = ncout.createVariable('lat', np.float32, ('lat',))
#re_dw_mean_o = ncout.createVariable('re_dw', np.float32, ('time', 'lat', 'lon'))
#tau_dw_mean_o = ncout.createVariable('tau_dw', np.float32, ('time', 'lat', 'lon'))
#lwp_dw_mean_o = ncout.createVariable('lwp_dw', np.float32, ('time', 'lat', 'lon'))
#qnc_dw_mean_o = ncout.createVariable('nd_dw', np.float32, ('time', 'lat', 'lon'))
#lon_o[:] = lon[:]
#lat_o[:] = lat[:]
#print(np.shape(re_dw_mean_o))
#re_dw_mean_o[:] = re_dw[:]
#tau_dw_mean_o[:] = tau_dw[:]
#lwp_dw_mean_o[:] = lwp_dw[:]
#qnc_dw_mean_o[:] = nd_dw[:]
re_dw[re_dw < 4e-6] = np.nan
tau_dw[tau_dw < 4] = np.nan
lwp_dw[lwp_dw < 0.0001] = np.nan
nd_dw = 1.37e-5 * (tau_dw ** 0.5) * (re_dw ** -2.5)

lon = nc_dw.variables['lon'][:]
lat = nc_dw.variables['lat'][:]

re_dw = re_dw * 1e6

re_dw = np.ma.masked_array(re_dw, np.isnan(re_dw))
tau_dw = np.ma.masked_array(tau_dw, np.isnan(tau_dw))
lwp_dw = np.ma.masked_array(lwp_dw, np.isnan(lwp_dw))
nd_dw = np.ma.masked_array(nd_dw, np.isnan(nd_dw))
# ctt_dw = np.ma.masked_array(ctt_dw,np.isnan(ctt_dw))
# print(np.amax(ctt_dw))
re_dw_mean = np.ma.mean(re_dw, axis = 0)
tau_dw_mean = np.ma.mean(tau_dw, axis = 0)
lwp_dw_mean = np.ma.mean(lwp_dw, axis = 0)
nd_dw_mean = np.ma.mean(nd_dw, axis = 0)
# ctt_dw_mean=np.ma.mean(ctt_dw,axis=0)
# print('mean')
# print(np.amax(ctt_dw_mean))
nd_dw_mean = nd_dw_mean * 1e-6

ncout.createDimension('lon', nlon)
ncout.createDimension('lat', nlat)
lon_o = ncout.createVariable('lon', np.float32, ('lon',))
lat_o = ncout.createVariable('lat', np.float32, ('lat',))
re_dw_mean_o = ncout.createVariable('re_dw', np.float32, ('lat', 'lon'))
tau_dw_mean_o = ncout.createVariable('tau_dw', np.float32, ('lat', 'lon'))
lwp_dw_mean_o = ncout.createVariable('lwp_dw', np.float32, ('lat', 'lon'))
qnc_dw_mean_o = ncout.createVariable('nd_dw', np.float32, ('lat', 'lon'))
# ctt_dw_mean_o= ncout.createVariable('ctt_dw',np.float32,('lat','lon'))
lon_o[:] = lon[:]
lat_o[:] = lat[:]
re_dw_mean_o[:] = re_dw_mean[:]
tau_dw_mean_o[:] = tau_dw_mean[:]
lwp_dw_mean_o[:] = lwp_dw_mean[:]
qnc_dw_mean_o[:] = nd_dw_mean[:]
print('done')