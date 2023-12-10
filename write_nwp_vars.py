import numpy as np
from netCDF4 import Dataset

ncout = Dataset(
    '/work/bb1036/b380900/work_levante/cosp_var_new_version_icon/compare_holo_nwp_vs_art/'
    'diff_timestep/6sep_con_nwp.nc',
    mode = "w", format = 'NETCDF4_CLASSIC')

ipath = '/scratch/b/b380900/work_levante/control_holo_nwp/'
fileList = open(
    '/scratch/b/b380900/work_levante/perturbed_holo_nwp/diff_timestep_file_list_6_sep.txt', 'r+')

nlon = 801
nlat = 301
nlev = 75
ns = 2

tqc = np.zeros((ns, nlat, nlon))
qnc = np.zeros((ns, nlev, nlat, nlon))
z = np.zeros((ns, nlev, nlat, nlon))
pres = np.zeros((ns, nlev, nlat, nlon))
temp = np.zeros((ns, nlev, nlat, nlon))
ih = 0

for FILE_NAME in fileList:
    FILE_NAME = FILE_NAME.strip()
    FILE_NAME = ipath + FILE_NAME
    print(FILE_NAME)
    nc_dw = Dataset(FILE_NAME, 'r')
    tqc[ih, :, :] = nc_dw.variables['tqc'][0, :, :]
    qnc[ih, :, :, :] = nc_dw.variables['qnc'][0, :, :, :]
    z[ih, :, :, :] = nc_dw.variables['geopot'][0, :, :, :]
    pres[ih, :, :, :] = nc_dw.variables['pres'][0, :, :, :]
    temp[ih, :, :, :] = nc_dw.variables['temp'][0, :, :, :]
    ih += 1
    print(ih)

lon = nc_dw.variables['lon'][:]
lat = nc_dw.variables['lat'][:]
lev = nc_dw.variables['height'][:]

tqc[tqc <= 0] = np.nan
qnc[qnc <= 0] = np.nan

tqc = np.ma.masked_array(tqc, np.isnan(tqc))
qnc = np.ma.masked_array(qnc, np.isnan(qnc))

# #compute density with ρ = P / (R * T)
R = 287.058
rho = pres/(R*temp)
rho_mean = np.ma.mean(rho,  axis = 0)
tqc_mean = np.ma.mean(tqc,  axis = 0)
qnc_mean = np.ma.mean(qnc, axis = 0)
z_mean = np.ma.mean(z, axis= 0)

ncout.createDimension('lon', nlon)
ncout.createDimension('lat', nlat)
ncout.createDimension('lev', nlev)
lon_o = ncout.createVariable('lon', np.float32, ('lon',))
lat_o = ncout.createVariable('lat', np.float32, ('lat',))
lev_o = ncout.createVariable('lev', np.float32, ('lev',))
tqc_mean_o = ncout.createVariable('tqc', np.float32, ('lat', 'lon'))
qnc_mean_o = ncout.createVariable('qnc', np.float32, ('lev', 'lat', 'lon'))
z_mean_o = ncout.createVariable('geopt', np.float32, ('lev', 'lat', 'lon'))
rho_mean_o = ncout.createVariable('rho', np.float32, ('lev', 'lat', 'lon'))
lon_o[:] = lon[:]
lat_o[:] = lat[:]
lev_o[:] = lev[:]
tqc_mean_o[:] = tqc_mean[:]
qnc_mean_o[:] = qnc_mean[:]
z_mean_o[:] = z_mean[:]
rho_mean_o[:] = rho_mean[:]

