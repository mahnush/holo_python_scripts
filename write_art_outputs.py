import numpy as np
from netCDF4 import Dataset
#
ncout = Dataset(
    '/work/bb1036/b380900/work_levante/cosp_var_new_version_icon/compare_holo_nwp_vs_art/diff_timestep/art_mistral'
    '/6sep_con_art.nc',
    mode = "w", format = 'NETCDF4_CLASSIC')

ipath = '/scratch/b/b380900/work_levante/art_con_mistral/'
fileList = open(
    '/scratch/b/b380900/work_levante/art_per_mistral/diff_file_list_6_sep.txt', 'r+')

nlon = 601
nlat = 301
nlev = 50
ns = 2

tqc = np.zeros((ns, nlat, nlon))
so4_at = np.zeros((ns, nlev, nlat, nlon))
so4_acc = np.zeros((ns, nlev, nlat, nlon))
qnc = np.zeros((ns, nlev, nlat, nlon))
pres = np.zeros((ns, nlev, nlat, nlon))
temp = np.zeros((ns, nlev, nlat, nlon))
so2 = np.zeros((ns, nlev, nlat, nlon))
ih = 0

for FILE_NAME in fileList:
    FILE_NAME = FILE_NAME.strip()
    FILE_NAME = ipath + FILE_NAME
    print(FILE_NAME)
    nc_dw = Dataset(FILE_NAME, 'r')
    tqc[ih, :, :] = nc_dw.variables['tqc'][0, :, :]
    so4_at[ih, :, :, :] = nc_dw.variables['so4_sol_ait'][0, :, :, :]
    so4_acc[ih, :, :, :] = nc_dw.variables['so4_sol_acc'][0, :, :, :]
    qnc[ih, :, :, :] = nc_dw.variables['qnc'][0, :, :, :]
    pres[ih, :, :, :] = nc_dw.variables['pres'][0, :, :, :]
    temp[ih, :, :, :] = nc_dw.variables['temp'][0, :, :, :]
    so2[ih, :, :, :] = nc_dw.variables['TRSO2_chemtr'][0, :, :, :]
    ih += 1
    print(ih)

lon = nc_dw.variables['lon'][:]
lat = nc_dw.variables['lat'][:]
lev = nc_dw.variables['height'][:]
z = nc_dw.variables['z_mc']

R = 287.058
#compute density with ρ = P / (R * T)
rho = pres/(R*temp)
rho_mean = np.ma.mean(rho,  axis = 0)


#tqc[tqc <= 0] = np.nan
#qnc[qnc <= 0] = np.nan
#so4_at[so4_at <= 0] = np.nan
#so4_acc[so4_acc <= 0] = np.nan

#tqc = np.ma.masked_array(tqc, np.isnan(tqc))
#qnc = np.ma.masked_array(qnc, np.isnan(qnc))
#so4_at = np.ma.masked_array(so4_at, np.isnan(so4_at))
#so4_acc = np.ma.masked_array(so4_acc, np.isnan(so4_acc))

tqc_mean = np.ma.mean(tqc,  axis = 0)
qnc_mean = np.ma.mean(qnc, axis = 0)
so4_acc_mean = np.ma.mean(so4_acc, axis = 0)
so4_at_mean = np.ma.mean(so4_at, axis = 0)
so2_mean = np.ma.mean(so2, axis = 0)
#z_mean = np.ma.mean(z, axis= 0)

ncout.createDimension('lon', nlon)
ncout.createDimension('lat', nlat)
ncout.createDimension('lev', nlev)
lon_o = ncout.createVariable('lon', np.float32, ('lon',))
lat_o = ncout.createVariable('lat', np.float32, ('lat',))
lev_o = ncout.createVariable('lev', np.float32, ('lev',))
tqc_mean_o = ncout.createVariable('tqc', np.float32, ('lat', 'lon'))
qnc_mean_o = ncout.createVariable('qnc', np.float32, ('lev', 'lat', 'lon'))
so4_acc_mean_o = ncout.createVariable('so4_acc', np.float32, ('lev', 'lat', 'lon'))
so4_at_mean_o = ncout.createVariable('so4_at', np.float32, ('lev', 'lat', 'lon'))
z_mean_o = ncout.createVariable('z_mc', np.float32, ('lev', 'lat', 'lon'))
rho_mean_o = ncout.createVariable('rho', np.float32, ('lev', 'lat', 'lon'))
so2_o = ncout.createVariable('so2', np.float32, ('lev', 'lat', 'lon'))

lon_o[:] = lon[:]
lat_o[:] = lat[:]
lev_o[:] = lev[:]
tqc_mean_o[:] = tqc_mean[:]
qnc_mean_o[:] = qnc_mean[:]
so4_acc_mean_o[:] = so4_acc_mean[:]
so4_at_mean_o[:] = so4_at_mean[:]
z_mean_o[:] = z[:]
rho_mean_o[:] = rho_mean[:]
so2_o[:] = so2_mean[:]
