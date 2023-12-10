#calc_lonlat_unrotate_pole_write_nc.py
'''
This program calculates geographycal (unrotated) longitudes and latitudes, which are 2-D variables.
Inputs are 1-D rotated longitudes and latitudes.
The input data needs to include 1-D rotated longitudes and latitudes as well as a 2-D field.
A column loading of so2 is currently used.

Below can be ignored because THESE ARE NOW DONE WITHIN THIS SCRIPT !!!
  (This is pre-calculated from mass mixing ratio using extract_mmr2load_pp2nc_vn11_nestingsuite_iris.py )
  (time dimension is removed by doing
     ncwa -a time *.nc /tmp/myoshioka/*_notime.nc
   Also preferably do below to change record dimension to fixed dimension 
     ncks -O --fix_rec_dmn grid_latitude /tmp/myoshioka/*_notime.nc /tmp/myoshioka/*_notime.nc )
'''

#ipython calc_lonlat_unrotate_pole_write_nc.py

import iris
import subprocess  #so bash command can be used as "subprocess.call(["ls -l "+diri+fnmi], shell=True)" etc
import sys         #to enable "sys.exit()" which is the same as "stop" in IDL
import numpy as np
#from iris.analysis.cartography import rotate_pole
from iris.analysis.cartography import unrotate_pole

s1 = ' '
s2 = '  '
s4 = '    '
hsh = '#'
scn = ';'
usc = '_'
dnc = '.nc'

"""
#centre of Regn1
lonc = -10.0
latc =  48.0

#DOMAIN CENTRE LOCATION ABOVE IS SET IN THE MODEL.
#BUT THE MODEL OUTPUT (NC FILE) CONTAINS THE INFO ABOUT WHERE THE ROTATED POLE IS.
#USING THIS IS EASIER AND SAFER.
#                rotated_latitude_longitude:grid_north_pole_latitude = 42. ;
#                rotated_latitude_longitude:grid_north_pole_longitude = 170. ;

#calculate longitude and latitude of rotated pole
if lonc < 0:
    lonp = lonc + 180.0
else:
    lonp = lonc - 180.0

if latc < 0:
    latp = latc + 90.0
else:
    latp = 90.0 - latc

print(lonc,latc) #(-10.0, 48.0)
print(lonp,latp) #(170.0, 42.0)
"""

#POLE FOR Regn1
lonp = 170.0; latp =  42.0 #old one using offsets
lonp = 174.7; latp =  39.4 #new one for swBritL with no offsets in cj069
lonp = 169.3; latp =  49.6 #new one for wPortoL with no offsets in cj046
lonp = 179.5; latp =  39.3 #for eChannel with no offsets in cj666 and cj728
lonp = 173.3; latp =  39.2 #c180WL2
lonp = 160.0; latp =  25.0 #ADVANCE NAtl_hilat

dom  = 'natl'
dom  = 'brit2port'
dom  = 'c182'
dom  = 'wporto'
dom  = 'swbrit'
dom  = 'swBritL'
dom  = 'wPortoL'
dom  = 'eChannel'
dom  = 'c180WL2'
dom  = 'NAtl_hilat'
rgndommdl = 'NAtl_hilat_km2p5_ra3_p3_casim_ukca'

job  = 'ch060'
date = '20190712'
hhh  = '012'

job  = 'cj069'
date = '20190710'
hhh  = '012'

job  = 'cj046'
date = '20190712'
hhh  = '012'

job  = 'cj666'
date = '20190710'
hhh  = '012'

job  = 'cj728'    #this will create the same thing as for cj666. I couldve copied this to cj728 directory. but did this anyway.
date = '20190718'
hhh  = '012'

vnmfi = 'load_so2' #any 2D data is fine -common for all above

job   = 'cl169'
date  = '20190711'
hhh   = '012'
vnmfi = 'cdnc_incloud_casim_sfc' #for casim run #made 2D

#ADVANCE
job   = 'cs159'
date  = '20140901'
hhh   = '024'
vnmfi = 'sw_up_toa' #for casim run #made 2D


#prepare data
diri = '/gws/nopw/j04/ukca_vol1/myoshioka/um/Dumps/Links2NC/'+job+'/'+date+'/'
diri = '/gws/nopw/j04/ukca_vol1/myoshioka/um/Dumps/Links2NC/'+job+'/'

dirt = '/tmp/myoshioka/'

fnm0 = vnmfi+'_'+job+'_'+dom+'_'+date+'_'+hhh+'.nc'
fnmi = vnmfi+'_'+job+'_'+dom+'_'+date+'_'+hhh+'_notime.nc'

fnm0 = vnmfi+'_'+job+'_'+rgndommdl+'_'+date+'_'+hhh+'.nc'
fnmi = vnmfi+'_'+job+'_'+rgndommdl+'_'+date+'_'+hhh+'_notime.nc'

'''
#THESE HAVE TO BE DONE ONLY ONCE -- UNCOMMENT AS NEEDED
#remove time dimension by doing: ncwa -a time *.nc /tmp/myoshioka/*_notime.nc
print "ncwa -a time "+diri+fnm0+" "+dirt+fnmi
subprocess.call(["ncwa -a time "+diri+fnm0+" "+dirt+fnmi], shell=True)
print

#change record dimension to fixed dimension
print "ncks -O --fix_rec_dmn grid_latitude "+dirt+fnmi+" "+dirt+fnmi
subprocess.call(["ncks -O --fix_rec_dmn grid_latitude "+dirt+fnmi+" "+dirt+fnmi], shell=True)
print
subprocess.call(["ls -l "+dirt+fnmi], shell=True)
print('subprocess.call([" ncdump '+dirt+fnmi+' |less"], shell=True)')
print
print "mv "+dirt+fnmi+' '+diri
subprocess.call(["mv "+dirt+fnmi+' '+diri], shell=True)
'''
subprocess.call(["ls -l "+diri+fnmi], shell=True)
print('subprocess.call([" ncdump '+diri+fnmi+' |less "], shell=True)')
print

#sys.exit() #stop
#ipython calc_lonlat_unrotate_pole_write_nc.py


cube = iris.load_cube(diri+fnmi)
cube #<iris 'Cube' of load_so2 / (kg m-2) (grid_latitude: 480; grid_longitude: 1100)>

lonname = 'grid_longitude'
latname = 'grid_latitude'

lon = cube.coord(lonname).points
lat = cube.coord(latname).points

nlon = len(lon)
nlat = len(lat)

nlon,nlat #(1100, 480)

nx = nlon-1
ix = nlon/2-1

ny = nlat-1
iy = nlat/2-1

print lon[:3],lon[ix:ix+3],lon[nx-2:] #[298.4   298.472  298.544   ] [337.92798       338.            338.072        ] [377.38397  377.456  377.52798 ]
print lat[:3],lat[iy:iy+3],lat[ny-2:] #[-17.28  -17.208  -17.136002] [ -7.2000504e-02  -1.9073486e-06   7.1998596e-02] [ 17.063997  17.136   17.207998]

lon2d = np.zeros((nlat, nlon))
for ilat in range(0, nlat):
    lon2d[ilat,:] = lon[:]

lon2d.shape #(480, 1100)
print lon2d[ 0,:3],    lon2d[ 0,ix:ix+3],lon2d[   0,nx-2:] #[298.3999939 298.47198486 298.54400635] [337.92797852 338.         338.07199097] [377.38397217 377.45599365 377.52798462]
print lon2d[iy,:3],    lon2d[iy,ix:ix+3],lon2d[  iy,nx-2:] #[298.3999939 298.47198486 298.54400635] [337.92797852 338.         338.07199097] [377.38397217 377.45599365 377.52798462]
print lon2d[  :3,   0],lon2d[  :3,   ix],lon2d[  :3,   nx] #[298.3999939 298.3999939  298.3999939]  [337.92797852 337.92797852 337.92797852] [377.52798462 377.52798462 377.52798462]
print lon2d[iy:iy+3,0],lon2d[iy:iy+3,ix],lon2d[iy:iy+3,nx] #[298.3999939 298.3999939  298.3999939]  [337.92797852 337.92797852 337.92797852] [377.52798462 377.52798462 377.52798462]

lat2d = np.zeros((nlat, nlon))
for ilon in range(0, nlon):
    lat2d[:,ilon] = lat[:]

lat2d.shape #(480, 1100)
print lat2d[:3, 0],    lat2d[iy:iy+3, 0],lat2d[ny-2:, 0]   #[-17.28000069 -17.20800018 -17.13600159] [-7.20005035e-02 -1.90734863e-06  7.19985962e-02] [17.06399727 17.13599968 17.20799828]
print lat2d[:3,ix],    lat2d[iy:iy+3,ix],lat2d[ny-2:,ix]   #[-17.28000069 -17.20800018 -17.13600159] [-7.20005035e-02 -1.90734863e-06  7.19985962e-02] [17.06399727 17.13599968 17.20799828]
print lat2d[0,  :3   ],lat2d[iy,  :3   ],lat2d[ny,  :3   ] #[-17.28000069 -17.28000069 -17.28000069] [-0.0720005 -0.0720005 -0.0720005] [17.20799828 17.20799828 17.20799828]
print lat2d[0,ix:ix+3],lat2d[iy,ix:ix+3],lat2d[ny,ix:ix+3] #[-17.28000069 -17.28000069 -17.28000069] [-0.0720005 -0.0720005 -0.0720005] [17.20799828 17.20799828 17.20799828]

#SO FAR ROTATED LONS&LATS. CALCULATE UNROTATED (GEOGRAPHYCAL) LONS&LATS
lon2du, lat2du = unrotate_pole(lon2d, lat2d, lonp, latp)

print lon2du.shape #(480, 1100)
print lon2du[ 0,:3],    lon2du[ 0,ix:ix+3],lon2du[ 0,nx-2:]   #[-68.01084595 -67.95868955 -67.90648229] [-33.81783693 -33.74434091 -33.67083841] [ 8.95853129  9.03428572  9.10997552]
print lon2du[iy,:3],    lon2du[iy,ix:ix+3],lon2du[iy,nx-2:]   #[-80.0562353  -80.00116922 -79.94604316] [-41.17741069 -41.08578355 -40.99411134] [15.04189816 15.13871493 15.23541366]
print lon2du[ny,:3],    lon2du[ny,ix:ix+3],lon2du[ny,nx-2:]   #[-94.28074255 -94.22917112 -94.17754034] [-53.94024544 -53.82818656 -53.71596153] [26.18734554 26.31365494 26.43968279]
print lon2du[  :3,   0],lon2du[  :3,   ix],lon2du[  :3,   nx] #[-68.01084595 -68.05884332 -68.10685221] [-33.81783693 -33.84332087 -33.86883564] [ 9.10997552  9.13087376  9.15179916]
print lon2du[iy:iy+3,0],lon2du[iy:iy+3,ix],lon2du[iy:iy+3,nx] #[-80.0562353  -80.11000253 -80.16380772] [-41.17741069 -41.21561312 -41.2539008 ] [15.23541366 15.26774416 15.300153  ]
print lon2du[ny-2:,  0],lon2du[ny-2:,  ix],lon2du[ny-2:,  nx] #[-94.14887588 -94.21477773 -94.28074255] [-53.79049141 -53.86523959 -53.94024544] [26.3028482  26.37112987 26.43968279]

print lat2du.shape #(480, 1100)
print lat2du[:3, 0],    lat2du[iy:iy+3, 0],lat2du[ny-2:, 0]   #[ 7.97520285  8.02928586  8.08336186] [20.64743332 20.69894247 20.7504363 ] [32.29286027 32.33849415 32.38409132]
print lat2du[:3,ix],    lat2du[iy:iy+3,ix],lat2du[ny-2:,ix]   #[27.31214028 27.38049    27.44883327] [43.45936843 43.52581922 43.59225902] [58.72809347 58.78876797 58.84939588]
print lat2du[:3,nx],    lat2du[iy:iy+3,nx],lat2du[ny-2:,nx]   #[28.54808592 28.6177085  28.68732602] [45.05613037 45.12441517 45.19269264] [60.90256452 60.96647052 61.03033848]
print lat2du[0,   :3  ],lat2du[iy,   :3  ],lat2du[ny,   :3  ] #[ 7.97520285  8.02056565  8.06592181] [20.64743332 20.69771577 20.74800193] [32.38409132 32.43732231 32.49057102]
print lat2du[0,ix:ix+3],lat2du[iy,ix:ix+3],lat2du[ny,ix:ix+3] #[27.31214028 27.33372283 27.35523341] [43.45936843 43.4870398  43.51462616] [58.84939588 58.88649779 58.92350843]
print lat2du[0,nx-2:],  lat2du[iy,nx-2:  ],lat2du[ny,nx-2:  ] #[28.58298941 28.56556739 28.54808592] [45.10159448 45.07889851 45.05613037] [61.09371738 61.06206869 61.03033848]

#ipython calc_lonlat_unrotate_pole_write_nc.py


#replace data -lon
vnmx           = 'longitude'
vnmfx          = 'lon2d_unrot'
cube.data      = lon2du
cube.var_name  = vnmx
cube.long_name = vnmx
cube.units     = 'degrees_east'

print cube

dirt = '/tmp/myoshioka/'
diro = diri
fnmo = 'lon_lat_2d_unrot_'+dom+'.nc'
fnmx = vnmfx+'_'+dom+'.nc'

print 'Output filename: ',diro+fnmo
print

#SAVE RESULT
print('Saving the result to netCDF file...')
iris.save(cube, dirt+fnmx, netcdf_format='NETCDF3_CLASSIC')
#subprocess.call(["ls -l "+dirt+fnmo], shell=True)
#print('subprocess.call([" ncdump '+dirt+fnmo+' |less "], shell=True)')

#DELETE ATTRIBUTES THAT ARE NO LONGER APPLICABLE
subprocess.call(["ncatted -a um_stash_source,"+vnmx+",d,, "+dirt+fnmx], shell=True)
subprocess.call(["ncatted -a cell_methods,"+vnmx+",d,, "+dirt+fnmx], shell=True)
subprocess.call(["ncatted -a grid_mapping,"+vnmx+",d,, "+dirt+fnmx], shell=True)
subprocess.call(["ncatted -a coordinates,"+vnmx+",d,, "+dirt+fnmx], shell=True)

##MOVE TO GWS
#subprocess.call(["mv "+dirt+fnmo+' '+diro], shell=True)

#CREATE OUTPUT DATA
subprocess.call(["ncks -O -v "+vnmx+" "+dirt+fnmx+" "+dirt+fnmo], shell=True)

print('Created;')
subprocess.call(["ls -l "+dirt+fnmo], shell=True)
#print('subprocess.call([" ncdump '+dirt+fnmo+' |less "], shell=True)')


#replace data -lat
vnmy           = 'latitude'
vnmfy          = 'lat2d_unrot'
cube.data      = lat2du
cube.var_name  = vnmy
cube.long_name = vnmy
cube.units     = 'degrees_north'

print cube

fnmy = vnmfy+'_'+dom+'.nc'

#SAVE RESULT
print('Saving the result to netCDF file...')
iris.save(cube, dirt+fnmy, netcdf_format='NETCDF3_CLASSIC')
#subprocess.call(["ls -l "+dirt+fnmy], shell=True)
#print('subprocess.call([" ncdump '+dirt+fnmy+' |less "], shell=True)')

#DELETE ATTRIBUTES THAT ARE NO LONGER APPLICABLE
subprocess.call(["ncatted -a um_stash_source,"+vnmy+",d,, "+dirt+fnmy], shell=True)
subprocess.call(["ncatted -a cell_methods,"+vnmy+",d,, "+dirt+fnmy], shell=True)
subprocess.call(["ncatted -a grid_mapping,"+vnmy+",d,, "+dirt+fnmy], shell=True)
subprocess.call(["ncatted -a coordinates,"+vnmy+",d,, "+dirt+fnmy], shell=True)

#MOVE VARIABLE TO OUTPUT FILE
subprocess.call(["ncks -A -v "+vnmy+" "+dirt+fnmy+" "+dirt+fnmo], shell=True)

##CHANGE RECORD DIMENSION INTO FIXED DIMENSION - not necessary if input data is treated
#subprocess.call(["ncks -O --fix_rec_dmn grid_latitude "+dirt+fnmo+" "+dirt+fnmo], shell=True)

#MOVE TO GWS
subprocess.call(["mv "+dirt+fnmo+' '+diro], shell=True)

print('Created;')
subprocess.call(["ls -l "+diro+fnmo], shell=True)
print('subprocess.call([" ncdump '+diro+fnmo+' |less "], shell=True)')

#ipython calc_lonlat_unrotate_pole_write_nc.py

