#!/bin/bash
for p in /dades/LIDAR/towers_detection/LAS_data_windows_200x200/RIBERA/*.las; do
  pdal translate $p $p hag_nn --writers.las.extra_dims="HeightAboveGround=float32"
  echo "$p"
done

for p in /dades/LIDAR/towers_detection/LAS_data_windows_200x200/CAT3/*.las; do
  pdal translate $p $p hag_nn --writers.las.extra_dims="HeightAboveGround=float32"
  echo "$p"
done

#for p in /dades/LIDAR/towers_detection/LAS_data_windows/BDN/w_towers_80x80_10p/tower_v*.las; do
#  pdal translate $p $p hag_nn --writers.las.extra_dims="HeightAboveGround=float32"
#  echo "$p"
#done

#for p in /dades/LIDAR/towers_detection/LAS_data_windows/datasets/CAT3/w_othertowers_100x100_50p/othertower_*.las; do
#  pdal translate $p $p hag_nn --writers.las.extra_dims="HeightAboveGround=float32"
#  echo "$p"
#done
#for p in /dades/LIDAR/towers_detection/LAS_data_windows/datasets/RIBERA/w_othertowers_100x100_50p/othertower_*.las; do
#  pdal translate $p $p hag_nn --writers.las.extra_dims="HeightAboveGround=float32"
#  echo "$p"
#done

# Rename files
#for f in /dades/LIDAR/towers_detection/LAS_data_windows/datasets/CAT3/w_no_towers_40x40/*pkl.las; do
#  mv -- "$f" "${f%.pkl.las}.las"
#done
# save HAG in z
#for p in /dades/LIDAR/towers_detection/datasets/pc_towers_40x40/las_files/*HAG.las; do
#  pdal translate $p $p hag_nn ferry --filters.ferry.dimensions="HeightAboveGround=Z"
#  echo "$p"
#done