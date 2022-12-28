#!/bin/bash
# set input data path
#LAS_files_path='/mnt/Lidar_O/DeepLIDAR/VolVegetacioRibera_ClassTorres-Linies/LAS'
LAS_files_path='/mnt/Lidar_M/DEMO_Productes_LIDARCAT3/LAS_def'
# set output data path
LAS_proc_path='/dades/LIDAR/towers_detection/LAS_data_windows'

# set variables
w_size=40
dataset_name='test_inference_CAT3'
n_points=2048
max_z=100.0
noise_labels=(135 106) # use spaces to separate elements

## 1.  Load LAS data and split into point clouds of smaller size of width=w_size size=[w_size, w_size]
##    Load LAS files from in_data_path, get its features (x, y, z, classification, intensity, red, green, blue, nir)
##    Noise labels are assigned to class 30, which is removed in next step
##    Each LAS point cloud is stored as an array of features
##    if dataset == 'BDN', NIR & RGB values are set to zero
##    NIR is stored in a different file than LAS as PDAL library does not enable NIR
#python3 /home/m.caros/work/objectDetection/data_proc/1_get_windows_inference.py --w_size $w_size --dataset_name $dataset_name --LAS_files_path $LAS_files_path --out_path $LAS_proc_path --noise_labels $noise_labels

## 2.  Use PDAL library to get height above ground (HAG)
files_path="$LAS_proc_path/$dataset_name/files_$w_size*/*.las"
echo "Normalizing heights... "  # Transformation of Height Above Sea (HAS) into Height Above Ground (HAG)...
for p in $files_path; do
  pdal translate "$p" "$p" hag_nn --writers.las.extra_dims="HeightAboveGround=float32"
  echo "$p"
done

#3. Remove ground, noise and outliers and apply constrained sampling
#    1- Remove ground and noise labeled points (by Terrasolid) to reduce noise and number of points
#    2- Add NIR from dictionary
#    3- Remove outliers defined as points > max_z and points < 0
#    4- Normalize data
#    5- Remove terrain points (up to n_points points in point cloud)
#    6- Add constrained sampling flag at column 10
in_path="$LAS_proc_path/$dataset_name/files_$w_size*"
out_path="/dades/LIDAR/towers_detection/datasets/test_inference_$w_size""x$w_size"
python3 /home/m.caros/work/objectDetection/data_proc/2_preprocessing_inference.py --out_path $out_path --in_path $in_path --dataset $dataset_name --n_points $n_points --max_z $max_z

# 4. Classification inference
# Detect towers within point clouds and generate a text file with point clouds to segment

# 5. Segmentation inference

# 6. Generate a LAS file per block with points labeled as tower


