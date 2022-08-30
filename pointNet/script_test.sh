
python pointNet/test_classification.py /dades/LIDAR/towers_detection/datasets classification pointNet/results/ --weighing_method EFS --number_of_points 2048 --number_of_workers 0 --model_checkpoint /home/m.caros/work/objectDetection/pointNet/checkpoints/checkpoint_05-11-12:540.999.pth

python pointNet/test_segmen.py /dades/LIDAR/towers_detection/datasets pointNet/results/ --number_of_points 2048 --number_of_workers 0 --model_checkpoint /home/m.caros/work/objectDetection/pointNet/checkpoints/checkpoint_05-11-19:22_seg.pth


python pointNet/test_RNN.py classification  /dades/LIDAR/towers_detection/datasets/pc_towers_40x40/sampled_2048  --path_list_files pointNet/data/train_test_files/RGBN

python pointNet/test_RNN_segmentation.py  /dades/LIDAR/towers_detection/datasets/pc_towers_40x40/sampled_2048  --path_list_files pointNet/data/train_test_files/RGBN