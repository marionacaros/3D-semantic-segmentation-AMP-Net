python pointNet/baseline/train_segmentation.py  /dades/LIDAR/towers_detection/datasets/pc_towers_40x40_10p/normalized_2048  --path_list_files /home/m.caros/work/objectDetection/train_test_files/RGBN_x10_40x40 --batch_size 32 --epochs 100 --learning_rate 0.001  --number_of_points 2048 --number_of_workers 8
python pointNet/train_RNN.py classification  /dades/LIDAR/towers_detection/datasets/pc_towers_40x40/sampled_2048  --path_list_files pointNet/data/train_test_files/RGBN --batch_size 16 --epochs 41 --learning_rate 0.001 --weighing_method EFS --beta 0.9999 --number_of_points 2048 --number_of_workers 4
python pointNet/rnn/train_pointNetGRU.py segmentation  /dades/LIDAR/towers_detection/datasets/new_kmeans  --path_list_files train_test_files/RGBN --batch_size 64 --epochs 80 --learning_rate 0.001 --number_of_points 2048 --number_of_workers 8
python pointNet/baseline/train_classification.py /dades/LIDAR/towers_detection/datasets/pc_towers_40x40_10p/normalized_2048 --path_list_files /home/m.caros/work/objectDetection/train_test_files/RGBN_x10_40x40  --batch_size 32  --epochs 70 --learning_rate 0.001 --weighing_method EFS --beta 0.999  --number_of_points 2048 --number_of_workers 8 --c_sample True