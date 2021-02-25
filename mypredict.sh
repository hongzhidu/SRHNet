
python predict.py --crop_height=384 \
                  --crop_width=1248 \
                  --max_disp=192 \
                  --data_path='/media/ubuntu/data/data_stereo_flow/testing/' \
                  --test_list='./lists/kitti2012_test.list' \
                  --save_path='./result/' \
                  --kitti=1\
                  --resume='./checkpoint/finetune2_kitti_epoch_8.pth'
exit


