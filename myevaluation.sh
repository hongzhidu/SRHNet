CUDA_VISIBLE_DEVICES=0 python evaluation.py --crop_height=576 \
                  --crop_width=960 \
                  --max_disp=192 \
                  --data_path='/media/ubuntu/data/datasets/' \
                  --test_list='./lists/sceneflow_test.list' \
                  --save_path='./result/' \
                  --resume='.checkpoint/sceneflow_epoch_10.pth' \
                  --threshold=1.0 
# 2>&1 |tee logs/log_evaluation.txt
exit

