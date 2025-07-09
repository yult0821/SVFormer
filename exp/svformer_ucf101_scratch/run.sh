work_path=$(dirname $0)
PYTHONPATH=$PYTHONPATH:./slowfast \
python tools/run_net.py \
  --cfg $work_path/config.yaml \
  DATA.PATH_TO_DATA_DIR ./data_list/ucf101/split1 \
  DATA.PATH_PREFIX /userhome/Datasets/ucf101/videos/ \
  TRAIN.EVAL_PERIOD 5 \
  TRAIN.CHECKPOINT_PERIOD 50 \
  TRAIN.BATCH_SIZE 64 \
  NUM_GPUS 4 \
  UNIFORMER.PRETRAIN_NAME '' \
  UNIFORMER.DROP_DEPTH_RATE 0. \
  DATA.NUM_FRAMES 16 \
  SOLVER.MAX_EPOCH 600 \
  SOLVER.BASE_LR 6e-3 \
  SOLVER.COSINE_END_LR 1e-6 \
  SOLVER.BASE_LR_SCALE_NUM_SHARDS False \
  SOLVER.WARMUP_EPOCHS 30. \
  DATA.TEST_CROP_SIZE 224 \
  TEST.BATCH_SIZE 128 \
  TEST.NUM_ENSEMBLE_VIEWS 5 \
  TEST.NUM_SPATIAL_CROPS 3 \
  RNG_SEED 6666 \
  OUTPUT_DIR $work_path

# UNIFORMER.T 16 \

# bash exp/svformer_ucf101_scratch/run.sh 2>&1 | tee -a exp/svformer_ucf101_scratch/run.log; bash exp/svformer_ucf101_scratch/test.sh 2>&1 | tee -a exp/svformer_ucf101_scratch/test.log;


# TRAIN.BATCH_SIZE 12, NUM_GPUS = 1 (bs_per_gpu = 12)

# TRAIN.BATCH_SIZE 16, NUM_GPUS = 1 (bs_per_gpu = 16)
# Mon Mar  4 10:03:20 2024       
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 450.119.04   Driver Version: 450.119.04   CUDA Version: 11.5     |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |                               |                      |               MIG M. |
# |===============================+======================+======================|
# |   0  Tesla V100-SXM3...  On   | 00000000:57:00.0 Off |                    0 |
# | N/A   49C    P0   115W / 350W |  31380MiB / 32510MiB |    100%      Default |
# |                               |                      |                  N/A |
# +-------------------------------+----------------------+----------------------+
                                                                               
# +-----------------------------------------------------------------------------+
# | Processes:                                                                  |
# |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
# |        ID   ID                                                   Usage      |
# |=============================================================================|
# +-----------------------------------------------------------------------------+

