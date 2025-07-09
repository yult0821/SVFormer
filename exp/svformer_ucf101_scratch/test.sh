work_path=$(dirname $0)
PYTHONPATH=$PYTHONPATH:./slowfast \
python tools/run_net.py \
  --cfg $work_path/config.yaml \
  DATA.PATH_TO_DATA_DIR ./data_list/ucf101/split1 \
  DATA.PATH_PREFIX /home/yult/datasets/ucf101/videos/ \
  TRAIN.EVAL_PERIOD 5 \
  TRAIN.CHECKPOINT_PERIOD 50 \
  TRAIN.BATCH_SIZE 6 \
  NUM_GPUS 1 \
  UNIFORMER.PRETRAIN_NAME '' \
  UNIFORMER.DROP_DEPTH_RATE 0. \
  DATA.NUM_FRAMES 16 \
  SOLVER.MAX_EPOCH 600 \
  SOLVER.BASE_LR 6e-3 \
  SOLVER.COSINE_END_LR 1e-6 \
  SOLVER.BASE_LR_SCALE_NUM_SHARDS False \
  SOLVER.WARMUP_EPOCHS 30. \
  DATA.TEST_CROP_SIZE 224 \
  TEST.BATCH_SIZE 22 \
  TEST.NUM_ENSEMBLE_VIEWS 5 \
  TEST.NUM_SPATIAL_CROPS 3 \
  RNG_SEED 6666 \
  TRAIN.ENABLE False \
  TEST.CHECKPOINT_FILE_PATH $work_path/checkpoints/best_Valtop1Err_checkpoint.pyth \
  OUTPUT_DIR $work_path

# UNIFORMER.T 16 \
# $work_path/checkpoints/best_Valtop1Err_checkpoint.pyth \
# exp/svformer_ucf101_scratch/checkpoints/checkpoint_epoch_00600.pyth

# bash exp/svformer_ucf101_scratch/test.sh 2>&1 | tee -a exp/svformer_ucf101_scratch/test.log;

# bash exp/svformer_ucf101_scratch/test.sh 2>&1 | tee -a exp/svformer_ucf101_scratch/test_robustness.log;


#############################################################################################################
# TEST.NUM_ENSEMBLE_VIEWS 1; TEST.NUM_SPATIAL_CROPS 1
#############################################################################################################

# work_path=$(dirname $0)
# PYTHONPATH=$PYTHONPATH:./slowfast \
# python tools/run_net.py \
#   --cfg $work_path/test.yaml \
#   DATA.PATH_TO_DATA_DIR ./data_list/ucf101/split1 \
#   DATA.PATH_PREFIX /userhome/Datasets/ucf101/videos/ \
#   TRAIN.EVAL_PERIOD 5 \
#   TRAIN.CHECKPOINT_PERIOD 50 \
#   TRAIN.BATCH_SIZE 64 \
#   NUM_GPUS 1 \
#   UNIFORMER.PRETRAIN_NAME '' \
#   UNIFORMER.DROP_DEPTH_RATE 0. \
#   DATA.NUM_FRAMES 16 \
#   SOLVER.MAX_EPOCH 600 \
#   SOLVER.BASE_LR 6e-3 \
#   SOLVER.COSINE_END_LR 1e-6 \
#   SOLVER.BASE_LR_SCALE_NUM_SHARDS False \
#   SOLVER.WARMUP_EPOCHS 30. \
#   DATA.TEST_CROP_SIZE 224 \
#   TEST.BATCH_SIZE 16 \
#   TEST.NUM_ENSEMBLE_VIEWS 1 \
#   TEST.NUM_SPATIAL_CROPS 1 \
#   RNG_SEED 6666 \
#   TRAIN.ENABLE False \
#   TEST.CHECKPOINT_FILE_PATH $work_path/checkpoints/best_Valtop1Err_checkpoint.pyth \
#   OUTPUT_DIR $work_path
  
# bash exp/svformer_ucf101_scratch/test.sh 2>&1 | tee -a exp/svformer_ucf101_scratch/test1x1.log;  
