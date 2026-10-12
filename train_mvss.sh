OUTPUT_DIR="results/rita_mvss"
DATA_PATH="/mnt/data0/public_datasets/IML/CASIA2.0"
mkdir -p ${OUTPUT_DIR}/logs
mkdir -p ${OUTPUT_DIR}/images
mkdir -p ${OUTPUT_DIR}/ckpts
mkdir -p ${OUTPUT_DIR}/runs
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train.py --path $OUTPUT_DIR --data_path $DATA_PATH --epoch 50 --batch_size 8 \
> $OUTPUT_DIR/logs/log_stdout.log 2> $OUTPUT_DIR/logs/log_stderr.log