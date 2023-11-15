CONFIG=$1
GPU=${2:-0}
SEED=${3:-0}

export CUDA_VISIBLE_DEVICES=$GPU
export OMP_NUM_THREADS=4

python TextEE/train.py -c $CONFIG --seed $SEED
