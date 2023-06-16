#!/usr/bin/env bash

MODEL_NAME=$1
CHECKPOINT=$2
GPUS=$3

CONFIG_DIR="configs/saod/test/"$MODEL_NAME/

# Check the dataset configuration
if [[ $CONFIG_DIR == *"coco"* ]]
then
  CONFIGS=($MODEL_NAME".py" $MODEL_NAME"_pseudoid.py" $MODEL_NAME"_pseudoood.py" $MODEL_NAME"_obj45k.py" $MODEL_NAME"_sinobj110kood.py")
else
  CONFIGS=($MODEL_NAME".py" $MODEL_NAME"_pseudoid.py" $MODEL_NAME"_pseudoood.py" $MODEL_NAME"_bdd45k.py" $MODEL_NAME"_sinobj110kood.py")
fi

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

for i in "${!CONFIGS[@]}"; do
  PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
	python -m torch.distributed.launch \
    		--nnodes=$NNODES \
    		--node_rank=$NODE_RANK \
	    	--master_addr=$MASTER_ADDR \
	    	--nproc_per_node=$GPUS \
	    	--master_port=$PORT \
	    	$(dirname "$0")/analysis_tools/saod_inference.py \
	    	"$CONFIG_DIR${CONFIGS[i]}"\
	    	"$CHECKPOINT" \
	    	--eval bbox \
	    	--out "detections/${CONFIGS[i]:0:-3}.pkl" \
	    	--launcher pytorch \
	    	${@:4} \
	    	>> "results/${CONFIGS[i]:0:-3}.txt"
	done