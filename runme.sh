#!/bin/bash
# You need to modify this path to your downloaded dataset directory
DATASET_DIR='/vol/vssp/cvpnobackup/scratch_4weeks/qk00006/dcase2019/task5/dataset_root'

# You need to modify this path to your workspace to store features and models
WORKSPACE='/vol/vssp/msos/qk/workspaces/dcase2019_task5'

# Hyper-parameters
GPU_ID=1
MODEL_TYPE='Cnn_9layers_AvgPooling'
BATCH_SIZE=32

# Calculate feature
python utils/features.py calculate_feature_for_all_audio_files --dataset_dir=$DATASET_DIR --data_type='train' --workspace=$WORKSPACE
python utils/features.py calculate_feature_for_all_audio_files --dataset_dir=$DATASET_DIR --data_type='validate' --workspace=$WORKSPACE
python utils/features.py calculate_feature_for_all_audio_files --dataset_dir=$DATASET_DIR --data_type='evaluate' --workspace=$WORKSPACE

# Calculate scalar
python utils/features.py calculate_scalar --data_type='train' --workspace=$WORKSPACE


############ Train and validate on development dataset ############

# Train & inference
for TAXONOMY_LEVEL in 'fine' 'coarse'
do
  # Train
  CUDA_VISIBLE_DEVICES=$GPU_ID python pytorch/main.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --taxonomy_level=$TAXONOMY_LEVEL --model_type=$MODEL_TYPE --holdout_fold=1 --batch_size=$BATCH_SIZE --cuda

  # Inference and evaluate
  CUDA_VISIBLE_DEVICES=$GPU_ID python pytorch/main.py inference_validation --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --taxonomy_level=$TAXONOMY_LEVEL --model_type=$MODEL_TYPE --holdout_fold=1 --iteration=5000 --batch_size=$BATCH_SIZE --cuda
done

# Plot statistics
python utils/plot_results.py --workspace=$WORKSPACE --taxonomy_level=fine


############ Train on full development dataset and inference on evaluation dataset ############

# Train on full development dataset
for TAXONOMY_LEVEL in 'fine' 'coarse'
do
  # Train
  CUDA_VISIBLE_DEVICES=$GPU_ID python pytorch/main.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --taxonomy_level=$TAXONOMY_LEVEL --model_type=$MODEL_TYPE --holdout_fold='none' --batch_size=$BATCH_SIZE --cuda
  
  # Inference
  CUDA_VISIBLE_DEVICES=$GPU_ID python pytorch/main.py inference_evaluation --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --taxonomy_level=$TAXONOMY_LEVEL --model_type=$MODEL_TYPE --holdout_fold='none' --iteration=3000 --batch_size=$BATCH_SIZE --cuda
done

############ END ############
