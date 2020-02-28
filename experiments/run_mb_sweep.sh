#!/bin/bash

echo -n "Enter a version [td3, mean_mb, sample_mb, maac_mb]: "
read VAR

if [ $VAR == "td3" ]
then
  echo "Running standard TD3 HRL"
  python run_hrl.py "AntGather" \
      --evaluate \
      --total_steps 5000000 \
      --relative_goals \
      --use_huber

elif [ $VAR == "mean_mb" ]
then
  echo "Running standard Model Based LLP (mean rollout) HRL"
  python run_hrl.py "AntGather" \
      --evaluate \
      --total_steps 5000000 \
      --relative_goals \
      --use_huber \
      --multistep_llp \
      --max_rollout_using_model 5

elif [ $VAR == "sample_mb" ]
then
  echo "Running standard Model Based LLP (sample rollout) HRL"
  python run_hrl.py "AntGather" \
      --evaluate \
      --total_steps 5000000 \
      --relative_goals \
      --use_huber \
      --multistep_llp \
      --max_rollout_using_model 5 \
      --use_sample_not_mean

elif [ $VAR == "maac_mb" ]
then
  echo "Running standard Model Augmented LLP (mean rollout) HRL"
  python run_hrl.py "AntGather" \
      --evaluate \
      --total_steps 5000000 \
      --relative_goals \
      --use_huber \
      --multistep_llp \
      --max_rollout_using_model 5 \
      --add_final_q_value

else
  echo "Version must be [td3, mean_mb, sample_mb, maac_mb]"

fi