#!/bin/bash

echo -n "Enter a version [td3, mean_mb, sample_mb, maac_mb]: "
read VAR

if [ $VAR == "td3" ]
then
  echo "Running standard TD3 HRL"
  for ((i=0; i<3; i+=1))
  do
    python run_hrl.py "AntGather" \
        --evaluate \
        --total_steps 5000000 \
        --relative_goals \
        --use_huber \
        --seed $i
  done

elif [ $VAR == "mean_mb" ]
then
  echo "Running standard Model Based LLP (mean rollout) HRL"
  for ((i=0; i<3; i+=1))
  do
    python run_hrl.py "AntGather" \
        --evaluate \
        --total_steps 5000000 \
        --relative_goals \
        --use_huber \
        --multistep_llp \
        --max_rollout_using_model 5 \
        --seed $i
  done

elif [ $VAR == "sample_mb" ]
then
  echo "Running standard Model Based LLP (sample rollout) HRL"
  for ((i=0; i<3; i+=1))
  do
    python run_hrl.py "AntGather" \
        --evaluate \
        --total_steps 5000000 \
        --relative_goals \
        --use_huber \
        --multistep_llp \
        --max_rollout_using_model 5 \
        --use_sample_not_mean \
        --seed $i
  done

elif [ $VAR == "maac_mb" ]
then
  echo "Running standard Model Augmented LLP (mean rollout) HRL"
  for ((i=0; i<3; i+=1))
  do
    python run_hrl.py "AntGather" \
        --evaluate \
        --total_steps 5000000 \
        --relative_goals \
        --use_huber \
        --multistep_llp \
        --max_rollout_using_model 5 \
        --add_final_q_value \
        --seed $i
  done

else
  echo "Version must be [td3, mean_mb, sample_mb, maac_mb]"

fi