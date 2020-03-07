#!/bin/bash

echo -n "Enter a version [td3, no_teacher, mean_mb, sample_mb, maac_mb]: "
read VAR

if [ $VAR == "td3" ]
then
  echo "Running TD3 HRL"
  for ((i=0; i<3; i+=1))
  do
    python run_hrl.py "AntGather" \
        --evaluate \
        --total_steps 5000000 \
        --relative_goals \
        --use_huber \
        --seed $i
  done

elif [ $VAR == "no_teacher" ]
then
  echo "Running Model Based LLP (no teacher forcing) HRL"
  for ((i=0; i<3; i+=1))
  do
    python run_hrl.py "AntGather" \
        --evaluate \
        --total_steps 5000000 \
        --relative_goals \
        --use_huber \
        --multistep_llp \
        --max_rollout_using_model 5 \
        --max_rollout_when_training 5 \
        --seed $i
  done

elif [ $VAR == "mean_mb" ]
then
  echo "Running Model Based LLP (mean rollout) HRL"
  for ((i=0; i<3; i+=1))
  do
    python run_hrl.py "AntGather" \
        --evaluate \
        --total_steps 5000000 \
        --relative_goals \
        --use_huber \
        --multistep_llp \
        --max_rollout_using_model 5 \
        --max_rollout_when_training 1 \
        --seed $i
  done

elif [ $VAR == "sample_mb" ]
then
  echo "Running Model Based LLP (sample rollout) HRL"
  for ((i=0; i<3; i+=1))
  do
    python run_hrl.py "AntGather" \
        --evaluate \
        --total_steps 5000000 \
        --relative_goals \
        --use_huber \
        --multistep_llp \
        --max_rollout_using_model 5 \
        --max_rollout_when_training 1 \
        --use_sample_not_mean \
        --seed $i
  done

elif [ $VAR == "maac_mb" ]
then
  echo "Running Model Augmented LLP (mean rollout) HRL"

  for ((i=0; i<3; i+=1))
  do
    python run_hrl.py "AntGather" \
        --evaluate \
        --total_steps 5000000 \
        --relative_goals \
        --use_huber \
        --multistep_llp \
        --max_rollout_using_model 1 \
        --max_rollout_when_training 1 \
        --add_final_q_value \
        --seed $i
  done

  for ((i=0; i<3; i+=1))
  do
    python run_hrl.py "AntGather" \
        --evaluate \
        --total_steps 5000000 \
        --relative_goals \
        --use_huber \
        --multistep_llp \
        --max_rollout_using_model 2 \
        --max_rollout_when_training 1 \
        --add_final_q_value \
        --seed $i
  done

  for ((i=0; i<3; i+=1))
  do
    python run_hrl.py "AntGather" \
        --evaluate \
        --total_steps 5000000 \
        --relative_goals \
        --use_huber \
        --multistep_llp \
        --max_rollout_using_model 5 \
        --max_rollout_when_training 1 \
        --add_final_q_value \
        --seed $i
  done

else
  echo "Version must be [td3, mean_mb, sample_mb, maac_mb]"

fi