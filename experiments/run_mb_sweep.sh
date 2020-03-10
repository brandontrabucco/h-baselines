#!/bin/bash

echo -n "Enter a version [td3, no_teacher, maac_mb]: "
read VAR0

echo -n "Enter a roll out horizon: "
read VAR1

if [ $VAR0 == "td3" ]
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

elif [ $VAR0 == "no_teacher" ]
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
        --max_rollout_using_model $VAR1 \
        --max_rollout_when_training $VAR1 \
        --seed $i
  done

elif [ $VAR0 == "maac_mb" ]
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
        --max_rollout_using_model $VAR1 \
        --max_rollout_when_training $VAR1 \
        --add_final_q_value \
        --seed $i
  done

else
  echo "Version must be [td3, no_teacher, maac_mb]"

fi