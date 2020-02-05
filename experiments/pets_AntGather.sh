#!/bin/bash

for ((i=0;i<3;i+=1))
do
  # running goal-conditioned experiments without any augmentations
	python experiments/run_hrl.py "AntGather" --relative_goals --use_huber --multistep_llp --seed $i &
done
