#!/bin/bash

for ((i=0;i<10;i+=1))
do
  # running fcnet experiments
	python fcnet_baseline.py "AntMaze" --seed $i --steps 10000000
	python fcnet_baseline.py "AntPush" --seed $i --steps 10000000
	python fcnet_baseline.py "AntFall" --seed $i --steps 10000000

  # running goal-directed experiments without any augmentations
	python hiro_baseline.py "AntMaze" --evaluate --total_steps 10000000 --noise 1 --relative_goals --use_huber --seed $i
done
