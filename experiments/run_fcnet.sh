#!/bin/bash

for ((i=0;i<10;i+=1))
do
	python fcnet_baseline.py "AntMaze" --seed $i --steps 10000000
	python fcnet_baseline.py "AntPush" --seed $i --steps 10000000
	python fcnet_baseline.py "AntFall" --seed $i --steps 10000000
done
