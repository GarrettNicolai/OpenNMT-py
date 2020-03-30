#!/bin/bash

#$ -N trainLow
#$ -cwd
#$ -o trainLow.dat 
#$ -j yes
#$ -pe smp 1
#$ -t 1-60 -tc 60 
#$ -V
#$ -S /bin/bash
#$ -l 'gpu=1,mem_free=5G,ram_free=5G'


filename='lowGPU.expts'

runningJobs=$(jobs | wc -l | xargs)

line=""
count=0

source ~/RNN/bin/activate

while read line; do
	count=$((count+1))

	if [[ "$count" -eq "$SGE_TASK_ID" ]]; then	
		echo $line
		break
	fi

done < $filename

	parts=(${line//	/ })


	mkdir -p "Models/GPU/"${parts[0]}"/"${parts[1]}
	mkdir -p "logs/GPU/"${parts[0]}"/"
	mkdir -p "Results/dev/"${parts[0]}"/low"
	echo ${parts[0]}"-"${parts[1]}"-"${parts[2]}"-"${parts[3]}
	python3 train.py -optim adam -gpu_ranks 0 -data "morphData/onmtdata/"${parts[0]}"-"${parts[1]} -save_model "Models/GPU/"${parts[0]}"/"${parts[1]}"/"${parts[3]}"_model"${parts[2]} --teacher_forcing ${parts[3]} --seed ${parts[2]} --valid_steps 500 --report_every 500 --save_checkpoint_steps 500 --train_steps 10000 --rnn_size 50 --copy_attn --copy_attn_type general --early_stopping_criteria accuracy &> "logs/GPU/"${parts[0]}"/"${parts[3]}"-"${parts[1]}"-"${parts[2]}".log" &

wait


	python getScores.py "logs/GPU/"${parts[0]}"/"${parts[3]}"-"${parts[1]}"-"${parts[2]}".log"
