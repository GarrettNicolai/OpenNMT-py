#!/bin/bash

#$ -N trainMedGPU
#$ -cwd
#$ -o trainMedGPU.dat 
#$ -j yes
#$ -pe smp 1
#$ -t 1-1 -tc 1 
#$ -V
#$ -S /bin/bash
#$ -l 'gpu=1,mem_free=5G,ram_free=5G'


filename='medGPU.expts'

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
	mkdir -p "Results/dev/"${parts[0]}"/medium/"

	echo ${parts[0]}"-"${parts[1]}"-"${parts[2]}"-"${parts[3]}
	os.environ["CUDA_VISIBLE_DEVICES"] = 0 #'/home/gkumar/scripts/free-gpu' 
	python3 train.py --optim adam --gpu_ranks 0 -data "morphData/onmtdata/"${parts[0]}"-"${parts[1]} -save_model "Models/GPU/"${parts[0]}"/"${parts[1]}"/"${parts[3]}"_model_"${parts[2]} --teacher_forcing ${parts[3]} --seed ${parts[2]} --valid_steps 5000 --report_every 5000 --save_checkpoint_steps 5000 --train_steps 50000 --rnn_size 200 --copy_attn --copy_attn_type general &> "logs/GPU/"${parts[0]}"/"${parts[3]}"-"${parts[1]}"-"${parts[2]}".log" &

wait

	python getScores.py "logs/GPU/"${parts[0]}"/"${parts[3]}"-"${parts[1]}"-"${parts[2]}".log"

