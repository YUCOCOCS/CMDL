#!/bin/sh                     
#BSUB -q normal             
#BSUB -o %J.out    
#BSUB -e %J.err       
#BSUB -n 1         
#BSUB -J JOBNAME	
#BSUB  -R span[ptile=1]   
#BSUB -m "node09" 
#BSUB  -gpu  num=4
export NCCL_P2P_LEVEL=NVL
export NCCL_DEBUG=INFO
PYTHONIOENCODING=utf-8  
CUDA_VISIBLE_DEVICES=2,3 nohup python -m torch.distributed.launch  --nproc_per_node 2 --master_port  21102 train.py >voc_psp50.log &


