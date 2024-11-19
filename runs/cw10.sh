#/bin/bash

# L2
CUDA_VISIBLE_DEVICES=$1 python main.py --prefix_name cw10 --seed 123 --continual_method l2 --save_path ./save/ --lamb 5000
CUDA_VISIBLE_DEVICES=$1 python main.py --prefix_name cw10 --seed 231 --continual_method l2 --save_path ./save/ --lamb 5000
CUDA_VISIBLE_DEVICES=$1 python main.py --prefix_name cw10 --seed 312 --continual_method l2 --save_path ./save/ --lamb 5000

#EWC
CUDA_VISIBLE_DEVICES=$1 python main.py --prefix_name cw10 --seed 123 --continual_method ewc --save_path ./save/ --lamb 5000
CUDA_VISIBLE_DEVICES=$1 python main.py --prefix_name cw10 --seed 231 --continual_method ewc --save_path ./save/ --lamb 5000
CUDA_VISIBLE_DEVICES=$1 python main.py --prefix_name cw10 --seed 312 --continual_method ewc --save_path ./save/ --lamb 5000

#MAS
CUDA_VISIBLE_DEVICES=$1 python main.py --prefix_name cw10 --seed 123 --continual_method mas --save_path ./save/ --lamb 10
CUDA_VISIBLE_DEVICES=$1 python main.py --prefix_name cw10 --seed 231 --continual_method mas --save_path ./save/ --lamb 10
CUDA_VISIBLE_DEVICES=$1 python main.py --prefix_name cw10 --seed 312 --continual_method mas --save_path ./save/ --lamb 10

#LwF
CUDA_VISIBLE_DEVICES=$1 python main.py --prefix_name cw10 --seed 123 --continual_method lwf --save_path ./save/ --lamb 1
CUDA_VISIBLE_DEVICES=$1 python main.py --prefix_name cw10 --seed 231 --continual_method lwf --save_path ./save/ --lamb 1
CUDA_VISIBLE_DEVICES=$1 python main.py --prefix_name cw10 --seed 312 --continual_method lwf --save_path ./save/ --lamb 1

#RWalk
CUDA_VISIBLE_DEVICES=$1 python main.py --prefix_name cw10 --seed 123 --continual_method r_walker --save_path ./save/ --lamb 1000
CUDA_VISIBLE_DEVICES=$1 python main.py --prefix_name cw10 --seed 231 --continual_method r_walker --save_path ./save/ --lamb 1000
CUDA_VISIBLE_DEVICES=$1 python main.py --prefix_name cw10 --seed 312 --continual_method r_walker --save_path ./save/ --lamb 1000

#VCL
CUDA_VISIBLE_DEVICES=$1 python main.py --prefix_name cw10 --seed 123 --continual_method vcl --save_path ./save/ --lamb 1
CUDA_VISIBLE_DEVICES=$1 python main.py --prefix_name cw10 --seed 231 --continual_method vcl --save_path ./save/ --lamb 1
CUDA_VISIBLE_DEVICES=$1 python main.py --prefix_name cw10 --seed 312 --continual_method vcl --save_path ./save/ --lamb 1

#Finetuning
CUDA_VISIBLE_DEVICES=$1 python main.py --prefix_name cw10 --seed 123 --save_path ./save/ 
CUDA_VISIBLE_DEVICES=$1 python main.py --prefix_name cw10 --seed 231 --save_path ./save/ 
CUDA_VISIBLE_DEVICES=$1 python main.py --prefix_name cw10 --seed 312 --save_path ./save/

#LoRA
CUDA_VISIBLE_DEVICES=$1 python main.py --prefix_name cw10 --seed 123 --continual_method lora --save_path ./save/ 
CUDA_VISIBLE_DEVICES=$1 python main.py --prefix_name cw10 --seed 231 --continual_method lora --save_path ./save/ 
CUDA_VISIBLE_DEVICES=$1 python main.py --prefix_name cw10 --seed 312 --continual_method lora --save_path ./save/ 

#PackNet
CUDA_VISIBLE_DEVICES=$1 python main.py --prefix_name cw10 --seed 123 --continual_method packnet --save_path ./save/ 
CUDA_VISIBLE_DEVICES=$1 python main.py --prefix_name cw10 --seed 231 --continual_method packnet --save_path ./save/ 
CUDA_VISIBLE_DEVICES=$1 python main.py --prefix_name cw10 --seed 312 --continual_method packnet --save_path ./save/ 

#Perfect Memory
CUDA_VISIBLE_DEVICES=$1 python main.py --prefix_name cw10 --seed 123 --continual_method perfect_memory --save_path ./save/ 
CUDA_VISIBLE_DEVICES=$1 python main.py --prefix_name cw10 --seed 231 --continual_method perfect_memory --save_path ./save/ 
CUDA_VISIBLE_DEVICES=$1 python main.py --prefix_name cw10 --seed 312 --continual_method perfect_memory --save_path ./save/ 

#A-GEM
CUDA_VISIBLE_DEVICES=$1 python main.py --prefix_name cw10 --seed 123 --continual_method agem --save_path ./save/ 
CUDA_VISIBLE_DEVICES=$1 python main.py --prefix_name cw10 --seed 231 --continual_method agem --save_path ./save/ 
CUDA_VISIBLE_DEVICES=$1 python main.py --prefix_name cw10 --seed 312 --continual_method agem --save_path ./save/ 

#Grow
CUDA_VISIBLE_DEVICES=$1 python main.py --prefix_name cw10 --seed 123 --continual_method grow --save_path ./save/ --recursion 
CUDA_VISIBLE_DEVICES=$1 python main.py --prefix_name cw10 --seed 231 --continual_method grow --save_path ./save/ --recursion 
CUDA_VISIBLE_DEVICES=$1 python main.py --prefix_name cw10 --seed 312 --continual_method grow --save_path ./save/ --recursion 

#Prune
CUDA_VISIBLE_DEVICES=$1 python main.py --prefix_name cw10 --seed 123 --continual_method prune --save_path ./save/ --recursion 
CUDA_VISIBLE_DEVICES=$1 python main.py --prefix_name cw10 --seed 231 --continual_method prune --save_path ./save/ --recursion 
CUDA_VISIBLE_DEVICES=$1 python main.py --prefix_name cw10 --seed 312 --continual_method prune --save_path ./save/ --recursion 


