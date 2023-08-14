CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --master_port 1034 --nproc_per_node 3 run_blocksworld.py --task mcts --model_name Vicuna --verbose False --data data/blocksworld/step_4.json --max_depth 4 --name run_5_Aug9 --rollouts 6 --model_path lmsys/vicuna-13b-v1.3 --num_gpus 4