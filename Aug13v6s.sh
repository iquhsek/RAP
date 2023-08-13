CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.run --master_port 29400 --nproc_per_node 1 run_vicuna.py --task mcts --model_name Vicuna --verbose False --data data/blocksworld/step_6.json --max_depth 6 --search_depth 1 --max_iter 6 --name Aug13v6s1 --rollouts 1 --sample_per_node 1 --model_path lmsys/vicuna-33b-v1.3 --num_gpus 4

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.run --master_port 29400 --nproc_per_node 1 run_vicuna.py --task mcts --model_name Vicuna --verbose False --data data/blocksworld/step_6.json --max_depth 6 --search_depth 1 --max_iter 6 --name Aug13v6s2 --rollouts 1 --sample_per_node 2 --model_path lmsys/vicuna-33b-v1.3 --num_gpus 4

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.run --master_port 29400 --nproc_per_node 1 run_vicuna.py --task mcts --model_name Vicuna --verbose False --data data/blocksworld/step_6.json --max_depth 6 --search_depth 1 --max_iter 6 --name Aug13v6s3 --rollouts 1 --sample_per_node 3 --model_path lmsys/vicuna-33b-v1.3 --num_gpus 4

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.run --master_port 29400 --nproc_per_node 1 run_vicuna.py --task mcts --model_name Vicuna --verbose False --data data/blocksworld/step_6.json --max_depth 6 --search_depth 2 --max_iter 3 --name Aug13v6s4 --rollouts 1 --sample_per_node 2 --model_path lmsys/vicuna-33b-v1.3 --num_gpus 4

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.run --master_port 29400 --nproc_per_node 1 run_vicuna.py --task mcts --model_name Vicuna --verbose False --data data/blocksworld/step_6.json --max_depth 6 --search_depth 1 --max_iter 6 --name Aug13v6s5 --rollouts 1 --sample_per_node 0 --model_path lmsys/vicuna-33b-v1.3 --num_gpus 4

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.run --master_port 29400 --nproc_per_node 1 run_vicuna.py --task mcts --model_name Vicuna --verbose False --data data/blocksworld/step_6.json --max_depth 6 --search_depth 3 --max_iter 2 --name Aug13v6s6 --rollouts 1 --sample_per_node 2 --model_path lmsys/vicuna-33b-v1.3 --num_gpus 4

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.run --master_port 29400 --nproc_per_node 1 run_vicuna.py --task mcts --model_name Vicuna --verbose False --data data/blocksworld/step_6.json --max_depth 6 --search_depth 2 --max_iter 3 --name Aug13v6s7 --rollouts 1 --sample_per_node 3 --model_path lmsys/vicuna-33b-v1.3 --num_gpus 4

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.run --master_port 29400 --nproc_per_node 1 run_vicuna.py --task mcts --model_name Vicuna --verbose False --data data/blocksworld/step_6.json --max_depth 6 --search_depth 2 --max_iter 3 --name Aug13v6s8 --rollouts 1 --sample_per_node 0 --model_path lmsys/vicuna-33b-v1.3 --num_gpus 4

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.run --master_port 29400 --nproc_per_node 1 run_vicuna.py --task mcts --model_name Vicuna --verbose False --data data/blocksworld/step_6.json --max_depth 6 --search_depth 3 --max_iter 2 --name Aug13v6s9 --rollouts 1 --sample_per_node 3 --model_path lmsys/vicuna-33b-v1.3 --num_gpus 4

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.run --master_port 29400 --nproc_per_node 1 run_vicuna.py --task mcts --model_name Vicuna --verbose False --data data/blocksworld/step_6.json --max_depth 6 --search_depth 6 --max_iter 1 --name Aug13v6s10 --rollouts 1 --sample_per_node 2 --model_path lmsys/vicuna-33b-v1.3 --num_gpus 4

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.run --master_port 29400 --nproc_per_node 1 run_vicuna.py --task mcts --model_name Vicuna --verbose False --data data/blocksworld/step_6.json --max_depth 6 --search_depth 3 --max_iter 2 --name Aug13v6s11 --rollouts 1 --sample_per_node 0 --model_path lmsys/vicuna-33b-v1.3 --num_gpus 4

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.run --master_port 29400 --nproc_per_node 1 run_vicuna.py --task mcts --model_name Vicuna --verbose False --data data/blocksworld/step_6.json --max_depth 6 --search_depth 6 --max_iter 1 --name Aug13v6s12 --rollouts 1 --sample_per_node 3 --model_path lmsys/vicuna-33b-v1.3 --num_gpus 4