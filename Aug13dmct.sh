for rollouts in 14 15 16 17 18 19 21 22 23 24 25 26 27 29 30 31 32 33 34 35 37 38 39 41 42 43 44 45 46 47 48 49 51 52 53 54 55 56 57 58 59
do
    CUDA_VISIBLE_DEVICES=0 nohup python -m torch.distributed.run --master_port 1034 --nproc_per_node 1 run_blocksworld.py --task mcts --model_name Vicuna --verbose False \
    --data data/blocksworld/step_4.json \
    --max_depth 4 \
    --name Aug13dmcts${rollouts} \
    --rollouts $rollouts \
    --model_path lmsys/vicuna-13b-v1.3 \
    --num_gpus 1
done