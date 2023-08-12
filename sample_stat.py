import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default='mcts-run_1_Aug2_search_depth_3_alpha_01_rollouts_10')
    args = parser.parse_args()
    data_path = os.path.join('logs', args.run_name, 'sample')
    sample_files = os.listdir(data_path)
    total_sample = 0
    total_success = 0
    for file_name in sample_files:
        with open(os.path.join(data_path, file_name), 'r') as f:
            line = f.readline()
            num_success, num_sample = line.split(',')
            total_success += int(num_success)
            total_sample += int(num_sample)
    print(total_success, total_sample)