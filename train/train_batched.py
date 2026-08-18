import argparse
import os
import random
import glob
import torch
import numpy as np
from tqdm.auto import tqdm
import concurrent.futures
import multiprocessing as mp
# Adjust imports based on your exact file structure
from core.deep_cfr_new import DeepCFRAgent
from gym_env import PokerEnv
from submission.player import PlayerAgent
from agents.prob_agent import ProbabilityAgent
from agents.test_agents import RandomAgent, AllInAgent
from parallelism_training.train_parallel import parallel_traversal_chunk

import os
import glob
import random
import re
import ray
import copy

def get_latest_checkpoint(checkpoint_dir, prefix="t_"):
    """
    80% chance: Returns the absolute latest checkpoint.
    20% chance: Returns a random checkpoint from the historical pool.
    """
    search_pattern = os.path.join(checkpoint_dir, f"{prefix}*.pt")
    checkpoints = glob.glob(search_pattern)
    
    if not checkpoints:
        return None
        
    # 1. Parse and sort checkpoints by iteration number (descending)
    # We use a lambda to extract the integer from the filename for sorting
    def get_iter(path):
        match = re.search(r'_(\d+)\.pt$', path)
        return int(match.group(1)) if match else -1

    # Sort so checkpoints[0] is the highest iteration (the latest)
    checkpoints.sort(key=get_iter, reverse=True)
    
    # 2. Roll the dice (80/20 split)
    roll = random.random()
    
    if roll < 0.80:
        # --- 80% LATEST ---
        # Pick the very first one in our sorted list
        print(f'Getting Latest Checkpoint: {checkpoints[0]}')
        chosen_checkpoint = checkpoints[0]
    else:
        # --- 20% HISTORICAL POOL ---
        # Pick any checkpoint from the list (including the latest)
        chosen_checkpoint = random.choice(checkpoints)
    
    return chosen_checkpoint
def env_maker():
    """Returns a fresh, isolated instance of the Poker environment."""
    # 1. Initialize the base environment
    env = PokerEnv()    
    return env
def main():
    parser = argparse.ArgumentParser(description="Deep CFR Curriculum Trainer")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of CFR iterations to run")
    parser.add_argument("--traversals", type=int, default=200, help="Games to play per iteration")
    parser.add_argument("--log-dir", type=str, default="logs/phase1", help="Directory for logs")
    parser.add_argument("--save-dir", type=str, default="models_final/phase1", help="Directory to save checkpoints")
    
    # Checkpoint loading
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume training from")
    
    # Phase 2 & 3 Arguments
    parser.add_argument("--self-play", action="store_true", help="Train against the fixed checkpoint provided")
    parser.add_argument("--mixed", action="store_true", help="Train against a pool of checkpoints")
    parser.add_argument("--checkpoint-dir", type=str, default="models_final/opps", help="Directory containing opponent checkpoints")
    parser.add_argument("--model-prefix", type=str, default="t_", help="Prefix for checkpoint files")
    parser.add_argument("--refresh-interval", type=int, default=20, help="How often to swap the opponent in mixed mode")
    
    args = parser.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda")
    print(f"🚀 Initializing Deep CFR Training on {device}")
    
    env = PokerEnv()

    # --- Initialize Main Agent (Player 0) ---
    main_agent = DeepCFRAgent(player_id=0, memory_size=3000000, device=device)
    # Load checkpoint if resuming
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"📥 Resuming main agent from {args.checkpoint}")
        main_agent.load_model(args.checkpoint)
    #opponent_agent = all_in_agent
    #main_agent.reset_adv()
    #main_agent.freeze_streams()
    # --- MASTER TRAINING LOOP ---
    start_iter = main_agent.iteration_count+1
    end_iter = start_iter + args.iterations

    for iteration in tqdm(range(start_iter, end_iter)):
        main_agent.iteration_count+=1
            #opponent_agent.agent.strategy_net = torch.compile(opponent_agent.agent.strategy_net, mode="reduce-overhead")

        # 1. Data Generation (Tree Traversals)
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        num_workers = 12
        if num_workers < 1: num_workers = 1

        traversals_per_worker = args.traversals // num_workers
        remainder = args.traversals % num_workers

        # Extract weights to CPU
        main_weights = {k: v.cpu() for k, v in main_agent.advantage_net.state_dict().items()}

        # 2. Put weights in Ray's zero-copy shared memory
        # This completely eliminates the overhead of sending weights to workers
        main_weights_id = ray.put(main_weights)
        print(f"Distributing {args.traversals} traversals across {num_workers} Ray workers...")
        # 3. Launch the remote tasks
        futures = []
        for i in range(num_workers):
            chunk = traversals_per_worker + (remainder if i == num_workers - 1 else 0)
            
            # Notice the .remote() syntax! 
            # We pass the Object IDs (main_weights_id) instead of the actual dictionaries
            future = parallel_traversal_chunk.remote(
                env_maker, 
                main_weights_id, 
                chunk, 
                iteration
            )
            futures.append(future)

        # 4. Gather the results using ray.get()
        # ray.get() blocks until all tasks are finished and returns the list of results
        # 4. Gather the results using ray.get()
        # ray.get() blocks until all tasks are finished and returns the list of results
        unfinished = futures
        while unfinished:
            # ray.wait returns the first worker that finishes
            done, unfinished = ray.wait(unfinished, num_returns=1)
            
            # Grab the result of just that one worker
            result = ray.get(done[0]) 
            adv_batch, strat_batch = result
            
            # Merge immediately and then the 'result' variable is cleared from RAM
            if adv_batch is not None:
                # Unpack the arrays
                main_agent.advantage_memory.add_batch(
                                    adv_batch["s"], adv_batch["a"], adv_batch["r"], adv_batch["i"], adv_batch["p"]
                                )
            
            if strat_batch is not None:
                main_agent.strategy_memory.add_batch(
                                    strat_batch["s"], strat_batch["p"], strat_batch["i"], strat_batch['rp']
                                )
            del result
            ray._private.internal_api.free(done)
        print(f"Ray generation complete! Advantage Buffer size: {len(main_agent.advantage_memory)}")
            
        # 2. Network Training
            # 🔴 UPGRADED: Larger batches and more epochs to handle the deep traversal data
        adv_loss = main_agent.train_advantage_network(batch_size=2048, epochs=7)
            
        
        print(f"Iter {iteration} | Adv Loss: {adv_loss:.4f} Buffer: {len(main_agent.advantage_memory)}")

        print(main_agent.metrics)
        # 3. Checkpoint Saving
        if iteration % 5 == 0:
            strat_loss = main_agent.train_strategy_network(batch_size=2048, epochs=15)
            print(f"Iter {iteration} | Strat Loss: {strat_loss:.4f}")
            #main_agent.save_model(save_path)
        if iteration % 40 == 0:
            print(f"💾 Saved Checkpoint at Iteration {iteration}")
            print("Saving new opponent")
            save_path = os.path.join(args.checkpoint_dir, "t_checkpoint")
            main_agent.save_model(save_path)
        if iteration % 50 == 0 and len(main_agent.strategy_memory)>=1000000:
            # Explicitly define the correct path for your main model logs
            mem_base_path = os.path.join(args.save_dir, "checkpoint")
            
            main_agent.strategy_memory.save_compressed(f"{mem_base_path}_iteration_{iteration}_str_mem.pt")
            print(f"📦 Saved massive replay buffers at Iteration {iteration}")
            

if __name__ == "__main__":
    main()