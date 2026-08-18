import argparse
import os
import random
import glob
import torch
import numpy as np
from tqdm.auto import tqdm

# Adjust imports based on your exact file structure
from core.deep_cfr_new import DeepCFRAgent
from gym_env import PokerEnv
from submission.player import PlayerAgent
from agents.prob_agent import ProbabilityAgent
from agents.test_agents import RandomAgent, AllInAgent

import os
import glob
import random
import re

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
        print(int(match.group(1)))
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

def main():
    parser = argparse.ArgumentParser(description="Deep CFR Curriculum Trainer")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of CFR iterations to run")
    parser.add_argument("--traversals", type=int, default=200, help="Games to play per iteration")
    parser.add_argument("--log-dir", type=str, default="logs/phase1", help="Directory for logs")
    parser.add_argument("--save-dir", type=str, default="models_new/phase1", help="Directory to save checkpoints")
    
    # Checkpoint loading
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume training from")
    
    # Phase 2 & 3 Arguments
    parser.add_argument("--self-play", action="store_true", help="Train against the fixed checkpoint provided")
    parser.add_argument("--mixed", action="store_true", help="Train against a pool of checkpoints")
    parser.add_argument("--checkpoint-dir", type=str, default="models/opps", help="Directory containing opponent checkpoints")
    parser.add_argument("--model-prefix", type=str, default="t_", help="Prefix for checkpoint files")
    parser.add_argument("--refresh-interval", type=int, default=25, help="How often to swap the opponent in mixed mode")
    
    args = parser.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda")
    print(f"🚀 Initializing Deep CFR Training on {device}")
    
    env = PokerEnv()

    # --- Initialize Main Agent (Player 0) ---
    main_agent = DeepCFRAgent(player_id=0, memory_size=5000000, device=device)
    prob_agent = ProbabilityAgent(stream=False)
    all_in_agent = AllInAgent(stream=False)
    # Load checkpoint if resuming
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"📥 Resuming main agent from {args.checkpoint}")
        main_agent.load_model(args.checkpoint)

    # --- Initialize Opponent Agent (Player 1) ---
    opponent_agent = RandomAgent(stream=False)
    
    if args.self_play or args.mixed:
        print("🤖 Setting up Trained Opponent...")
        opponent_agent = PlayerAgent(stream=False)
        opponent_agent.agent.player_id = 1
        if args.self_play and args.checkpoint:
            print(f"⚔️ Phase 2: Fixed Opponent loaded from {args.checkpoint}")
            opponent_agent.agent.load_model(args.checkpoint)
            #opponent_agent.agent.load_model('./models/phase1/t_checkpoint_iteration_630.pt')

            
        elif args.mixed:
            opp_ckpt = get_latest_checkpoint(args.checkpoint_dir, args.model_prefix)
            if opp_ckpt:
                print(f"🌪️ Phase 3: Mixed Opponent initially loaded from {opp_ckpt}")
                opponent_agent.agent.load_model(opp_ckpt)
            else:
                print("⚠️ No checkpoints found for mixed pool. Defaulting to untrained opponent.")
        #opponent_agent.agent.strategy_net = torch.compile(opponent_agent.agent.strategy_net, mode="reduce-overhead")

    #opponent_agent = all_in_agent
    #main_agent.reset_adv()
    #main_agent.freeze_streams()
    # --- MASTER TRAINING LOOP ---
    start_iter = main_agent.iteration_count+1
    end_iter = start_iter + args.iterations

    for iteration in tqdm(range(start_iter, end_iter)):
        main_agent.iteration_count+=1

        # Mixed Mode: Swap out the opponent every refresh_interval
        if args.mixed and ((iteration-1) % args.refresh_interval == 0):
            opp_ckpt = get_latest_checkpoint(args.checkpoint_dir, args.model_prefix)
            opponent_agent = PlayerAgent(stream=False)
            opponent_agent.agent.player_id = 1
            
            opp_ckpt = get_latest_checkpoint(args.checkpoint_dir, args.model_prefix)
            print(f"🔄 Swapping opponent pool: Loading {opp_ckpt}")
            opponent_agent.agent.load_model(opp_ckpt) 
            #opponent_agent.agent.strategy_net = torch.compile(opponent_agent.agent.strategy_net, mode="reduce-overhead")

        # 1. Data Generation (Tree Traversals)
        for _ in tqdm(range(args.traversals)):
            env.reset(options={'small_blind_player': random.randint(0,1)})
            # Pass the opponent_agent (will be None in Phase 1, using Random)
            # randomly use probability agent for robustness - more explotability but less likely to be fucking stupid
            #if random.random()<.5:
                #main_agent.cfr_traverse(env, iteration, opponent_agent=all_in_agent, depth=0)
            #else:  
            main_agent.cfr_traverse(env, iteration, opponent_agent=opponent_agent, depth=0)
            
        # 2. Network Training
            # 🔴 UPGRADED: Larger batches and more epochs to handle the deep traversal data
        adv_loss = main_agent.train_advantage_network(batch_size=1024, epochs=10)
            
        
        print(f"Iter {iteration} | Adv Loss: {adv_loss:.4f} Buffer: {len(main_agent.advantage_memory)}")

        print(main_agent.metrics)
        # 3. Checkpoint Saving
        if iteration % 100 == 0:
            strat_loss = main_agent.train_strategy_network(batch_size=4096, epochs=150)
            print(f"Iter {iteration} | Strat Loss: {strat_loss:.4f}")
            save_path = os.path.join(args.save_dir, "checkpoint")
            main_agent.save_model(save_path)
            print(f"💾 Saved Checkpoint at Iteration {iteration}")
            print("Saving new opponent")
            save_path = os.path.join(args.checkpoint_dir, "t_checkpoint")
            main_agent.save_model(save_path)
        """
        if iteration % 100 == 0 and main_agent.advantage_memory.get_memory_stats()['size']>=4000000:
            # Explicitly define the correct path for your main model logs
            mem_base_path = os.path.join(args.save_dir, "checkpoint")
            
            main_agent.advantage_memory.save(f"{mem_base_path}_iteration_{iteration}_adv_mem.pt")
            torch.save(list(main_agent.strategy_memory), f"{mem_base_path}_iteration_{iteration}_str_mem.pt")
            print(f"📦 Saved massive replay buffers at Iteration {iteration}")
        """
if __name__ == "__main__":
    main()