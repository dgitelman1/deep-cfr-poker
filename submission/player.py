import os
import torch
from agents.agent import Agent
from gym_env import PokerEnv

# Import your Deep CFR components
from core.deep_cfr_new import DeepCFRAgent

class PlayerAgent(Agent):
    def __init__(self, stream: bool = True, checkpoint_path: str = None):
        super().__init__(stream)
        self.action_types = PokerEnv.ActionType
        
        # 1. Setup PyTorch Device
        self.device = torch.device("cuda")
        self.logger.info(f"Initializing Deep CFR PlayerAgent on {self.device}")
        
        # 3. Create the Deep CFR Agent
        # We set memory_size tiny since we are only doing inference, no training
        self.agent = DeepCFRAgent(
            player_id=0,
            memory_size=10, 
            device=self.device
        )
        self.pot = 0
        self.num_iterations = 0
        
        # 4. Load the Trained Weights
        # Point this default path to your best checkpoint from Phase 3
        if checkpoint_path is None:
            # Assuming you run the tournament script from the repo root
            #checkpoint_path = os.path.join("models", "phase1", "checkpoint_iteration_10000.pt")
            #1310
            #
            checkpoint_path = os.path.join("models_final", "opps", "t_checkpoint_iteration_40.pt")
            
        if os.path.exists(checkpoint_path):
            self.logger.info(f"Loading Strategy Weights from {checkpoint_path}")
            self.agent.load_model(checkpoint_path)
        else:
            self.logger.warning(f"CRITICAL: No checkpoint found at {checkpoint_path}. Playing randomly!")
            
        # 5. Lock network into evaluation mode (disables gradients/dropout)
        self.agent.strategy_net.eval()

    def __name__(self):
        return "DeepCFR_Bot"

    def act(self, observation, reward, terminated, truncated, info):
        """
        Takes the Gym observation dict, asks the Deep CFR Strategy Network,
        and returns the formatted Gym action tuple.
        """
        # In actual tournament play, you don't need to learn anymore. 
        # You just use the distilled Strategy Network to output probabilities.
        num_iterations_left = 1000-self.num_iterations

        if self.pot - (num_iterations_left//2 + num_iterations_left) > 0:
            return (0, 0, 0, 0)
        
        # The agent.choose_action() method expects an observation tuple (obs0, obs1)
        # and looks up index [self.player_id]. Since we locked player_id to 0 in __init__,
        # we can just pass a tuple containing this single observation twice.
        dummy_obs_tuple = (observation, observation)
        
        # The agent natively returns (action_type, raise_amount, keep1, keep2)
        action_tuple = self.agent.choose_action(dummy_obs_tuple)
        
        # Optional: Log the decision
        #self.logger.info(f"DeepCFR chose action: {action_tuple}")
        
        return action_tuple
    
    def observe(self, observation, reward, terminated, truncated, info):
        if reward and terminated:
            self.pot+=reward
            self.num_iterations+=1
        if terminated and abs(reward) > 20:
            self.logger.info(f"Hand ended with reward: {reward}")
        if "player_0_cards" in info:
            self.logger.info(
                f"Showdown: {info['player_0_cards']} vs {info['player_1_cards']} "
                f"board {info['community_cards']}"
            )