from rlgym.utils.reward_functions import RewardFunction, CombinedReward
from rlgym.utils.gamestates import GameState, PlayerData
import numpy as np
from typing import Tuple, Optional, List, Any


N_UPDATES = "num-updates"


class RLCombinedLogReward(CombinedReward):

    def __init__(
            self,
            redis: Any,
            logger: Any,
            reward_names: List[str],
            reward_functions: Tuple[RewardFunction, ...],
            reward_weights: Optional[Tuple[float, ...]] = None,
    ):
        """
        Creates the combined reward using multiple rewards, and a potential set
        of weights for each reward. Will also log the weighted rewards to
        the WandB logger.
        :param redis: The redis instance
        :param logger: WandB logger
        :param reward_names: The list of reward names
        :param reward_functions: Each individual reward function.
        :param reward_weights: The weights for each reward.
        """
        super().__init__(reward_functions, reward_weights)

        self.redis = redis
        self.reward_names = reward_names
        self.logger = logger

        # Initiates the array that will store the episode totals
        self.returns = np.zeros(len(self.reward_functions))

    def reset(self, initial_state: GameState):
        self.returns = np.zeros(len(self.reward_functions))
        super().reset(initial_state)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        rewards = [
            func.get_reward(player, state, previous_action)
            for func in self.reward_functions
        ]

        self.returns += [a * b for a, b in zip(rewards, self.reward_weights)]  # store the rewards

        return float(np.dot(self.reward_weights, rewards))

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        rewards = [
            func.get_final_reward(player, state, previous_action)
            for func in self.reward_functions
        ]
        # Add the rewards to the cumulative totals with numpy broadcasting
        self.returns += [a * b for a, b in zip(rewards, self.reward_weights)]

        upd_num = int(self.redis.get(N_UPDATES))
        # Log each reward
        reward_dict = dict()
        for n, names in enumerate(self.reward_names):
            reward_dict[names] = self.returns[n]
        # Keep in mind this logs once per episode (usually once per goal scored)
        self.logger.log(reward_dict, step=upd_num, commit=False)

        return float(np.dot(self.reward_weights, rewards))
