## Stochastic
##
## DQN Keras 2018: https://github.com/keras-rl/keras-rl/blob/master/rl/agents/dqn.py
##
## gamma = 0.75
##
## 300k steps (~30 steps per battle. Up to 10k battles. Divided into 30 epochs)
##

import numpy as np
import tensorflow as tf
import asyncio

from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.player.random_player import RandomPlayer
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player.battle_order import ForfeitBattleOrder


from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

import neptune.new as neptune
import nest_asyncio

import pandas as pd
import time
import json
import os
import matplotlib
from collections import defaultdict
from datetime import date
from itertools import product
from scipy.interpolate import griddata

debug = True
use_neptune = True

nest_asyncio.apply()
np.random.seed(0)

if use_neptune:
    run = neptune.init(project='leolellisr/rl-pokeenv',
                       api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1NjY1YmJkZi1hYmM5LTQ3M2QtOGU1ZC1iZTFlNWY4NjE1NDQifQ==',
                       tags=["DeepRL", "Deep Q-learning", "DQN", "stochastic", "elu", "keras2018", "300k steps"])

# our team

OUR_TEAM = """ 
Pikachu-Original (M) @ Light Ball  
Ability: Static  
EVs: 252 Atk / 4 SpD / 252 Spe  
Adamant Nature  
- Volt Tackle  
- Nuzzle  
- Iron Tail  
- Knock Off  

Charizard @ Life Orb  
Ability: Solar Power  
EVs: 252 SpA / 4 SpD / 252 Spe  
Timid Nature  
IVs: 0 Atk  
- Flamethrower  
- Dragon Pulse  
- Roost  
- Sunny Day  

Blastoise @ White Herb  
Ability: Torrent  
EVs: 4 Atk / 252 SpA / 252 Spe  
Mild Nature  
- Scald  
- Ice Beam  
- Earthquake  
- Shell Smash  

Venusaur @ Black Sludge  
Ability: Chlorophyll  
EVs: 252 SpA / 4 SpD / 252 Spe  
Modest Nature  
IVs: 0 Atk  
- Giga Drain  
- Sludge Bomb  
- Sleep Powder  
- Leech Seed  

Sirfetch’d @ Aguav Berry  
Ability: Steadfast  
EVs: 248 HP / 252 Atk / 8 SpD  
Adamant Nature  
- Close Combat  
- Swords Dance  
- Poison Jab  
- Knock Off  

Tauros (M) @ Assault Vest  
Ability: Intimidate  
EVs: 252 Atk / 4 SpD / 252 Spe  
Adamant Nature  
- Double-Edge  
- Earthquake  
- Megahorn  
- Iron Head  
"""

OP_TEAM = """
Eevee @ Eviolite  
Ability: Adaptability  
EVs: 252 HP / 252 Atk / 4 SpD  
Jolly Nature  
- Quick Attack  
- Flail  
- Facade  
- Wish  

Vaporeon @ Leftovers  
Ability: Hydration  
EVs: 252 HP / 252 Def / 4 SpA  
Bold Nature  
IVs: 0 Atk  
- Scald  
- Shadow Ball  
- Toxic  
- Wish  

Sylveon @ Aguav Berry  
Ability: Pixilate  
EVs: 252 HP / 252 SpA / 4 SpD  
Modest Nature  
IVs: 0 Atk  
- Hyper Voice  
- Mystical Fire  
- Psyshock  
- Calm Mind  

Jolteon @ Assault Vest  
Ability: Quick Feet  
EVs: 252 SpA / 4 SpD / 252 Spe  
Timid Nature  
IVs: 0 Atk  
- Thunderbolt  
- Hyper Voice  
- Volt Switch  
- Shadow Ball  

Leafeon @ Life Orb  
Ability: Chlorophyll  
EVs: 252 Atk / 4 SpD / 252 Spe  
Jolly Nature  
- Leaf Blade  
- Knock Off  
- X-Scissor  
- Swords Dance  

Umbreon @ Iapapa Berry  
Ability: Inner Focus  
EVs: 252 HP / 4 Atk / 252 SpD  
Careful Nature  
- Foul Play  
- Body Slam  
- Toxic  
- Wish  
"""


class DQL_RLPlayer(Gen8EnvSinglePlayer):
    def __init__(self, battle_format, team, mode):
            super().__init__(battle_format=battle_format, team=team)
            self.mode = mode 
            self.num_battles = 0
            self.num_battles_avg = 0
            self.n_won_battles_avg = self.n_won_battles
            self._ACTION_SPACE = list(range(4 + 5))
    def embed_battle(self, battle):
        # -1 indicates that the move does not have a base power
        # or is not available
    #    moves_base_power = -np.ones(4)
    #    moves_dmg_multiplier = np.ones(4)
    #    for i, move in enumerate(battle.available_moves):
    #        moves_base_power[i] = (
    #            move.base_power / 100
    #        )  # Simple rescaling to facilitate learning
    #        if move.type:
    #            moves_dmg_multiplier[i] = move.type.damage_multiplier(
    #                battle.opponent_active_pokemon.type_1,
    #                battle.opponent_active_pokemon.type_2,
    #            )

        # We count how many pokemons have not fainted in each team
    #    remaining_mon_team = (
    #        len([mon for mon in battle.team.values() if mon.fainted]) / 6
    #    )
    #    remaining_mon_opponent = (
    #        len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
    #    )

        # Final vector with 10 components
    #    return np.concatenate(
    #        [
    #            moves_base_power,
    #            moves_dmg_multiplier,
    #            [remaining_mon_team, remaining_mon_opponent],
    #        ]
    #    )

        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                    move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                )

        # We count how many pokemons have not fainted in each team
        n_fainted_mon_team = (
            len([mon for mon in battle.team.values() if mon.fainted])
        )
        n_fainted_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted])
        )

    #    state = list()
    #    state.append(NAME_TO_ID_DICT[str(battle.active_pokemon).split(' ')[0]])
    #    state.append(NAME_TO_ID_DICT[str(battle.opponent_active_pokemon).split(' ')[0]])
    #    for move_base_power in moves_base_power:
    #        state.append('{0:.2f}'.format(move_base_power))
    #    for move_dmg_multiplier in moves_dmg_multiplier:
    #        state.append('{0:.2f}'.format(move_dmg_multiplier))
    #    state.append(n_fainted_mon_team)
    #    state.append(n_fainted_mon_opponent)
        state= np.concatenate([
            [NAME_TO_ID_DICT[str(battle.active_pokemon).split(' ')[0]]],
            [NAME_TO_ID_DICT[str(battle.opponent_active_pokemon).split(' ')[0]]],
            [move_base_power for move_base_power in moves_base_power],
            [move_dmg_multiplier for move_dmg_multiplier in moves_dmg_multiplier],
            [n_fainted_mon_team,
            n_fainted_mon_opponent]])
            
    
        return state

    # Computing rewards
    
    def reward_computing_helper(
            self,
            battle: AbstractBattle,
            *,
            fainted_value: float = 0.15,
            hp_value: float = 0.15,
            number_of_pokemons: int = 6,
            starting_value: float = 0.0,
            status_value: float = 0.15,
            victory_value: float = 1.0
    ) -> float:
        # 1st compute
        if battle not in self._reward_buffer:
            self._reward_buffer[battle] = starting_value
        current_value = 0

        # Verify if pokemon have fainted or have status
        for mon in battle.team.values():
            current_value += mon.current_hp_fraction * hp_value
            if mon.fainted:
                current_value -= fainted_value
            elif mon.status is not None:
                current_value -= status_value

        current_value += (number_of_pokemons - len(battle.team)) * hp_value

        # Verify if opponent pokemon have fainted or have status
        for mon in battle.opponent_team.values():
            current_value -= mon.current_hp_fraction * hp_value
            if mon.fainted:
                current_value += fainted_value
            elif mon.status is not None:
                current_value += status_value

        current_value -= (number_of_pokemons - len(battle.opponent_team)) * hp_value

        # Verify if we won or lost
        if battle.won:
            current_value += victory_value
        elif battle.lost:
            current_value -= victory_value

        # Value to return
        to_return = current_value - self._reward_buffer[battle]
        self._reward_buffer[battle] = current_value
        if use_neptune:
            run[f'{self.mode} reward_buffer'].log(current_value)
            run[f'{self.mode} accum. reward_buffer'].log(sum(_reward_buffer))
        return to_return

    # Calling reward_computing_helper
    def compute_reward(self, battle) -> float:
        return self.reward_computing_helper(battle, fainted_value=2, hp_value=1, victory_value=15)
        
    def _battle_finished_callback(self, battle):
        self.num_battles += 1
        self.num_battles_avg += 1
        if use_neptune:
            run[f'{self.mode} win_acc'].log(self.n_won_battles / self.num_battles)
            run[f'{self.mode} win_acc avg'].log(self.n_won_battles_avg / self.num_battles_avg)

        if self.num_battles%100==0:
            self.num_battles_avg = 0
            self.n_won_battles_avg = 0

        self._observations[battle].put(self.embed_battle(battle))

class MaxDamagePlayer(RandomPlayer):
    def choose_move(self, battle):
        # If the player can attack, it will
        if battle.available_moves:
            # Finds the best move among available ones
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)

EPOCHS = 30
NB_TRAINING_EPISODES = 10000
NB_TRAINING_STEPS = NB_TRAINING_EPISODES*EPOCHS
NB_EVALUATION_EPISODES = int(NB_TRAINING_EPISODES/3)

N_STATE_COMPONENTS = 12
# num of features = num of state components + action
N_FEATURES = N_STATE_COMPONENTS + 1

N_OUR_MOVE_ACTIONS = 4
N_OUR_SWITCH_ACTIONS = 5
N_OUR_ACTIONS = N_OUR_MOVE_ACTIONS + N_OUR_SWITCH_ACTIONS

ALL_OUR_ACTIONS = np.array(range(0, N_OUR_ACTIONS))

N_HIDDEN = 128

NAME_TO_ID_DICT = {
    "pikachuoriginal": 0,
    "charizard": 1,
    "blastoise": 2,
    "venusaur": 3,
    "sirfetchd": 4,
    "tauros": 5,
    "eevee": 6,
    "vaporeon": 7,
    "sylveon": 8,
    "jolteon": 9,
    "leafeon": 10,
    "umbreon": 11
}

tf.random.set_seed(0)
np.random.seed(0)


# This is the function that will be used to train the dqn
def dqn_training(player, dqn, nb_steps):
    dqn.fit(player, nb_steps=nb_steps)
    player.complete_current_battle()


def dqn_evaluation(player, dqn, nb_episodes):
    # Reset battle statistics
    player.reset_battles()
    dqn.test(player, nb_episodes=nb_episodes, visualize=False, verbose=False)

    print(
        "DQN Evaluation: %d victories out of %d episodes"
        % (player.n_won_battles, nb_episodes)
    )


if __name__ == "__main__":
    env_player = DQL_RLPlayer(battle_format="gen8ou", team=OUR_TEAM, mode = "train")

    second_opponent = RandomPlayer(battle_format="gen8ou", team=OP_TEAM)
    opponent= MaxDamagePlayer(battle_format="gen8ou", team=OP_TEAM)

    # Output dimension
    n_action = len(env_player.action_space)

    model = Sequential()
    model.add(Dense(N_HIDDEN, activation="elu", input_shape=(1, N_STATE_COMPONENTS)))

    # Our embedding have shape (1, 12), which affects our hidden layer
    # dimension and output dimension
    # Flattening resolve potential issues that would arise otherwise
    model.add(Flatten())
    model.add(Dense(int(N_HIDDEN/2), activation="elu"))
    model.add(Dense(n_action, activation="linear"))

    memory = SequentialMemory(limit=NB_TRAINING_STEPS, window_length=1)

    # epsilon greedy
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.05,
        value_test=0,
        nb_steps=NB_TRAINING_STEPS,
    )

    # Defining our DQN / https://github.com/keras-rl/keras-rl/blob/master/rl/agents/dqn.py

    dqn = rl.agents.dqn.DQNAgent(
        model=model,
        nb_actions=len(env_player.action_space),
        policy=policy,
        memory=memory,
        nb_steps_warmup=int(NB_TRAINING_STEPS/10),
        gamma=0.75,
        target_model_update=1,
        delta_clip=0.01,
        enable_double_dqn=True,
    )

    dqn.compile(Adam(lr=0.00025), metrics=["mae"])

    # Training
    env_player.play_against(
        env_algorithm=dqn_training,
        opponent=opponent,
        env_algorithm_kwargs={"dqn": dqn, "nb_steps": NB_TRAINING_STEPS},
    )
    model.save("model_%d" % NB_TRAINING_STEPS)
    env_player.mode = "val_max"
    # Evaluation
    print("Results against max player:")
    env_player.num_battles=0
    env_player.play_against(
        env_algorithm=dqn_evaluation,
        opponent=opponent,
        env_algorithm_kwargs={"dqn": dqn, "nb_episodes": NB_EVALUATION_EPISODES},
    )
    env_player.mode = "val_rand"
    print("\nResults against random player:")
    env_player.num_battles=0
    env_player.play_against(
        env_algorithm=dqn_evaluation,
        opponent=second_opponent,
        env_algorithm_kwargs={"dqn": dqn, "nb_episodes": NB_EVALUATION_EPISODES},
    )







