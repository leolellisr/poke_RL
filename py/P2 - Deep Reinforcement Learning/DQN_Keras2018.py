## Code for training and testing with DQN-Keras in Pokémon Showdown
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

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from distutils.util import strtobool
import neptune.new as neptune

import pandas as pd
import time
import json
import os

from collections import defaultdict
from datetime import date
from itertools import product
from scipy.interpolate import griddata
import argparse

# Definition of the agent stochastic team (Pokémon Showdown template)
OUR_TEAM_STO = """ 
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

# Definition of the opponent stochastic team (Pokémon Showdown template)
OP_TEAM_STO = """
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

# Encoding stochastic Pokémon Name for ID
NAME_TO_ID_DICT_STO = {
    "pikachuoriginal": 0,
    "charizard": 1,
    "blastoise": 2,
    "venusaur": 3,
    "sirfetchd": 4,
    "tauros": 5,
    "eevee": 0,
    "vaporeon": 1,
    "sylveon": 2,
    "jolteon": 3,
    "leafeon": 4,
    "umbreon": 5
}  

# Definition of the agent deterministic team (Pokémon Showdown template)
OUR_TEAM = """
Turtonator @ White Herb  
Ability: Shell Armor  
EVs: 4 Atk / 252 SpA / 252 Spe  
Rash Nature  
- Flamethrower  
- Dragon Pulse  
- Earthquake  
- Shell Smash  

Lapras @ Leftovers  
Ability: Shell Armor  
EVs: 252 HP / 252 SpA / 4 SpD  
Modest Nature  
IVs: 0 Atk  
- Freeze-Dry  
- Surf  
- Thunderbolt  
- Toxic  

Armaldo @ Assault Vest  
Ability: Battle Armor  
EVs: 252 HP / 252 Atk / 4 SpD  
Adamant Nature  
- Earthquake  
- Knock Off  
- X-Scissor  
- Aqua Jet  

Drapion @ Life Orb  
Ability: Battle Armor  
EVs: 252 Atk / 4 SpD / 252 Spe  
Jolly Nature  
- Poison Jab  
- Knock Off  
- Earthquake  
- X-Scissor  

Kabutops @ Aguav Berry  
Ability: Battle Armor  
EVs: 252 Atk / 4 SpD / 252 Spe  
Jolly Nature  
- Liquidation  
- Leech Life  
- Knock Off  
- Swords Dance  

Falinks @ Iapapa Berry  
Ability: Battle Armor  
EVs: 252 HP / 252 Atk / 4 SpD  
Adamant Nature  
- Close Combat  
- Poison Jab  
- Iron Head  
- No Retreat  

"""

# Definition of the opponent deterministic team (Pokémon Showdown template)
OP_TEAM = """
Cloyster @ Assault Vest  
Ability: Shell Armor  
EVs: 248 HP / 252 Atk / 8 SpA  
Naughty Nature  
- Icicle Spear  
- Surf  
- Tri Attack  
- Poison Jab  

Omastar @ White Herb  
Ability: Shell Armor  
EVs: 252 SpA / 4 SpD / 252 Spe  
Modest Nature  
IVs: 0 Atk  
- Surf  
- Ancient Power  
- Earth Power  
- Shell Smash  

Crustle @ Leftovers  
Ability: Shell Armor  
EVs: 252 HP / 252 Atk / 4 SpD  
Adamant Nature  
- Earthquake  
- Knock Off  
- X-Scissor  
- Stealth Rock  

Escavalier @ Life Orb  
Ability: Shell Armor  
EVs: 248 HP / 252 Atk / 8 SpD  
Adamant Nature  
- Knock Off  
- Swords Dance  
- Iron Head  
- Poison Jab  

Drednaw @ Aguav Berry  
Ability: Shell Armor  
EVs: 248 HP / 252 Atk / 8 SpD  
Adamant Nature  
- Liquidation  
- Earthquake  
- Poison Jab  
- Swords Dance  

Type: Null @ Eviolite  
Ability: Battle Armor  
EVs: 252 HP / 252 Atk / 4 SpD  
Adamant Nature  
- Facade  
- Knock Off   
- Shadow Claw  
- Rest  

"""

# Encoding deterministic Pokémon Name for ID
NAME_TO_ID_DICT = {
"turtonator": 0,
"lapras": 1,
"armaldo": 2,
"drapion": 3,
"kabutops": 4,
"falinks": 5,
"cloyster": 0,
"omastar": 1,
"crustle": 2,
"escavalier": 3,
"drednaw": 4,
"typenull": 5
}                

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
        help='the name of this experiment')
    parser.add_argument('--debug', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='if toggled, debug will be enabled')
    parser.add_argument('--neptune', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='if toggled, neptune will be enabled')   
    parser.add_argument('--train', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, train will be realized') 
    parser.add_argument('--saved', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='if toggled, use saved trained model will be realized')     
    parser.add_argument('--model-folder', type=str, default="/model",
        help='folder of trained model (just for validation)')       
    parser.add_argument('--env', type=str, default="stochastic",    
        help='type of environment (stochastic or deterministic). Define teams. OBS: must change showdown too.')

    # Agent parameters
    parser.add_argument('--policy', type=str, default="lin_epsGreedy",
        help='applied policy')            
    parser.add_argument('--hidden', type=int, default=128,
        help="Hidden layers applied on our nn") 
    parser.add_argument('--gamma', type=float, default=0.75,
        help="gamma value used on DQN") 
    parser.add_argument('--enable-double-dqn', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='enable doubleDQN')
    parser.add_argument('--adamlr', type=float, default=0.00025,
        help="learning rate used on Adam")    

    # Training parameters
    parser.add_argument('--epochs', type=int, default=30,
        help="n epochs")    
    parser.add_argument('--battles', type=int, default=10000,
        help="n steps per epoch")
    
    args = parser.parse_args()
    return args

np.random.seed(0)

# Definition of DQN player
class DQL_RLPlayer(Gen8EnvSinglePlayer):
    def __init__(self, battle_format, team, mode):
            super().__init__(battle_format=battle_format, team=team)
            self.mode = mode 
            self.num_battles = 0
            self._ACTION_SPACE = list(range(4 + 5))
    def embed_battle(self, battle):
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
        if args.neptune:
            run[f'{self.mode} reward_buffer'].log(current_value)
        #    run[f'{self.mode} accum. reward_buffer'].log(sum(self._reward_buffer.values()))
        return to_return

    # Calling reward_computing_helper
    def compute_reward(self, battle) -> float:
        return self.reward_computing_helper(battle, fainted_value=2, hp_value=1, victory_value=15)
        
    def _battle_finished_callback(self, battle):
        self.num_battles += 1
        if args.neptune:
            run[f'{self.mode} win_acc'].log(self.n_won_battles / self.num_battles)

        self._observations[battle].put(self.embed_battle(battle))

# Definition of MaxDamagePlayer
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


# This is the function that will be used to train the dqn
def dqn_training(player, dqn, nb_steps):
    dqn.fit(player, nb_steps=nb_steps, verbose=0)
    player.complete_current_battle()

# This is the function that will be used to evaluate the dqn
def dqn_evaluation(player, dqn, nb_episodes):
    # Reset battle statistics
    player.reset_battles()
    dqn.test(player, nb_episodes=nb_episodes, visualize=False, verbose=False)

    print(
        "DQN Evaluation: %d victories out of %d episodes"
        % (player.n_won_battles, nb_episodes)
    )

# Main program
if __name__ == "__main__":

    args = parse_args()

    if args.neptune:
        run = neptune.init(project='your_project',
                        api_token='your_api_token==',
                        tags=["DeepRL", args.exp_name, args.env, str(args.epochs)+"epochs"])

    

    EPOCHS = args.epochs
    NB_TRAINING_EPISODES = args.battles
    NB_TRAINING_STEPS = NB_TRAINING_EPISODES*EPOCHS
    NB_EVALUATION_EPISODES = int(NB_TRAINING_EPISODES/3)
    N_STATE_COMPONENTS = 12

    # num of features = num of state components + action
    N_FEATURES = N_STATE_COMPONENTS + 1

    N_OUR_MOVE_ACTIONS = 4
    N_OUR_SWITCH_ACTIONS = 5
    N_OUR_ACTIONS = N_OUR_MOVE_ACTIONS + N_OUR_SWITCH_ACTIONS

    ALL_OUR_ACTIONS = np.array(range(0, N_OUR_ACTIONS))

    if args.train:
        env_player = DQL_RLPlayer(battle_format="gen8ou", team=OUR_TEAM, mode = "train")
    else: env_player = DQL_RLPlayer(battle_format="gen8ou", team=OUR_TEAM, mode = "val")

    second_opponent = RandomPlayer(battle_format="gen8ou", team=OP_TEAM)
    opponent= MaxDamagePlayer(battle_format="gen8ou", team=OP_TEAM)

    # Output dimension
    n_action = len(env_player.action_space)

    if args.saved: 
        modelfolder = args.model_folder
        model = tf.keras.models.load_model(modelfolder)
    else: 
        N_HIDDEN = args.hidden
        model = Sequential()
        model.add(Dense(N_HIDDEN, activation="elu", input_shape=(1, N_STATE_COMPONENTS)))

        # Our embedding have shape (1, 12), which affects our hidden layer
        # dimension and output dimension
        # Flattening resolve potential issues that would arise otherwise
        model.add(Flatten())
        model.add(Dense(int(N_HIDDEN/2), activation="elu"))
        model.add(Dense(n_action, activation="linear"))

    memory = SequentialMemory(limit=NB_TRAINING_STEPS, window_length=1)

    if args.policy == "lin_epsGreedy":
        # linear annealing epsilon greedy https://github.com/keras-rl/keras-rl/blob/master/rl/policy.py
        policy = LinearAnnealedPolicy(
            EpsGreedyQPolicy(),
            attr="eps",
            value_max=1.0,
            value_min=0.05,
            value_test=0,
            nb_steps=NB_TRAINING_STEPS,
        )

    # Defining our DQN / https://github.com/keras-rl/keras-rl/blob/master/rl/agents/dqn.py

    dqn = DQNAgent(
        model=model,
        nb_actions=len(env_player.action_space),
        policy=policy,
        memory=memory,
        nb_steps_warmup=int(NB_TRAINING_STEPS/10),
        gamma=args.gamma,
        target_model_update=1,
        delta_clip=0.01,
        enable_double_dqn=args.enable_double_dqn
    )

    dqn.compile(Adam(lr=args.adamlr), metrics=["mae"])

    if args.train:
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







