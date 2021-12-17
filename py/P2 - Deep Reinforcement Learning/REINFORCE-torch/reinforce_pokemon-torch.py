import numpy as np
import torch
import asyncio

from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.player.random_player import RandomPlayer
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player.battle_order import ForfeitBattleOrder

#import torch
import torch
from torch.autograd import Variable
import torch.nn.utils as utils
from normalized_actions import NormalizedActions
from reinforce_discrete import REINFORCE
from distutils.util import strtobool
import neptune.new as neptune
#import nest_asyncio

import pandas as pd
import time
import json
import os
#import matplotlib
from collections import defaultdict
from datetime import date
from itertools import product
from scipy.interpolate import griddata
import argparse

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

NAME_TO_ID_DICT = {
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

OUR_TEAM_DET = """
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

OP_TEAM_DET = """
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
- Sleep Talk  
- Shadow Claw  
- Rest  
"""

NAME_TO_ID_DICT_DET = {
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
    parser.add_argument('--debug', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, debug will be enabled')
    parser.add_argument('--neptune', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
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
    parser.add_argument('--gamma', type=float, default=0.99,
        help="gamma value used on DQN") 
    parser.add_argument('--adamlr', type=float, default=0.00005,
        help="learning rate used on Adam")    

    # Training parameters
    parser.add_argument('--epochs', type=int, default=30,
        help="n epochs")    
    parser.add_argument('--battles', type=int, default=30000,
        help="n steps per epoch")
    parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                    help='number of episodes (default: 128)')
    
    args = parser.parse_args()
    return args

#nest_asyncio.apply()
np.random.seed(0)

class ReinforcePlayer(Gen8EnvSinglePlayer):
    def __init__(self, battle_format, team, agent):
        super().__init__(battle_format=battle_format, team=team)
        self.agent = agent
        self.state = None
        self.action = None
        self.reward = 0
        self.num_battles = 0
        self.entropies = []
        self.log_probs = []
        self.rewards = []

    def choose_move(self, battle):
        if self.state is not None:
            # observe R, S'
            self.reward = self.compute_reward(battle)
            next_state = self.embed_battle(battle)
            # S <- S'
            self.state = next_state
        else:
            # S first initialization
            self.state = self.embed_battle(battle)

        # choose A from S using policy pi
        self.action, entropy, log_prob = self.agent.select_action(torch.from_numpy(np.array(self.state)))

        self.entropies.append(entropy)
        self.log_probs.append(log_prob)
        self.rewards.append(torch.from_numpy(np.array(self.reward)))

        # if the selected action is not possible, perform a random move instead
        if self.action == -1:
            return ForfeitBattleOrder()
        elif self.action < 4 and self.action < len(battle.available_moves) and not battle.force_switch:
            return self.create_order(battle.available_moves[self.action])
        elif 0 <= self.action - 4 < len(battle.available_switches):
            return self.create_order(battle.available_switches[self.action - 4])
        else:
            return self.choose_random_move(battle)

    def embed_battle(self, battle):
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
        state = np.concatenate([
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
            run[f'{"train"} reward_buffer'].log(current_value)
        return to_return

    # Calling reward_computing_helper
    def compute_reward(self, battle) -> float:
        return self.reward_computing_helper(battle, fainted_value=2, hp_value=1, victory_value=15)

    def _battle_finished_callback(self, battle):
        self.num_battles += 1
        if args.neptune:
            run[f'{"train"} win_acc'].log(self.n_won_battles / self.num_battles)
        self.agent.update_parameters(self.rewards, self.log_probs, self.entropies, args.gamma)

        self.entropies = []
        self.log_probs = []
        self.rewards = []


class ValidationPlayer(Gen8EnvSinglePlayer):
    def __init__(self, battle_format, team, agent, env_player_mode):
        super().__init__(battle_format=battle_format, team=team)
        self.agent = agent
        self.env_player_mode = env_player_mode
        self.state = None
        self.action = None
        self.num_battles = 0

    def choose_move(self, battle):
        # call to compute reward for logging reasons only
        self.compute_reward(battle)
        self.state = self.embed_battle(battle)
        # choose A from S using policy pi
        self.action, entropy, log_prob = self.agent.select_action(torch.from_numpy(np.array(self.state)))

        # if the selected action is not possible, perform a random move instead
        if self.action == -1:
            return ForfeitBattleOrder()
        elif self.action < 4 and self.action < len(battle.available_moves) and not battle.force_switch:
            return self.create_order(battle.available_moves[self.action])
        elif 0 <= self.action - 4 < len(battle.available_switches):
            return self.create_order(battle.available_switches[self.action - 4])
        else:
            return self.choose_random_move(battle)

    def embed_battle(self, battle):
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
        state = np.concatenate([
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
            run[f'{self.env_player_mode} reward_buffer'].log(current_value)
        return to_return

    # Calling reward_computing_helper
    def compute_reward(self, battle) -> float:
        return self.reward_computing_helper(battle, fainted_value=2, hp_value=1, victory_value=15)

    def _battle_finished_callback(self, battle):
        self.num_battles += 1
        if args.neptune:
            run[f'{self.env_player_mode} win_acc'].log(self.n_won_battles / self.num_battles)


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

if __name__ == "__main__":

    args = parse_args()

    if args.neptune:
        run = neptune.init(project='leolellisr/rl-pokeenv',
                        api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1NjY1YmJkZi1hYmM5LTQ3M2QtOGU1ZC1iZTFlNWY4NjE1NDQifQ==',
                        tags=["REINFORCE-torch", args.exp_name, args.env, str(args.epochs)+"episodes"])

    
    # environment: stochastic or deterministic
    if args.env == "deterministic":
        OUR_TEAM = OUR_TEAM_DET
        OP_TEAM = OP_TEAM_DET
    else:
        OUR_TEAM = OUR_TEAM
        OP_TEAM = OP_TEAM

    # extract the args
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


     # training
    async def do_battle_training():
        # REINFORCE agent
        reinforce_agent = REINFORCE(args.hidden_size, N_STATE_COMPONENTS, N_OUR_ACTIONS, lr = args.adamlr)
        # our player
        player = ReinforcePlayer(battle_format="gen8ou", team=OUR_TEAM, agent=reinforce_agent)
        # opponent's player
        opponent = MaxDamagePlayer(battle_format="gen8ou", team=OP_TEAM)
        await player.battle_against(opponent=opponent, n_battles=NB_TRAINING_EPISODES)
        # logs
        if args.debug:
            print("training: num battles (episodes)=%d, wins=%d, winning %%=%.2f" %
                  (
                      NB_TRAINING_EPISODES,
                      player.n_won_battles,
                      round((player.n_won_battles / NB_TRAINING_EPISODES) * 100, 2)
                  ))
        return reinforce_agent

    reinforce_agent = None
    if args.train:
        loop = asyncio.get_event_loop()
        reinforce_agent = loop.run_until_complete(loop.create_task(do_battle_training()))
    elif args.saved:
        reinforce_agent = REINFORCE(args.hidden_size, N_STATE_COMPONENTS, N_OUR_ACTIONS, lr = args.adamlr)
        reinforce_agent.load_model(args.model_folder)


    if reinforce_agent is None:
        exit("Error: no agent")
    
    # validation
    async def do_battle_validation(env_player_mode):
        # REINFORCE agent validation mode (no learning)
        # our player
        player = ValidationPlayer(
            battle_format="gen8ou",
            team=OUR_TEAM,
            agent=reinforce_agent,
            env_player_mode=env_player_mode
        )
        if env_player_mode == "val_max":
            print("\nResults against max player:")
            opponent = MaxDamagePlayer(battle_format="gen8ou", team=OUR_TEAM)
        else:
            print("\nResults against random player:")
            opponent = RandomPlayer(battle_format="gen8ou", team=OP_TEAM)
        await player.battle_against(opponent=opponent, n_battles=NB_EVALUATION_EPISODES)
        # logs
        if args.debug:
            print("validation: num battles (episodes)=%d, wins=%d, winning %%=%.2f" %
                  (
                      NB_EVALUATION_EPISODES,
                      player.n_won_battles,
                      round((player.n_won_battles / NB_EVALUATION_EPISODES) * 100, 2)
                  ))

    # evaluation against max player
    loop = asyncio.get_event_loop()
    loop.run_until_complete(loop.create_task(do_battle_validation("val_max")))

    # evaluation against random player
    loop = asyncio.get_event_loop()
    loop.run_until_complete(loop.create_task(do_battle_validation("val_rand")))