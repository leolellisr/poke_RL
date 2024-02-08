# Notebook for training and testing a Player in Pokemon Showdown using Monte Carlo with Function Approximation in Stochastic Environment

# Comparative Table: https://prnt.sc/1ytqrzm
# Action space: 4 moves + 5 switches
# poke-env installed in C:\\Users\\-\\anaconda3\\envs\\poke_env\\lib\\site-packages

# imports

import asyncio
import json
import os
import matplotlib
import neptune.new as neptune
import nest_asyncio
import numpy as np
import pandas as pd
import time

from collections import defaultdict
from datetime import date
from itertools import product
from matplotlib import pyplot
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player.battle_order import ForfeitBattleOrder
from poke_env.player.player import Player
from poke_env.player.random_player import RandomPlayer
from scipy.interpolate import griddata
from src.playerMC_FA import Player as PlayerMC_FA


# global configs

debug = True
save_to_json_file = False
use_validation = True
use_neptune = False

nest_asyncio.apply()
np.random.seed(0)

if use_neptune:
    run = neptune.init(name= 'MCControlFAStochastic', tags=['Function Approximation', 'MC Control', 'Stochastic', 'Train'], project='your project',
                       api_token='your api token')


# Definition of agent's team (Pokémon Showdown template)

OUR_TEAM = """
Pikachu-Original (M) @ Light Ball  
Ability: Static  
EVs: 252 Atk / 4 SpD / 252 Spe  
Jolly Nature  
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
Jolly Nature  
- Double-Edge  
- Earthquake  
- Megahorn  
- Iron Head  
"""

# Definition of opponent's team (Pokémon Showdown template)

OP_TEAM = """
Eevee @ Eviolite  
Ability: Adaptability  
EVs: 252 HP / 252 Atk / 4 SpD  
Adamant Nature  
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
Adamant Nature  
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


N_STATE_COMPONENTS = 12
# num of features = num of state components + action
N_FEATURES = N_STATE_COMPONENTS + 1

N_OUR_MOVE_ACTIONS = 4
N_OUR_SWITCH_ACTIONS = 5
N_OUR_ACTIONS = N_OUR_MOVE_ACTIONS + N_OUR_SWITCH_ACTIONS

ALL_OUR_ACTIONS = np.array(range(0, N_OUR_ACTIONS))

# Encoding Pokémon Name for ID
def name_to_id(name):
    if(name == 'venusaur'): return 0
    if(name == 'pikachuoriginal'): return 1 
    if(name == 'tauros'): return 2 
    if(name == 'sirfetchd'): return 3 
    if(name == 'blastoise'): return 4 
    if(name == 'charizard'): return 5 
    if(name == 'eevee'): return 0
    if(name == 'vaporeon'): return 1 
    if(name == 'leafeon'): return 2 
    if(name == 'sylveon'): return 3 
    if(name == 'jolteon'): return 4 
    if(name == 'umbreon'): return 5     


# Definition of MaxDamagePlayer
class MaxDamagePlayer(Player):
    def choose_move(self, battle):
        if battle.available_moves:
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)
        else:
            return self.choose_random_move(battle)

# Definition of Monte Carlo with Function Approximation Player
class MCPlayer(PlayerMC_FA):
    def choose_move(self, battle):
        # In the 1st state of all we don't append yet;
        # Other states: using previous state and action with actual reward (actual battle) 
        if self.previous_action != -10: self.episode.append((self.previous_state, self.previous_action, self.compute_reward(battle)))
            
        # Getting state s (-> embed battle)
        s = self.embed_battle(battle)
        # 1st move will be random wout policy    
        if(self.aux == 0):
            self.aux = 1
            action = np.random.choice(self.action_space)

        # Other moves have policy    
        else: action = np.random.choice(self.action_space, p=self.policy(s))

        # Saving action and state to append later. We can compute the reward only after the move    
        self.previous_action = action
        self.previous_state = s

        # Choose move according to action index
        if action == -1:
            return ForfeitBattleOrder()
        elif (
            action < 4
            and action < len(battle.available_moves)
            and not battle.force_switch
        ):
            return self.create_order(battle.available_moves[action])
        elif 0 <= action - 4 < len(battle.available_switches):
            return self.create_order(battle.available_switches[action - 4])
        else:
            return self.choose_random_move(battle)
        
    
    def x(self, state, action):
        state = np.array(state).astype(float)
        return np.append(state, action)

    # q^(S, A, W)
    def q_approx(self, state, action, w):
        state = np.array(state).astype(float)
        return np.dot(self.x(state, action), w)

    # max(a, q^(S, a', W))
    def max_q_approx(self, state, w):
        state = np.array(state).astype(float)
        return max(np.array([self.q_approx(state, action, w) for action in range(N_OUR_ACTIONS)]))

    def _battle_finished_callback(self, battle):
        rewards = [reward for state,action,reward in self.episode]
        states = [state for state,action,reward in self.episode]
        actions = [action for state,action,reward in self.episode]
        t_array = range(len(self.episode)+1)
        # Computing w and N
        for idx, state in enumerate(states):
            action = actions[idx]
            if state not in self.visited_states:
                self.N[str(state)][action] += 1
                returnGt = sum([reward*pow(self.gamma, t) for reward, t in zip(rewards[idx:], t_array[:(-idx-1)])]) 

               
                # step-size: 1./N[state][action]
                delta =                 returnGt - self.q_approx(state, action, self.w)
                self.w += (1/self.N[str(state)][action])* delta * self.x(state, action)            

                self.visited_states.append(state)
                
        self.visited_states = []
        self.episode = []
        if(self.aux == 1):
            self.aux == 0
            
        # Define new policy with updated w and N
        self.policy = self.update_epsilon_greedy_policy(self.w, self.n0, self.N)
        if use_neptune: run[f'N0: {self.n0} gamma: {self.gamma} win_acc'].log(self.n_won_battles/len(self._reward_buffer))
            
    ''' Helper functions '''

    # the embed battle is our state
    # 10 factors: 4 moves base power, 4 moves multipliers, active pokemon and active opponent pokemon 
    def embed_battle(self, battle):
        # -1 indicates that the move does not have a base power or not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling 
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                )

        # how many pokemons have fainted in each team
        remaining_mon_team = (
            len([mon for mon in battle.team.values() if mon.fainted]) / 6
        )
        remaining_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        state = list()       
        state.append('{0:.2f}'.format(name_to_id(str(battle.active_pokemon).split(' ')[0])))
        state.append('{0:.2f}'.format(name_to_id(str(battle.opponent_active_pokemon).split(' ')[0])))    
        for move_base_power in moves_base_power:
            state.append('{0:.2f}'.format(move_base_power))
        for move_dmg_multiplier in moves_dmg_multiplier:
            state.append('{0:.2f}'.format(move_dmg_multiplier))
        state.append('{0:.2f}'.format(remaining_mon_team))
        state.append('{0:.2f}'.format(remaining_mon_opponent))
        # Convert to string so we can use as hash
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
        if use_neptune: run[f'N0: {self.n0} gamma: {self.gamma} reward_buffer'].log(current_value)
        return to_return

    # Calling reward_computing_helper
    def compute_reward(self, battle) -> float:
        return self.reward_computing_helper(battle, fainted_value=2, hp_value=1, victory_value=15)

# Definition of Monte Carlo with function approximation validation player
class ValidationPlayer(PlayerMC_FA):
    def __init__(self, battle_format, team, w, N, n0):
        super().__init__(battle_format=battle_format, team=team)
        self.w = w
        self.N = N
        self.n0 = n0
        self.policy = self.update_epsilon_greedy_policy(self.w, self.n0, self.N)

    def x(self, state, action):
        state = np.array(state).astype(float)
        return np.append(state, action)

    # q^(S, A, W)
    def q_approx(self, state, action, w):
        state = np.array(state).astype(float)
        
        return np.dot(self.x(state, action), w)

    # max(a, q^(S, a', W))
    def max_q_approx(self, state, w):
        state = np.array(state).astype(float)
        return max(np.array([self.q_approx(state, action, w) for action in range(N_OUR_ACTIONS)]))
    
    def choose_move(self, battle):
        state = self.embed_battle(battle)
        # let's get the greedy action. Ties must be broken arbitrarily

        
        action = np.random.choice(self.action_space, p=self.policy(state))
       

        # if the selected action is not possible, perform a random move instead
        if action == -1:
            return ForfeitBattleOrder()
        elif action < 4 and action < len(battle.available_moves) and not battle.force_switch:
            return self.create_order(battle.available_moves[action])
        elif 0 <= action - 4 < len(battle.available_switches):
            return self.create_order(battle.available_switches[action - 4])
        else:
            return self.choose_random_move(battle)

    def _battle_finished_callback(self, battle):
        pass

    # the embed battle is our state
    # 11 factors: 4 moves base power, 4 moves multipliers, active pokemon and active opponent pokemon 
    def embed_battle(self, battle):
        # -1 indicates that the move does not have a base power or not available

        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling 
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                )

        # how many pokemons have fainted in each team
        remaining_mon_team = (
            len([mon for mon in battle.team.values() if mon.fainted]) / 6
        )
        
        remaining_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )
        
        state = list()
        state.append('{0:.2f}'.format(name_to_id(str(battle.active_pokemon).split(' ')[0])))
        state.append('{0:.2f}'.format(name_to_id(str(battle.opponent_active_pokemon).split(' ')[0])))    
        for move_base_power in moves_base_power:
            state.append('{0:.2f}'.format(move_base_power))
        for move_dmg_multiplier in moves_dmg_multiplier:
            state.append('{0:.2f}'.format(move_dmg_multiplier))
        state.append('{0:.2f}'.format(remaining_mon_team))
        state.append('{0:.2f}'.format(remaining_mon_opponent))
        # Convert to string so we can use as hash
        return state
    
    

# global parameters

# possible values for num_battles (number of episodes)
n_battles_array = [10000]
# exploration schedule from MC, i. e., epsilon(t) = N0 / (N0 + N(S(t)))
n0_array = [0.0001, 0.001, 0.01]

# possible values for gamma (discount factor)
gamma_array = [0.75]


list_of_params = [
    {
        'n_battles': n_battles,
        'n0': n0,
        'gamma': gamma
    } for n_battles, n0, gamma in product(n_battles_array, n0_array, gamma_array)
]


# json helper functions

def save_array_to_json(path_dir, filename, data):
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    full_filename = path_dir + "/" + filename
    # write
    with open(full_filename, "w") as file:
        json.dump(data if isinstance(data, list) else data.tolist(), file)
        file.close()


def save_dict_to_json(path_dir, filename, value):
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    full_filename = path_dir + "/" + filename
    file = open(full_filename, "w")
    value_dict = dict()
    for key in value:
        value_dict[key] = value[key].tolist()
    json.dump(value_dict, file)
    file.close()


def read_array_from_json(path_dir, filename):
    full_filename = path_dir + "/" + filename
    if not os.path.exists(full_filename):
        return None
    file = open(full_filename, "r")
    data = json.load(file)
    file.close()
    return data


def read_dict_from_json(path_dir, filename):
    full_filename = path_dir + "/" + filename
    if not os.path.exists(full_filename):
        return None
    file = open(full_filename, "r")
    data = json.load(file)
    for key in data:
        data[key] = np.array(data[key])
    file.close()
    return data

# main (let's battle!)

# training
async def do_battle_training():
    for params in list_of_params:
        start = time.time()
        params['player'] = MCPlayer(battle_format="gen8ou", team=OUR_TEAM, n0=params['n0'], gamma=params['gamma'])
        params['opponent'] = MaxDamagePlayer(battle_format="gen8ou", team=OP_TEAM)
        await params['player'].battle_against(opponent=params['opponent'], n_battles=params['n_battles'])
        if debug:
            print("training: num battles (episodes)=%d, N0=%.4f, gamma=%.2f, wins=%d, winning %%=%.2f, total time=%s sec" %
                  (
                      params['n_battles'],
                      round(params['n0'], 4),
                      round(params['gamma'], 2),
                      params['player'].n_won_battles,
                      round((params['player'].n_won_battles / params['n_battles']) * 100, 2),
                      round(time.time() - start, 2)
                  ))

        # save w to json file
        if save_to_json_file:
            today_s = str(date.today())
            n_battle_s = str(params['n_battles'])
            n0_s = str(round(params['n0'], 4))
            gamma_s = str(round(params['gamma'], 4))
            winning_percentage_s = str(round((params['player'].n_won_battles / params['n_battles']) * 100, 2))

            filename = "W_" + today_s + "_" + n_battle_s + "_" + n0_s + "_" + gamma_s + "_" + winning_percentage_s+".json"  
            save_array_to_json("./MC_Control_FA_w", filename, params['player'].w)
            
            filename = "N_" + today_s + "_" + n_battle_s + "_" + n0_s + "_" + gamma_s + "_" + winning_percentage_s+".json"           
            save_dict_to_json("./MC_Control_FA_N", filename, params['player'].N)
        key = str(params['n_battles']) + "_" + str(round(params['n0'], 4)) + "_" + str(round(params['gamma'], 2))
        winning_status = list()
        for battle in params['player']._battles.values():
            if battle.won:
                winning_status.append(True)
            else:
                winning_status.append(False)
        # save statistics json file (append)
        data = dict()
        data[key] = winning_status
        save_dict_to_json("./MC_Control_FA_statistics", "statistics.json", data)


loop = asyncio.get_event_loop()
loop.run_until_complete(loop.create_task(do_battle_training()))




# validation  - vs MaxPlayer

async def do_battle_validation_params(params):
    for parm in params:
        # learned feature vector
        w = parm['player'].w
        N = parm['player'].N
        # params: n_battles, n0, gamma
        n_battles = parm['n_battles']
        n0 = parm['n0']
        gamma = parm['gamma']

        # validation (play 1/3 of the battles using Q-learned table)
        start = time.time()
        validation_player = ValidationPlayer(battle_format="gen8ou", team=OUR_TEAM, w=w, N=N, n0=n0)
        opponent = MaxDamagePlayer(battle_format="gen8ou", team=OP_TEAM)
        n_battles_validation = int(n_battles / 3)
        await validation_player.battle_against(opponent=opponent, n_battles=n_battles_validation)
        print("validation: num battles (episodes)=%d, N0=%.4f, gamma=%.2f, wins=%d, winning %%=%.2f, total time=%s sec" %
              (
                  n_battles_validation,
                  n0,
                  gamma,
                  validation_player.n_won_battles,
                  round((validation_player.n_won_battles / n_battles_validation) * 100, 2),
                  round(time.time() - start, 2)
              ))


if use_validation:
    loop = asyncio.get_event_loop()
    loop.run_until_complete(loop.create_task(do_battle_validation_params(list_of_params)))


# validation - vs RandomPlayer

async def do_battle_validation_params(params):
    for parm in params:
        # learned feature vector
        w = parm['player'].w
        N = parm['player'].N
        # params: n_battles, n0, gamma
        n_battles = parm['n_battles']
        n0 = parm['n0']
        gamma = parm['gamma']

        # validation (play 1/3 of the battles using Q-learned table)
        start = time.time()
        validation_player = ValidationPlayer(battle_format="gen8ou", team=OUR_TEAM, w=w, N=N, n0=n0)
        opponent = RandomPlayer(battle_format="gen8ou", team=OP_TEAM)
        n_battles_validation = int(n_battles / 3)
        await validation_player.battle_against(opponent=opponent, n_battles=n_battles_validation)
        print("validation: num battles (episodes)=%d, N0=%.4f, gamma=%.2f, wins=%d, winning %%=%.2f, total time=%s sec" %
              (
                  n_battles_validation,
                  n0,
                  gamma,
                  validation_player.n_won_battles,
                  round((validation_player.n_won_battles / n_battles_validation) * 100, 2),
                  round(time.time() - start, 2)
              ))


if use_validation:
    loop = asyncio.get_event_loop()
    loop.run_until_complete(loop.create_task(do_battle_validation_params(list_of_params)))


