#!/usr/bin/env python
# coding: utf-8

# In[30]:


# imports

import asyncio
import os
import re
from datetime import date

import json
import matplotlib
import neptune.new as neptune
import nest_asyncio
import numpy as np
import pandas as pd
import time

from collections import defaultdict
from itertools import product
from matplotlib import pyplot
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player.battle_order import ForfeitBattleOrder
from poke_env.player.player import Player
# from poke_env.player.random_player import RandomPlayer
from scipy.interpolate import griddata
from src.PlayerQLearning import Player as PlayerSarsa

use_neptune = True
if use_neptune:
    run = neptune.init(project='leolellisr/rl-pokeenv',
                       api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1NjY1YmJkZi1hYmM5LTQ3M2QtOGU1ZC1iZTFlNWY4NjE1NDQifQ==',
                       name= 'SarsaDeterministic', tags=['Bruno_', 'SarsaFA', 'Deterministic', 'Train'])


# In[31]:


# global configs

debug = True
save_to_json_file = True
use_validation = False

nest_asyncio.apply()
np.random.seed(0)


# In[32]:


# our team

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


# In[33]:


# opponent's team

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
- Sleep Talk  
- Shadow Claw  
- Rest  

"""


# In[34]:


N_STATE_COMPONENTS = 12
# num of features = num of state components + action
N_FEATURES = N_STATE_COMPONENTS + 1

N_OUR_MOVE_ACTIONS = 4
N_OUR_SWITCH_ACTIONS = 5
N_OUR_ACTIONS = N_OUR_MOVE_ACTIONS + N_OUR_SWITCH_ACTIONS

ALL_OUR_ACTIONS = np.array(range(0, N_OUR_ACTIONS))

NAME_TO_ID_DICT = {
    "turtonator": 0,
    "lapras": 1,
    "armaldo": 2,
    "drapion": 3,
    "kabutops": 4,
    "falinks": 5,
    "cloyster": 6,
    "omastar": 7,
    "crustle": 8,
    "escavalier": 9,
    "drednaw": 10,
    "typenull": 11
}


# In[35]:


# Max-damage player

class MaxDamagePlayer(Player):
    def choose_move(self, battle):
        if battle.available_moves:
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)
        else:
            return self.choose_random_move(battle)


# In[36]:


# SARSA player

class SARSAFAPlayer(PlayerSarsa):
    def __init__(self, battle_format, team, n0, gamma, lambda_):
        super().__init__(battle_format=battle_format, team=team)
        self.N = defaultdict(lambda: np.zeros(N_OUR_ACTIONS))
        self.F = defaultdict(lambda: np.zeros(N_FEATURES))
        self.e = np.zeros(N_FEATURES)
        self.w = np.random.rand(N_FEATURES)
        self.n0 = n0
        self.gamma = gamma
        self.lambda_ = lambda_
        self.state = None
        self.action = None
        
    def choose_move(self, battle):
        
        if self.state is not None:
            # observe R, next_state and next_action
            reward = self.compute_reward(battle)
            next_state = self.embed_battle(battle)
            next_action = self.policy_fn(next_state, self.w)
            delta = reward
            #alpha
            self.N[str(self.state)][self.action] += 1
            alpha = 0.01 / self.N[str(self.state)][self.action]
            
             #Accumulative
            self.e += 1
            delta = reward + self.gamma * self.q_approx(next_state, next_action, self.w) - self.q_approx(self.state, self.action, self.w)
            
            # update theta
            self.w += alpha * delta * self.e
            
            #update e
            self.e *= self.gamma * self.lambda_
           
            # S <- S'
            self.state = next_state
            # Choose action
            self.action = self.policy_fn(self.state, self.w)
            
        else:
            # S first initialization
            self.state = self.embed_battle(battle)
            self.action = self.policy_fn(self.state, self.w)
        
        # if the selected action is not possible, perform a random move instead
        if self.action == -1:
            return ForfeitBattleOrder()
        elif self.action < 4 and self.action < len(battle.available_moves) and not battle.force_switch:
            return self.create_order(battle.available_moves[self.action])
        elif 0 <= self.action - 4 < len(battle.available_switches):
            return self.create_order(battle.available_switches[self.action - 4])
        else:
            return self.choose_random_move(battle)

    def _battle_finished_callback(self, battle):
        self.e = np.zeros(N_FEATURES)
        if use_neptune: run[f'N0: {self.n0} gamma: {self.gamma} lambda: {self.lambda_} win_acc'].log(self.n_won_battles/len(self._reward_buffer))

    
    ''' Helper functions '''
    
    # feature vector
    @staticmethod
    def x(state, action):
        state = np.array(state).astype(float)
        return np.append(state, action)
    
    # q^(S, A, W)
    def q_approx(self, state, action, w):
        state = np.array(state).astype(float)
        return np.dot(self.x(state, action), w)

    # epsilon-greedy policy
    #Function to choose the next action
    def policy_fn(self, state, w):
        epsilon = self.n0 / (self.n0 + np.sum(self.N[str(state)]))
        # let's get the greedy action. Ties must be broken arbitrarily
        q_approx = np.array([self.q_approx(state, action, w) for action in range(N_OUR_ACTIONS)])
        greedy_action = np.argmax(np.where(q_approx == q_approx.max())[0])
        action_pick_probability = np.full(N_OUR_ACTIONS, epsilon / N_OUR_ACTIONS)
        action_pick_probability[greedy_action] += 1 - epsilon
        return np.random.choice(ALL_OUR_ACTIONS, p=action_pick_probability)
    

    # the embed battle is our state
    # 12 factors: our active mon, opponent's active mon, 4 moves base power, 4 moves multipliers, num fainted mons
    @staticmethod
    def embed_battle(battle):
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

        state = list()
        state.append(NAME_TO_ID_DICT[str(battle.active_pokemon).split(' ')[0]])
        state.append(NAME_TO_ID_DICT[str(battle.opponent_active_pokemon).split(' ')[0]])
        for move_base_power in moves_base_power:
            state.append('{0:.2f}'.format(move_base_power))
        for move_dmg_multiplier in moves_dmg_multiplier:
            state.append('{0:.2f}'.format(move_dmg_multiplier))
        state.append(n_fainted_mon_team)
        state.append(n_fainted_mon_opponent)

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
            run[f'N0: {self.n0} gamma: {self.gamma} lambda: {self.lambda_} reward_buffer'].log(current_value)
        return to_return

    # Calling reward_computing_helper
    def compute_reward(self, battle) -> float:
        return self.reward_computing_helper(battle, fainted_value=2, hp_value=1, victory_value=15)


# In[41]:


# sarsa V player

class ValidationPlayer(PlayerSarsa):
    def __init__(self, battle_format, team, w):
        super().__init__(battle_format=battle_format, team=team)
        self.w = w

    def choose_move(self, battle):
        state = self.embed_battle(battle)
        # let's get the greedy action. Ties must be broken arbitrarily
        q_approx = np.array([self.q_approx(state, action, self.w) for action in range(N_OUR_ACTIONS)])
        action = np.argmax(q_approx)
        #action = np.argmax(np.where(q_approx == q_approx.max())[0])

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

    ''' Helper functions '''

    # feature vector
    @staticmethod
    def x(state, action):
        state = np.array(state).astype(float)
        return np.append(state, action)

    # q^(S, A, W)
    def q_approx(self, state, action, w):
        state = np.array(state).astype(float)
        return np.dot(self.x(state, action), w)

    # the embed battle is our state
    # 12 factors: our active mon, opponent's active mon, 4 moves base power, 4 moves multipliers, remaining mons
    @staticmethod
    def embed_battle(battle):
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
        fainted_mon_team = (
            len([mon for mon in battle.team.values() if mon.fainted])
        )
        fainted_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted])
        )

        state = list()
        state.append(NAME_TO_ID_DICT[str(battle.active_pokemon).split(' ')[0]])
        state.append(NAME_TO_ID_DICT[str(battle.opponent_active_pokemon).split(' ')[0]])
        for move_base_power in moves_base_power:
            state.append('{0:.2f}'.format(move_base_power))
        for move_dmg_multiplier in moves_dmg_multiplier:
            state.append('{0:.2f}'.format(move_dmg_multiplier))
        state.append(fainted_mon_team)
        state.append(fainted_mon_opponent)

        return state


# In[42]:


# global parameters

# possible values for num_battles (number of episodes)
n_battles_array = [10000]
# exploration schedule from MC, i. e., epsilon(t) = N0 / (N0 + N(S(t)))
n0_array = [0.0001, 0.001, 0.01]
# possible values for gamma (discount factor)
gamma_array = [0.75] 
#Lambda
lambda_ = [0, 0.2, 0.4, 0.6, 0.8, 1]

list_of_params = [
    {
        'n_battles': n_battles,
        'n0': n0,
        'gamma': gamma,
        'lambda': lambda_
    } for n_battles, n0, gamma, lambda_ in product(n_battles_array, n0_array, gamma_array, lambda_)
]


# In[44]:


# let's battle!
async def lets_battle():
    for params in list_of_params:
        start = time.time()
        if use_neptune:
            run['params'] = params
        params['player'] = SARSAFAPlayer(battle_format="gen8ou", team=OUR_TEAM, n0=params['n0'], gamma=params['gamma'], lambda_= params['lambda'])
        params['opponent'] = MaxDamagePlayer(battle_format="gen8ou", team=OP_TEAM)
        await params['player'].battle_against(opponent=params['opponent'], n_battles=params['n_battles'])
        if debug:
            print("training: num battles (episodes)=%d, N0=%.4f, gamma=%.2f, lambda=%.2f, wins=%d, winning %%=%.2f, total time=%s sec" %
                  (
                      params['n_battles'],
                      round(params['n0'], 4),
                      round(params['gamma'], 2),
                      round(params['lambda'], 4),
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
            lambda_s = str(round(params['lambda'], 8))
            winning_percentage_s = str(round((params['player'].n_won_battles / params['n_battles']) * 100, 2))
            filename = "WDeterministic" + today_s + "_" + n_battle_s + "_" + n0_s + "_" + gamma_s + "_" + lambda_s + "_" + winning_percentage_s + ".json"
            save_array_to_json("./SarsaDeterministic_FA_w_10k2", filename, params['player'].w)

        # statistics: key: "n_battles, n0, alpha, gamma", values: list of win or lose
        key = str(params['n_battles']) + "_" + str(round(params['n0'], 4)) + "_" + str(round(params['gamma'], 2)) + "_" + str(round(params['lambda'], 2))
        winning_status = list()
        for battle in params['player']._battles.values():
            if battle.won:
                winning_status.append(True)
            else:
                winning_status.append(False)
        # save statistics json file (append)
        data = dict()
        data[key] = winning_status
        save_dict_to_json("./SarsaDeterministic_Learning_FA_statistics2", "statistics.json", data)


# In[45]:


loop = asyncio.get_event_loop()
loop.run_until_complete(loop.create_task(lets_battle()))


# In[13]:


if use_neptune: run.stop()


# In[21]:


# validation max player

async def do_battle_validation(path_dir):
    # read from json
    for filename in os.listdir(path_dir):
        # learned feature vector
        w = np.array(read_array_from_json(path_dir, filename))
        # params: n_battles, n0, gamma
        params = filename.split("_")
        n_battles = int(params[1])
        n0 = float(params[2])
        gamma = float(params[3])
        lambda_ = float(params[4])

        # validation (play 1/3 of the battles using Q-learned table)
        start = time.time()
        validation_player = ValidationPlayer(battle_format="gen8ou", team=OUR_TEAM, w=w)
        opponent = MaxDamagePlayer(battle_format="gen8ou", team=OP_TEAM)
        n_battles_validation = int(n_battles / 3)
        await validation_player.battle_against(opponent=opponent, n_battles=n_battles_validation)
        print("validation: num battles (episodes)=%d, N0=%.4f, gamma=%.2f, lambda=%.2f, wins=%d, winning %%=%.2f, total time=%s sec" %
              (
                  n_battles_validation,
                  n0,
                  gamma,
                  lambda_,
                  validation_player.n_won_battles,
                  round((validation_player.n_won_battles / n_battles_validation) * 100, 2),
                  round(time.time() - start, 2)
              ))


# In[ ]:


loop = asyncio.get_event_loop()
loop.run_until_complete(loop.create_task(do_battle_validation("./SarsaDeterministic_FA_w_10k")))


# In[24]:


from poke_env.player.player import Player
from poke_env.player.random_player import RandomPlayer

randomPlayer = RandomPlayer(battle_format="gen8ou", team=OUR_TEAM)


# In[27]:


# validation random player

async def do_battle_validation(path_dir):
    # read from json
    for filename in os.listdir(path_dir):
        # learned feature vector
        w = np.array(read_array_from_json(path_dir, filename))
        # params: n_battles, n0, gamma
        params = filename.split("_")
        n_battles = int(params[1])
        n0 = float(params[2])
        gamma = float(params[3])
        lambda_ = float(params[4])

        # validation (play 1/3 of the battles using Q-learned table)
        start = time.time()
        validation_player = ValidationPlayer(battle_format="gen8ou", team=OUR_TEAM, w=w)
        opponent = RandomPlayer(battle_format="gen8ou", team=OUR_TEAM)
        n_battles_validation = int(n_battles / 3)
        await validation_player.battle_against(opponent=opponent, n_battles=n_battles_validation)
        print("validation: num battles (episodes)=%d, N0=%.4f, gamma=%.2f, lambda=%.2f, wins=%d, winning %%=%.2f, total time=%s sec" %
              (
                  n_battles_validation,
                  n0,
                  gamma,
                  lambda_,
                  validation_player.n_won_battles,
                  round((validation_player.n_won_battles / n_battles_validation) * 100, 2),
                  round(time.time() - start, 2)
              ))


# In[ ]:


loop = asyncio.get_event_loop()
loop.run_until_complete(loop.create_task(do_battle_validation("./SarsaDeterministic_FA_w_10k")))

