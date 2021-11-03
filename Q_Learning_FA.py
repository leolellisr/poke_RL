# imports

import asyncio
import json
import os
import matplotlib
import neptune.new as neptune
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
# from poke_env.player.random_player import RandomPlayer
from scipy.interpolate import griddata
from src.PlayerQLearning import Player as PlayerQLearning

# global configs
debug = True
save_to_json_file = True
use_validation = False
use_neptune = False

np.random.seed(0)

if use_neptune:
    run = neptune.init(project='project', api_token='token')

# our team
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

# opponent's team

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


# Max-damage player

class MaxDamagePlayer(Player):
    def choose_move(self, battle):
        if battle.available_moves:
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)
        else:
            return self.choose_random_move(battle)


# Q-learning FA player

class QLearningFAPlayer(PlayerQLearning):
    def __init__(self, battle_format, team, n0, alpha, gamma):
        super().__init__(battle_format=battle_format, team=team)
        self.N = defaultdict(lambda: np.zeros(N_OUR_ACTIONS))
        self.w = np.random.rand(N_FEATURES)
        self.n0 = n0
        self.alpha = alpha
        self.gamma = gamma
        self.state = None
        self.action = None

    def choose_move(self, battle):
        if self.state is not None:
            # observe R, S'
            reward = self.compute_reward(battle)
            next_state = self.embed_battle(battle)
            # Q-learning
            self.N[str(self.state)][self.action] += 1
            delta = \
                reward + self.gamma * self.max_q_approx(next_state, self.w) - self.q_approx(self.state, self.action, self.w)
            self.w += self.alpha * delta * self.x(self.state, self.action)
            # S <- S'
            self.state = next_state
        else:
            # S first initialization
            self.state = self.embed_battle(battle)

        # Choose A from S using epsilon-greedy policy
        self.action = self.pi(self.state, self.w)

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

    # max(a, q^(S, a', W))
    def max_q_approx(self, state, w):
        state = np.array(state).astype(float)
        return max(np.array([self.q_approx(state, action, w) for action in range(N_OUR_ACTIONS)]))

    # epsilon-greedy policy
    def pi(self, state, w):
        epsilon = self.n0 / (self.n0 + np.sum(self.N[str(state)]))
        # let's get the greedy action. Ties must be broken arbitrarily
        q_approx = np.array([self.q_approx(state, action, w) for action in range(N_OUR_ACTIONS)])
        greedy_action = np.random.choice(np.where(q_approx == q_approx.max())[0])
        action_pick_probability = np.full(N_OUR_ACTIONS, epsilon / N_OUR_ACTIONS)
        action_pick_probability[greedy_action] += 1 - epsilon
        return np.random.choice(ALL_OUR_ACTIONS, p=action_pick_probability)

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
        return to_return

    # Calling reward_computing_helper
    def compute_reward(self, battle) -> float:
        return self.reward_computing_helper(battle, fainted_value=2, hp_value=1, victory_value=15)


# validation player

class ValidationPlayer(PlayerQLearning):
    def __init__(self, battle_format, team, w):
        super().__init__(battle_format=battle_format, team=team)
        self.w = w

    def choose_move(self, battle):
        state = self.embed_battle(battle)
        # let's get the greedy action. Ties must be broken arbitrarily
        q_approx = np.array([self.q_approx(state, action, self.w) for action in range(N_OUR_ACTIONS)])
        action = np.random.choice(np.where(q_approx == q_approx.max())[0])

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


# global parameters

# possible values for num_battles (number of episodes)
n_battles_array = [100, 1000]
# exploration schedule from MC, i. e., epsilon(t) = N0 / (N0 + N(S(t)))
n0_array = [0.0001, 0.001, 0.01]
# possible values for alpha (learning rate)
alpha_array = [0.01]
# possible values for gamma (discount factor)
gamma_array = [0.75]

list_of_params = [
    {
        'n_battles': n_battles,
        'n0': n0,
        'alpha': alpha,
        'gamma': gamma
    } for n_battles, n0, alpha, gamma in product(n_battles_array, n0_array, alpha_array, gamma_array)
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


def save_dict_to_json(path_dir, filename, data, append=True):
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    full_filename = path_dir + "/" + filename
    if os.path.exists(full_filename) and append:
        with open(full_filename, "r") as file:
            value_dict = json.load(file)
            for key in data:
                value_dict[key] = data[key] if isinstance(data[key], list) else data[key].tolist()
            file.close()
    else:
        value_dict = dict()
        for key in data:
            value_dict[key] = data[key] if isinstance(data[key], list) else data[key].tolist()
    # write
    with open(full_filename, "w") as file:
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
    file.close()
    return data


# main (let's battle!)

# training
async def do_battle_training():
    for params in list_of_params:
        start = time.time()
        params['player'] = QLearningFAPlayer(battle_format="gen8ou", team=OUR_TEAM, n0=params['n0'], alpha=params['alpha'], gamma=params['gamma'])
        params['opponent'] = MaxDamagePlayer(battle_format="gen8ou", team=OP_TEAM)
        await params['player'].battle_against(opponent=params['opponent'], n_battles=params['n_battles'])
        if debug:
            print("training: num battles (episodes)=%d, N0=%.4f, alpha=%.2f, gamma=%.2f, wins=%d, winning %%=%.2f, total time=%s sec" %
                  (
                      params['n_battles'],
                      round(params['n0'], 4),
                      round(params['alpha'], 4),
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
            alpha_s = str(round(params['alpha'], 2))
            gamma_s = str(round(params['gamma'], 4))
            winning_percentage_s = str(round((params['player'].n_won_battles / params['n_battles']) * 100, 2))
            filename = "W_" + today_s + "_" + n_battle_s + "_" + n0_s + "_" + alpha_s + "_" + gamma_s + "_" + winning_percentage_s + ".json"
            save_array_to_json("./Q_Learning_FA_w", filename, params['player'].w)

        # statistics: key: "n_battles, n0, alpha, gamma", values: list of win or lose
        key = str(params['n_battles']) + "_" + str(round(params['n0'], 4)) + "_" + str(round(params['alpha'], 2)) + "_" + str(round(params['gamma'], 2))
        winning_status = list()
        for battle in params['player']._battles.values():
            if battle.won:
                winning_status.append(True)
            else:
                winning_status.append(False)
        # save statistics json file (append)
        data = dict()
        data[key] = winning_status
        save_dict_to_json("./Q_Learning_FA_statistics", "statistics.json", data)


loop = asyncio.get_event_loop()
loop.run_until_complete(loop.create_task(do_battle_training()))


# plotting helper functions

def plot_2d(path, title, x_label, x_array, y_label, y_array):
    # print("plotting %s" % title)
    # set labels and plot surface
    figure = matplotlib.pyplot.figure(figsize=(20, 10))
    ax = figure.gca()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.plot(x_array, y_array)
    # pyplot.show()
    if not os.path.exists(path):
        os.makedirs(path)
    filename = path + "/" + title + "_" + x_label + "_" + y_label + "_" + ".pdf"
    figure.savefig(filename, dpi=figure.dpi)
    pyplot.close(figure)
    # print("done!")


def plot_3d(path, title, x_label, x_array, y_label, y_array, z_label, z_array):
    # print("plotting %s" % title)
    xyz = {'x': x_array, 'y': y_array, 'z': z_array}
    df = pd.DataFrame(xyz, index=range(len(xyz['x'])))
    xv, yv = np.meshgrid(x_array, y_array)
    zv = griddata((df['x'], df['y']), df['z'], (xv, yv), method='nearest')
    # set labels and plot surface
    figure = matplotlib.pyplot.figure(figsize=(20, 10))
    ax = figure.gca(projection='3d')
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    surface = ax.plot_surface(xv, yv, zv, rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, linewidth=0,
                              antialiased=False)
    figure.colorbar(surface)
    # pyplot.show()
    if not os.path.exists(path):
        os.makedirs(path)
    filename = path + "/" + title + ".pdf"
    figure.savefig(filename, dpi=figure.dpi)
    pyplot.close(figure)
    # print("done!")


# plot additional statistics

def plot_statistics_json(path_dir, filename="statistics.json"):
    # plots from statistics.json
    statistics = read_dict_from_json(path_dir, filename)
    # win/lost vs. episode number
    for key in statistics.keys():
        key_elements = key.split("_")
        n_battles = key_elements[0]
        n0 = key_elements[1]
        alpha = key_elements[2]
        gamma = key_elements[3]
        value = statistics[key]
        plot_2d(path="./Q_Learning_FA_plot",
                title="acc_victories_n_battles_" + n_battles + "_N0_" + n0 + "_alpha_" + alpha + "_gamma_" + gamma,
                x_label="episodes",
                x_array=np.array(range(0, len(value))),
                y_label="acc victory",
                y_array=np.cumsum(value).astype(np.int))

    # winning % by set of parameters
    n_battles = ""
    alpha = ""
    x_values = []
    y_values = []
    z_values = []
    for key in statistics.keys():
        key_elements = key.split("_")
        n_battles = key_elements[0]
        n0 = key_elements[1]
        alpha = key_elements[2]
        gamma = key_elements[3]
        value = statistics[key]
        x_values.append(n0)
        y_values.append(gamma)
        z_values.append(value.count(True) / len(value))
    plot_3d(path="./Q_Learning_FA_plot",
            title="winning_percentage_n_battles_" + n_battles + "_alpha_" + alpha,
            x_label="N0",
            x_array=np.array(x_values).astype(np.float),
            y_label="gamma",
            y_array=np.array(y_values).astype(np.float),
            z_label="winning %",
            z_array=np.array(z_values))


# plots from statistics
plot_statistics_json("./Q_Learning_FA_statistics")


# validation

async def do_battle_validation(path_dir):
    # read from json
    for filename in os.listdir(path_dir):
        # learned feature vector
        w = np.array(read_array_from_json(path_dir, filename))
        # params: n_battles, n0, gamma
        params = filename.split("_")
        n_battles = int(params[2])
        n0 = float(params[3])
        alpha = float(params[4])
        gamma = float(params[5])

        # validation (play 1/3 of the battles using Q-learned table)
        start = time.time()
        validation_player = ValidationPlayer(battle_format="gen8ou", team=OUR_TEAM, w=w)
        opponent = MaxDamagePlayer(battle_format="gen8ou", team=OP_TEAM)
        n_battles_validation = int(n_battles / 3)
        await validation_player.battle_against(opponent=opponent, n_battles=n_battles_validation)
        print("validation: num battles (episodes)=%d, N0=%.4f, alpha=%.2f, gamma=%.2f, wins=%d, winning %%=%.2f, total time=%s sec" %
              (
                  n_battles_validation,
                  n0,
                  alpha,
                  gamma,
                  validation_player.n_won_battles,
                  round((validation_player.n_won_battles / n_battles_validation) * 100, 2),
                  round(time.time() - start, 2)
              ))


if use_validation:
    loop = asyncio.get_event_loop()
    loop.run_until_complete(loop.create_task(do_battle_validation("./Q_Learning_FA_w")))