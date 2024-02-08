# Code for training and testing with REINFORCE in Pokémon Showdown

import argparse
import asyncio
import os
from distutils.util import strtobool

import neptune.new as neptune
import nest_asyncio
import numpy as np
import tensorflow as tf
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player.battle_order import ForfeitBattleOrder
from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.player.random_player import RandomPlayer

from REINFORCEAgent import REINFORCEAgent

# Definition of the agent stochastic team (Pokémon Showdown template)
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

# Definition of the opponent stochastic team (Pokémon Showdown template)
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

# Encoding stochastic Pokémon Name for ID
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

# Definition of the agent deterministic team (Pokémon Showdown template)
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

# Definition of the opponent deterministic team (Pokémon Showdown template)
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

# Encoding deterministic Pokémon Name for ID
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
    parser.add_argument('--debug', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, debug will be enabled')
    parser.add_argument('--neptune', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='if toggled, neptune will be enabled')
    parser.add_argument('--train', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, train will be realized')
    parser.add_argument('--saved', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='if toggled, use saved trained model will be realized')
    parser.add_argument('--model-folder', type=str, default="/model",
                        help='folder of trained model (just for validation)')
    parser.add_argument('--env', type=str, default="stochastic",
                        help='type of environment (stochastic or deterministic). Define teams. OBS: must change showdown too.')

    # Agent parameters
    parser.add_argument('--hidden', type=int, default=128,
                        help="Hidden layers applied on our nn")
    parser.add_argument('--gamma', type=float, default=0.99,
                        help="gamma value used on DQN")
    parser.add_argument('--adamlr', type=float, default=0.0005,
                        help="learning rate used on Adam")

    # Training parameters
    parser.add_argument('--battles', type=int, default=10000,
                        help="n steps per epoch")

    return parser.parse_args()


nest_asyncio.apply()
np.random.seed(0)

# Definition of REINFORCE player
class MCPlayer(Gen8EnvSinglePlayer):
    def __init__(self, battle_format, team, agent):
        super().__init__(battle_format=battle_format, team=team)
        self.agent = agent
        self.state = None
        self.action = None
        self.reward = 0
        self.num_battles = 0

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
        self.action = self.agent.get_action(self.state)
        # store the (state, action, reward) tuple into the agent
        self.agent.add_sar(self.state, self.action, self.reward)
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
        self.agent.fit()

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

# Definition of REINFORCE validation player
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
        self.action = self.agent.get_action(self.state)
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


tf.random.set_seed(0)
np.random.seed(0)

# Main program
if __name__ == "__main__":

    args = parse_args()

    if args.neptune:
        run = neptune.init(project='your_project',
                           api_token='your_api_token==',
                           tags=["REINFORCE", args.exp_name, args.env, str(args.battles) + "episodes"])

    # environment: stochastic or deterministic
    if args.env == "deterministic":
        our_team = OUR_TEAM_DET
        op_team = OP_TEAM_DET
    else:
        our_team = OUR_TEAM
        op_team = OP_TEAM

    # extract the args
    N_STATE_COMPONENTS = 12
    N_OUR_MOVE_ACTIONS = 4
    N_OUR_SWITCH_ACTIONS = 5
    N_OUR_ACTIONS = N_OUR_MOVE_ACTIONS + N_OUR_SWITCH_ACTIONS
    NB_HIDDEN = args.hidden
    ALPHA = args.adamlr
    GAMMA = args.gamma
    NB_TRAINING_EPISODES = args.battles
    NB_EVALUATION_EPISODES = int(NB_TRAINING_EPISODES / 3)

    # training
    async def do_battle_training():
        # REINFORCE agent
        reinforce = REINFORCEAgent(
            nb_states=N_STATE_COMPONENTS,
            nb_actions=N_OUR_ACTIONS,
            nb_hidden=NB_HIDDEN,
            alpha=ALPHA,
            gamma=GAMMA
        )
        # our player
        player = MCPlayer(battle_format="gen8ou", team=our_team, agent=reinforce)
        # opponent's player
        opponent = MaxDamagePlayer(battle_format="gen8ou", team=op_team)
        await player.battle_against(opponent=opponent, n_battles=NB_TRAINING_EPISODES)
        # logs
        if args.debug:
            print("training: num battles (episodes)=%d, wins=%d, winning %%=%.2f" %
                  (
                      NB_TRAINING_EPISODES,
                      player.n_won_battles,
                      round((player.n_won_battles / NB_TRAINING_EPISODES) * 100, 2)
                  ))
        return reinforce

    reinforce_agent = None
    if args.train:
        loop = asyncio.get_event_loop()
        reinforce_agent = loop.run_until_complete(loop.create_task(do_battle_training()))
    elif args.saved:
        reinforce_agent = REINFORCEAgent(
            nb_states=N_STATE_COMPONENTS,
            nb_actions=N_OUR_ACTIONS,
            nb_hidden=NB_HIDDEN
        )
        reinforce_agent.load_model(args.model_folder)
    if reinforce_agent is None:
        exit("Error: no agent")

    # evaluation
    async def do_battle_validation(env_player_mode):
        # REINFORCE agent validation mode (no learning)
        # our player
        player = ValidationPlayer(
            battle_format="gen8ou",
            team=our_team,
            agent=reinforce_agent,
            env_player_mode=env_player_mode
        )
        if env_player_mode == "val_max":
            print("\nResults against max player:")
            opponent = MaxDamagePlayer(battle_format="gen8ou", team=op_team)
        else:
            print("\nResults against random player:")
            opponent = RandomPlayer(battle_format="gen8ou", team=op_team)
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
