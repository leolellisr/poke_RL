
import numpy as np
from src.playerMC import Player as PlayerMC
from poke_env.player.player import Player 
from poke_env.player.random_player import RandomPlayer

from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.data import GenData

import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.append('../')

from py.variables.utils import name_to_id_sto as name_to_id

# Definition of Monte Carlo Player

class MCPlayer(PlayerMC):
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
        

    # the embed battle is our state
    # 10 factors: 4 moves base power, 4 moves multipliers, active pokemon and active opponent pokemon 
    def embed_battle(self, battle):
        # -1 indicates that the move does not have a base power or not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        type_chart = GenData(9).type_chart
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling 
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    type_chart=type_chart
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
        return str(state)

        

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
        return self.reward_computing_helper(            
            battle, fainted_value=2, hp_value=1, status_value=1, victory_value=15
        )

    # Battle (episode) has finished
    def _battle_finished_callback(self, battle):
        rewards = [reward for state,action,reward in self.episode]
        states = [state for state,action,reward in self.episode]
        actions = [action for state,action,reward in self.episode]
        t_array = range(len(self.episode)+1)
        # Computing Q and N
        for idx, state in enumerate(states):
            action = actions[idx]
            if state not in self.visited_states:
                self.N[state][action] += 1
                returnGt = sum([reward*pow(self.gamma, t) for reward, t in zip(rewards[idx:], t_array[:(-idx-1)])]) 

                # incremental update of Q value is more efficient than keeping a record of all rewards
                # and averaging after every new reward
                # step-size: 1./N[state][action]
                
                self.Q[state][action] += (1/self.N[state][action])*(returnGt-self.Q[state][action]) 
                self.visited_states.append(state)
                
        self.visited_states = []
        self.episode = []
        if(self.aux == 1):
            self.aux == 0
            
        # Define new policy with updated Q and N
        self.policy = self.update_epsilon_greedy_policy(self.Q, self.n0, self.N)
        