Code repository with classical reinforcement learning and deep reinforcement learning methods using [poke_env environment](poke-env.readthedocs.io/en/latest/).

# Tabular Methods and Function Approximation implemented
* Monte Carlo Control First-Visit;
* Function Approximation with Monte Carlo Control First-Visit;
* Q-Learning;
* Function Approximation with Q-Learning;
* SARSA($\lambda$)
* Function Approximation with SARSA($\lambda$)


# Deep Reinforcement Learning Methods implemented
* [DQN - Keras 2018](https://notebooks.githubusercontent.com/view/github.com/keras-rl/keras-rl/blob/master/rl/agents/dqn.py)
* [Double-DQN - Keras 2018](https://notebooks.githubusercontent.com/view/github.com/keras-rl/keras-rl/blob/master/rl/agents/dqn.py)
* [PPO - Stable Baselines 2021](https://notebooks.githubusercontent.com/view/github.com/Stable-Baselines-Team/stable-baselines/blob/master/stable_baselines/ppo2/ppo2.py)
* REINFORCE - Keras 2018
* DQN - Pytorch
* REINFORCE - Pytorch

# Requirements

It is necessary to install the requirements available at [requirements_poke_env.yml](https://github.com/leolellisr/poke_RL/blob/master/requirements_poke_env.yml).
To use PPO, it is necessary to install the requirements available at [requirements_poke_env_ppo.yml](https://github.com/leolellisr/poke_RL/blob/master/requirements_poke_env_ppo.yml).

For training, it is necessary to run [Pokémon Showdown](https://play.pokemonshowdown.com) on localhost. Showdown is [open-source](https://github.com/smogon/pokemon-showdown.git).

# Benchmark

Graphics available at:
* [Graphs MC Control, MC Control FA, Q-Learning, Q-Learning FA, SARSA($\lambda$) Deterministic, SARSA($\lambda$) FA, DQN and Double-DQN](https://app.neptune.ai/leolellisr/rl-pokeenv)
* [Graphs SARSA($\lambda$) Stochastic](https://app.neptune.ai/mauricioplopes/poke-env)
* [Graphs PPO](https://github.com/leolellisr/poke_RL/tree/master/images/report/ppo_results)
 
Notebooks of implemented Tabular Methods and Function Approximation Methods available [here](https://github.com/leolellisr/poke_RL/tree/master/notebooks)

Python files of implemented methods available [here](https://github.com/leolellisr/poke_RL/tree/master/py)

json files of trained methods are available [here](https://drive.google.com/drive/folders/1GwNQSsOR0PPtPKlbIy9NzWvTnTtMOD8_?usp=sharing)

Trained models of deep reinforcement learning methods (DQN-Keras; Double DQN-Keras; PPO2-Stable Baselines) available [here](https://drive.google.com/drive/folders/17_Gn1RWOCh-ekiRhhj40ehPz9nv_d-Fy?usp=sharing)

# Goal and Motivation

* This project aims to employ different reinforcement learning techniques to train agents in a Pokémon battle simulator;

* Our motivation is that the trainer automatically learns, by making decisions through the analysis of states and rewards related to their performance, how to win battles throughout the episodes, noticing:
  * the different types between Pokémon;
  * which moves cause more damage to the opponent's Pokémon;
  * what are the possible strategies using no-damage moves;
  * and the best times to switch Pokémon. 
  * Move's effects: Some moves have [additional effects](https://bulbapedia.bulbagarden.net/wiki/Additional_effect). e.g.: Iron Head have 30% chance of flinching the target (target cannot move in the turn).

#  **The  problem addressed**
* [Pokémon](https://www.pokemon.com) is a popular Japanese RPG (Role Playing Game) which stands a world championship every year; 
* One single [battle](https://bulbapedia.bulbagarden.net/wiki/Pokémon_battle) of Pokémon has two players. Each player has a 6-Pokémon team; 
* Each Pokémon has:
  * 6 [stats](https://bulbapedia.bulbagarden.net/wiki/Stat) (Health Points, Attack, Defense, Special Attack, Special Defense, Speed). The first 5 are used in the damage calculation. The speed defined which Pokémon moves first in the turn.
    * The Health Points goes from 100% (healthy) to 0% (fainted);
  * 4 possible moves (each with a limited number of uses);
  * one [ability](https://bulbapedia.bulbagarden.net/wiki/Ability) that has special effects in the field;
  * one [nature](https://bulbapedia.bulbagarden.net/wiki/Nature) that specifies which stats are higher and which are lower;
  * one [item](https://bulbapedia.bulbagarden.net/wiki/Item), that can  restore Health Points or increase the Power of an Attack.
* The winner of the battle is the player that makes all Pokémon of the oposing team to faint (all oposing Pokémon with health points equals zero, "last man standing" criteria);
* Only one Pokémon of each team can be at the battle field at the same time;
* Every turn, each players select one action: one of the 4 moves of their active Pokémon or [switching](https://bulbapedia.bulbagarden.net/wiki/Recall) for one of other non-fainted Pokémon of their team;

* Pokémon can be summarized as an analyze state (turn) -> take action sequence game. 

* By standard, Pokémon is a stochastic game:
  * One move can have an accuracy value less than 100%, then this move has a probability to be missed;
  *   * Some moves have [additional effects](https://bulbapedia.bulbagarden.net/wiki/Additional_effect). e.g.: Iron Head have 30% chance of flinching the target (target cannot move in the turn);
  * The damage moves (attacks) have the following [damage calculation](https://bulbapedia.bulbagarden.net/wiki/Damage):
  ![Damage](https://wikimedia.org/api/rest_v1/media/math/render/svg/b8c51fed93bb9a80ae8febc13700a40b8a5da402)

 where:
  *  **[Level](https://bulbapedia.bulbagarden.net/wiki/Level)** (the level of the attacking Pokémon);
  *  **A** is the effective Attack stat of the attacking Pokémon if the used move is a physical move, or the effective Special Attack stat of the attacking Pokémon if the used move is a special move;
  *  **D** is the effective Defense stat of the target if the used move is a physical move or a special move that uses the target's Defense stat, or the effective Special Defense of the target if the used move is an other special move;
  *  **[Power](https://bulbapedia.bulbagarden.net/wiki/Power)** is the effective power of the used move;
  *  **Weather** is 1.5 if a Water-type move is being used during rain or a Fire-type move during harsh sunlight, and 0.5 if a Water-type move is used during harsh sunlight or a Fire-type move during rain, and 1 otherwise.
  *  **[Critical](https://bulbapedia.bulbagarden.net/wiki/Critical_hit)** has 6.25% chance of occurs and multiplies the damage by 1.5;
  *  **random** is a random factor between 0.85 and 1.00 (inclusive):
  *  **[STAB](https://bulbapedia.bulbagarden.net/wiki/Same-type_attack_bonus)** is the same-type attack bonus. This is equal to 1.5 if the move's type matches any of the user's types, 2 if the user of the move additionally has the ability Adaptability, and 1 if otherwise;
  *  **[Type](https://bulbapedia.bulbagarden.net/wiki/Type)** is the type effectiveness. This can be 0 (ineffective); 0.25, 0.5 (not very effective); 1 (normally effective); 2, or 4 (super effective), depending on both the move's and target's types;
  *  **[Burn](https://bulbapedia.bulbagarden.net/wiki/Burn_(status_condition))** is 0.5 (from Generation III onward) if the attacker is burned, its Ability is not Guts, and the used move is a physical move (other than Facade from Generation VI onward), and 1 otherwise.
  *  **other** is 1 in most cases, and a different multiplier when specific interactions of moves, Abilities, or items take effect. In this work, this is applied just to Pokémon that has the item  **Life Orb**, which multiplies the damage by 1.3.
  
  * **Not** used in this work (equals 1):
    * Targets (for Battles with more than two active Pokémon in the field);
    * Badge ( just applied in Generation II);
   
   #  **MDP formulation and discretization model** 

## Original (stochastic)

We considered our original (stochastic) MDP as a tuple M = (S, A, phi, R), where:
* **S** is the whole set of possible states. One state  **s in S**  is defined at each turn with 12 battle elements concatenated, that correspond to:
  * [0] Our Active Pokémon index (0: Venusaur,  1: Pikachu, 2: Tauros, 3: Sirfetch'd, 4: Blastoise, 5: Charizard);
  * [1] Opponent Active Pokémon index (0: Eevee,  1: Vaporeon, 2: Leafeon, 3: Sylveon, 4: Jolteon, 5: Umbreon);
  * [2-5] Active Pokémon moves base power (if a move doesn't have base power, default to -1);
  * [6-9] Active Pokémon moves damage multipliers;
  * [10] Our remaining Pokémon;
  * [11] Opponent remaining Pokémon.
 
* **A** is the whole set of possible actions. Our action space is a range [0, 8]. One action  **a in A** is one of the possible choices:
  * [0] 1st Active Pokémon move;
  * [1] 2nd Active Pokémon move;
  * [2] 3rd Active Pokémon move;
  * [3] 4th Active Pokémon move;
  * [4] Switch to 1st next Pokémon;
  * [5] Switch to 2nd next Pokémon;
  * [6] Switch to 3rd next Pokémon;
  * [7] Switch to 4th next Pokémon;
  * [8] Switch to 5th next Pokémon.

When a selected action cannot be executed, we random select another possible action.

* **phi** is a stochastic transition function that occurs from state  **s** to state  **s'**, by taking an action  **a**. The following parameters are part of our  stochastic transition function:
  * Move's accuracy (chance of the move successfully occurs or to fail);
  * Damage calculation: The  *[Critical](https://bulbapedia.bulbagarden.net/wiki/Critical_hit) * parameter (6.25% chance of occurs) and the  *random * parameter, ranging from 0.85 and 1.00 (inclusive).
 * Move's effects: Some moves have [additional effects](https://bulbapedia.bulbagarden.net/wiki/Additional_effect). e.g.: Iron Head have 30% chance of flinching the target (target cannot move in the turn).

* **R** is a set of rewards. A reward  **r in R** is acquired in state  **s** by taking the action  **a**. The rewards are calculated at the end of the turn. The value of reward  **r** is defined by:
  * +Our Active Pokémon current Health Points;
  * -2 if our Active Pokémon fainted;
  * -1 if our Active Pokémon have a [negative status condition](https://bulbapedia.bulbagarden.net/wiki/Status_condition);
  * +Number of remaining Pokémon in our team;
  * -Opponent Active Pokémon current Health Points;
  * +2 if opponent Active Pokémon fainted;
  * +1 if opponent Active Pokémon have a [negative status condition](https://bulbapedia.bulbagarden.net/wiki/Status_condition);
  * -Number of remaining Pokémon in opponent team;
  * +15 if we won the battle;
  * -15 if we lost the battle.
 
### Stochastic Team

Our stochastic team, with each Pokémon, their abilities, natures, items, moves (base power and accuracy) and possible switches are shown in [Team](https://imgur.com/KSXvlmO).

The stochastic opponent team, with each Pokémon, their abilities, natures, items, moves (base power and accuracy) and possible switches are shown in [Opponent Team](https://imgur.com/rLF5Cli).

## Deterministic

To adapt Pokémon to a deterministic environment, we use Pokémon that cannot receive a critical hit, moves with only 100% accuracy and edit the server code to ignore the random parameter in damage calculation, removing the stochastic transition function \phi from our MDP. Therefore, now our MDP is a tuple M = (S, A, R), where:
* **S** is the whole set of possible states. One state  **s in S**  is defined at each turn with 12 battle elements concatenated, that correspond to:
  * [0] Our Active Pokémon index ;
  * [1] Opponent Active Pokémon index ;
  * [2-5] Active Pokémon moves base power (if a move doesn't have base power, default to -1);
  * [6-9] Active Pokémon moves damage multipliers;
  * [10] Our remaining Pokémon;
  * [11] Opponent remaining Pokémon.
 
* **A** is the whole set of possible actions. Our action space is a range [0, 8] (len: 9). One action  **a in A** is one of the possible choices:
  * [0] 1st Active Pokémon move;
  * [1] 2nd Active Pokémon move;
  * [2] 3rd Active Pokémon move;
  * [3] 4th Active Pokémon move;
  * [4] Switch to 1st next Pokémon;
  * [5] Switch to 2nd next Pokémon;
  * [6] Switch to 3rd next Pokémon;
  * [7] Switch to 4th next Pokémon;
  * [8] Switch to 5th next Pokémon.

When a selected action cannot be executed, we random select another possible action.

* **R*** is a set of rewards. A reward  **r in R** is acquired in state  **s** by taking the action  **a**. The rewards are calculated at the end of each turn. The value of reward  **r** is defined by:
  * +Our Active Pokémon current Health Points;
  * -2 if our Active Pokémon fainted;
  * -1 if our Active Pokémon have a [negative status condition](https://bulbapedia.bulbagarden.net/wiki/Status_condition);
  * +Number of remaining Pokémon in our team;
  * -Opponent Active Pokémon current Health Points;
  * +2 if opponent Active Pokémon fainted;
  * +1 if opponent Active Pokémon have a [negative status condition](https://bulbapedia.bulbagarden.net/wiki/Status_condition);
  * -Number of remaining Pokémon in opponent team;
  * +15 if we won the battle;
  * -15 if we lost the battle.
 
### Deterministic Team

Our deterministic team, with each Pokémon, their abilities, natures, items, moves (base power and accuracy) and possible switches are shown in [Team](https://imgur.com/DeRAEQb).

The deterministic opponent team, with each Pokémon, their abilities, natures, items, moves (base power and accuracy) and possible switches are shown in [Opponent Team](https://imgur.com/Hltn5OO).

We use on both teams only Pokémon with Battle Armor or Shell Armor abilities, which prevent critical hits from being performed. Also, we use in both teams only moves with 100% accuracy, with no chance of getting it missed, and the move haven't additional effects.

## Search space

The features that integrate our states are shown in [this figure](https://imgur.com/tREjWCG). For a single battle between two players with 6 Pokémon each, we have $1.016.064$ possible states.

Since we have 9 possible actions for each Pokémon, we total $9.144.576$ possibilities for each battle.

#  **The environments built**

The environment used is [Pokémon Showdown](https://play.pokemonshowdown.com), a [open-source](https://github.com/smogon/pokemon-showdown.git) Pokémon battle simulator.

[Example](https://imgur.com/hjHikuc) of one battle in Pokémon Showdown.

To communicate our agents with Pokémon Showdown we used [poke-env](https://poke-env.readthedocs.io/en/latest/) a Python environment for interacting in pokemon showdown battles.

We used separated Python classes for define the Players that are trained with each method. These classes communicates with Pokémon Showdown and implements the poke-env methods to:
* Create battles;
* Accept battles;
* Send orders (actions) to Pokémon Showdown.

## Original (stochastic)

To speed up the battles, we hosted our own server of Pokemon Showdown in localhost. It requires Node.js v10+.

## Deterministic environment

To adapt our environment to a deterministic setup, we had to establish the following parameters:
* We removed the random component of sim/battle.ts from the Pokémon Showdown simulator;
* We use on both teams only Pokémon with Battle Armor or Shell Armor abilities, which prevent critical hits from being performed;
* We used in both teams only moves with 100% accuracy, with no chance of getting it missed;
* We didn't use any move with additional effect. 

#  **Characteristics of  the problem**

* Both of our environments (stochastic and deterministic) are episodic. One state occurs after another;

* Our terminal states are:
  * When all our Pokémon are fainted (we lose);
  * When all opponent Pokémon are fainted (we won).

* As specified before, a reward  **r** is calculated at the end of a turn. The value of reward  **r** is defined by:
  * +Our Active Pokémon current Health Points;
  * -2 if our Active Pokémon fainted;
  * -1 if our Active Pokémon have a [negative status condition](https://bulbapedia.bulbagarden.net/wiki/Status_condition);
  * +Number of remaining Pokémon in our team;
  * -Opponent Active Pokémon current Health Points;
  * +2 if opponent Active Pokémon fainted;
  * +1 if opponent Active Pokémon have a [negative status condition](https://bulbapedia.bulbagarden.net/wiki/Status_condition);
  * -Number of remaining Pokémon in opponent team;
  * +15 if we won the battle;
  * -15 if we lost the battle.


# Comparisons

One table with performance comparisons in Validation between our Tabular Methods and Function Approximation implemented and some methods proposed in literature are showed in [this figure](https://imgur.com/vwUOnDZ).

One table with performance comparisons in Validation between our Deep Reinforcement Learning methods implemented, our best performance method from P1 and one method proposed in literature are showed in [this figure](https://imgur.com/4A4X54z).

The Function Approximation method with the best performance against both Players (MaxDamagePlayer and RandomPlayer) was Q-Learning Function Approximation in the Stochastic environment.

The Deep-RL method with the best performance in the Stochastic environment against both Players (MaxDamagePlayer and RandomPlayer) was Proximal Policy Optimization (PPO2) - Stable Baselines (2021) trained by 300k steps. In the Deterministic environment, against both Players (MaxDamagePlayer and RandomPlayer),  the best performance was Proximal Policy Optimization (PPO2) - Stable Baselines (2021) trained by 900k steps.

# Limitations

The only limitations of our project are in the use of the **deterministic** environment. Given the need to remove **randomness**, our deterministic solutions require the use of modified Pokémon Showdown, without the random parameter in damage calculation, and the Pokémon on both teams with:
- Shell Armor or Battle Armor abilities, to prevent critical hits;
- Moves with 100% accuracy and no side effects likely to occur.

Our **stochastic** solutions can be applied to any case and with any team formation.

We also found some libraries limitations in the use of PPO2, from Stable Baselines (2021). The requirements listed [here](https://github.com/leolellisr/poke_RL/blob/master/requirements_poke_env_ppo.yml) must be used. Neptune cannot be used.


# Saving and Loading models

## Tabular models

### Saving Tabular models

The tabular models weights are saved as json. 

We use the following function to save the model:
  
```# helper function: save to json file
def save_to_json(path, params, name, value):
    today_s = str(date.today())
    n_battle_s = str(params['n_battles'])
    ... 
    if not os.path.exists(path):
        os.makedirs(path)
    filename = path + "/" + name  today_s + "_n_battles_" + n_battle_s + ...
    filename=re.sub(r'(?<=\d)[,\.-]','',filename)+".json"
    file = open(filename, "w")
    value_dict = dict()
    for key in value:
        value_dict[key] = value[key].tolist()
    json.dump(value_dict, file)
    file.close()
```
The function is used at async function do_battle_training: `save_array_to_json("./name", filename, params['player'])`.


### Loading Tabular models

After defining your model, you can load the weights with the functions:

```def read_array_from_json(path_dir, filename):
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
    return data`

```

We use the functions at the function do_battle_validation:

` w = np.array(read_array_from_json(path_dir, filename))`

## Deep Reinforcement Learning models

## Saving DRL models

The models were saved in .pb with the function `model.save()`

[Keras reference](https://www.tensorflow.org/tutorials/keras/save_and_load?hl=pt-br)

## Loading DRL models

The SavedModel format is a way to serialize models. Models saved in this format can be restored using tf.keras.models.load_model and are compatible with TensorFlow Serving. 

```# Create and train a new model instance.
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# Save the entire model as a SavedModel.
!mkdir -p saved_model
model.save('saved_model/my_model')

```

The SavedModel format is a directory containing a protobuf binary and a TensorFlow checkpoint.

Inspect the saved model directory and reload a fresh Keras model from the saved model:

```# my_model directory
ls saved_model

# Contains an assets folder, saved_model.pb, and variables folder.
ls saved_model/my_model

new_model = tf.keras.models.load_model('saved_model/my_model')

# Check its architecture
new_model.summary()```
