import time
import sys
import os
from datetime import date
today = date.today()
import json
import re
import numpy as np

# Encoding Pokémon Name for ID
def name_to_id_sto(name):
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



# helper function: save to json file
def save_to_json(path, params, name, value):
    today_s = str(date.today())
    n_battle_s = str(params['n_battles'])
    n0_s = str(round(params['n0'], 8))
    gamma_s = str(round(params['gamma'], 8))
    winning_percentage_s = str(round((params['player'].n_won_battles / params['n_battles']) * 100, 2))
    if not os.path.exists(path):
        os.makedirs(path)
    filename = path + "/" + name + today_s + "_n_battles_" + n_battle_s + "_N0_" + n0_s + "_gamma_" + gamma_s + "_wining_" + winning_percentage_s
    filename=re.sub(r'(?<=\d)[,\.-]','',filename)+".json"
    file = open(filename, "w")
    value_dict = dict()
    for key in value:
        value_dict[key] = value[key].tolist()
    json.dump(value_dict, file)
    file.close()    


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



def save_to_json_file(name, params):
    today_s = str(date.today())
    n_battle_s = str(params['n_battles'])
    n0_s = str(round(params['n0'], 4))
    gamma_s = str(round(params['gamma'], 4))
    winning_percentage_s = str(round((params['player'].n_won_battles / params['n_battles']) * 100, 2))

    filename = "W_" + today_s + "_" + n_battle_s + "_" + n0_s + "_" + gamma_s + "_" + winning_percentage_s+".json"  
    save_array_to_json("./"+name+"_w", filename, params['player'].w)
            
    filename = "N_" + today_s + "_" + n_battle_s + "_" + n0_s + "_" + gamma_s + "_" + winning_percentage_s+".json"           
    save_dict_to_json("./"+name+"_N", filename, params['player'].N)
    # statistics: key: "n_battles, n0, alpha, gamma", values: list of win or lose
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
    save_dict_to_json("./"+name+"_statistics", "statistics.json", data)