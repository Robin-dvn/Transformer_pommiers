from markov import Markov, MarkovModel
from tqdm import tqdm
from sequences import generate_sequence,terminal_fate,_generate_random_draw_sequence
import os
import pathlib
import random
from typing import Dict, Tuple
import numpy as np
import toml
import io
from enums import Observation
from itertools import product
import pandas as pd
from datetime import datetime
from time import time
from colorama import Fore,Style
from HSMM import HSMM

def get_shared_data_path(path: str) -> str:
    return str((pathlib.Path(__file__).parent / "./data" / path).resolve())

class DatasetCreator():


    _markov: Markov
    _markov_models: Dict[Tuple[str, int], MarkovModel] 
    _output_path: pathlib.Path
    _number_samples_per_model: int
    _seed: int
    year_no:int
    _rng: np.random.Generator
    dataset: pd.DataFrame
 

    def __init__(self, outpath,seed,min_length,max_length,nb_samples_per_model) -> None:
        self.year_no = 0
        self._number_samples_per_model = nb_samples_per_model
        self._output_path = pathlib.Path(outpath or os.getcwd() + "/output")
        self.dataset = pd.DataFrame()
        self.tokenised_dataset = pd.DataFrame()
        self._seed = seed
        self._rng = np.random.default_rng(0)
        random.seed(self._seed)

        self._markov = Markov(
            generator=self._rng,
            minimum_length=min_length,
            maximum_length=max_length,
        )
        
        self._markov_models = {}
        for path in os.listdir(get_shared_data_path("markov")):
            path = pathlib.Path(get_shared_data_path("markov")) / path
            if path.is_file() and path.suffix == ".toml":
                with io.open(path) as file:
                    model = MarkovModel(**toml.loads(file.read()))
                    self._markov_models[(model.length, model.year)] = model
    
    
    def _set_markov_model(self):
        if self.year_no == 0:

            self._markov.set_models(
                medium=self._markov_models[("MEDIUM", 3)],
                long=self._markov_models[("LONG", 1)],
            )   
        if self.year_no == 1:
            self._markov.set_models(
                medium=self._markov_models[("MEDIUM", 3)],
                long=self._markov_models[("LONG", 1)],
            )

        elif self.year_no == 2:
            self._markov.set_models(
                medium=self._markov_models[("MEDIUM", 3)],
                long=self._markov_models[("LONG", 3)],
            )

        elif self.year_no == 3:
            self._markov.set_models(
                medium=self._markov_models[("MEDIUM", 4)],
                long=self._markov_models[("LONG", 4)],
            )

        else:
            self._markov.set_models(
                medium=self._markov_models[("MEDIUM", 5)],
                long=self._markov_models[("LONG", 5)],
            )

    def create_data(self,to_CSV = False,rewrite = True):
        usedObservations =  [
            Observation.SMALL,
            Observation.FLORAL,
            Observation.LARGE,
            Observation.MEDIUM
        ]
        start_time =time() 
        dataset = []
        for obs in usedObservations:
            print(Fore.CYAN + f"🔹 Début de la génération pour le type {obs}" + Style.RESET_ALL)
            
            for year in range(5):
                print(Fore.GREEN + f"🕒 Début de l'année {year+1}" + Style.RESET_ALL)
                self.year_no = year
                self._set_markov_model()
                print(year,"   ",self._markov._long.year)
                for _ in tqdm(range(self._number_samples_per_model)):
                    terminal = terminal_fate(self.year_no, obs) if obs != Observation.FLORAL else Observation.DORMANT
                    sequence = ''.join([str(t[1]) for t in generate_sequence(obs, self._markov, self.year_no, True, select_trunk=0)]) 
                    
                    sample = {
                        "Observation": obs.value,
                        "Year": "Y"+str(year + 1),
                        "Sequence": sequence,
                        "Terminal Fate": terminal
                    }

                    dataset.append(sample)

        end_time = time()
        print(Fore.YELLOW + f"⚡ [INFO] Temps total : {end_time - start_time:.2f} secondes" + Style.RESET_ALL)

        self.dataset = pd.DataFrame(dataset)

        print(Fore.MAGENTA + "📂 [INFO] Conversion en CSV..." + Style.RESET_ALL)
        if to_CSV:
            self._output_path.mkdir(exist_ok=True)
            path = self._output_path/"dataset100.csv" if rewrite else self._output_path/f'dataset_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv'
            self.dataset.to_csv(path,index=False)

    def load_data(self,path):        
        self.dataset = pd.read_csv(path)


class DatasetCreatorCustomHSMM():
    def __init__(self,outpath,nb_samples_per_model,min_length,max_length) -> None:
        self.year_no = 0
        self._number_samples_per_model = nb_samples_per_model
        self.max_length = max_length
        self.min_length = min_length
        self.dataset = pd.DataFrame()
        self._output_path = pathlib.Path(outpath or os.getcwd() + "/output")
        self.mappings = {
            1:{Observation.LARGE: "data/markov/fuji_long_year_1.toml", Observation.MEDIUM: "data/markov/fuji_medium_year_3.toml"},
            2:{Observation.LARGE: "data/markov/fuji_long_year_2.toml", Observation.MEDIUM: "data/markov/fuji_medium_year_3.toml"},
            3:{Observation.LARGE: "data/markov/fuji_long_year_3.toml", Observation.MEDIUM: "data/markov/fuji_medium_year_3.toml"},
            4:{Observation.LARGE: "data/markov/fuji_long_year_4.toml", Observation.MEDIUM: "data/markov/fuji_medium_year_4.toml"},
            5:{Observation.LARGE: "data/markov/fuji_long_year_5.toml", Observation.MEDIUM: "data/markov/fuji_medium_year_5.toml"}
        }
        self.starting_states = [
            Observation.SMALL,
            Observation.FLORAL,
            Observation.LARGE,
            Observation.MEDIUM,
        ]
    def generate_seq(self,starting_state,year,hsmm =None):
        if starting_state == Observation.FLORAL:
            return [0,0,0,0]
        if starting_state == Observation.SMALL:
            return [0,0,0,0]
        return hsmm.generate_sequence(self.min_length, self.max_length)[1]

    def create_data(self,to_CSV = False,rewrite = True):
        start_time = time()
        dataset = []
        for starting_state in self.starting_states:
            for year in range(1,6):
                
                    
                print(Fore.CYAN + f"🔹 Début de la génération pour le type {starting_state} en année {year}" + Style.RESET_ALL)
         
                hsmm_model = HSMM(self.mappings[year][starting_state]) if starting_state != Observation.FLORAL and starting_state != Observation.SMALL else None

                for _ in tqdm(range(self._number_samples_per_model)):
                    terminal = terminal_fate(year, starting_state) if starting_state != Observation.FLORAL else Observation.DORMANT
                    seq =  _generate_random_draw_sequence() if year == 2 and starting_state == Observation.LARGE else  self.generate_seq(starting_state,year,hsmm_model)
                    seq = [el[1] for el in seq] if year == 2 and starting_state == Observation.LARGE else seq
                    sample = {
                        "Observation": starting_state.value,
                        "Year": "Y"+str(year),
                        "Sequence": ''.join([str(obs) for obs in seq]),
                        "Terminal Fate": terminal
                    }
                    self.dataset = dataset.append(sample)
        
        
        end_time = time()
        print(Fore.YELLOW + f"⚡ [INFO] Temps total : {end_time - start_time:.2f} secondes" + Style.RESET_ALL)

        self.dataset = pd.DataFrame(dataset)

        print(Fore.MAGENTA + "📂 [INFO] Conversion en CSV..." + Style.RESET_ALL)
        if to_CSV:
            self._output_path.mkdir(exist_ok=True)
            path = self._output_path/f"datasetcustom{self._number_samples_per_model}.csv" if rewrite else self._output_path/f'dataset_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv'
            self.dataset.to_csv(path,index=False)

if __name__ == "__main__":
    datasetcreator = DatasetCreatorCustomHSMM("out/",10000,4,70)
    datasetcreator.create_data(True)