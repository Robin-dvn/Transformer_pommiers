from markov import Markov, MarkovModel
from tqdm import tqdm
from sequences import generate_sequence,terminal_fate
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

    def create_dataset(self,to_CSV = False,rewrite = True):
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
                        "Year": year + 1,
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
            path = self._output_path/"dataset.csv" if rewrite else self._output_path/f'dataset_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv'
            self.dataset.to_csv(path,index=False)
            









       

