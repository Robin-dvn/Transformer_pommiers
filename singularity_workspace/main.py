import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from openalea.sequence_analysis import HiddenSemiMarkov, sequences, I_DEFAULT
import json
import io
import os
from collections import OrderedDict
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon

import numpy as np
from openalea.sequence_analysis import Estimate
from openalea.sequence_analysis import Plot
import os
import pandas as pd
from openalea.sequence_analysis import HiddenSemiMarkov,Sequences
import matplotlib.pyplot as plt
from openalea.sequence_analysis._sequence_analysis import _SemiMarkovData

from tqdm  import tqdm
from time import time
import math

def process_csv_estimation(csvpath):
    # Charger le fichier
    file_path = csvpath
    # Lire le fichier ligne par ligne
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f.readlines()]

    # Identifier les indices des sections
    initial_prob_index = lines.index("INITIAL_PROBABILITIES") + 1
    transition_prob_index = lines.index("TRANSITION_PROBABILITIES") + 1

    # Extraire les probabilites initiales
    initial_probabilities = lines[initial_prob_index].split("\t")
    # print(initial_probabilities)
    # Extraire les probabilites de transition
    transition_values = []
    i = transition_prob_index
    while i < len(lines) and not lines[i].startswith("transient"):  # Tant qu'on ne tombe pas sur une nouvelle section
        transition_values.extend(lines[i].split("\t"))
        i += 1
    # print(transition_values)
    # Extraire les distributions d'observation
    observation_values = []
    for line in lines:
        if line.startswith("OUTPUT") and not line.startswith("OUTPUT_PROCESS"):
            observation_values.extend(line.split("\t")[2:])  # Ignorer "OUTPUT"
    # print(observation_values)
    # Fusionner toutes les valeurs dans un seul vecteur
    all_probabilities = initial_probabilities + transition_values[:-1] + observation_values
    numeric_probabilities = []
    for value in all_probabilities:
        try:
            numeric_probabilities.append(float(value))  # Convertit en float si possible
        except ValueError:
            pass  # Ignore les valeurs non convertibles
    # Afficher les 10 premieres valeurs pour verification
#     print(numeric_probabilities)
    return numeric_probabilities

def rmse(y_true, y_pred):
    """Calcule la Root Mean Squared Error (RMSE) entre deux listes."""
    if len(y_true) != len(y_pred):
        raise ValueError("Les listes doivent avoir la meme longueur.")
    
    error_squared = [(y_t - y_p) ** 2 for y_t, y_p in zip(y_true, y_pred)]
    mean_error = sum(error_squared) / len(y_true)
    
    return math.sqrt(mean_error)


class Validator():
    def __init__(self):
        self.stats = {}

    def save_stats(self, filepath):
        """
        Sauvegarde les statistiques dans un fichier JSON.

        Args:
            filepath (str): Chemin vers le fichier JSON des statistiques.
        """
        with io.open(filepath, 'w', encoding='utf-8') as f:
            json_str = json.dumps(self.stats, indent=4, ensure_ascii=False,sort_keys=True)  # JSON en unicode
            f.write(unicode(json_str))

        print(u"Statistics saved to {0}".format(filepath))

    def load_stats(self, filepath):
        """
        Charge les statistiques a partir d'un fichier JSON.

        Args:
            filepath (str): Chemin vers le fichier JSON des statistiques.
        """
        if os.path.exists(filepath):
            with io.open(filepath, 'r',encoding='utf-8') as f:
                stats_str_keys = json.load(f)  
            # Reconvertir les cles en tuples
            self.stats = dict((key, value) for key, value in stats_str_keys.items())

            print("Statistics loaded from {0}".format(filepath))
        else:
            print("File {0} does not exist".format(filepath))
    
    def rmse_markov_model_parameters(self, generated_dataset_path,validation_folder_path):


        
        def analyze_markov_from_csv(dataset_path, generated_dataset_path, year, type):

            print("Analyse des rmse de markov de l'observation {} en Y{}".format(type, year))

            innit_guide_path = "data/guide/innit_{0}_y{1}.hsmc".format(type, year)

            dic = {"long": "LARGE", "medium": "MEDIUM"}

            def process_file(csv_path):

                file_path = csv_path  
                df = pd.read_csv(file_path)
                filtered_df = df[(df['Observation'].str.contains(dic[type])) & (df['Year'] == "Y{0}".format(year))]
 
                sequences = filtered_df['Sequence'].tolist()
                sequences = [[[int(char)] for char in seq] for seq in sequences]
                for (i,seq) in enumerate(sequences):
                    sequences[i].append([5])

                for i in range(len(sequences)):
                    if sequences[i][0] !=  [0]:
                        sequences[i][0] = [0]

                ms= Sequences(sequences)
                return ms
            
            # ms_generated_by_markov_python = process_file(dataset_path)
            ms_generated_by_transformer = process_file(generated_dataset_path)
            hmsc_init_guide = HiddenSemiMarkov(innit_guide_path)
            try:
                hsmc_transformer_guide = Estimate(ms_generated_by_transformer,"HIDDEN_SEMI-MARKOV",hmsc_init_guide,Nbiteration= 1000)
            except:
                return None
            # hsmc_markov_python_guide = Estimate(ms_generated_by_markov_python,"HIDDEN_SEMI-MARKOV",hmsc_init_guide,Nbiteration= 1000) 

            print(validation_folder_path+"spread_transformer_type_{0}_year_{1}.csv".format(type,year))
            hsmc_transformer_guide.spreadsheet_write(validation_folder_path+"spread_transformer_type_{0}_year_{1}.csv".format(type,year))
            # hsmc_markov_python_guide.spreadsheet_write(validation_folder_path+"spread_markov_python_type_{0}_year_{1}.csv".format(type,year))
            vec_markov = process_csv_estimation("data/guide/"+"spread_markov_python_type_{0}_year_{1}.csv".format(type,year))
            vec_transformer = process_csv_estimation(validation_folder_path+"spread_transformer_type_{0}_year_{1}.csv".format(type,year)) 

            return rmse(vec_markov,vec_transformer)
        
        for year in tqdm(range(1, 6)):
            for type in ["long", "medium"]:
                rmse_error = analyze_markov_from_csv("markov_python_generated_dataset10000.csv", generated_dataset_path, year, type)
                key = "LARGE_Y{0}".format(year) if type == "long" else "MEDIUM_Y{0}".format(year)
                if rmse_error is not None:
                    if key not in self.stats:
                        self.stats[key] = {"rmse_error": rmse_error}
                    else:
                        self.stats[key]["rmse_error"] = rmse_error

    def log_prob_distribution_of_sequences(self, generated_dataset_path):
        """
        Analyse la distribution des log-probabilites des sequences generees et les compare avec le dataset original.

        Args:
            generated_dataset_path (str): Chemin vers le fichier CSV des donnees generees.
        """
        def analyze_sequences_from_csv(dataset_path, generated_dataset_path, year, type):
            print("Analyse des sequences de l'observation {} en Y{}".format(type, year))
            if year == 1 or year == 2:
                if type == "long":

                    toml_file = "data/markov/fmodel_fuji_y12.txt"
                else:

                    toml_file = "data/markov/fmodel_fuji_5_15_y3_96.txt"
            else:
                type_real = "16_65" if type == "long" else "5_15"
                real_year = {3: 96, 4: 97, 5: 98}
                
                toml_file = "data/markov/fmodel_fuji_{0}_y{1}_{2}.txt".format(type_real, year,real_year[year])

            hsmm_model = HiddenSemiMarkov(toml_file)
            dic = {"long": "LARGE", "medium": "MEDIUM"}

            def process_file(file):
                df = pd.read_csv(file)
                filtered_df = df[(df["Observation"] == dic[type]) & (df["Year"] == "Y{0}".format(year))]
                filtered_df = filtered_df[["Sequence"]]
                pylist = []
                for seq_str in filtered_df["Sequence"]:
                    # Chaque caractere devient un entier encapsule dans une liste
                    seq = [[int(char)] for char in str(seq_str)]
                    # print(seq)
                    pylist.append(seq)
                # Cree l'objet Sequences a partir de la liste des sequences
                pyseq = sequences.Sequences(pylist)
                post_probs = []
                liste_probs_normal = []
                # Calcule la probabilite pour chaque sequence normalisee par sa longueur
                for i in range(len(pylist)):
                    p = hsmm_model.likelihood_computation(pyseq, post_probs, i) / float(len(pylist[i]))
                    # print(p)
                    liste_probs_normal.append(p)
                # log_probabilities = np.log(liste_probs_normal)
                return liste_probs_normal

            log_prob_dataset = process_file(dataset_path)
            # print(log_prob_dataset)
            log_prob_generated = process_file(generated_dataset_path)
            # print(log_prob_generated)
            min_log_prob = min(min(log_prob_dataset), min(log_prob_generated))
            max_log_prob = max(max(log_prob_dataset), max(log_prob_generated))
            bin_width = (max_log_prob - min_log_prob) / 1000.0
            bins = np.arange(min_log_prob, max_log_prob + bin_width, bin_width)

            # # Creation de la figure avec matplotlib 2.0.2
            # fig, ax = plt.subplots(figsize=(12, 8))

            # # Tracer les histogrammes
            # ax.hist(log_prob_dataset, bins=bins, color='green', alpha=0.5, label='Original')
            # ax.hist(log_prob_generated, bins=bins, color='blue', alpha=0.5, label='Generee')

            # Calcul et trace des courbes KDE
            kde_dataset = gaussian_kde(log_prob_dataset, bw_method=0.05)
            kde_generated = gaussian_kde(log_prob_generated, bw_method=0.05)
            x = np.linspace(min_log_prob, max_log_prob, 1000)
            kde_dataset_values = kde_dataset(x) * len(log_prob_dataset) * bin_width
            kde_generated_values = kde_generated(x) * len(log_prob_generated) * bin_width

            # ax.plot(x, kde_dataset_values, color='green', lw=2, label='KDE File 1')
            # ax.plot(x, kde_generated_values, color='blue', lw=2, label='KDE File 2')

            dx = (max_log_prob - min_log_prob) / (len(x) - 1)
            P = kde_dataset(x)
            Q = kde_generated(x)
            P_norm = P / np.sum(P * dx)
            Q_norm = Q / np.sum(Q * dx)
            js_distance = jensenshannon(P_norm, Q_norm)

            # # Annotation de la distance Jensen-Shannon dans le coin superieur droit
            # ax.text(0.95, 0.95, "Jensen-Shannon Distance: {:.4f}".format(js_distance),
            #         transform=ax.transAxes, ha='right', va='top',
            #         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

            # ax.set_title("Distribution des log-probabilites pour l'observation {} en Y{}".format(dic[type], year))
            # ax.set_xlabel("Log-Probabilite de la sequence")
            # ax.set_ylabel("Nombre de sequences")
            # ax.legend()

            # # if self.show:
            # #     plt.show()
            # # Enregistrer la figure via la methode de l'objet
            # self.save_figure(fig, "log_prob_distribution_of_sequences", type, year)
            # plt.close(fig)
            print("js_distance: ", js_distance)
            return js_distance

        for year in range(1, 6):
            for type in ["long", "medium"]:
                js_distance = analyze_sequences_from_csv("markov_python_generated_dataset10000.csv", generated_dataset_path, year, type)
                key = "LARGE_Y{0}".format(year) if type == "long" else "MEDIUM_Y{0}".format(year)
                if key not in self.stats:
                    self.stats[key] = {"js_distance": js_distance}
                else:
                    self.stats[key]["js_distance"] = js_distance








if __name__ == "__main__":
    json_path = sys.argv[1]
    generated_dataset_path = sys.argv[2]
    validation_folder_path = sys.argv[3]
    validator = Validator()
    validator.load_stats(json_path)
    st = time()
    validator.rmse_markov_model_parameters(generated_dataset_path,validation_folder_path)
    et = time()
    print("Time taken to calculate rmse: ", et-st)
    st = time()
    validator.log_prob_distribution_of_sequences(generated_dataset_path)
    et = time()
    print("Time taken to calculate js_distance: ", et-st)
    validator.save_stats(json_path)
