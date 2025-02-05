import numpy as np
import plotly.express as px
from HSMM import HSMM
from tqdm import tqdm
import pandas as pd

def generate_and_analyze_sequences(toml_file="data/markov/fuji_long_year_4.toml", num_sequences=10000, min_length=1, max_length=50, num_bins=2000):
    """
    Génère un grand nombre de séquences, calcule leur probabilité et affiche un histogramme
    interactif des occurrences des séquences en fonction de leur probabilité.
    
    Paramètres :
    - toml_file : chemin vers le fichier TOML du modèle HSMM
    - num_sequences : nombre de séquences à générer
    - min_length : longueur minimale des séquences
    - max_length : longueur maximale des séquences
    - num_bins : nombre de bins pour l'histogramme
    """
    hsmm_model = HSMM(toml_file)
    probabilities = []
    
    for _ in tqdm(range(num_sequences)):
        _, observations = hsmm_model.generate_sequence(min_length, max_length)
        prob_O = hsmm_model.forward_algorithm(observations)
        probabilities.append(prob_O)
    
    # Convertir en log-probabilités pour meilleure visualisation
    log_probabilities = np.log(probabilities)
    
    # Créer l'histogramme avec Plotly
    fig = px.histogram(x=log_probabilities, nbins=num_bins, title="Distribution des log-probabilités des séquences générées",
                        labels={'x': 'Log-Probabilité de la séquence', 'y': 'Nombre de séquences'})
    fig.show()


def generate_and_analyze_sequences_from_csv(csv_file,year, type,toml_file="data/markov/fuji_long_year_3.toml", num_bins=2000):
    """
    Charge un fichier CSV, filtre les séquences où Observation = "LARGE" et Year = "Y3",
    applique l'algorithme Forward et trace un histogramme des log-probabilités.
    
    Paramètres :
    - csv_file : chemin vers le fichier CSV contenant les séquences
    - toml_file : fichier TOML pour initialiser le modèle HSMM
    - num_bins : nombre de bins pour l'histogramme
    """
    # Charger le fichier CSV
    df = pd.read_csv(csv_file)
    toml_file =f"data/markov/fuji_{type}_year_{year}.toml"

    # Filtrer les séquences où Observation = "LARGE" et Year = "Y4"
    dic = {"long":"LARGE","medium":"MEDIUM"}
    filtered_df = df[(df["Observation"] == dic[type]) & (df["Year"] == f"Y{year}")]
    
    # Initialiser le modèle HSMM
    hsmm_model = HSMM(toml_file)

    probabilities = []
    
    # Appliquer l'algorithme Forward sur chaque séquence filtrée
    for sequence in tqdm(filtered_df["Sequence"]):

        # Transformer la séquence en liste d'entiers
 
        observations = [int(x) for x in str(sequence)]
        prob_O = hsmm_model.forward_algorithm(observations)
        probabilities.append(prob_O)
    
    # Convertir en log-probabilités
    log_probabilities = np.log(probabilities)
    
    # Tracer l'histogramme avec Plotly
    fig = px.histogram(x=log_probabilities, nbins=num_bins, title="Distribution des log-probabilités des séquences filtrées",
                        labels={'x': 'Log-Probabilité de la séquence', 'y': 'Nombre de séquences'})
    fig.show()

# Exemple d'utilisation
if __name__ == "__main__":
    # generate_and_analyze_sequences()
    year = 5
    type = "medium"
    generate_and_analyze_sequences_from_csv("out/generated_datasetcustom10000.csv",year,type)
    generate_and_analyze_sequences_from_csv("out/datasetcustom10000.csv",year,type)