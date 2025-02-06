import torch
import pandas as pd
from transformer import Transformer
from tqdm import tqdm
import plotly.express as px


from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon
from HSMM import HSMM
import numpy as np
import plotly.graph_objects as go
import json
import os

class Validator:
    def __init__(self, model:Transformer, device, token_to_id,datapath= None, ):
        self.model = model
        self.device = device
        self.datapath = datapath
        self.token_to_id = token_to_id
        self.id_to_token = {v: k for k, v in token_to_id.items()}
        self.stats = {}
        self.df = pd.DataFrame(columns=["Observation", "Year", "Sequence", "Terminal Fate"])
    
    def generate_data(self, nb_samples, output_path, end_toks_list):
        sequences_generees = []
        for type in range(8, 12):
            for year in tqdm(range(12, 17)):
                
                if nb_samples > 1000:
                    for i in range(0, nb_samples, 1000):
                        batch_size = min(1000, nb_samples - i )
                        start_seq = torch.tensor([[type, year]] * batch_size, device=self.device)

                        generated_seq = self.model.generate_batch(start_seq, 1, self.device, end_toks_list, batch_size=int(batch_size))
                        sequences_generees.extend(torch.cat((start_seq, generated_seq[:, 1:]), dim=1).to('cpu').tolist())
                else:
                    start_seq = torch.tensor([[type, year]] * nb_samples, device=self.device)

                    generated_seq = self.model.generate_batch(start_seq, 1, self.device, end_toks_list, batch_size=nb_samples)
                    sequences_generees.extend(torch.cat((start_seq, generated_seq[:, 1:]), dim=1).to('cpu').tolist())

      
        print(f"[INFO] Generated {len(sequences_generees)} sequences")
        print(f"[INFO] converting to dataset: ")
        data_generated = []
        for seq in tqdm(sequences_generees):
            datasetform = []
            digits = ""

            for item in seq:

                if item in self.id_to_token:

                    if self.id_to_token[item].isdigit():
                        digits += self.id_to_token[item]
                        continue
 
                    if digits != "":
                        datasetform.append(digits)
                    datasetform.append(self.id_to_token[item])
                    digits = ""

                    if item in end_toks_list and len(datasetform) !=1:
                        break   
            data_generated.append(datasetform)

        self.df = pd.DataFrame(data_generated, columns=["Observation", "Year", "Sequence", "Terminal Fate"])
        print(f"[INFO] Saving to {output_path}")
        self.df.to_csv(output_path, index=False)


    def load_data(self,data_path = None):
        self.datapath = data_path
        assert self.datapath is not None, "No data path provided"
        self.df = pd.read_csv(self.datapath)


    def markov_model_validation(self,generated_dataset_path):

        dataset = self.df
        generated_dataset = pd.read_csv(generated_dataset_path)

        # assert len(dataset) == len(generated_dataset), "Datasets have different lengths"
        
        # Ajouter une colonne 'Source' pour indiquer la provenance des données
        dataset['Source'] = 'Dataset'
        generated_dataset['Source'] = 'Generated Dataset'

        # Combiner les deux datasets
        combined_dataset = pd.concat([dataset, generated_dataset])

        # Obtenir les couples uniques (Observation, Year)
        unique_pairs = combined_dataset[['Observation', 'Year']].drop_duplicates()

        for _, row in unique_pairs.iterrows():
            observation = row['Observation']
            year = row['Year']

            # Filtrer les données pour le couple actuel
            subset = combined_dataset[(combined_dataset['Observation'] == observation) & (combined_dataset['Year'] == year)]

            # Compter les occurrences de Terminal Fate pour chaque source
            counts = subset.groupby(['Terminal Fate', 'Source']).size().reset_index(name='Count')
            
            # Créer un DataFrame avec toutes les combinaisons possibles de Terminal Fate et Source
            all_combinations = pd.MultiIndex.from_product([counts['Terminal Fate'].unique(), ['Dataset', 'Generated Dataset']], names=['Terminal Fate', 'Source']).to_frame(index=False)

            # Fusionner avec le DataFrame counts pour ajouter les combinaisons manquantes
            counts = all_combinations.merge(counts, on=['Terminal Fate', 'Source'], how='left').fillna(1) 


            
            # Calculer l'erreur en pourcentage pour chaque Terminal Fate
            terminal_fates = counts['Terminal Fate'].unique()
            percentage_errors = {}
            for fate in terminal_fates:
                dataset_count = counts[(counts['Terminal Fate'] == fate) & (counts['Source'] == 'Dataset')]['Count'].values[0]
                generated_count = counts[(counts['Terminal Fate'] == fate) & (counts['Source'] == 'Generated Dataset')]['Count'].values[0]
                percentage_error = abs(dataset_count - generated_count) / dataset_count * 100
                percentage_errors[fate] = percentage_error

            # Mettre à jour le dictionnaire stats
            self.stats[(observation, year)] = {
                "percentage_errors": percentage_errors
            } if (observation, year) not in self.stats else self.stats[(observation, year)].update({
                "percentage_errors": percentage_errors
            })
            # print(self.stats)
            # Créer le graphique avec des couleurs explicites
            fig = px.bar(counts, x='Terminal Fate', y='Count', color='Source', barmode='group',
                        title=f'Comparison of Terminal Fate for {observation} in {year}',
                        color_discrete_map={'Dataset': 'green', 'Generated Dataset': 'blue'})
            # fig.show()

    def sequence_length_validation(self,generated_dataset_path):
        dataset = self.df
        generated_dataset = pd.read_csv(generated_dataset_path)

        # Ajouter une colonne 'Source' pour indiquer la provenance des données
        dataset['Source'] = 'Dataset'
        generated_dataset['Source'] = 'Generated Dataset'

        # Combiner les deux datasets
        combined_dataset = pd.concat([dataset, generated_dataset])

        # Obtenir les couples uniques (Observation, Year)
        unique_pairs = combined_dataset[['Observation', 'Year']].drop_duplicates()

# Créer un graphique pour chaque couple (Observation, Year)
        for _, row in unique_pairs.iterrows():
            observation = row['Observation']
            year = row['Year']

            # Filtrer les données pour le couple actuel
            subset = combined_dataset[(combined_dataset['Observation'] == observation) & (combined_dataset['Year'] == year)].copy()

            # Calculer la longueur des séquences
            subset.loc[:, 'Sequence Length'] = subset['Sequence'].apply(len)

            # Calculer la moyenne et l'écart type des longueurs de séquence pour chaque source
            stats = subset.groupby('Source')['Sequence Length'].agg(['mean', 'std']).reset_index()
           
            # Calculer l'erreur de moyenne et de std
            mean_error = abs(stats.loc[stats['Source'] == 'Dataset', 'mean'].values[0] - stats.loc[stats['Source'] == 'Generated Dataset', 'mean'].values[0])
            std_error = abs(stats.loc[stats['Source'] == 'Dataset', 'std'].values[0] - stats.loc[stats['Source'] == 'Generated Dataset', 'std'].values[0])

            # Mettre à jour le dictionnaire stats
            # Mettre à jour le dictionnaire stats
            if (observation, year) not in self.stats:
                self.stats[(observation, year)] = {
                    "mean_error": mean_error,
                    "std_error": std_error
                }
            else:
                self.stats[(observation, year)]["mean_error"] = mean_error
                self.stats[(observation, year)]["std_error"] = std_error

            # Ajouter une colonne pour les annotations
            stats['text'] = stats.apply(lambda row: f"Mean: {row['mean']:.2f}<br>Std: {row['std']:.2f}", axis=1)

            # Définir les limites de l'axe des y
            y_max = stats['mean'].max() + 10 * stats['std'].max()
            y_min = stats['mean'].min() - 10 * stats['std'].max()

            # Créer le graphique
            fig = px.scatter(stats, x='Source', y='mean', error_y='std', 
                            title=f'Sequence Length Comparison for {observation} in {year}',
                            labels={'mean': 'Average Sequence Length', 'std': 'Standard Deviation'},
                            color='Source', color_discrete_map={'Dataset': 'green', 'Generated Dataset': 'blue'},
                            text='text', size_max=10)  # Ajuster la taille maximale des points
            fig.update_traces(marker=dict(size=12, opacity=0.8), error_y=dict(width=5), textposition='top right')  # Ajuster la taille et l'opacité des points, et la largeur des barres d'erreur
            fig.update_yaxes(range=[y_min, y_max])  # Ajuster les limites de l'axe des y

            # Ajuster les limites de l'axe des x pour espacer les points
            fig.update_xaxes(range=[-8, 9])  # Ajuster les limites de l'axe des x pour espacer les points

            # fig.show()

    def sequence_digit_stats(self,generated_dataset_path):
        dataset = self.df
        generated_dataset = pd.read_csv(generated_dataset_path)

        # Ajouter une colonne 'Source' pour indiquer la provenance des données
        dataset['Source'] = 'Dataset'
        generated_dataset['Source'] = 'Generated Dataset'

        # Combiner les deux datasets
        combined_dataset = pd.concat([dataset, generated_dataset])

        # Obtenir les couples uniques (Observation, Year)
        unique_pairs = combined_dataset[['Observation', 'Year']].drop_duplicates()

        # Créer un graphique pour chaque couple (Observation, Year)
        for _, row in unique_pairs.iterrows():
            observation = row['Observation']
            year = row['Year']

            # Filtrer les données pour le couple actuel
            subset = combined_dataset[(combined_dataset['Observation'] == observation) & (combined_dataset['Year'] == year)].copy()

            # Séparer les données par source
            dataset_subset = subset[subset['Source'] == 'Dataset']
            generated_subset = subset[subset['Source'] == 'Generated Dataset']

            # Compter les occurrences de chaque chiffre dans les séquences pour chaque source
            dataset_digit_counts = dataset_subset['Sequence'].apply(lambda seq: pd.Series([int(char) for char in seq if char.isdigit() and int(char) <= 4]).value_counts()).fillna(0)
            generated_digit_counts = generated_subset['Sequence'].apply(lambda seq: pd.Series([int(char) for char in seq if char.isdigit() and int(char) <= 4]).value_counts()).fillna(0)

            # Aligner les index des séries résultantes
            dataset_digit_counts = dataset_digit_counts.reindex(columns=range(5), fill_value=0)
            generated_digit_counts = generated_digit_counts.reindex(columns=range(5), fill_value=0)

            # Calculer la moyenne et la variance des occurrences de chaque chiffre pour chaque source
            dataset_mean_counts = dataset_digit_counts.mean()
            dataset_var_counts = dataset_digit_counts.var()
            generated_mean_counts = generated_digit_counts.mean()
            generated_var_counts = generated_digit_counts.var()

            # Calculer l'erreur standard et l'erreur moyenne pour chaque chiffre
            mean_errors = abs(dataset_mean_counts - generated_mean_counts)
            std_errors = abs(dataset_var_counts - generated_var_counts)

            # Mettre à jour le dictionnaire stats
            if (observation, year) not in self.stats:
                self.stats[(observation, year)] = {
                    "digit_mean_errors": mean_errors.to_dict(),
                    "digit_std_errors": std_errors.to_dict()
                }
            else:
                self.stats[(observation, year)]["digit_mean_errors"] = mean_errors.to_dict()
                self.stats[(observation, year)]["digit_std_errors"] = std_errors.to_dict()


            # Créer un DataFrame pour les statistiques
            stats = pd.DataFrame({
                'Digit': dataset_mean_counts.index,
                'Dataset Mean': dataset_mean_counts.values,
                'Dataset Var': dataset_var_counts.values,
                'Generated Mean': generated_mean_counts.values,
                'Generated Var': generated_var_counts.values
            })

            # Préparer les données pour le graphique
            plot_data = pd.DataFrame({
                'Digit': list(stats['Digit']) * 2,
                'Mean': list(stats['Dataset Mean']) + list(stats['Generated Mean']),
                'Variance': list(stats['Dataset Var']) + list(stats['Generated Var']),
                'Source': ['Dataset'] * len(stats) + ['Generated Dataset'] * len(stats),
                'text': [f"Mean: {mean:.2f}<br>Var: {var:.2f}" for mean, var in zip(stats['Dataset Mean'], stats['Dataset Var'])] +
                        [f"Mean: {mean:.2f}<br>Var: {var:.2f}" for mean, var in zip(stats['Generated Mean'], stats['Generated Var'])]
            })

            # Définir les limites de l'axe des y
            y_max = plot_data['Mean'].max() + 10 * plot_data['Variance'].max()
            y_min = plot_data['Mean'].min() - 10 * plot_data['Variance'].max()

            # Créer le graphique
            fig = px.scatter(plot_data, x='Digit', y='Mean', error_y='Variance', 
                             title=f'Digit Occurrence Stats for {observation} in {year}',
                             labels={'Mean': 'Average Occurrence', 'Variance': 'Variance'},
                             color='Source', color_discrete_map={'Dataset': 'green', 'Generated Dataset': 'blue'},
                             text='text', size_max=10)  # Ajuster la taille maximale des points
            fig.update_traces(marker=dict(size=12, opacity=0.8), error_y=dict(width=5))  # Ajuster la taille et l'opacité des points, et la largeur des barres d'erreur

            # Ajouter des annotations spécifiques pour chaque source
            for trace in fig.data:
                if trace.name == 'Dataset':
                    trace.textposition = 'top left'
                elif trace.name == 'Generated Dataset':
                    trace.textposition = 'top right'

            # Ajouter des annotations pour l'erreur de moyenne et de variance
            for digit in stats['Digit']:
                mean_error = abs(stats.loc[digit, 'Dataset Mean'] - stats.loc[digit, 'Generated Mean'])
                var_error = abs(stats.loc[digit, 'Dataset Var'] - stats.loc[digit, 'Generated Var'])
                fig.add_annotation(
                    x=digit, y=stats.loc[digit, 'Dataset Mean'],
                    text=f"Mean Error: {mean_error:.2f}<br>Var Error: {var_error:.2f}",
                    showarrow=False,
                    font=dict(color="red"),
                    xshift=-50, yshift=-30  # Positionner l'annotation en bas à gauche du point
                )

            fig.update_yaxes(range=[y_min, y_max])  # Ajuster les limites de l'axe des y

            # fig.show()

    def log_prob_distribution_of_sequences(self,generated_dataset_path):

                
        def generate_and_analyze_sequences_from_csv(dataset_path, generated_dataset_path, year, type):
            toml_file = f"data/markov/fuji_{type}_year_{year}.toml"
            hsmm_model = HSMM(toml_file)
            dic = {"long": "LARGE", "medium": "MEDIUM"}

            def process_file(file):
                df = pd.read_csv(file)
                filtered_df = df[(df["Observation"] == dic[type]) & (df["Year"] == f"Y{year}")]
                probabilities = []
                for sequence in tqdm(filtered_df["Sequence"]):
                    observations = [int(x) for x in str(sequence)]
                    prob_O = hsmm_model.forward_algorithm(observations)
                    probabilities.append(prob_O)
                log_probabilities = np.log(probabilities)
                return log_probabilities

            log_probabilities_dataset = process_file(dataset_path)
            log_probabilities_generated = process_file(generated_dataset_path)

            min_log_prob = min(min(log_probabilities_dataset), min(log_probabilities_generated))
            max_log_prob = max(max(log_probabilities_dataset), max(log_probabilities_generated))
            bin_width = (max_log_prob - min_log_prob) / 1000
            xbins = go.histogram.XBins(start=min_log_prob, end=max_log_prob, size=bin_width)

            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=log_probabilities_dataset, xbins=xbins,
                name='Original', marker_color='green', opacity=0.5))
            fig.add_trace(go.Histogram(
                x=log_probabilities_generated, xbins=xbins,
                name='Générée', marker_color='blue', opacity=0.5))

            kde_dataset = gaussian_kde(log_probabilities_dataset,bw_method=0.05)
            kde_generated = gaussian_kde(log_probabilities_generated,bw_method=0.05)
            x = np.linspace(min_log_prob, max_log_prob, 1000)
            # Mise à l'échelle des KDE pour qu'ils correspondent aux comptes des histogrammes
            kde_dataset_values = kde_dataset(x) * len(log_probabilities_dataset) * bin_width
            kde_generated_values = kde_generated(x) * len(log_probabilities_generated) * bin_width

            fig.add_trace(go.Scatter(
                x=x, y=kde_dataset_values, mode='lines',
                name='KDE File 1', line=dict(color='green', width=2)))
            fig.add_trace(go.Scatter(
                x=x, y=kde_generated_values, mode='lines',
                name='KDE File 2', line=dict(color='blue', width=2)))

            # Pour la mesure de similarité, on normalise les KDE en distributions de probabilité
            dx = (max_log_prob - min_log_prob) / (len(x) - 1)
            P = kde_dataset(x)
            Q = kde_generated(x)
            P_norm = P / np.sum(P * dx)
            Q_norm = Q / np.sum(Q * dx)
            
            # Calcul de la distance de Jensen-Shannon
            js_distance = jensenshannon(P_norm, Q_norm)
            
            # Affichage du résultat sur le graphique dans un encart
            fig.add_annotation(
                x=0.95, y=0.95, xref="paper", yref="paper",
                text=f"Jensen-Shannon Distance: {js_distance:.4f}",
                showarrow=False,
                bordercolor="black",
                borderwidth=1,
                borderpad=4,
                bgcolor="white",
                opacity=0.8
            )
            fig.update_layout(
                title="Distribution des log-probabilités des séquences générées et originales",
                xaxis_title="Log-Probabilité de la séquence",
                yaxis_title="Nombre de séquences",
                barmode='overlay'
            )

            fig.show()
            return js_distance
        
        
        for year in range(3, 6):
            for type in ["long", "medium"]:
                js_distance = generate_and_analyze_sequences_from_csv(self.datapath, generated_dataset_path, year, type)
                
                if (year, type) not in self.stats:
                    self.stats[(year, type)] = {
                        "js_distance": js_distance
                    }
                else:
                    self.stats[(year, type)]["js_distance"] = js_distance 


    def plot_stats(self):
        # Préparer les données pour le tableau
        headers = ["Observation", "Year", "Stat Name", "Value"]
        rows = []
        fill_colors = []

        color1 = 'lavender'
        color2 = 'lightgrey'
        current_color = color1

        for key, value in self.stats.items():
            observation, year = key
            start_index = len(rows)
            for stat_name, stat_value in value.items():
                if isinstance(stat_value, dict):
                    for sub_key, sub_value in stat_value.items():
                        rows.append([observation, year, f"{stat_name} - {sub_key}", f"{sub_value:.4f}"])
                else:
                    rows.append([observation, year, stat_name, f"{stat_value:.4f}"])
            end_index = len(rows)
            fill_colors.extend([current_color] * (end_index - start_index))
            current_color = color2 if current_color == color1 else color1

        # Créer le tableau Plotly
        fig = go.Figure(data=[go.Table(
            header=dict(values=headers, fill_color='paleturquoise', align='left'),
            cells=dict(values=[list(col) for col in zip(*rows)], fill_color=[fill_colors] * len(headers), align='left')
        )])

        fig.update_layout(title="Statistics Table")
        fig.show()


    def save_stats(self, filepath):
        # Convertir les clés en chaînes de caractères
        stats_str_keys = {f"{key[0]}_{key[1]}": value for key, value in self.stats.items()}
        with open(filepath, 'w') as f:
            json.dump(stats_str_keys, f, indent=4)
        print(f"Statistics saved to {filepath}")

    def load_stats(self, filepath):
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                stats_str_keys = json.load(f)
            # Reconvertir les clés en tuples
            self.stats = {tuple(key.split('_')): value for key, value in stats_str_keys.items()}
            print(f"Statistics loaded from {filepath}")
        else:
            print(f"File {filepath} does not exist")
if __name__ == "__main__":

    vocab_to_id ={'<PAD>': 0, '<SOS>': 1, '0': 2, '1': 3, '2': 4, '3': 5, '4': 6, 'DORMANT': 7, 'FLORAL': 8, 'LARGE': 9, 'MEDIUM': 10, 'SMALL': 11, 'Y1': 12, 'Y2': 13, 'Y3': 14, 'Y4': 15, 'Y5': 16} 
    id_to_vocab = {v: k for k, v in vocab_to_id.items()}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Transformer(17,12,128,4,0)
    state_dict = torch.load("10epochavecdormant-4_128_512.pth",map_location=torch.device(device))
    model.load_state_dict(state_dict)
    model.eval().to(device=device)


    validator = Validator(model, device, token_to_id=vocab_to_id)
    nb_samples =1100
    # validator.load_data("out/datasetcustom10000.csv") 
    # validator.markov_model_validation("out/generated_datasetcustom10000.csv")
    # validator.sequence_length_validation("out/generated_datasetcustom10000.csv")
    # validator.sequence_digit_stats("out/generated_datasetcustom10000.csv")
    # validator.log_prob_distribution_of_sequences("out/generated_datasetcustom10000.csv")
    # validator.plot_stats()
    # validator.save_stats("out/stats_2.json")
    validator.load_stats("out/stats.json")
    validator.plot_stats()
    

    # validator.sequence_digit_stats("out/generated_dataset10000.csv")
    # validator.generate_data(nb_samples, f"out/generated_dataset{nb_samples}.csv", end_toks_list=[7,8,9,10,11]sqi<tab>ii