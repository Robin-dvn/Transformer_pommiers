import torch
import pandas as pd
from transformer import Transformer,TransformerDecoderOnly
from tqdm import tqdm
import plotly.express as px
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon
from HSMM import HSMM
import numpy as np
import plotly.graph_objects as go
import json
import os
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path  # si besoin d'utiliser pathlib ailleurs

class Validator: 
    """
    Classe Validator pour valider et analyser des séquences générées par un modèle Transformer.

    Attributs:
        model (Transformer): Instance du modèle Transformer utilisé pour générer des séquences.
        device (str): Dispositif sur lequel les calculs sont effectués (CPU ou GPU).
        datapath (str): Chemin vers le fichier de données de référence.
        token_to_id (dict): Dictionnaire mappant des tokens à leurs identifiants numériques.
        id_to_token (dict): Dictionnaire inverse mappant des identifiants numériques à leurs tokens.
        stats (dict): Dictionnaire stockant diverses statistiques calculées lors des validations.
        show (bool): Booléen indiquant si les graphiques doivent être affichés lors de leur création.
        df (pd.DataFrame): DataFrame pandas contenant les données de référence ou générées.
        simu_folder (str): Dossier où les figures générées sont sauvegardées.
    """
    def __init__(self, model:Transformer, device, token_to_id,datapath= None,show=False ):
        """
        Initialise le validateur avec un modèle, un dispositif, et un mappage de tokens.

        Args:
            model (Transformer): Modèle Transformer utilisé pour générer des séquences.
            device (str): Dispositif sur lequel les calculs sont effectués (CPU ou GPU).
            token_to_id (dict): Dictionnaire mappant des tokens à leurs identifiants numériques.
            datapath (str, optional): Chemin vers le fichier de données de référence.
            show (bool, optional): Indique si les graphiques doivent être affichés.
        """

        self.model = model
        self.device = device
        self.datapath = datapath
        self.token_to_id = token_to_id
        self.id_to_token = {v: k for k, v in token_to_id.items()}
        self.stats = {}
        self.show = show
        self.df = pd.DataFrame(columns=["Observation", "Year", "Sequence", "Terminal Fate"])
        self.simu_folder = os.path.join("assets", datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.simu_folder, exist_ok=True)
        
    def save_figure(self, fig, validation_type, observation, year):
        """
        Sauvegarde une figure dans le dossier de simulation.

        Args:
            fig (plotly.graph_objs._figure.Figure): Figure à sauvegarder.
            validation_type (str): Type de validation (utilisé pour le nom du fichier).
            observation (str): Observation associée à la figure.
            year (int): Année associée à la figure.
        """
        
        folder_path = os.path.join(self.simu_folder, validation_type)
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, f"{observation}_{year}_{validation_type}.png")
        fig.write_image(file_path)
    
    def generate_data(self, nb_samples, output_path, end_toks_list):
        """

        Génère des données en utilisant le modèle Transformer.

        Args:
            nb_samples (int): Nombre d'échantillons à générer.
            output_path (str): Chemin où sauvegarder les données générées.
            end_toks_list (list): Liste des tokens de fin.
        """

        sequences_generees = []
        decoder_only = True
        for type in tqdm(range(8, 12)):
            for year in range(12, 17):
                
                if nb_samples > 1000:
                    for i in range(0, nb_samples, 1000):
                        batch_size = min(1000, nb_samples - i )
                        start_seq = torch.tensor([[type, year]] * batch_size, device=self.device)

                        generated_seq = self.model.generate_batch(start_seq, 1, self.device, end_toks_list, batch_size=int(batch_size))
                        if not decoder_only:
                          sequences_generees.extend(torch.cat((start_seq, generated_seq[:, 1:]), dim=1).to('cpu').tolist())
                        else:
                          sequences_generees.extend(torch.cat((start_seq, generated_seq[:, 3:]), dim=1).to('cpu').tolist())
                else:
                    start_seq = torch.tensor([[type, year]] * nb_samples, device=self.device)

                    generated_seq = self.model.generate_batch(start_seq, 1, self.device, end_toks_list, batch_size=nb_samples)
                    if not decoder_only:
                      sequences_generees.extend(torch.cat((start_seq, generated_seq[:, 1:]), dim=1).to('cpu').tolist())
                    else:
                      sequences_generees.extend(torch.cat((start_seq, generated_seq[:, 3:]), dim=1).to('cpu').tolist())

      
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

        """
        Charge les données à partir d'un fichier CSV.

        Args:
            data_path (str, optional): Chemin vers le fichier de données.
        """
        self.datapath = data_path
        assert self.datapath is not None, "No data path provided"
        self.df = pd.read_csv(self.datapath)


    def markov_model_validation(self,generated_dataset_path):
        """
        Valide les séquences générées en comparant les distributions des états terminaux avec celles du dataset original.

        Args:
            generated_dataset_path (str): Chemin vers le fichier CSV des données générées.
        """

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
            if self.show: fig.show()
            fig.update_layout(
                title_text=f'Terminal fates for {observation} in {year}',
                height=800,
                width=1200,
                margin=dict(t=100, b=100, l=50, r=50)
            )
            
            self.save_figure(fig, "markov_model_validation", observation, year)

    def sequence_length_validation(self, generated_dataset_path):
        """
        Compare la longueur des séquences générées avec celles du dataset original.

        Args:
            generated_dataset_path (str): Chemin vers le fichier CSV des données générées.
        """
        

        dataset = self.df
        generated_dataset = pd.read_csv(generated_dataset_path)

        dataset['Source'] = 'Dataset'
        generated_dataset['Source'] = 'Generated Dataset'
        combined_dataset = pd.concat([dataset, generated_dataset])
        unique_pairs = combined_dataset[['Observation', 'Year']].drop_duplicates()

        for _, row in unique_pairs.iterrows():
            observation = row['Observation']
            year = row['Year']
            subset = combined_dataset[(combined_dataset['Observation'] == observation) &
                                    (combined_dataset['Year'] == year)].copy()
            subset['Sequence Length'] = subset['Sequence'].apply(len)
            stats = subset.groupby('Source')['Sequence Length'].agg(['mean', 'std']).reset_index()

            mean_error = abs(stats.loc[stats['Source'] == 'Dataset', 'mean'].values[0] -
                            stats.loc[stats['Source'] == 'Generated Dataset', 'mean'].values[0])
            std_error = abs(stats.loc[stats['Source'] == 'Dataset', 'std'].values[0] -
                            stats.loc[stats['Source'] == 'Generated Dataset', 'std'].values[0])
            self.stats[(observation, year)] = {"mean_error": mean_error, "std_error": std_error}

            y_max = stats['mean'].max() + 10 * stats['std'].max()
            y_min = stats['mean'].min() - 10 * stats['std'].max()

            # On retire les annotations sur les points en n'utilisant pas le paramètre 'text'
            fig = px.scatter(stats, x='Source', y='mean', error_y='std',
                            title=f'Sequence Length Comparison for {observation} in {year}',
                            labels={'mean': 'Average Sequence Length', 'std': 'Standard Deviation'},
                            color='Source',
                            color_discrete_map={'Dataset': 'green', 'Generated Dataset': 'blue'},
                            size_max=10)
            fig.update_traces(marker=dict(size=12, opacity=0.8), error_y=dict(width=5))
            fig.update_yaxes(range=[y_min, y_max])
            fig.update_xaxes(range=[-1, 2])

            # Positionner la légende à droite et agrandir la marge droite pour faire de la place
            fig.update_layout(
                legend=dict(x=1.02, y=1, font=dict(size=12)),
                margin=dict(l=50, r=200, t=50, b=50)
            )
            # Combiner les stats dans une annotation unique formatée avec des retours à la ligne
            annotation_text = (
                f"Dataset<br>Mean: {stats.loc[stats['Source']=='Dataset', 'mean'].values[0]:.2f}<br>"
                f"Std: {stats.loc[stats['Source']=='Dataset', 'std'].values[0]:.2f}<br><br>"
                f"Generated Dataset<br>Mean: {stats.loc[stats['Source']=='Generated Dataset', 'mean'].values[0]:.2f}<br>"
                f"Std: {stats.loc[stats['Source']=='Generated Dataset', 'std'].values[0]:.2f}"
            )

            # Placer l'annotation en bas à droite du graph
            fig.update_layout(
                annotations=[
                    dict(
                        x=0.98, y=0.02, xref='paper', yref='paper',
                        text=annotation_text,
                        showarrow=False, font=dict(size=12),
                        xanchor='right', yanchor='bottom'
                    )
                ]
            )


            if self.show: fig.show()
            fig.update_layout(
                title_text=f'Sequence length for {observation} in {year}',
                height=800,
                width=1200,
                margin=dict(t=100, b=100, l=50, r=50)
            )
            self.save_figure(fig, "sequence_length_validation", observation, year)

    def sequence_digit_stats(self, generated_dataset_path):
        """
        Analyse les statistiques des chiffres dans les séquences générées et les compare avec le dataset original.

        Args:
            generated_dataset_path (str): Chemin vers le fichier CSV des données générées.
        """


        # Préparation des données
        dataset = self.df.copy()
        generated_dataset = pd.read_csv(generated_dataset_path)
        dataset['Source'] = 'Dataset'
        generated_dataset['Source'] = 'Generated Dataset'
        combined_dataset = pd.concat([dataset, generated_dataset])
        unique_pairs = combined_dataset[['Observation', 'Year']].drop_duplicates()

        for _, row in unique_pairs.iterrows():
            observation = row['Observation']
            year = row['Year']
            subset = combined_dataset[(combined_dataset['Observation'] == observation) &
                                    (combined_dataset['Year'] == year)].copy()
            ds_subset = subset[subset['Source'] == 'Dataset']
            gen_subset = subset[subset['Source'] == 'Generated Dataset']

            # Comptage des occurrences de chiffres (0 à 4)
            count_digits = lambda seq: pd.Series([int(ch) for ch in seq if ch.isdigit() and int(ch) <= 4]).value_counts()
            ds_counts = ds_subset['Sequence'].apply(count_digits).fillna(0).reindex(columns=range(5), fill_value=0)
            gen_counts = gen_subset['Sequence'].apply(count_digits).fillna(0).reindex(columns=range(5), fill_value=0)

            # Moyennes et variances par chiffre
            ds_mean = ds_counts.mean()
            ds_var = ds_counts.var()
            gen_mean = gen_counts.mean()
            gen_var = gen_counts.var()

            # Calcul des erreurs
            mean_errors = abs(ds_mean - gen_mean)
            var_errors = abs(ds_var - gen_var)

            # Mise à jour de self.stats
            key = (observation, year)
            self.stats[key] = {
                "digit_mean_errors": mean_errors.to_dict(),
                "digit_std_errors": var_errors.to_dict()
            }

            # Construction d'un DataFrame de stats et calcul des erreurs par ligne
            stats_df = pd.DataFrame({
                'Digit': ds_mean.index,
                'Dataset Mean': ds_mean.values,
                'Dataset Var': ds_var.values,
                'Generated Mean': gen_mean.values,
                'Generated Var': gen_var.values
            }).set_index('Digit')
            stats_df['Mean Error'] = abs(stats_df['Dataset Mean'] - stats_df['Generated Mean'])
            stats_df['Var Error'] = abs(stats_df['Dataset Var'] - stats_df['Generated Var'])
            table_df = stats_df.reset_index()

            # Préparation des données pour le scatter plot
            plot_data = pd.DataFrame({
                'Digit': list(ds_mean.index) * 2,
                'Mean': list(ds_mean.values) + list(gen_mean.values),
                'Variance': list(ds_var.values) + list(gen_var.values),
                'Source': ['Dataset'] * len(ds_mean) + ['Generated Dataset'] * len(gen_mean),
                'text': ([f"Mean: {m:.2f}<br>Var: {v:.2f}" for m, v in zip(ds_mean.values, ds_var.values)] +
                        [f"Mean: {m:.2f}<br>Var: {v:.2f}" for m, v in zip(gen_mean.values, gen_var.values)])
            })
            y_max = plot_data['Mean'].max() + 10 * plot_data['Variance'].max()
            y_min = plot_data['Mean'].min() - 10 * plot_data['Variance'].max()

            # Création du scatter plot avec Plotly Express
            fig_scatter = px.scatter(plot_data, x='Digit', y='Mean', error_y='Variance',
                                    title=f'Digit Occurrence Stats for {observation} in {year}',
                                    labels={'Mean': 'Average Occurrence', 'Variance': 'Variance'},
                                    color='Source',
                                    color_discrete_map={'Dataset': 'green', 'Generated Dataset': 'blue'},
                                    size_max=10)
            fig_scatter.update_traces(marker=dict(size=12, opacity=0.8), error_y=dict(width=5))

            fig_scatter.update_yaxes(range=[y_min, y_max])

            # Préparation du tableau des annotations (une ligne par chiffre)
            # Colonnes: Digit, Dataset Mean, Generated Mean, Mean Error, Dataset Var, Generated Var, Var Error
            table_header = ["Digit", "Dataset Mean", "Generated Mean", "Mean Error", "Dataset Var", "Generated Var", "Var Error"]
            annotation_rows = [
                [digit,
                f"{row['Dataset Mean']:.2f}",
                f"{row['Generated Mean']:.2f}",
                f"{row['Mean Error']:.2f}",
                f"{row['Dataset Var']:.2f}",
                f"{row['Generated Var']:.2f}",
                f"{row['Var Error']:.2f}"]
                for digit, row in stats_df.iterrows()
            ]
            table_trace = go.Table(
                header=dict(values=table_header, fill_color='paleturquoise', align='left'),
                cells=dict(values=list(zip(*annotation_rows)), fill_color='lavender', align='left')
            )

            # Création d'une figure en deux parties (graph + tableau)
            fig = make_subplots(rows=2, cols=1,
                                row_heights=[0.7, 0.3],
                                vertical_spacing=0.1,
                                specs=[[{"type": "xy"}],
                                    [{"type": "table"}]])
            for trace in fig_scatter.data:
                fig.add_trace(trace, row=1, col=1)
            fig.add_trace(table_trace, row=2, col=1)
            fig.update_layout(title_text=f'Digit Occurrence Stats and Annotations for {observation} in {year}')
            if self.show: fig.show()        
            # Définir une hauteur explicite pour que le tableau soit entièrement visible
            fig.update_layout(
                title_text=f'Digit Occurrence Stats & Details for {observation} in {year}',
                height=800,
                width=1200,
                margin=dict(t=100, b=100, l=50, r=50)
            )

            self.save_figure(fig, "sequence_digit_stats", observation, year)


    def log_prob_distribution_of_sequences(self,generated_dataset_path):
        """
        Analyse la distribution des log-probabilités des séquences générées et les compare avec le dataset original.

        Args:
            generated_dataset_path (str): Chemin vers le fichier CSV des données générées.
        """

                
        def analyze_sequences_from_csv(dataset_path, generated_dataset_path, year, type):
            if year == 1 or year ==2:
                if type == "long":
                    toml_file = f"data/markov/fuji_{type}_year_1.toml"
                else:
                    toml_file = f"data/markov/fuji_{type}_year_3.toml"
            else:
                toml_file = f"data/markov/fuji_{type}_year_{year}.toml"

            hsmm_model = HSMM(toml_file)
            dic = {"long": "LARGE", "medium": "MEDIUM"}

            def process_file(file):
                df = pd.read_csv(file)
                filtered_df = df[(df["Observation"] == dic[type]) & (df["Year"] == f"Y{year}")]
                probabilities = []
                # print(filtered_df)
                for sequence in tqdm(filtered_df["Sequence"]):
                    # print(sequence)
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
                title=f"Distribution des log-probabilités pour l'observation {dic[type]} en Y{year}",
                xaxis_title="Log-Probabilité de la séquence",
                yaxis_title="Nombre de séquences",
                barmode='overlay'
            )


            if self.show: fig.show()
            fig.update_layout(
                title_text=f"Distribution des log-probabilités pour l'observation {dic[type]} en Y{year}",
                height=800,
                width=1200,
                margin=dict(t=100, b=100, l=50, r=50)
            )
            self.save_figure(fig, "log_prob_distribution_of_sequences", type, year)
            return js_distance
        
        
        for year in range(1, 6):
            for type in ["long", "medium"]:
                js_distance = analyze_sequences_from_csv(self.datapath, generated_dataset_path, year, type)
                
                if (year, type) not in self.stats:
                    self.stats[(year, type)] = {
                        "js_distance": js_distance
                    }
                else:
                    self.stats[(year, type)]["js_distance"] = js_distance 


    def plot_stats(self):
        """
        Affiche un tableau des statistiques calculées.
        """
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
        if self.show: fig.show()


    def save_stats(self, filepath):
        """
        Sauvegarde les statistiques calculées dans un fichier JSON.

        Args:
            filepath (str): Chemin où sauvegarder le fichier JSON des statistiques.
        """
        # Convertir les clés en chaînes de caractères
        stats_str_keys = {f"{key[0]}_{key[1]}": value for key, value in self.stats.items()}
        with open(filepath, 'w') as f:
            json.dump(stats_str_keys, f, indent=4)
        print(f"Statistics saved to {filepath}")

    def load_stats(self, filepath):
        """
        Charge les statistiques à partir d'un fichier JSON.

        Args:
            filepath (str): Chemin vers le fichier JSON des statistiques.
        """
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

    model = TransformerDecoderOnly(17,32,4,15,0)
    # model = Transformer(17,12,128,4,0)
    state_dict = torch.load("DecoderOnly_32_layers_15_epochs_400_1024.pth",map_location=torch.device(device))
    model.load_state_dict(state_dict["model_state_dict"])
    model.eval().to(device=device)


    validator = Validator(model, device, token_to_id=vocab_to_id)
    validator.show = True
    # nb_samples =10000
    # validator.generate_data(nb_samples, f"out/generated_transformer_128_3_40_10000.csv", end_toks_list=[7,8,9,10,11])
    validator.load_data("out/markov_python_generated_dataset10000.csv") 
    validator.markov_model_validation("out/generated_DecoderOnly_32_layers_15_epochs_400_1024.csv")
    validator.sequence_length_validation("out/generated_DecoderOnly_32_layers_15_epochs_400_1024.csv")
    validator.sequence_digit_stats("out/generated_DecoderOnly_32_layers_15_epochs_400_1024.csv")
    validator.log_prob_distribution_of_sequences("out/generated_DecoderOnly_32_layers_15_epochs_400_1024.csv")
    validator.plot_stats()
    validator.save_stats("out/stats_enc_dec_only_runpod_32.json")
    # validator.load_stats("out/stats.json")
    # validator.plot_stats()
    

    # validator.sequence_digit_stats("out/generated_dataset10000.csv")
    # validator.generate_data(nb_samples, f"out/generated_dataset{nb_samples}.csv", end_toks_list=[7,8,9,10,11]sqi<tab>ii