import torch
import pandas as pd
from transformer import Transformer
from tqdm import tqdm
import plotly.express as px
class Validator:
    def __init__(self, model:Transformer, device, token_to_id,datapath= None, ):
        self.model = model
        self.device = device
        self.datapath = datapath
        self.token_to_id = token_to_id
        self.id_to_token = {v: k for k, v in token_to_id.items()}
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

        assert len(dataset) == len(generated_dataset), "Datasets have different lengths"
        
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

            # Créer le graphique avec des couleurs explicites
            fig = px.bar(counts, x='Terminal Fate', y='Count', color='Source', barmode='group',
                        title=f'Comparison of Terminal Fate for {observation} in {year}',
                        color_discrete_map={'Dataset': 'green', 'Generated Dataset': 'blue'})
            fig.show()

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
            print(stats)
            
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

            fig.show()

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

            fig.show()





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
    validator.load_data("out/dataset.csv") 
    validator.sequence_digit_stats("out/generated_dataset10000.csv")
    # validator.generate_data(nb_samples, f"out/generated_dataset{nb_samples}.csv", end_toks_list=[7,8,9,10,11]sqi<tab>ii