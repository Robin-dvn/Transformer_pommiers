import pandas as pd
import plotly.express as px

# Charger les deux fichiers CSV
df1 = pd.read_csv('out/probs_sequence_ananlysys.csv')
df2 = pd.read_csv('out/probs_dataset_sequenceanamysis_perso.csv')
# df1 = df1.head(1000)
# df2 = df2.head(1000)

# Ajouter une colonne pour identifier la source des données
df1['Source'] = 'Fichier 1'
df2['Source'] = 'Fichier 2'

# Fusionner les deux DataFrames pour un affichage commun
df_combined = pd.concat([df1, df2], ignore_index=True)

# Créer l'histogramme avec Plotly Express
fig = px.histogram(df_combined, x='Probs', color='Source', nbins=500,
                   barmode='overlay',  # Superposer les histogrammes
                   title="Comparaison des Distributions de Log-Probabilités",
                   labels={'Probs': 'Log-Probabilité de la Séquence'},
                   color_discrete_map={'Fichier 1': 'skyblue', 'Fichier 2': 'salmon'})

# Personnalisation de l'apparence
fig.update_layout(
    xaxis_title="Log-Probabilité de la Séquence",
    yaxis_title="Nombre de Séquences",
    bargap=0.1,
    template='plotly_white'
)

# Afficher l'histogramme
fig.show()

# Calcul de la différence entre les colonnes 'Probs'
# On ajuste pour s'assurer que les deux DataFrames ont la même longueur
min_length = min(len(df1), len(df2))
diff = df1['Probs'].iloc[:min_length].reset_index(drop=True) - df2['Probs'].iloc[:min_length].reset_index(drop=True)

# Calcul des statistiques
resultats = {
    'Moyenne des différences': diff.mean(),
    'Écart-type des différences': diff.std(),
    'Différence maximale': diff.max(),
    'Différence minimale': diff.min()
}

# Affichage des résultats
for stat, valeur in resultats.items():
    print(f"{stat} : {valeur}")
