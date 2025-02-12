import pandas as pd
import plotly.express as px

def compare_probabilities():
    """
    Compare les probabilités générées par deux modèles, celui de sequence analysis et celui de HSMM.
    """
    # Charger les deux fichiers CSV
    df_algo_forward = pd.read_csv('out/probs_sequence_ananlysys.csv')
    df_algo_sequence_analysis = pd.read_csv('out/probs_dataset_sequence_analysis_perso_corrige_a_garder.csv')
    # df_algo_sequence_analysis = pd.read_csv('out/probs_dataset_sequenceanamysis_perso1000lignes.csv')
    # df_algo_forward = df_algo_forward.head(1000)
    # df_algo_sequence_analysis = df_algo_sequence_analysis.head(1000)

    # Ajouter une colonne pour identifier la source des données
    df_algo_forward['Source'] = 'Algo Forward'
    df_algo_sequence_analysis['Source'] = 'Algo Sequence Analysis'

    # Fusionner les deux DataFrames pour un affichage commun
    df_combined = pd.concat([df_algo_forward, df_algo_sequence_analysis], ignore_index=True)

    # Créer l'histogramme avec Plotly Express
    fig = px.histogram(df_combined, x='Probs', color='Source', nbins=500,
                       barmode='overlay',  # Superposer les histogrammes
                       title="Comparaison des Distributions de Log-Probabilités",
                       labels={'Probs': 'Log-Probabilité de la Séquence'},
                       color_discrete_map={'Algo Forward': 'skyblue', 'Algo Sequence Analysis': 'salmon'})

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
    min_length = min(len(df_algo_forward), len(df_algo_sequence_analysis))
    diff = df_algo_forward['Probs'].iloc[:min_length].reset_index(drop=True) - df_algo_sequence_analysis['Probs'].iloc[:min_length].reset_index(drop=True)

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

if __name__ == "__main__":
    compare_probabilities()
