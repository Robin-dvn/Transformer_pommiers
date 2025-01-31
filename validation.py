import torch
from transformer import Transformer
import pandas as pd
from tqdm import tqdm


############ A RECODER C'EST PAS BON #############
if __name__ == "__main__":
    vocab_to_token = {
            "LARGE": "L", "MEDIUM": "M", "SMALL": "S", "FLORAL": "F","DORMANT":"D",
            "Y1": "Y1", "Y2": "Y2", "Y3": "Y3", "Y4": "Y4", "Y5": "Y5"
    }
    token_to_vocab = {v: k for k, v in vocab_to_token.items()}

    vocab_to_id ={'<PAD>': 0, '<SOS>': 1, '0': 2, '1': 3, '2': 4, '3': 5, '4': 6, 'D': 7, 'F': 8, 'L': 9, 'M': 10, 'S': 11, 'Y1': 12, 'Y2': 13, 'Y3': 14, 'Y4': 15, 'Y5': 16} 
    id_to_vocab = {v: k for k, v in vocab_to_id.items()}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Transformer(17,12,128,4,0)

    state_dict = torch.load("10epochavecdormant-4_128_512.pth",map_location=torch.device(device))
    model.load_state_dict(state_dict)
    model.eval().to(device=device)
    sequences_generees = []

    for type in range(8,12):
        for year in range(12,17):
            for _ in tqdm(range(100)):

                start_seq = torch.tensor([[type,year]],device=device)
                generated_seq = model.generate_batch(start_seq,1,device,[7,8,9,10,11])
                generated_seq_token = [id_to_vocab[n] for n in generated_seq]
                full_seq = start_seq.to(device='cpu').tolist()[0]
                full_seq = [id_to_vocab[n] for n in full_seq] + generated_seq_token
                datasetform = []    
                digits = ""
                for item in full_seq:
                        
                    if item in token_to_vocab:
                        if digits !="":
                            datasetform.append(digits)
                        datasetform.append(token_to_vocab[item])
                        digits= ""
                    elif item.isdigit():

                        digits+=item  # Chaque chiffre devient un token
                sequences_generees.append(datasetform)
                
    # Ordre défini par le CSV
    order = {'SMALL': 0, 'FLORAL': 1, 'LARGE': 2, 'MEDIUM': 3}

    # Tri de la liste en fonction de l'ordre défini
    sorted_data = sorted(sequences_generees, key=lambda x: (order.get(x[0], float('inf')), x[1], x[2]))
    sorted_df = pd.DataFrame(sorted_data, columns=["Observation", "Year", "Sequence", "Terminal Fate"])
    sorted_df.to_csv("out/generated_dataset100.csv", index=False)




import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Charger les fichiers CSV
df1 = pd.read_csv("out/dataset100.csv")
df2 = pd.read_csv("out/generated_dataset.csv")

import pandas as pd
import plotly.express as px


# Ajouter une colonne pour identifier les datasets
df1["Dataset"] = "Dataset 1"
df2["Dataset"] = "Dataset 2"

# Fusionner les deux DataFrames
df = pd.concat([df1, df2])

# Compter les occurrences de "Terminal Fate" pour chaque (Observation, Year, Dataset)
grouped = df.groupby(["Observation", "Year", "Terminal Fate", "Dataset"]).size().reset_index(name="Count")

# Générer un graphe interactif pour chaque couple (Observation, Year)
for (obs, year), subset in grouped.groupby(["Observation", "Year"]):
    fig = px.bar(
        subset,
        x="Terminal Fate",
        y="Count",
        color="Dataset",
        barmode="group",
        title=f"Observation: {obs}, Year: {year}",
        labels={"Count": "Nombre", "Terminal Fate": "Terminal Fate"}
    )
    fig.show()

print("Les graphiques interactifs ont été générés avec Plotly.")
