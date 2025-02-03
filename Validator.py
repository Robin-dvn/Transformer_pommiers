import torch
import pandas as pd
from transformer import Transformer
from tqdm import tqdm

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


    def load_data(self):
        assert self.datapath is not None, "No data path provided"
        self.df = pd.read_csv(self.datapath)

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

    validator.generate_data(nb_samples, f"out/generated_dataset{nb_samples}.csv", end_toks_list=[7,8,9,10,11])