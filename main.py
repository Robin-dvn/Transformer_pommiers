from DatasetCreator import DatasetCreator
from pathlib import Path
if __name__ == "__main__":
    
    outpath = "out"
    data_creator = DatasetCreator(outpath,1234,4,70,10000)
    if Path(outpath+"/dataset.csv").exists():
        data_creator.load_data(outpath+"/dataset.csv")
    else:
        data_creator.create_data(True,True)
    if Path(outpath+"/dataset_tokenised.csv").exists():
        data_creator.load_tokenised_data(outpath+"/dataset_tokenised.csv")
        data_creator.tokens_to_id()
    else:
        data_creator.tokenise_data(rewrite=True)
        