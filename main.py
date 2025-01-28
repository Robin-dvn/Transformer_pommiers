from DatasetCreator import DatasetCreator

if __name__ == "__main__":
    data_creator = DatasetCreator("out",1234,4,70,10)
    data_creator.create_dataset(True,False)