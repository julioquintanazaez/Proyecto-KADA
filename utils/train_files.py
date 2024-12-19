import os

# Create classes here

class Train_files:
    """This class is a util for loading model file
    """
    def load_latest_file(self):
        """This method is a util for loading a model from a directory, no parameters required.
            The models are stored in a directory called "train_saves" in which are stored
            all the models trained by the application.
            The models are saved with the following name format:
                - model_name_date

           return the last model by a date
        """
        print("Loading models files......")
        file_folder = 'train_saves'
        train_files = [f for f in os.listdir(file_folder) if f.endswith('.csv')]
        if not train_files:
            return None
        train_files.sort(key=lambda x: os.path.getmtime(os.path.join(file_folder, x)))
        print(train_files[-1])
        return os.path.join(file_folder, train_files[-1])

