from fuzzywuzzy import fuzz
import pandas as pd

class fuzzy_features:
    def __init__(self, my_dataframe):
        self.my_dataframe = my_dataframe
        
        
    def create_fuzzy_features(self):
        self.my_dataframe['fuzz_ratio'] = self.my_dataframe.apply(lambda x: fuzz.ratio(str(x['question1']), str(x['question2'])), axis=1)
        self.my_dataframe['fuzz_partial_ratio'] = self.my_dataframe.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
        self.my_dataframe['fuzz_partial_token_sort_ratio'] = self.my_dataframe.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
        self.my_dataframe['fuzz_partial_token_set_ratio'] = self.my_dataframe.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
        self.my_dataframe['fuzz_token_set_ratio'] = self.my_dataframe.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
        self.my_dataframe['fuzz_token_sort_ratio'] = self.my_dataframe.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
       
        


if __name__ == '__main__':

   
    csv_path = ""  # Path of cleaned_text csv file
    data = pd.read_csv(csv_path)
    obj = fuzzy_features(data)
    obj.create_fuzzy_features()

