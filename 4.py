import pandas as pd
def find_s_algorithm(file_path):
    data = pd.read_csv(file_path)
    attributes = data.columns[:-1]           
    class_label = data.columns[-1]           
    hypothesis = ['?'] * len(attributes)    
    for  _,row in data.iterrows():
        if row[class_label] == 'Yes':        
            for i in range(len(attributes)):
                if hypothesis[i] == '?':
                    hypothesis[i] = row[i]
                elif hypothesis[i] != row[i]:
                    hypothesis[i] = '?'

    return hypothesis
file_path = 'training_data.csv'
final_hypothesis = find_s_algorithm(file_path)
print("\nFinal Hypothesis:", final_hypothesis)

