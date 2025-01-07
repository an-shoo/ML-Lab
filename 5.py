import pandas as pd

# 1. Load the playtennis.csv dataset
data = pd.read_csv('playtennis.csv')

# 2. Initialize hypothesis
def candidate_elimination(data):
    specific_hypothesis = ['0'] * len(data.columns[:-1])  # Start with the most specific hypothesis
    general_hypothesis = [['?'] * len(data.columns[:-1])]

    for _, row in data.iterrows():
        if row['PlayTennis'] == 'Yes':  # Positive example
            for i in range(len(specific_hypothesis)):
                if specific_hypothesis[i] == '0':
                    specific_hypothesis[i] = row[i]
                elif specific_hypothesis[i] != row[i]:
                    specific_hypothesis[i] = '?'
            general_hypothesis = [g for g in general_hypothesis if all(g[i] == '?' or g[i] == row[i] for i in range(len(row) - 1))]
        else:  # Negative example
            new_general_hypothesis = []
            for g in general_hypothesis:
                for i in range(len(g)):
                    if g[i] == '?':
                        for value in set(data.iloc[:, i]) - {row[i]}:
                            new_general_hypothesis.append(g[:i] + [value] + g[i + 1:])
            general_hypothesis = new_general_hypothesis

    return specific_hypothesis, general_hypothesis

specific, general = candidate_elimination(data)
print("Specific Hypothesis:", specific)
print("General Hypothesis:", general)
