import pandas as pd


TRAIN_CSV_PATH = "./files/train.csv"

df = pd.read_csv(TRAIN_CSV_PATH)

# Languages: ['spanish' 'portuguese']
# Qualities: ['unreliable' 'reliable']
# Split by Label quality & Language
df_esp_ok = df[(df['label_quality'] == 'reliable') & (df['language'] == 'spanish')]
df_esp_doubt = df[(df['label_quality'] == 'unreliable') & (df['language'] == 'spanish')]
df_por_ok = df[(df['label_quality'] == 'reliable') & (df['language'] == 'portuguese')]
df_por_doubt = df[(df['label_quality'] == 'unreliable') & (df['language'] == 'portuguese')]

print("Data in Spanish with Label quality verified: " + str(df_esp_ok.size))
print("Data in Spanish with Label quality not verified: " + str(df_esp_doubt.size))
print("Data in Portuguese with Label quality verified: " + str(df_por_ok.size))
print("Data in Portuguese with Label quality not verified: " + str(df_por_doubt.size))

# write a small data set
df_esp_ok.to_csv('./files/train_esp_ok.csv')
df_esp_doubt.to_csv('./files/train_esp_doubt.csv')
df_por_ok.to_csv('./files/train_por_ok.csv')
df_por_doubt.to_csv('./files/train_por_doubt.csv')
