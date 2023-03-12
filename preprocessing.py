import pandas as pd

from lib import cleansing, tokenize_text, stemming_text, remove_punctuation
from sklearn.preprocessing import LabelEncoder

file_dataset = 'datasets/shopee_reviews.csv'
df = pd.read_csv(file_dataset)

df = df[['userName','reviewId','content','score']]
df = df.dropna(subset=['content'])

# df['score'].value_counts().sort_values(ascending=True)

print("\nTotal number of reviews: ",len(df))

print("=======================================================================")

print("\nTotal number of reviewer: ", len(set(df['userName'])))

print("=======================================================================")

print("\nPercentage of reviews with positive sentiment : {:.2f}%"\
    .format(df[df['score']>=4]["content"].count()/len(df)*100))

print("=======================================================================")

print("\nPercentage of reviews with negative sentiment : {:.2f}%"\
    .format(df[df['score']<=3]["content"].count()/len(df)*100))
print("=======================================================================")

df['content'] = df['content'].apply(cleansing)

label=[]
for index, row in df.iterrows():
    if row['score']>=4:
        label.append("Positive")
    else:
        label.append("Negative")

encoder = LabelEncoder()

df['label']= encoder.fit_transform(label)
df=df.drop(columns='score', axis=1)

# print("\nTotal number of label = 1 : ", label.count(1))
# print("\nTotal number of label = 0 : ", label.count(0))

s1=df[df.label==1].sample(4000, replace=True)
s2=df[df.label==0].sample(3404, replace=True)

print('\n')
df=pd.concat([s1,s2])
print(df.shape)
print(df.label.value_counts(normalize=True))

df['content'] = df['content'].apply(remove_punctuation)

df['content']= tokenize_text(df)

df.to_csv('datasets/ready_datasets.csv', index=False)
