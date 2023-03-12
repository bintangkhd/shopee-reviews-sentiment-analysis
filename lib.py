import string, re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def cleansing(df):
    #lowertext
    df=df.lower()
    
    #Remove Punctuation
    remove=string.punctuation
    translator=str.maketrans(remove,' '*len(remove))
    df=df.translate(translator)
    
    #Remove ASCII & UNICODE
    df=df.encode('ascii','ignore').decode('utf-8')
    df=re.sub(r'[^\x00-\x7f]',r'', df)
    
    #Remove Newline
    df=df.replace('\n',' ')
    
    return df


def remove_punctuation(text):
    '''Menghilangkan tanda baca dari teks'''
    return re.sub(r'[^\w\s]', '', text)

def tokenize_text(df):
    review = []
    for index, row in df.iterrows():
        temp = word_tokenize(row['content'])
        tokens = [word for word in temp if not word in stopwords.words()]
        review.append(tokens)
    return review
    
