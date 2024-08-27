import string
import re
import nltk
from nltk.stem import PorterStemmer

''' Gettin stopwordsand punctuation list to use for removing stopwords'''
punctuation = string.punctuation
nltk.download('punkt')
''' Instantiating word stemmer to reduce words to stem'''
stemmer=PorterStemmer()



'''tokenizing text, essentially separating each word and making each an element in a list'''
def tokenize_text(text):
    return text.split()

'''removes stopwords from the article text'''
def remove_stopwords(text):
    stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", 
                  "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", 
                  "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", 
                  "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", 
                  "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", 
                  "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", 
                  "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", 
                  "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", 
                  "should", "now"]
    clean_text = []
    for word in text:
        if (word not in stop_words):
            clean_text.append(word)
    return clean_text

'''removing punctuation, hyperlinks, html synta, etc. Used google for some for this'''
def remove_punctuation(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'" + '_'
    for p in punctuations:
        text = text.replace(p,'') #removing punctuations
    return text

'''reduces each word in the article to just its stems for consistency,, done uses nltk stemmer'''
def word_stems(clean_text):
    stem_text = []
    for word in clean_text:
        stem_word = stemmer.stem(word)
        stem_text.append(stem_word)
        
    return stem_text

'''uses functions created to prepare the article text for training'''
def process_article_text(text):
    text = remove_punctuation(text)
    text = tokenize_text(text)
    clean_text = remove_stopwords(text)
    stemmed_text = word_stems(clean_text)

    return(stemmed_text)