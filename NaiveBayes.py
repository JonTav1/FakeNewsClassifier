
from PreProcessingNB import process_article_text
import math

'''gets the frequency of each word within each class'''
def frequencies(text_list, outcomes):
    word_freqs = {}
    for text, outcome in zip(text_list, outcomes): #iterates through each article text, also looking if its real or fake news
        processed_text = process_article_text(text)
      
        for word in processed_text:
            pair = (word, outcome)
            if pair in word_freqs:#if the word-classification pair is already in the dict, incrmeent its freq by one
                word_freqs[pair] += 1
            else:
                word_freqs[pair] = 1
    return word_freqs



'''calculates the prior probability of the classes, and the word likelihoods for each class'''
def train_NB(word_freqs, train_df_outcomes):
    #likelihood of an article being real/fake
    prior_probabilities = {} 
    #likelihood of a word being in a real/fake article
    word_likelihoods= {}
    
    #calculating the number of unique words to calculate probabilkities
    unique_words = set([pair[0] for pair in word_freqs.keys()])
    
    amount_unique_words = len(unique_words)
    
    #calculate the amount of real/fake article words
    
    num_real, num_fake = 0, 0

    for pair in word_freqs.keys(): # a value of 0 is fake, 1 real
        if pair[1] > 0:
            num_real += word_freqs[(pair)] #if it's real, add the frequency/count of the word to num_real, else the same for num_fake
        if pair[1] == 0:
            num_fake += word_freqs[(pair)]
    #total num of articles
    
    total_articles = train_df_outcomes.shape[0]
    
    #total num of real articles, since all real articles have value of 1 just sum the values for total num of articles    
    real_articles = sum(train_df_outcomes)
    
    #total num of fake articles
    
    fake_articles = total_articles - real_articles
    
    prior_probabilities['real'] = real_articles / total_articles
    prior_probabilities['fake'] = fake_articles / total_articles
    
    # calculate word likelihoods for both classes
    for word in unique_words:
        real_freq = word_freqs.get((word, 1), 0)
        fake_freq = word_freqs.get((word, 0), 0)
        
        #to avoid zero probabilites, which can be a problem for a product, add 1 to every word.
        # Google calls this "Laplace Smoothing"
        
        word_likelihoods[(word, 'real')] = (real_freq + 1) / (num_real + amount_unique_words)
        word_likelihoods[(word, 'fake')] = (fake_freq + 1) / (num_fake + amount_unique_words)
    
    return prior_probabilities, word_likelihoods

def predict_NB(text, prior_probabilities, word_likelihoods):
    """Naive Bayes prediction, using prior probabilities, and the likelihoopd of each word. 
    probabilities were changed to addition operations by taking the log, to avoid floating point number issues."""
    word_list = process_article_text(text)
    
    #using log probabilities to add, used multiplication before but made model significantly worse
    log_p_real = math.log(prior_probabilities['real'])
    log_p_fake = math.log(prior_probabilities['fake'])
    
    # update log probabilities based on word likelihoods, adding these instead of multiplying the probabilities helps
    #mitigate issues with small numbers. If word is nto found, just give it an arbitray small value
    for word in word_list:
        
        if (word, 'real') in word_likelihoods:
            log_p_real += math.log(word_likelihoods[(word, 'real')])
        else:
            log_p_real += math.log(1e-10)
        
        if (word, 'fake') in word_likelihoods:
            log_p_fake += math.log(word_likelihoods[(word, 'fake')])
        else:

            log_p_fake += math.log(1e-10)
    
    #comparing the log probabilities
    if log_p_real > log_p_fake:
        return 1  # article is real
    else:
        return 0  # article is fake
    