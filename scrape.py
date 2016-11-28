import requests, re
from bs4 import BeautifulSoup
import numpy as np
from nltk.corpus import stopwords

url = "https://en.wikipedia.org/wiki/Apple_Inc."

def get_all_words(url):

	source = requests.get(url)
	soup = BeautifulSoup(source.text,'lxml')
	all_words=[]
	all_divs = soup.find_all('div')
	for div in all_divs:
		div_text = div.text.lower()
		words_in_div = div_text.split()
		for each_word in words_in_div:
			cleaned_word = clean_word(each_word)
			if len(cleaned_word)>0:
				all_words.append(cleaned_word)	
	return all_words

def clean_word(w):
	# regex to get just the word and get rid of punctuations and numbers
	cleaned_word = re.sub('[^A-Za-z]+', '', w)
	return cleaned_word

def remove_stop_words(word_array):
	stop_words = stopwords.words('english')
	temp_words = []
	for word in word_array:
		if word not in stop_words:
			temp_words.append(word)
	return temp_words


all_words = get_all_words(url)
cleaned_words = remove_stop_words(all_words)
print(np.shape(all_words),np.shape(cleaned_words))
print(cleaned_words[:500])
