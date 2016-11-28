import requests, re, operator
from bs4 import BeautifulSoup
import numpy as np
from tabulate import tabulate
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

def create_frequency_table(word_array):
	freq_table = {}
	for word in word_array:
		if word in freq_table:
			freq_table[word] += 1
		else:
			freq_table[word] = 1
	return freq_table

def calculate_word_percentage(freq_table):
	total_words = 0
	percentage_added_list = []

	for word,occurence in freq_table:
		total_words = total_words + occurence
	
	for key,value in freq_table:
		percentage_value = float(value * 100) / total_words
		percentage_added_list.append([key, value, round(percentage_value, 4)])
	
	return percentage_added_list

def pretty_table_print(final_freq_table,count):
	print_headers = ['Word', 'Frequency', 'Frequency Percentage']
	print(tabulate(final_freq_table[:count], headers=print_headers, tablefmt='orgtbl'))

all_words = get_all_words(url)
cleaned_words = remove_stop_words(all_words)
word_frequency = create_frequency_table(cleaned_words)

print(np.shape(all_words),np.shape(cleaned_words))
freq_table = create_frequency_table(cleaned_words)
sorted_freq_table = sorted(freq_table.items(), key=operator.itemgetter(1), reverse=True)
final_freq_table = calculate_word_percentage(sorted_freq_table)

pretty_table_print(final_freq_table,10)
