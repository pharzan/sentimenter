import requests, re, operator
from bs4 import BeautifulSoup
import numpy as np
from tabulate import tabulate
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

url = ["https://en.wikipedia.org/wiki/Iran",
"https://en.wikipedia.org/wiki/Economy_of_Iran",
"https://en.wikipedia.org/wiki/History_of_Iran",
"https://en.wikipedia.org/wiki/Politics_of_Iran",
"https://en.wikipedia.org/wiki/Education_in_Iran",
"https://en.wikipedia.org/wiki/Cinema_of_Iran",
"https://en.wikipedia.org/wiki/Name_of_Iran"]

def get_all_words(url):
	vectorizer = CountVectorizer(analyzer='word',min_df=1,ngram_range=(1,3),stop_words='english')
	analyze = vectorizer.build_analyzer()
	soups=[]
	for eachUrl in url:
		source = requests.get(eachUrl)
		soups.append(BeautifulSoup(source.text,'lxml').body)
	all_words=[]
	
	for soup in soups:
		all_divs = soup.find_all('p')
		for div in all_divs:
			div_text = div.text.lower()
			# words_in_div = div_text.split()
			words_in_div = analyze(div_text)
			for each_word in words_in_div:
				cleaned_word = remove_numbers(each_word)
				if len(cleaned_word.strip())>0:
					all_words.append(cleaned_word)	
	return all_words

def remove_numbers(w):
	# regex to get just the word and get rid of punctuations and numbers
	# cleaned_word = re.sub('[^A-Za-z]+', '', w)
	cleaned_word = re.sub('[0-9*]','',w)
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
	print('total words: ', total_words)
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

freq_table = create_frequency_table(cleaned_words)
sorted_freq_table = sorted(freq_table.items(), key=operator.itemgetter(1), reverse=True)
final_freq_table = calculate_word_percentage(sorted_freq_table)

def add_IR_tag(freq_list):
	for key,row in enumerate(freq_list):
		freq_list[key].append('IR')
	return freq_list


pretty_table_print(final_freq_table,320)