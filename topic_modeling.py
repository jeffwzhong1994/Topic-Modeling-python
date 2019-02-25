"""
Project Name:Document Clustering and Topic Modeling 

Author: Jeff Wenlue Zhong

Date: 02/25/19

"""

#Import Package:
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import nltk
import re
import os

from sklearn import decomposition
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('stopwords')

#Import Files:
titles = open('title_list.txt').read().split('\n')
titles = titles[:100]

#The Wiki synopses and imdb synopses of each movie is separated by the keywords 'BREAKS HERE'.
#Each synoposes may consist of multiple paragraphs.
synopses_wiki = open('synopses_list_wiki.txt').read().split('\n BREAKS HERE')
synopses_wiki = synopses_wiki[:100]

synopses_imdb = open('synopses_list_imdb.txt').read().split('\n BREAKS HERE')
synopses_imdb = synopses_imdb[:100]

#Combines imdb and wiki to get full synoposes for the top 100 movies.
synoposes = []
for i in range(len(synopses_wiki)):
	item = synopses_wiki[i] + synopses_imdb[i]
	synoposes.append(item)

#Generate a list of ordered numbers for future usage.
ranks = range(len(titles))


"""
Tokenizing and Stemming:

Load Stopwords and stemmer function from NLTK library. Stop words like 'a', 'the', 'in'
don't really convey any significant meaning. Stemming is the process of breakingt a word 
down into its root.

"""

stopwords = nltk.corpus.stopwords.words('english')
print("We use " + str(len(stopwords)) + " stopwords from NLTK library")
print("Top 10 stopwords: ", stopwords[:10])

#Read Snowball libray:
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

#Main Function:
def tokenization_and_stemming(text):
	tokens = []
	for sent in nltk.sent_tokenize(text):
		for word in nltk.word_tokenize(sent):
			if word not in stopwords:
				tokens.append(word)

	#Filter out any tokens not containing letters: (Numeric tokens, Raw punctuation)
	filtered_tokens = []
	for token in tokens:
		if re.search('[a-zA-Z]', token):
			filtered_tokens.append(token)

	stems = [stemmer.stem(t) for t in filtered_tokens]
	return stems

def tokenization(text):
	tokens = []
	for sent in nltk.sent_tokenize(text):
		for word in nltk.word_tokenize(sent):
			if word not in stopwords:
				tokens.append(word)

	filtered_tokens = []
	for token in tokens:
		if re.search('[a-zA-Z]', token):
			filtered_tokens.append(token)
	return filtered_tokens

#Test the main function:
# print(tokenization_and_stemming("she looked at her father's arm "))

#Use defined functions to analyze(tokenize, stem) our synoposes.
docs_stemmed = []
docs_tokenized = []
for i in synoposes:
	tokenization_and_stemming_results = tokenization_and_stemming(i)
	docs_stemmed.extend(tokenization_and_stemming_results)

	tokenized_results = tokenization(i)
	docs_tokenized.extend(tokenized_results)

#Create a Mapping:
vocab_frame_dict = {docs_stemmed[x]: docs_tokenized[x] for x in range(len(docs_stemmed))}
print("The mapping of word angel is: ",  vocab_frame_dict['angel'])
print()

"""
TF-IDF: 
Tf-IDF stands for term frequency- inverse document frequency, hwhich is a weight 
often used in information retrieval and text mining. It's a statistical measure used to
evaluate how important a word is to a document in a collection or a corpus.

"""

#Create Vectorizer parameters and fit the vectorizer to synopses.
tfidf =  TfidfVectorizer(max_df = 0.8, max_features = 2000, min_df = 0.2, stop_words = 'english',  use_idf = True, 
						tokenizer = tokenization_and_stemming, ngram_range = (1,1))
tfidf_matrix = tfidf.fit_transform(synoposes)

#Save the terms identified by TF-IDF.
tf_selected_words = tfidf.get_feature_names()

#Print out the matrix, parameters and main features of the TF-IDF Vector.
print("In total, there are " + str(tfidf_matrix.shape[0]) + " synoposes and " + str(tfidf_matrix.shape[1]) + " terms.")
print("The Paramter of TFIDF Vector is: ", tfidf.get_params())
print()
print("<TFIDF-Matrix>")
print(tfidf_matrix)
print()
print("<Selected Feature Names>")
print(tf_selected_words) 
print()

#Calculate Document Similarity:
from sklearn.metrics.pairwise import cosine_similarity
cos_matrix = cosine_similarity(tfidf_matrix)
print(cos_matrix)


"""
K Means Clustering Technique:

A simple unsupervised learning clsutering algorithms.  It aims to partition a set of observations into a number
of clusters, resulting in the partitioning of the data into Voronoi cells. It can be considered a method of
finding out which group a certain object really belongs to.

"""
from sklearn.cluster import KMeans

num_of_clusters = 5
km = KMeans(n_clusters = num_of_clusters)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()

#K-Mean Result Analysis:
films = {'title': titles, 'rank': ranks, 'synopsis': synoposes, 'cluster': clusters}
frame = pd.DataFrame(films, index = [clusters], columns =['rank', 'title', 'cluster'])
print(frame.head(50))

#Films included in each clusters:
print("Number of films included in each cluster:")
frame['cluster'].value_counts().to_frame()

#Average Rank:
grouped = frame['rank'].groupby(frame['cluster'])
print("Average rank (1 to 100) per cluster: ")
grouped.mean().to_frame()

#Clustering from K Means:
print("<Document clustering result by K-means>")
#Cluters_centers denotes the importances of each items in centroid.
#We need to sort it in decreasing-order and get the top k items.
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
cluster_keyword_summary = {}
for i in range(num_of_clusters):
	print("Cluster " + str(i) + " words: ", end = '')
	cluster_keyword_summary[i] = []
	for ind in order_centroids[i, :6]:
		cluster_keyword_summary[i].append(vocab_frame_dict[tf_selected_words[ind]])
		print(vocab_frame_dict[tf_selected_words[ind]] + "," , end = "")
	print()

	#Print out the ClusterID of each item. Without tolist, the values result from 
	#dataframe is <type 'numpy.ndarray'>
	cluster_movies = frame.ix[i]['title'].values.tolist()
	print("Cluster " + str(i) + " titles (" + str(len(cluster_movies)) + " movies): ")
	print(", ".join(cluster_movies))
	print()
#Plotting 
pca = decomposition.PCA(n_components = 2)
tfidf_matrix_np = tfidf_matrix.toarray()
pca.fit(tfidf_matrix_np)
X = pca.transform(tfidf_matrix_np)
xs, ys = X[:,0], X[:,1]

#Using a dictionary to set up colors per cluster:
colors  = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}
#Use a dictionary to set up cluster names:
cluster_names = {}
for i in range(num_of_clusters):
	cluster_names[i] = ", ".join(cluster_keyword_summary[i])

#Create Dataframe with PCA cluster results:
df = pd.DataFrame(dict(x = xs, y = ys, label = clusters, title = titles))
groups = df.groupby(clusters)

#Plot it:
fig, ax = plt.subplots(figsize = (16, 9))
for name, group in groups:
	ax.plot(group.x, group.y, marker = 'o', linestyle = '', ms = 12,
		label = cluster_names[name], color = colors[name], mec = 'none')

ax.legend(numpoints = 1, loc = 4)
plt.title("K-Mean Plot using PCA")
plt.show()

"""
Topic Modeling using Latent Dirichlet Allocation:

In NLP, LDA is a generative statistical model that allows sets of observations to be 
explained by unobserved groups that explain why some parts of data are similar. (Def from Wiki)

"""
from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components = 5, learning_method = 'online')

tfidf_matrix_lda = (tfidf_matrix * 100).astype(int)
lda.fit(tfidf_matrix_lda)
topic_word = lda.components_

n_top_words = 7
topic_keywords_list = []
for i, topic_dist in enumerate(topic_word):
	lda_topic_words = np.array(tf_selected_words)[np.argsort(topic_dist)][:-n_top_words:-1]
	for j in range(len(lda_topic_words)):
		lda_topic_words[j] = vocab_frame_dict[lda_topic_words[j]]
	topic_keywords_list.append(lda_topic_words.tolist())
doc_topic = lda.transform(tfidf_matrix_lda)

#Shape:
print(topic_word.shape)
print(doc_topic.shape)

#Document clustering by LDA:
topic_doc_dict = {}
print("<Document clustering result by LDA>")
for i in range(len(doc_topic)):
	topicID = doc_topic[i].argmax()
	if topicID not in topic_doc_dict:
		topic_doc_dict[topicID] = [titles[i]]
	else:
		topic_doc_dict[topicID].append(titles[i])

for i in topic_doc_dict:
	print("Cluster " + str(i) + " words: ")
	print("Cluster " + str(i) + " titles (" + str(len(topic_doc_dict[i])) + " movies): ")
	print(",".join(topic_doc_dict[i]))
	print()

#More K-means：
from sklearn.datasets.samples_generator import make_blobs
x, y = make_blobs(n_samples = 300, centers = 4, random_state = 0, cluster_std = 0.60)
plt.scatter(x[:,0], x[:,1], s = 50)

est = KMeans(4)
est.fit(x)
y_kmeans = est.predict(x)
plt.scatter(x[:,0], x[:,1], c = y_kmeans, s = 50)