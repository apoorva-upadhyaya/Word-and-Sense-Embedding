from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
import pickle
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pylab import rcParams
from sklearn import manifold
from sklearn.manifold import TSNE
import warnings
rcParams['figure.figsize'] = 15, 10

glove = pickle.load(open('glove_50.pkl','rb'))

def create_sense_bags(inputWord):
	set_words=set()
	
	gloss_lemmas=set([WordNetLemmatizer().lemmatize(word) for word in synset.definition().split()])
	set_words=set_words.union(gloss_lemmas)
	for sentence in synset.examples() :
		#print("sentence",sentence)
			for word in sentence.split():
				# print("word",WordNetLemmatizer().lemmatize(word))
				set_words.add(WordNetLemmatizer().lemmatize(word))
	for word in synset.hypernyms():
		word=set(word.lemma_names())
		# print(word)
		set_words=set_words.union(word)
	for word in synset.hyponyms():
		word=set(word.lemma_names())
		# print(word)
		set_words=set_words.union(word)

	# print("set_words",set_words)
	
	
	return set_words

def get_embedding(sense_bag):
	N=len(sense_bag)
	count=0
	datav = []
	for word in sense_bag:
		tempv=[]
		try:
			tempv.extend(glove[word])
			count=count+1
		except:
			tempv.extend([0]*50)
		datav.append(tempv)
	# print("len:",N)
	# print("count",count)
	list_=map(sum, zip(*datav))
	list1=list(list_)
	newList = [i/count for i in list1]
	# print("newList",newList)
	return newList			



def find_cosine_sim(inputWord,sense_bag_embed) :
	vA=np.array(inputWord)
	vB=np.array(sense_bag_embed)
	cos = np.dot(vA, vB) / (np.sqrt(np.dot(vA,vA)) * np.sqrt(np.dot(vB,vB)))
	# print("cos",cos)
	return cos

def plot_embed(inputWord_embed,dmerge,inWord):
	list_=[]
	list_labels=[]
	for key in dmerge:
		myarray = np.asarray(dmerge[key])
		list_.append(myarray)
		list_labels.append(key)
	list_.append(inputWord_embed)
	list_labels.append(inWord)
	list_v=np.array(list_)

	
	with warnings.catch_warnings():
		pca_ = PCA(n_components=2)
		viz_data = pca_.fit_transform(list_v)
		plt.scatter(viz_data[:,0],viz_data[:,1],cmap=plt.get_cmap('Spectral'))
		for label,x,y in zip(list_labels,viz_data[:,0],viz_data[:,1]):
			plt.annotate(label,xy=(x,y),xytext=(-5,5), textcoords='offset points',arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
		plt.xlabel('PCA Component 1 ')
		plt.ylabel('PCA Component 2')
		plt.title('PCA representation for Word Embedding')
		# plt.xlim(-10,10)
		plt.ylim(-5,6)
		
	plt.show()

if __name__ == '__main__':
	inWord = input('Enter the input word : ')
	# print('The input Word is ', inWord)
	inWord=inWord.lower()
	syn=wordnet.synsets(inWord)
	dict_cos={}
	dict_embed={}
	print("synsets are",syn)
	if (syn) :
		for synset in syn:
			dict_key=synset.name()
			set_words=create_sense_bags(inWord)
			# print("len set_words",len(set_words))
			word_embed=get_embedding(set_words)
			dict_embed[dict_key]=word_embed
			
			try :
				inputWord_embed=(glove[inWord])
			except:
				inputWord_embed=([0]*50)

		
			cos=find_cosine_sim(inputWord_embed,word_embed)	
			dict_cos[synset]=cos
			

		# print("dict_cos",dict_cos)
		sorted_result = sorted(dict_cos.items(), key=lambda x: x[1],reverse=True)
		# print("dict_cos",sorted_result)
		print("Most Frequent Sense of inputWord is :",sorted_result[0])

		plot_embed(inputWord_embed,dict_embed,inWord)
	else :
		print("Sorry, No senses are found in wordnet")