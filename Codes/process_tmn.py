import numpy as np
import os
import sys
import pickle
import json

# TOPIC MODELLING:
# Topic modeling is an unsupervised machine learning technique that's capable of scanning a set of documents, detecting word and phrase
# patterns within them, and automatically clustering word groups and similar expressions that best characterize a set of documents.

import gensim   #used for topic modelling, document indexing and similarity retreival
from scipy import sparse        #used to create sparse matrices
from gensim.parsing.preprocessing import STOPWORDS
import logging      #This module defines functions and classes which implement a flexible event logging system for applications and libraries.

#%%==================================================================================================================================================================================

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
# logging.info - Logs a message with level INFO on the root logger
logging.root.level = logging.INFO


#%%==================================================================================================================================================================================
#if proper arguments not passes while running the file in the terminal
if len(sys.argv) != 2:
    print("Usage:\npython process_tmn.py <input_data_file>")
    exit(0)

#%%==================================================================================================================================================================================
#opening the data file, allocating the directory and encoding the whole set of documents using the utf-8 encoding method
#converting each string to 8 bit unicode
#why use unicode??
#For a computer to be able to store text and numbers that humans can understand, there needs to be a code that transforms characters into numbers. for this ascii 
# can also be used but its not universal an does not cover all the characters

data_file = sys.argv[1]
data_dir = os.path.dirname(data_file)
with open(os.path.join(data_file), 'r') as fin:
    text = gensim.utils.to_unicode(fin.read(), 'utf8').strip()

#%%==================================================================================================================================================================================

#splitting each of the instance from the whole document set and creating a list of instances
news_lst = text.split("\n")
for i in news_lst[:5]:
	print(i)	
	print("==================================================")

msgs = []
labels = []
label_dict = {}


print("total number of instances in the data file: ",len(news_lst))
print("==================================================")

#%%==================================================================================================================================================================================

errors=0
passed=0

for n_i, line in enumerate(news_lst):  #enumerate function adds a counter to an iterable, here that being the  news_lst
	try:
	    msg, label = line.strip().split("######")
        #.tokenise ===== Iteratively yield tokens as unicode strings, optionally removing accent marks and lowercasing it.
	    msg = list(gensim.utils.tokenize(msg, lower=True))
        #for eg, statement 1 is converted to : ['there', 'were', 'mass', 'shootings', 'in', 'texas', 'last', 'week', 'but', 'only', 'on', 'tv']
	    msgs.append(msg)
	    if label not in label_dict:
	        label_dict[label] = len(label_dict)  #later on used to factorise the labels using this dictionary, generated somewhat like 
        #crime:1, entertainment:2, politics :3 ....
	    labels.append(label_dict[label]) #factorisation process: the labels are converted from strings to numbers using the dictionary created in the above step
	    passed+=1
	except:  #to handle all the text instances that were not in the correct format required
		errors+=1
		# print("problem statement was: ==============================")
		# print(line)


print("==================================================")
print("==================================================")
print("==================================================")
print("total instances: ",len(news_lst))
print("passed instances: ",passed)
print("failed instances: ",errors)
print("==================================================")
print("==================================================")
print("==================================================")


#%%==================================================================================================================================================================================



#read this link to understand gensim.corpora.dictionary better
# https://radimrehurek.com/gensim/auto_examples/core/run_corpora_and_vector_spaces.html


# build dictionary
dictionary = gensim.corpora.Dictionary(msgs)
#convert the tokens returned from the gensim.utils.tokenise to a proper BOW representation 
#the dictionary in this case is formed of 84788 tokens, i.e, on analysing all the documents in the dataset, we found the main tokens were 84788 in count
#and each of teh document will now be represented with a vector of lenght 84788 in a BOW representation

# dictionary.dfs[id]:  represent the document count for the token at the key id, ie, in how many documents the token is present


#python is based completely  references ad addresses, so normal copying using the "=" operator do not the way its expected to
#any change made in any of the documents is reflected in the other one, thats why we use copy library's deepcopy method

import copy
bow_dictionary = copy.deepcopy(dictionary)
#this next step takes the tokens from the dictionary and matched them with the available stopewords 
#to find all the stopwords present in the tokens and then filter them out of the BOW token representation 
# list(map(bow_dictionary.token2id.get, STOPWORDS)  returns something like [723, 2009, 1258, 1483, 63276] with a length of 337

bow_dictionary.filter_tokens(list(map(bow_dictionary.token2id.get, STOPWORDS)))
#tokens reduced from 84788 to 84461

#this command filters out all the one letter long tokens from the dictionary that were created due to some sort of anomalies in the dataset at the first place
#here len(len_1_words) = 46
# also so as to confirm the working of this command:
# count=0
# for i in bow_dictionary.values():
#     if len(i)==1:
#         count+=1
# print(count)

#returns the same result of 46
len_1_words = list(filter(lambda w: len(w) == 1, bow_dictionary.values()))

#all the one letter long tokens are removed from the dictionary
bow_dictionary.filter_tokens(list(map(bow_dictionary.token2id.get, len_1_words)))

#removes all the tokens which occour in less than 3 documents
bow_dictionary.filter_extremes(no_below=3, keep_n=None)

# after some tokens have been removed via filter_tokens and there are gaps in the id series. Calling this method will remove the gaps.
bow_dictionary.compactify()


def get_wids(text_doc, seq_dictionary, bow_dictionary, ori_labels):
    seq_doc = []
    # build bow
    row = []
    col = []
    value = []
    row_id = 0
    m_labels = []

    for d_i, doc in enumerate(text_doc):
        if len(bow_dictionary.doc2bow(doc)) < 3:    # filter too short
            continue
        for i, j in bow_dictionary.doc2bow(doc):
# doc2bow returns something llike == [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]=== where in each tuple, the first number is the id of the token in the 
# bow_dictionary and the second number is the frequency occurance of that token in the document ==== bag-of-words format
            row.append(row_id)
            col.append(i)
            value.append(j)
        row_id += 1
#this above mentioned for loop creates a bow representation of each of the tokenised document using the bow_dictionary, and append the info in respective lists
# after first iteration usnng the example mentioned in the comments
# row=[0,0,0,0,0]
# col=[0,1,2,3,4]
# value=[1,1,1,1,1]
#row_id+=1

#convert each tokenised doc to a array of ids using hte original dictionary ------------- so that, this vector can be used later as word seq vector for the document
        wids = list(map(seq_dictionary.token2id.get, doc))
#applying the filter
        wids = np.array(list(filter(lambda x: x is not None, wids))) + 1
#appending the labels
        m_labels.append(ori_labels[d_i])
        seq_doc.append(wids)
#storing the length of each of the toeknised doc
    lens = list(map(len, seq_doc))
#creating a sparse matrix for the bow representation of the tokenised document
    bow_doc = sparse.coo_matrix((value, (row, col)), shape=(row_id, len(bow_dictionary)))
    logging.info("get %d docs, avg len: %d, max len: %d" % (len(seq_doc), np.mean(lens), np.max(lens)))
    return seq_doc, bow_doc, m_labels

#msgs - contains each text_instance in a tokenised format
#dictionary is the originally created one without any changes and filtering
#bow_dictionary is the updated version of the dictionary formed earlier
#labels - factorised array of labels of the text_instances
seq_title, bow_title, label_title = get_wids(msgs, dictionary, bow_dictionary, labels)

# split data
indices = np.arange(len(seq_title))         #total length of text instances
np.random.shuffle(indices)                  #random shuffling of indexes for forming train ans test dataset
nb_test_samples = int(0.2 * len(seq_title))         #20% as test data
seq_title = np.array(seq_title)[indices]        #forming a numpy array of the array of inndex vectors
seq_title_train = seq_title[:-nb_test_samples]      #training dataset
seq_title_test = seq_title[-nb_test_samples:]   #testing dataset
bow_title = bow_title.tocsr()                     #converting sparse matrix to compressed sparse row format
bow_title = bow_title[indices]                      #
bow_title_train = bow_title[:-nb_test_samples]      #train data sparse matrix
bow_title_test = bow_title[-nb_test_samples:]       #test data sparse matrix
label_title = np.array(label_title)[indices]        #
label_title_train = label_title[:-nb_test_samples]  #train data labels
label_title_test = label_title[-nb_test_samples:]   #test data labels

# save
# logging.info("save data...")
# pickle.dump(seq_title, open(os.path.join(data_dir, "dataMsg"), "wb"))
# pickle.dump(seq_title_train, open(os.path.join(data_dir, "dataMsgTrain"), "wb"))
# pickle.dump(seq_title_test, open(os.path.join(data_dir, "dataMsgTest"), "wb"))
# pickle.dump(bow_title, open(os.path.join(data_dir, "dataMsgBow"), "wb"))
# pickle.dump(bow_title_train, open(os.path.join(data_dir, "dataMsgBowTrain"), "wb"))
# pickle.dump(bow_title_test, open(os.path.join(data_dir, "dataMsgBowTest"), "wb"))
# pickle.dump(label_title, open(os.path.join(data_dir, "dataMsgLabel"), "wb"))
# pickle.dump(label_title_train, open(os.path.join(data_dir, "dataMsgLabelTrain"), "wb"))
# pickle.dump(label_title_test, open(os.path.join(data_dir, "dataMsgLabelTest"), "wb"))
# dictionary.save(os.path.join(data_dir, "dataDictSeq"))
# bow_dictionary.save(os.path.join(data_dir, "dataDictBow"))
# json.dump(label_dict, open(os.path.join(data_dir, "labelDict.json"), "w"), indent=4)
# logging.info("done!")



# logging.info("save data...")



# with open(os.path.join(data_dir, "word_index_complete"),"w") as f:
#     f.write(seq_title)



# np.savetxt(open(os.path.join(data_dir, "word_index_complete"),"w"),seq_title)





pickle.dump(seq_title, open(os.path.join(data_dir, "word_index_complete"), "wb"))
pickle.dump(seq_title_train, open(os.path.join(data_dir, "word_index_train"), "wb"))
pickle.dump(seq_title_test, open(os.path.join(data_dir, "word_index_test"), "wb"))

pickle.dump(bow_title, open(os.path.join(data_dir, "bow_sparse_matrix_complete"), "wb"))
pickle.dump(bow_title_train, open(os.path.join(data_dir, "bow_sparse_matrix_train"), "wb"))
pickle.dump(bow_title_test, open(os.path.join(data_dir, "bow_sparse_matrix_test"), "wb"))

pickle.dump(label_title, open(os.path.join(data_dir, "labels_complete"), "wb"))
pickle.dump(label_title_train, open(os.path.join(data_dir, "labels_train"), "wb"))
pickle.dump(label_title_test, open(os.path.join(data_dir, "labels_test"), "wb"))

dictionary.save(os.path.join(data_dir, "original_unfiltered_dictionary"))
bow_dictionary.save(os.path.join(data_dir, "filtered_dictionary"))
json.dump(label_dict, open(os.path.join(data_dir, "labelDict.json"), "w"), indent=4)
logging.info("done!")






# #renaming the files for better understanding
# dataMsg          ---------         word_index_complete
# dataMsgTrain     ---------         word_index_train
# dataMsgTest      ---------         word_index_test

# dataMsgBow       ---------         bow_sparse_matrix_complete
# dataMsgBowTrain  ---------         bow_sparse_matrix_train
# dataMsgBowTest   ---------         bow_sparse_matrix_test

# dataMsgLabel      ---------        labels_complete
# dataMsgLabelTrain ---------        labels_train
# dataMsgLabelTest  ---------        labels_test



# dataDictSeq       ---------        original_unfiltered_dictionary
# dataDictBow       ---------        filtered_dictionary
# label_dict        ---------        #dictionary of all the unique lables present in the dataset along with unique ids.