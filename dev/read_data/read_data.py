import pandas
import re
from collections import Counter

path = '../data/icd9'

#Step 1: splite words in pre-operation to create features
'''
feature = []
for i in range(2011,2018):
	df = pandas.read_excel(path+str(i)+'.xls')
	#print(df)
	df['pre_operation'] = df['pre_operation'].apply(lambda x: str(x).lower())
	df['pre_operation'] = df['pre_operation'].apply(lambda x: re.sub('[^a-z]', ' ', x).split(' '))
	df['pre_operation'] = df['pre_operation'].apply(lambda x: list(filter(None, x)))
	
	s = df['pre_operation'].apply(pandas.Series).stack().value_counts().to_frame()
	l = s.index.tolist()
	feature = list(set(feature + l))
	
	#print(df['pre_operation'].values.flatten().tolist())
	print('finished '+str(i))

df = pandas.DataFrame(columns=feature)
df.to_csv('feature.csv')
'''

#Step 2: create a training set
'''
features = []
for i in range(2011,2018):
	df = pandas.read_excel(path+str(i)+'.xls')
	f = pandas.read_csv('feature.csv')
	df['pre_operation'] = df['pre_operation'].apply(lambda x: str(x).lower())
	df['pre_operation'] = df['pre_operation'].apply(lambda x: re.sub('[^a-z]', ' ', x).split(' '))
	df['pre_operation'] = df['pre_operation'].apply(lambda x: list(filter(None, x)))
	feature = f.columns.tolist()
	s = df.pre_operation.apply(lambda x: pandas.Series(x)).unstack()
	df2 = df.join(pandas.DataFrame((s.reset_index(level=0, drop=True)))).rename(columns={0:'feature'})
	df2 = df2[['feature','icd9']]
	df2 = df2[df2.feature.notnull()]
	df2 = df2[df2.icd9.notnull()]
	df2['icd9'] = df2['icd9'].apply(lambda x: '#'+str(x))
	features.append(df2)
	print(i)
df = pandas.concat(features)
df.to_csv('trainingset.csv')
'''

#Step 3: create a word2vec model capturing association between pre-operation words
'''
df = pandas.read_csv('trainingset.csv')
df = df[['feature','icd9']]
df = df[df.feature.notnull()]
import gensim
model = gensim.models.Word2Vec(df.values.tolist(), min_count=1)
model.save('model')
'''

#Step 4: use the word2vec model to find the associated words and link them back to icd-9

import gensim
df_t = pandas.read_csv('trainingset.csv')
model = gensim.models.Word2Vec.load('model')
f = pandas.read_csv('feature.csv')
feature = f.columns.tolist()
dftest = pandas.read_excel(path+'2018.xls')
dftest['pre_operation_list'] = dftest['pre_operation'].apply(lambda x: str(x).lower())
dftest['pre_operation_list'] = dftest['pre_operation_list'].apply(lambda x: re.sub('[^a-z]', ' ', x).split(' '))
dftest['pre_operation_list'] = dftest['pre_operation_list'].apply(lambda x: list(filter(None, x)))
#dftest['recommendation_level'] = 0
#dftest['probability'] = ''
#dftest['recommended_icd9'] = ''
#dftest['recommended_icd9_probability'] = ''
#dftest = dftest.head(20)
for index,row in dftest.iterrows():
	if 'nan' not in row['pre_operation_list']:
		v = [x for x in row['pre_operation_list'] if x in f]
		if len(v) > 0:
			similar_words = model.most_similar(positive=v, topn=10)
			words = []
			for i in similar_words:
				words.append(i[0])
			df = df_t.copy()
			df = df[df.feature.isin(words)]
			df = pandas.DataFrame(df['icd9'].value_counts(normalize=True)).reset_index()
			df = df.rename(columns={'index':'icd9','icd9':'probability'})
			df['icd9'].replace({'#': ''}, inplace=True, regex=True)
			#dftest.at[index,'recommended_icd9'] = str(df.head(5)['icd9'].tolist())
			#dftest.at[index,'recommended_icd9_probability'] = str(df.head(5)['probability'].tolist())
			result = df[df['icd9'] == str(row['icd9'])]
			
			if len(result) == 1:
				dftest.at[index,'recommendation_level'] = result.index.values[0]+1
				dftest.at[index,'probability'] = result['probability'].values.tolist()[0]
			
				
print(dftest)
dftest.replace({';': ''}, inplace=True, regex=True)
dftest = dftest[['icd9','descs','pre_operation','recommendation_level']]
dftest.to_csv('2018_result.csv')

#ML model: not used
'''
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

def get_dataset(trainingset, validation_size):
	n = len(trainingset.columns)-1
	array = trainingset.values
	X = array[:,0:n]
	Y = array[:,n]
	seed = 7
	X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
	X_train, Y_train = shuffle(X_train, Y_train, random_state=0)
	return X_train, X_validation, Y_train, Y_validation

le = preprocessing.LabelEncoder()
f = pandas.read_csv('feature.csv')
feature = f.columns.tolist()
le.fit(feature)
df['feature_code'] = le.transform(df.feature.values.tolist())
df = df[['feature_code','icd9']]

df['icd9'] = df.icd9.astype(str)
X_train, X_validation, Y_train, Y_validation = get_dataset(df, 0.2)
c = SVC()
c.fit(X_train, Y_train)
p = c.predict(X_validation)
cf = confusion_matrix(Y_validation, p)
print(cf)
cr = classification_report(Y_validation, p)
print(cr)
'''

















