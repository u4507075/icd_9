import pandas

path = '../data/'

for i in range(2011,2018):
	df = pandas.read_excel(path+str(i)+'.xls')
	print(df)
