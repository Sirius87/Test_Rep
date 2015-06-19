from pyspark import SparkContext
import numpy as np
from pyspark.mllib.recommendation import ALS


sc=SparkContext(appName="collaborative filtering test")
txtfile=sc.textFile("outputfood.txt")


def nonEmpty(s):
	return not s==""

txtfilewithoutempty=txtfile.filter(nonEmpty)

def create_tuples(s):
	elements=s.split("::")
	return (hash(elements[0]),hash(elements[1]),np.double(elements[2]))

rdd_entier=txtfilewithoutempty.map(create_tuples)
distinct_couples=rdd_entier.map(lambda r:(r[0],r[1])).distinct().count()

len1=rdd_entier.count()

#pourcentage du dataset utilise pour le training
frac=0.1

trainratings=rdd_entier.sample(False,frac,0)
len2=trainratings.count()
nb_users=rdd_entier.map(lambda r:r[0]).distinct().count()

#construction du modele a partir de la fraction des donnees
model=ALS.train(trainratings,20,10)
usersinsample=trainratings.map(lambda r:r[0]).distinct().count()
#on utilise le set entier comme test data
test_data=rdd_entier.map(lambda p: (p[0],p[1]))

len3=test_data.count()

predicted=model.predictAll(test_data)
nbusersinpredicted=predicted.map(lambda r:r[0]).distinct().count()
len4=predicted.count()
predicted.saveAsTextFile("predictedamazon")
print "training using "+str(len2)+" values out of "+str(len1)+" trying on "+str(len3)+" test data, size of prediction : "+str(len4)+" and there are "+str(nb_users)+"users and "+str(distinct_couples)+" distinct couples in input dataset computed with hashcode"
print str(usersinsample)+" distinct users in sample used for training and "+str(nbusersinpredicted)+" unique users in model prediction"



