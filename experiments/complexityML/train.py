#!/usr/bin/python
import sys
sys.path.append("../../scripts/")
from ITS import *
import cPickle 

from pybrain.datasets            import ClassificationDataSet
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer

from sklearn import cross_validation
from sklearn import svm
from sklearn import preprocessing

def saveForPython(db, filename):
    with open(filename,'wb') as f:
        cPickle.dump(db, f)
        
def percentErrorOnANN(out, target):
    wrong = 0
    for i in range(len(out)):
        if target[i][out[i]] != 1:
            wrong = wrong + 1
    return 100. * float(wrong) / float(len(out)) 
    
def trainANN(tstdata, trndata, hiddenNodes):
    fnn = buildNetwork( trndata.indim, hiddenNodes, trndata.outdim, outclass=SoftmaxLayer, bias=True )
    
    trainer = BackpropTrainer( fnn, dataset=trndata, verbose=True)

    print ""
    print "ANN with " + str(hiddenNodes) + " hidden nodes"

    trainer.trainEpochs( 50 )

    trnresult = percentErrorOnANN( trainer.testOnClassData(dataset=trndata), trndata['target'] )
    tstresult = percentErrorOnANN( trainer.testOnClassData(dataset=tstdata), tstdata['target'] )


    print "epoch: %4d" % trainer.totalepochs, \
          "  train error: %0f%%" % trnresult, \
          "  test error: %0f%%" % tstresult


    saveForPython(fnn, "classifiers/ANN_hiddenNodes=" + str(hiddenNodes))
    
def trainSVM(clf, name, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    
    saveForPython(clf, "classifiers/" + name + "")
    
    print ""
    print name
    print "score on testing set: %0f" % clf.score(X_test, y_test)
    
    scores = cross_validation.cross_val_score(clf, X_train, y_train, cv=10)
    print "cross validation: Accuracy: %0f (+/- %0f)" % (scores.mean(), scores.std() * 2)


Y = cPickle.load(open("Y.dat", 'rb'))
featuresX = cPickle.load(open("featuresX.dat", 'rb'))

# shuffle data
p = np.random.permutation(len(Y))
featuresX = map(lambda i: featuresX[i], p)
Y = map(lambda i: Y[i], p)

# normalize data
scaler = preprocessing.StandardScaler().fit(featuresX)
featuresX = scaler.transform(featuresX) 

saveForPython(scaler, "classifiers/dataScaler")


### ANN

inputData = featuresX

DS = ClassificationDataSet(len(inputData[0]), nb_classes=2, class_labels=['Simple', 'Complex'])

for i in range(len(inputData)):
    DS.appendLinked(inputData[i], Y[i])

DS._convertToOneOfMany( )

tstdata, trndata = DS.splitWithProportion( 0.25 )

print "Number of training patterns: ", len(trndata)
print "Input and output dimensions: ", trndata.indim, trndata.outdim


trainANN(tstdata, trndata, 2)
trainANN(tstdata, trndata, 3)
trainANN(tstdata, trndata, 4)
trainANN(tstdata, trndata, 6)
trainANN(tstdata, trndata, 8)
trainANN(tstdata, trndata, 10)
trainANN(tstdata, trndata, 12)
trainANN(tstdata, trndata, 14)


### SVM

X_train, X_test, y_train, y_test = cross_validation.train_test_split(featuresX, Y, test_size=0.25)


trainSVM(svm.SVC(kernel='linear', cache_size=2000), "SVM_kernel=linear", X_train, X_test, y_train, y_test)
trainSVM(svm.SVC(kernel='poly', degree = 3, cache_size=2000), "SVM_kernel=poly_3", X_train, X_test, y_train, y_test)
trainSVM(svm.SVC(kernel='poly', degree = 7, cache_size=2000), "SVM_kernel=poly_7", X_train, X_test, y_train, y_test)
trainSVM(svm.SVC(kernel='rbf', cache_size=2000), "SVM_kernel=rbf", X_train, X_test, y_train, y_test)
trainSVM(svm.SVC(kernel='sigmoid', cache_size=2000), "SVM_kernel=sigmoid", X_train, X_test, y_train, y_test)
trainSVM(svm.LinearSVC(dual=False), "LinearSVM_dual=false", X_train, X_test, y_train, y_test)
trainSVM(svm.LinearSVC(dual=True), "LinearSVM_dual=true", X_train, X_test, y_train, y_test)

