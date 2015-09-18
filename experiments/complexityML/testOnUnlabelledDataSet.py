#!/usr/bin/python
import sys
sys.path.append("../../scripts/")
from ITS import *

features = []
features.append(its.NormalizedVariance())
features.append(its.MaxAbsLaplacian())
features.append(its.Tenengrad())
features.append(its.Choppiness())  
features.append(its.RelaxedSymmetry())
features.append(its.GlobalContrastFactor())
features.append(its.JpegImageComplexity())  

def complexityStr(value):
    return "Complex image" if value == 1 else "Simple image"

def getResultsANN(fnn, scaler):
    results = []
    for imageFilename in glob.glob("testingSet/*.png"):
        image = loadImage(imageFilename)
        
        imgFeatures = map(lambda f: f.Evaluate(image), features)
        
        x = scaler.transform(imgFeatures) 
        results.append(np.argmax(fnn.activate(x)))
    return results
    
def getResultsSVM(clf, scaler):
    results = []
    for imageFilename in glob.glob("testingSet/*.png"):
        image = loadImage(imageFilename)
        
        imgFeatures = map(lambda f: f.Evaluate(image), features)
        
        x = scaler.transform(imgFeatures) 
        results.append(clf.predict(x))
    return results

def compareResults(r1, r2):
    i = 0
    for imageFilename in glob.glob("testingSet/*.png"):
        if r1[i] != r2[i]:
            image = loadImage(imageFilename)
            
            print "First: " + complexityStr(r1[i])
            print "Second: "+ complexityStr(r2[i]) 
            
            plt.imshow(image,cmap = 'gray', interpolation = 'bicubic')
            plt.axis("off")
            plt.show()
        i = i + 1

def showResultsANN(fnn, scaler):
    for imageFilename in glob.glob("testingSet/*.png"):
        image = loadImage(imageFilename)
        
        imgFeatures = map(lambda f: f.Evaluate(image), features)
        
        x = scaler.transform(imgFeatures) 

        print fnn.activate(x)
        print complexityStr(np.argmax(fnn.activate(x)))

        plt.imshow(image,cmap = 'gray', interpolation = 'bicubic')
        plt.axis("off")
        plt.show()

def showResultsSVM(clf, scaler):
    for imageFilename in glob.glob("testingSet/*.png"):
        image = loadImage(imageFilename)
        
        imgFeatures = map(lambda f: f.Evaluate(image), features)
        
        x = scaler.transform(imgFeatures) 

        print clf.decision_function(x)
        print complexityStr(clf.predict(x))

        plt.imshow(image,cmap = 'gray', interpolation = 'bicubic')
        plt.axis("off")
        plt.show()


# show classification of the final complexity classifier on unlabelled data set

showResultsANN(loadForPython("chosen/ANN_hiddenNodes=10"), loadForPython("chosen/dataScaler"))


# compare results of classifiers on unlabelled data set

#scaler = loadForPython("classifiers/dataScaler")
#
#ann2 = getResultsANN(loadForPython("classifiers/ANN_hiddenNodes=2"), scaler)
#ann3 = getResultsANN(loadForPython("classifiers/ANN_hiddenNodes=3"), scaler)
#ann4 = getResultsANN(loadForPython("classifiers/ANN_hiddenNodes=4"), scaler)
#ann6 = getResultsANN(loadForPython("classifiers/ANN_hiddenNodes=6"), scaler)
#ann8 = getResultsANN(loadForPython("classifiers/ANN_hiddenNodes=8"), scaler)
#ann10 = getResultsANN(loadForPython("classifiers/ANN_hiddenNodes=10"), scaler)
#ann12 = getResultsANN(loadForPython("classifiers/ANN_hiddenNodes=12"), scaler)
#ann14 = getResultsANN(loadForPython("classifiers/ANN_hiddenNodes=14"), scaler)
#
#svm_linear = getResultsSVM(loadForPython("classifiers/SVM_kernel=linear"), scaler)
#svm_rbf = getResultsSVM(loadForPython("classifiers/SVM_kernel=rbf"), scaler)
#svm_poly = getResultsSVM(loadForPython("classifiers/SVM_kernel=poly_3"), scaler)
#linearSvm = getResultsSVM(loadForPython("classifiers/LinearSVM_dual=false"), scaler)
#
#
#compareResults(ann4, ann10)
#compareResults(svm_poly, svm_rbf)
#compareResults(svm_poly, ann10)
