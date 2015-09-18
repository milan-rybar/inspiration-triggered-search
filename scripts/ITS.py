#!/usr/bin/python
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../build"))
import cv2
import MultiNEAT as NEAT
import its
import multiprocessing
from multiprocessing import Pool
import os
import glob
import numpy as np
import shutil
import cPickle 
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import csv

globalPhenotypeFunction = its.ImagePhenotype(256)
    
def createNumberedDirectory(directoryPath):
    number = 1
    while True:
        directoryName = "{0}{1}".format(directoryPath, number)
        if not os.path.isdir(directoryName):
            try:
                os.makedirs(directoryName)
                return directoryName
            except OSError, e:
                if e.errno != 17:
                    raise
                pass
        number = number + 1
        
def mkdir(directoryName):
    while True:
        if not os.path.isdir(directoryName):
            try:
                os.makedirs(directoryName)
                return directoryName
            except OSError, e:
                if e.errno != 17:
                    raise
                pass
        else: return directoryName
        
def loadImage(filename):
    return cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY)

def getImageFeatureValues(a, image):
    values = []
    for f in a.Features:
        values.append(f.Evaluate(image))
    return values
    
def save(data, filename):
    f = open(filename, "w")
    for item in data:
        f.write("%s\n" % item)
    f.close()
    
def saveIndividualImageFromFile(individualFilename, saveDirectory):
    g = NEAT.Genome(individualFilename) # load individual
    globalPhenotypeFunction.Create(g) # create image
    outputFilename = os.path.join(saveDirectory, os.path.basename(individualFilename) + ".bmp")
    cv2.imwrite(outputFilename, globalPhenotypeFunction.Image) # save image
    
def saveIndividualImageFromFiles(individualFilenames, saveDirectory):
    for f in individualFilenames:
        saveIndividualImageFromFile(f, saveDirectory)
                
def showIndividualImage(ind):
    print "individual " + str(ind.GetID())
    plt.imshow(globalPhenotypeFunction.Image,cmap = 'gray', interpolation = 'bicubic')
    plt.axis("off")
    plt.show()
    
def showImage(image):
    plt.imshow(image,cmap = 'gray', interpolation = 'bicubic')
    plt.axis("off")
    plt.show()

def createAndShowImage(ind):
    globalPhenotypeFunction.Create(ind)
    showIndividualImage(ind)

def showNetwork(ind):
    img = np.zeros((250, 250, 3), dtype=np.uint8)
    NEAT.DrawPhenotype(img, (0, 0, 250, 250), globalPhenotypeFunction.NeuralNetwork )
    print "individual " + str(ind.GetID())
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), interpolation = 'bicubic')
    plt.axis("off")
    plt.show()

def saveForPython(db, filename):
    with open(filename,'wb') as f:
        cPickle.dump(db, f)

def loadForPython(filename):
    return cPickle.load(open(filename, 'rb'))

def deleteFile(filename):
    if os.path.exists(filename):
        os.remove(filename)
        
        
# complexity by machine learning

features = []
features.append(its.NormalizedVariance())
features.append(its.MaxAbsLaplacian())
features.append(its.Tenengrad())
features.append(its.Choppiness())  
features.append(its.RelaxedSymmetry())
features.append(its.GlobalContrastFactor())
features.append(its.JpegImageComplexity())

featuresNames = []
featuresNames.append("NormalizedVariance")
featuresNames.append("MaxAbsLaplacian")
featuresNames.append("Tenengrad")
featuresNames.append("Choppiness")
featuresNames.append("RelaxedSymmetry")
featuresNames.append("GlobalContrastFactor")
featuresNames.append("JpegImageComplexity")

featuresHumanNames = []
featuresHumanNames.append("Normalized variance")
featuresHumanNames.append("Maximum of absolute Laplacian")
featuresHumanNames.append("Tenengrad")
featuresHumanNames.append("Choppiness")
featuresHumanNames.append("Relaxed symmetry")
featuresHumanNames.append("Global contrast factor")
featuresHumanNames.append("Image complexity by JPEG compression")

def complexityStr(value):
    return "Complex image" if value == 1 else "Simple image"

def getComplexity(image):
    imgFeatures = map(lambda f: f.Evaluate(image), features)
    
    # possible problem with NormalizedVariance when mean = 0
    if math.isnan(imgFeatures[0]):
        imgFeatures[0] = 0
    
    x = scaler.transform(imgFeatures) 
    return fnn.activate(x)

def printComplexity(image):
    output = getComplexity(image)
    classification = np.argmax(output)
    print complexityStr(classification) + " - " + str(output)

def complexityMetric(x):
    return x[1] - x[0]

def getML():
    scaler = loadForPython(os.path.join(os.path.dirname(__file__), "../experiments/complexityML/chosen/dataScaler"))
    fnn = loadForPython(os.path.join(os.path.dirname(__file__), "../experiments/complexityML/chosen/ANN_hiddenNodes=10"))
    return scaler, fnn
    
scaler, fnn = getML()


# graphs

mpl.rcParams['axes.color_cycle'] = [u'b', u'g', u'c', u'm', u'y', u'k']
     
def plotDataWithFig(data, fig, errorevery=10):
    dataMean = map(lambda i: np.mean(data[i]), range(len(data)))
    dataUpperQuartile = map(lambda i: abs(np.percentile(data[i], 75)-dataMean[i]), range(len(data)))
    dataLowerQuartile = map(lambda i: abs(dataMean[i]-np.percentile(data[i], 25)), range(len(data)))

    fig.errorbar(range(len(data)), dataMean, yerr=[dataLowerQuartile,dataUpperQuartile], errorevery=errorevery)
     
def plotDataWithFigWithoutLines(data, fig, errorevery=10):
    dataMean = map(lambda i: np.mean(data[i]), range(len(data)))
    dataUpperQuartile = map(lambda i: abs(np.percentile(data[i], 75)-dataMean[i]), range(len(data)))
    dataLowerQuartile = map(lambda i: abs(dataMean[i]-np.percentile(data[i], 25)), range(len(data)))

    fig.errorbar(range(len(data)), dataMean, yerr=[dataLowerQuartile,dataUpperQuartile], fmt='+', errorevery=errorevery)
     
def plotData2WithFig(x, data, fig, errorevery=10):
    dataMean = map(lambda i: np.mean(data[i]), range(len(data)))
    dataUpperQuartile = map(lambda i: abs(np.percentile(data[i], 75)-dataMean[i]), range(len(data)))
    dataLowerQuartile = map(lambda i: abs(dataMean[i]-np.percentile(data[i], 25)), range(len(data)))

    fig.errorbar(x, dataMean, yerr=[dataLowerQuartile,dataUpperQuartile], fmt="+")#, ecolor="green", color="black")
        
def numWindowsGraph(numWindows, ax):
    plotDataWithFig(numWindows, ax)
    ax.set_xlabel("Evaluations")
    ax.set_ylabel("Number of windows in objective function")

def complexPartGraph(complexities, complexitiesBest, ax, its=True):
    ax.plot(map(lambda pop: max(map(lambda x: x[1], pop)), complexities), "+")
    plotDataWithFig(map(lambda pop: map(lambda x: x[1], pop), complexities), ax)
    if len(complexitiesBest[0]) > 1:
        plotDataWithFig(map(lambda pop: map(lambda x: x[1], pop), complexitiesBest), ax)
    else:
        ax.plot(map(lambda pop: map(lambda x: x[1], pop), complexitiesBest), "+")
    ax.legend([ "Max", "Average", "Best individual",  ], loc="best")
    ax.set_xlabel("Evaluations" if its else "Generations")
    ax.set_ylabel("Complexity node value")
    
def simplePartGraph(complexities, complexitiesBest, ax, its=True):
    ax.plot(map(lambda pop: max(map(lambda x: x[0], pop)), complexities), "+")
    plotDataWithFig(map(lambda pop: map(lambda x: x[0], pop), complexities), ax)
    if len(complexitiesBest[0]) > 1:
        plotDataWithFig(map(lambda pop: map(lambda x: x[0], pop), complexitiesBest), ax)
    else:
        ax.plot(map(lambda pop: map(lambda x: x[0], pop), complexitiesBest), "+")
    ax.legend([ "Max", "Average", "Best individual",  ], loc="best")
    ax.set_xlabel("Evaluations" if its else "Generations")
    ax.set_ylabel("Simplicity node value")
    
def complexityMetricGraph(complexities, complexitiesBest, ax, its=True):
    ax.plot(map(lambda pop: max(map(lambda x: x[1] - x[0], pop)), complexities), "+")
    plotDataWithFig(map(lambda pop: map(complexityMetric, pop), complexities), ax)
    if len(complexitiesBest[0]) > 1:
        plotDataWithFig(map(lambda pop: map(complexityMetric, pop), complexitiesBest), ax)
    else:
        ax.plot(map(lambda pop: map(complexityMetric, pop), complexitiesBest), "+")
    ax.legend([ "Max", "Average", "Best individual",  ], loc="best")
    ax.set_xlabel("Evaluations" if its else "Generations")
    ax.set_ylabel("Classification difference")
    
def complexityRatioGraph(complexities, ax, its=True):
    if len(complexities[0]) == 150:
        ax.plot(map(lambda pop: sum(map(np.argmax, pop)) / float(len(pop)), complexities), "+")
    else:
        complexityOverTime = map(lambda x: [], range(len(complexities)))
        for i in range(len(complexityOverTime)):
            for p in range(0, len(complexities[i]), 150):
                pop = complexities[i][p:(p+150)]
                complexityOverTime[i].append(sum(map(np.argmax, pop)) / float(len(pop)))
        plotDataWithFigWithoutLines(complexityOverTime, ax)
        
        ax.plot(map(lambda pop: sum(map(np.argmax, pop)) / float(len(pop)), complexities), "+")
        ax.legend([ "Average", "Global" ], loc="best")
        
    ax.set_xlabel("Evaluations" if its else "Generations")
    ax.set_ylabel("Ratio of complex images in population")

def numNeuronsGraph(numNeurons, ax, its=True):
    plotDataWithFig(numNeurons, ax)
    ax.set_xlabel("Evaluations" if its else "Generations")
    ax.set_ylabel("Number of neurons")
    
def metricGraph(metric, ax):
    X = []
    Y = []
    for i in range(len(metric)):
        if len(metric[i]) > 0:
            X.append(i)
            Y.append(metric[i])
            
    ax.plot(X, map(max, Y), "+")     
    plotData2WithFig(X, Y, ax)
    ax.legend([ "Max", "Average" ], loc="best")
    ax.set_xlabel("Evaluations")
    ax.set_ylabel("Description metric value")

def EGraph(metric, ax):
    X = []
    Y = []
    for i in range(len(metric)):
        if len(metric[i]) > 0:
            X.append(i)
            Y.append(metric[i])
            
    ax.plot(X, map(max, Y), "+")     
    plotData2WithFig(X, Y, ax)
    ax.legend([ "Max", "Average" ], loc="best")
    ax.set_xlabel("Evaluations")
    ax.set_ylabel("Extension value")
    
def CGraph(metric, ax):
    X = []
    Y = []
    for i in range(len(metric)):
        if len(metric[i]) > 0:
            X.append(i)
            Y.append(metric[i])
            
    ax.plot(X, map(max, Y), "+")     
    plotData2WithFig(X, Y, ax)
    ax.legend([ "Max", "Average" ], loc="best")
    ax.set_xlabel("Generations")
    ax.set_ylabel("Coverage value")
    
def featureGraph(best, pop, featureName, ax, its=True):
    plotDataWithFig(best, ax)
    plotDataWithFig(pop, ax)
    ax.legend([ "Best individual", "Average" ], loc="best")
    ax.set_xlabel("Evaluations" if its else "Generations")
    ax.set_ylabel("Feature value")
    ax.set_title(featureName)
    
