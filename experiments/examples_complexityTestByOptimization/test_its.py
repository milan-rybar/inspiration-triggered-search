#!/usr/bin/python
import sys
sys.path.append("../../scripts/")
from ITS import *
import os
import timeit
import cPickle 
        
def loadDb(filename):
    return cPickle.load(open(filename, 'rb'))
    
# settings
jobID = 0
workingDirectory = ""

def runExperiment(image, imageIndex, trial):
    printMessage = "image index: " + str(imageIndex) + ", trial: " + str(trial)
    print "START: ", printMessage
    
    # create experiment directory
    directory = createNumberedDirectory(os.path.join(workingDirectory, "test_its/" + str(imageIndex) + "/trial_"))

    # prepare experiment
    a = its.AlgorithmTemplate()
    a.PhenotypeFunction = its.ImagePhenotype(256)
    
    # feature
    a.Features.append(its.DistanceInPixelSpace(image))

    # aggregation function
    a.Attach(its.Sum())
    
    # statistics about the whole population
    allStats = its.InformationForStatistics()
    allStats.StoreIndividuals = True
    a.Attach(allStats)
    
    # additional features
    globalFeatures = its.FeaturesStatistics()
    globalFeatures.Features.append(its.NormalizedVariance())
    globalFeatures.Features.append(its.MaxAbsLaplacian())
    globalFeatures.Features.append(its.Tenengrad())
    globalFeatures.Features.append(its.Choppiness())  
    globalFeatures.Features.append(its.RelaxedSymmetry())
    globalFeatures.Features.append(its.GlobalContrastFactor())
    globalFeatures.Features.append(its.JpegImageComplexity())
    a.Attach(globalFeatures)
    
    # check everything is ready
    if not a.Init(): return
    
    startTime = timeit.default_timer()
        
    # run experiment
    for i in range(6):
        if i > 0:
            # save current temporal results
            allStats.Save(os.path.join(directory, "temp_statistics"))
            globalFeatures.Save(os.path.join(directory, "temp_features"))
            a.Population.Save(os.path.join(directory, "temp_population"))
            
        a.RunGenerations(100)
        
    stopTime = timeit.default_timer()
    
    # save information for statistics
    allStats.Save(os.path.join(directory, "statistics"))
    
    # save additional features values
    globalFeatures.Save(os.path.join(directory, "features"))

    # save the last population
    a.Population.Save(os.path.join(directory, "lastPopulation"))

    # save time of the experiment
    with open(os.path.join(directory, "time"), "w") as f:
        f.write(str(stopTime - startTime))
        
    # remove temporal files
    os.remove(os.path.join(directory, "temp_statistics"))
    os.remove(os.path.join(directory, "temp_features"))
    os.remove(os.path.join(directory, "temp_population"))

    print "END: ", printMessage

if __name__ == '__main__':
    dbFilename = "ImagesDatabase_ITS.dat"
    db = loadDb(dbFilename)
    
    jobID = int(os.environ.get("PBS_ARRAYID", "0"))
    workingDirectory = os.environ.get("PC2WORK", ".")
    
    imageIndex = jobID % len(db)
    
    runExperiment(db[imageIndex], imageIndex, jobID / len(db) + 1)
