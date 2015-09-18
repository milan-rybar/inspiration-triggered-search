#!/usr/bin/python
import sys
sys.path.append("../../scripts/")
from ITS import *
import os
import timeit

# settings
jobID = 0
workingDirectory = ""

def runExperiment(trial, f, name):
    printMessage = name, " trial: ", trial
    print "START: ", printMessage

    # create experiment directory
    directory = createNumberedDirectory(os.path.join(workingDirectory, name + "_doublePenalty/trial_"))

    # prepare experiment
    a = its.AlgorithmTemplate()
    a.PhenotypeFunction = its.ImagePhenotype(256)
    
    # feature
    for feature in f:
        a.Features.append(feature)
    a.Features.append(its.JpegImageComplexityPenalty())
    a.Features.append(its.ChoppinessPenalty())
    
    # aggregation function
    a.Attach(its.AdaptiveScalingSumWithPenalties(2))
    
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
    for i in range(7):
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
    features = []
    featuresName = []
    
    features.append([its.NormalizedVariance(), its.RelaxedSymmetry(), its.GlobalContrastFactor()])
    featuresName.append("NV+RS+GCF")

    features.append([its.NormalizedVariance(), its.RelaxedSymmetry(), its.Tenengrad()])
    featuresName.append("NV+RS+T")

    features.append([its.NormalizedVariance(), its.RelaxedSymmetry(), its.GlobalContrastFactor(), its.Tenengrad()])
    featuresName.append("NV+RS+GCF+T")
    
    jobID = int(os.environ.get("PBS_ARRAYID", "0"))
    workingDirectory = os.environ.get("PC2WORK", ".")
    
    featureIndex = jobID % len(features)
    
    runExperiment(jobID / len(features) + 1, features[featureIndex], featuresName[featureIndex])
