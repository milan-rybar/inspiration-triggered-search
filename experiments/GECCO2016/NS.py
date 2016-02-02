#!/usr/bin/python
import sys
sys.path.append("../../scripts/")
from ITS import *
import os
import timeit

# settings
jobID = 0
workingDirectory = ""

def runExperiment(trial, f, penalty, name):
    printMessage = name, " trial: ", trial
    print "START: ", printMessage

    # create experiment directory
    directory = createNumberedDirectory(os.path.join(workingDirectory, "NS256/" + name + "/trial_"))

    # prepare experiment
    a = its.MultiObjectiveNoveltySearchTemplate()
    a.PhenotypeFunction = its.ImagePhenotype(256)
    
    # feature
    for feature in f:
        a.Features.append(feature)

    if penalty:
        a.Features.append(its.JpegImageComplexityPenalty())
        a.Features.append(its.ChoppinessPenalty())
        
        # aggregation function
        a.Attach(its.AdaptiveScalingSumWithPenalties(2))
    else:
        # aggregation function
        a.Attach(its.AdaptiveScalingSum())
    
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
    for i in range(20):
        if i > 0:
            # save current temporal results
            allStats.Save(os.path.join(directory, "temp_statistics"))
            globalFeatures.Save(os.path.join(directory, "temp_features"))
            a.Population.Save(os.path.join(directory, "temp_population"))
            
        a.RunGenerations(100)
        print "###Generation: " + str(a.Generation)
        
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
    penalties = []
    
    features.append([its.NormalizedVariance()])
    featuresName.append("NV")
    penalties.append(False)

    features.append([its.RelaxedSymmetry()])
    featuresName.append("RS")
    penalties.append(False)
 
    features.append([its.GlobalContrastFactor()])
    featuresName.append("GCF_doublePenalty")
    penalties.append(True)

    features.append([its.Tenengrad()])
    featuresName.append("T_doublePenalty")
    penalties.append(True)
 
    features.append([its.NormalizedVariance(), its.RelaxedSymmetry()])
    featuresName.append("NV+RS")
    penalties.append(False)
 
    features.append([its.NormalizedVariance(), its.RelaxedSymmetry(), its.GlobalContrastFactor()])
    featuresName.append("NV+RS+GCF_doublePenalty")
    penalties.append(True)
    
    features.append([its.NormalizedVariance(), its.RelaxedSymmetry(), its.Tenengrad()])
    featuresName.append("NV+RS+T_doublePenalty")
    penalties.append(True)
    
    features.append([its.NormalizedVariance(), its.RelaxedSymmetry(), its.GlobalContrastFactor(), its.Tenengrad()])
    featuresName.append("NV+RS+GCF+T_doublePenalty")
    penalties.append(True)
 
 
 
    jobID = int(os.environ.get("PBS_ARRAYID", "0"))
    workingDirectory = os.environ.get("PC2WORK", ".")
    
    featureIndex = jobID % len(features)
    
    runExperiment(jobID / len(features) + 1, features[featureIndex], penalties[featureIndex], featuresName[featureIndex])

