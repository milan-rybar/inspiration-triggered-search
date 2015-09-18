#!/usr/bin/python
import sys
sys.path.append("../../scripts/")
from ITS import *
import os
import timeit

# settings
jobID = 0
workingDirectory = ""

def runExperiment(trial, name, feature, removeWindow, removeWindowParam, doublePenalty, inspirationCriterionThreshold):
    printMessage = name, " trial: ", trial
    print "START: ", printMessage

    # create experiment directory
    directory = createNumberedDirectory(os.path.join(workingDirectory, name + "/trial_"))
    
    # prepare experiment
    a = its.ITS()

    a.OptimizationIterations = 10
    a.StagnationLimit = 10
    a.InspirationCriterionThreshold = inspirationCriterionThreshold
    
    # feature
    a.Features.append(feature)

    # penalties
    if doublePenalty:
        a.FeaturesPenalties.append(its.JpegImageComplexityPenalty())
        a.FeaturesPenalties.append(its.ChoppinessPenalty())

    # view features
    a.ViewFeatures.append(its.ContoursView())
    
    # modify objective function operators
    if removeWindow:
        remOp = its.RemoveRandomlyWindowByC()
        remOp.ThresholdToRemove = removeWindowParam
        a.ModifyObjectiveOperators.append(remOp)
    
    a.ModifyObjectiveOperators.append(its.AddRandomlyWindowByE())
    
    # selector of description to modify objective function
    a.DescriptionToModifyObjective = its.BestByMetric()
    
    # diversification operator
    a.Diversification = its.MutatePopulation()
    
    # statistics
    stats = its.ITSStatistics()
    stats.StoreIndividuals = True
    a.Attach(stats)
    
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
    for i in range(40):
        if i > 0:
            # save current temporal results
            stats.Save(os.path.join(directory, "temp_statistics"))
            globalFeatures.Save(os.path.join(directory, "temp_features"))
            a.Population.Save(os.path.join(directory, "temp_population"))
            
        a.OneCreativeCycle()
        
    stopTime = timeit.default_timer()
    
    # save information for statistics
    stats.Save(os.path.join(directory, "statistics"))
    
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
    names = []
    features = []
    removeWindow = []
    removeWindowParams = []    
    doublePenalty = []
    inspirationCriterionThreshold = []
    
    for DP in [False]:
        for ins in [0.01, 0.02]:
            for remPar in [0.2, 0.3]:
                for rem in [True, False]:
                    features.append(its.NormalizedVariance())
                    removeWindow.append(rem)
                    removeWindowParams.append(remPar)
                    doublePenalty.append(DP)
                    inspirationCriterionThreshold.append(ins)
                    names.append("its/NormalizedVariance/" + "inspCrit=" + str(ins) + ("_remWin=" + str(remPar) if rem else "_noRemWin") + ("_doublePenalty" if DP else "_noPenalty"))


    for DP in [False]:
        for ins in [0.01, 0.02]:
            for remPar in [0.3, 0.4]:
                for rem in [True, False]:
                    features.append(its.RelaxedSymmetry())
                    removeWindow.append(rem)
                    removeWindowParams.append(remPar)
                    doublePenalty.append(DP)
                    inspirationCriterionThreshold.append(ins)
                    names.append("its/RelaxedSymmetry/" + "inspCrit=" + str(ins) + ("_remWin=" + str(remPar) if rem else "_noRemWin") + ("_doublePenalty" if DP else "_noPenalty"))
    
    
    for DP in [True, False]:
        for ins in [0.1]:
            for remPar in [0.4, 0.5]:
                for rem in [True, False]:
                    features.append(its.GlobalContrastFactor())
                    removeWindow.append(rem)
                    removeWindowParams.append(remPar)
                    doublePenalty.append(DP)
                    inspirationCriterionThreshold.append(ins)
                    names.append("its/GlobalContrastFactor/" + "inspCrit=" + str(ins) + ("_remWin=" + str(remPar) if rem else "_noRemWin") + ("_doublePenalty" if DP else "_noPenalty"))
    
    for DP in [True, False]:
        for ins in [0.1]:
            for remPar in [0.4, 0.5]:
                for rem in [True, False]:
                    features.append(its.Tenengrad())
                    removeWindow.append(rem)
                    removeWindowParams.append(remPar)
                    doublePenalty.append(DP)
                    inspirationCriterionThreshold.append(ins)
                    names.append("its/Tenengrad/" + "inspCrit=" + str(ins) + ("_remWin=" + str(remPar) if rem else "_noRemWin") + ("_doublePenalty" if DP else "_noPenalty"))
    
    
    jobID = int(os.environ.get("PBS_ARRAYID", "0"))
    workingDirectory = os.environ.get("PC2WORK", ".")
    
    index = jobID % len(features)

    runExperiment(jobID / len(features) + 1, names[index], features[index], removeWindow[index], removeWindowParams[index], doublePenalty[index], inspirationCriterionThreshold[index])

    # len(names) = 32
