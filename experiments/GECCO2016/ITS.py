#!/usr/bin/python
import sys
sys.path.append("../../scripts/")
from ITS import *
import os
import timeit
import time

# settings
jobID = 0
workingDirectory = ""

def runExperiment(trial, name, feature, removeWindow, removeWindowParam, doublePenalty, inspirationCriterionThreshold):
    printMessage = name, " trial: ", trial
    print "START: ", printMessage

    # create experiment directory
    directory = createNumberedDirectory(os.path.join(workingDirectory, "ITS256/" + name + "/trial_"))
    
    # prepare experiment
    a = its.ITS()

    a.OptimizationIterations = 10
    a.StagnationLimit = 20
    a.InspirationCriterionThreshold = inspirationCriterionThreshold
    
    # features
    for f in feature:
        a.Features.append(f)
        
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
    
    def saveResults(prefix, mustPass):    
        try: 
            # save information for statistics
            stats.Save(os.path.join(directory, prefix + "statistics"))
            # save additional features values
            globalFeatures.Save(os.path.join(directory, prefix + "features"))
            # save the last population
            a.Population.Save(os.path.join(directory, prefix + "population"))
        except RuntimeError as e:
            print "Exception: " + str(e)
            if mustPass:
                time.sleep(30)
                saveResults(prefix, saveResults)
                
    startTime = timeit.default_timer()

    # run experiment
    i = 0 
    while a.Generation < 2000:
        if i > 0:
            # save current temporal results
            saveResults("temp_", False)
  
        a.OneCreativeCycle()
        print "###Generation: " + str(a.Generation)
        i = i + 1
        
    stopTime = timeit.default_timer()
    
    # save results
    saveResults("", True)
            
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
        for ins in [0.01]:
            for remPar in [0.3]:
                for rem in [True]:
                    features.append([its.NormalizedVariance()])
                    removeWindow.append(rem)
                    removeWindowParams.append(remPar)
                    doublePenalty.append(DP)
                    inspirationCriterionThreshold.append(ins)
                    names.append("NV/" + "inspCrit=" + str(ins) + ("_remWin=" + str(remPar) if rem else "_noRemWin") + ("_doublePenalty" if DP else "_noPenalty"))


    for DP in [False]:
        for ins in [0.01]:
            for remPar in [0.4]:
                for rem in [True]:
                    features.append([its.RelaxedSymmetry()])
                    removeWindow.append(rem)
                    removeWindowParams.append(remPar)
                    doublePenalty.append(DP)
                    inspirationCriterionThreshold.append(ins)
                    names.append("RS/" + "inspCrit=" + str(ins) + ("_remWin=" + str(remPar) if rem else "_noRemWin") + ("_doublePenalty" if DP else "_noPenalty"))
    
    
    for DP in [True]:
        for ins in [0.1]:
            for remPar in [0.4]:
                for rem in [True]:
                    features.append([its.GlobalContrastFactor()])
                    removeWindow.append(rem)
                    removeWindowParams.append(remPar)
                    doublePenalty.append(DP)
                    inspirationCriterionThreshold.append(ins)
                    names.append("GCF/" + "inspCrit=" + str(ins) + ("_remWin=" + str(remPar) if rem else "_noRemWin") + ("_doublePenalty" if DP else "_noPenalty"))
    
    for DP in [True]:
        for ins in [0.04]:
            for remPar in [0.4]:
                for rem in [True]:
                    features.append([its.Tenengrad()])
                    removeWindow.append(rem)
                    removeWindowParams.append(remPar)
                    doublePenalty.append(DP)
                    inspirationCriterionThreshold.append(ins)
                    names.append("T/" + "inspCrit=" + str(ins) + ("_remWin=" + str(remPar) if rem else "_noRemWin") + ("_doublePenalty" if DP else "_noPenalty"))

    for DP in [False]:
        for ins in [0.04]:
            for remPar in [0.3]:
                for rem in [True]:
                    features.append([its.NormalizedVariance(), its.RelaxedSymmetry()])
                    removeWindow.append(rem)
                    removeWindowParams.append(remPar)
                    doublePenalty.append(DP)
                    inspirationCriterionThreshold.append(ins)
                    names.append("NV+RS/" + "inspCrit=" + str(ins) + ("_remWin=" + str(remPar) if rem else "_noRemWin") + ("_doublePenalty" if DP else "_noPenalty"))
    

    for DP in [True]:
        for ins in [0.04]:
            for remPar in [0]:
                for rem in [False]:
                    features.append([its.NormalizedVariance(), its.RelaxedSymmetry(), its.GlobalContrastFactor()])
                    removeWindow.append(rem)
                    removeWindowParams.append(remPar)
                    doublePenalty.append(DP)
                    inspirationCriterionThreshold.append(ins)
                    names.append("NV+RS+GCF/" + "inspCrit=" + str(ins) + ("_remWin=" + str(remPar) if rem else "_noRemWin") + ("_doublePenalty" if DP else "_noPenalty"))
    
    
    for DP in [True]:
        for ins in [0.1]:
            for remPar in [0.5]:
                for rem in [True]:
                    features.append([its.NormalizedVariance(), its.RelaxedSymmetry(), its.Tenengrad()])
                    removeWindow.append(rem)
                    removeWindowParams.append(remPar)
                    doublePenalty.append(DP)
                    inspirationCriterionThreshold.append(ins)
                    names.append("NV+RS+T/" + "inspCrit=" + str(ins) + ("_remWin=" + str(remPar) if rem else "_noRemWin") + ("_doublePenalty" if DP else "_noPenalty"))
    
    for DP in [True]:
        for ins in [0.1]:
            for remPar in [0.5]:
                for rem in [True]:
                    features.append([its.NormalizedVariance(), its.RelaxedSymmetry(), its.GlobalContrastFactor(), its.Tenengrad()])
                    removeWindow.append(rem)
                    removeWindowParams.append(remPar)
                    doublePenalty.append(DP)
                    inspirationCriterionThreshold.append(ins)
                    names.append("NV+RS+GCF+T/" + "inspCrit=" + str(ins) + ("_remWin=" + str(remPar) if rem else "_noRemWin") + ("_doublePenalty" if DP else "_noPenalty"))
    
    
    
    
    jobID = int(os.environ.get("PBS_ARRAYID", "0"))
    workingDirectory = os.environ.get("PC2WORK", ".")
    
    index = jobID % len(features)

    runExperiment(jobID / len(features) + 1, names[index], features[index], removeWindow[index], removeWindowParams[index], doublePenalty[index], inspirationCriterionThreshold[index])

