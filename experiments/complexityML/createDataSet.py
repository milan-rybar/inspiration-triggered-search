#!/usr/bin/python
import sys
sys.path.append("../../scripts/")
from ITS import *


images = []
Y = []

for imageFile in glob.glob("../simple/*.png"):
    image = loadImage(imageFile)
    images.append(image)
    Y.append(False)
    
for imageFile in glob.glob("../complex/*.png"):
    image = loadImage(imageFile)
    images.append(image)
    Y.append(True)


features = []
features.append(its.NormalizedVariance())  
features.append(its.MaxAbsLaplacian())
features.append(its.Tenengrad())
features.append(its.Choppiness())  
features.append(its.RelaxedSymmetry())
features.append(its.GlobalContrastFactor())
features.append(its.JpegImageComplexity())  

    
featuresX = []


print "creating"

for image in images:
    imageFeatures = map(lambda f: f.Evaluate(image), features)
    featuresX.append(imageFeatures)


print "saving"

saveForPython(Y, "Y.dat")
saveForPython(featuresX, "featuresX.dat")

