#!/usr/bin/python
import sys
sys.path.append("../../scripts/")
from ITS import *
 

### show images from uninspired search used for the test by complexity 

# load database
db = loadForPython("ImagesDatabase_BasicFeatures.dat")
dbInfoName = loadForPython("ImagesDatabase_BasicFeatures_Info.dat")

# show images
for i in range(len(db)):
    print dbInfoName[i]
    showImage(db[i])


# show images from older experiment without additonal information
map(showImage, loadForPython("ImagesDatabase_oneFeature.dat"))




### show images from ITS used for the test by complexity 

# load database
db = loadForPython("ImagesDatabase_ITS.dat")
dbInfoName = loadForPython("ImagesDatabase_ITS_Info.dat")

# show images
for i in range(len(db)):
    print dbInfoName[i]
    showImage(db[i])

