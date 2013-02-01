#!/usr/bin/python

from AtlasWerksTestUtils import *
import os;
import os.path;
import sys;

#if(len(sys.argv) > 1):
#    basePath = sys.argv[1] + os.sep
#else:
#    basePath = "./"
     
imageDisplaySize = (128, 128);

warpDir = "RegressionTemp/"
imageDir = "images/"
baselineDir = "Baseline/"
parsedXML = warpDir + "AtlasWerksParsedOutput.xml"

nImages = int(GetXMLElement(parsedXML, "NumFiles")[0]);
sliceToExtract = 16

# High eps, will make a more exact match in next version (GreedyAtlas)
eps = 500

ErrorHtmlSection = "<hr><h1>Errors:</h1><br>"

TestsPass = True

print GetStartHTML("Greedy Atlas Tests")

# energy graph
print "<h2>Greedy Atlas Convergence</h2>"
print "<img src=\"AtlasEnergy.png\">"

# link to xml file
print "<br><a href=\"" + parsedXML + "\">Warp XML</a>"

# get original image format
inputFormat = GetXMLElement(parsedXML, "FormatString");
inputFormat = warpDir + inputFormat[0];

# get mean image
imageData = "averageImage.nhdr"
(passTest, meanHtml, errHtml) = TestImages(warpDir+imageData, baselineDir+imageData, 
                                          imageDir, sliceToExtract, imageDisplaySize, imageDir, "", eps)
if(passTest == False):
    TestsPass = False

print "<h2>Mean Image</h2>"
print meanHtml
ErrorHtmlSection = ErrorHtmlSection + errHtml

print "<h2>Greedy Atlas Images</h2>"

print "<table>"
# Greedy Atlas Images
print "<tr>"
print "<td></td>"
print "<td>Input</td>"
print "<td>Deformed</td>"
print "</tr>"
for i in range(0,nImages):
    print "<tr>"
    print "<td>Image %d</td>" % i
    # Print initial image
    imageData = inputFormat % i
    print "<td>" + GetImageHTML(imageData, imageDir, sliceToExtract, imageDisplaySize) + "</td>"
    # Print deformed image
    imageData = "deformedImage%04d.nhdr" % i
    (passTest, imgHtml, errHtml) = TestImages(warpDir+imageData, baselineDir+imageData, 
                                                  imageDir, sliceToExtract, imageDisplaySize, imageDir, "", eps)
    if(passTest == False):
        TestsPass = False
        
    print "<td>" + imgHtml + "</td>"
    ErrorHtmlSection = ErrorHtmlSection + errHtml
    print "</tr>"
print "</table>"

print ErrorHtmlSection

print GetEndHTML()

if(TestsPass):
    sys.exit(0)
else:
    sys.exit(-1)
            
