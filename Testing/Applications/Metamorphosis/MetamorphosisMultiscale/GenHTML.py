#!/usr/bin/python

from AtlasWerksTestUtils import *
import os.path;

imageDisplaySize = (128, 128);

warpDir = "RegressionTemp/"
imageDir = "images/"
baselineDir = "Baseline/"
parsedXML = warpDir + "MetamorphosisMultiscaleParsedOutput.xml"

nTimeSteps = int(GetXMLElement(parsedXML, "NTimeSteps")[0]);
sliceToExtract = 16

ErrorHtmlSection = "<hr><h1>Errors:</h1><br>"

TestsPass = True

print GetStartHTML("Metamorphosis Tests")

# energy graph
print "<h2>Metamorphosis Warp Convergence</h2>"
print "<img src=\"WarpEnergy.png\">"

# link to xml file
print "<br><a href=\"" + parsedXML + "\">Warp XML</a>"

# get input images
inputImage = GetXMLElement(parsedXML, "InitialImage")[0];
finalImage = GetXMLElement(parsedXML, "FinalImage")[0];

print "<h3>Initial Image</h3>"
print GetImageHTML(warpDir+inputImage, imageDir, sliceToExtract, imageDisplaySize)
print "<h3>Final Image</h3>"
print GetImageHTML(warpDir+finalImage, imageDir, sliceToExtract, imageDisplaySize)

print "<h2>Metamorphosis Warp Images</h2>"
print "<table>"
print "<tr>"
print "<td>Metamorphosis Warp</td>"
for t in range(0,nTimeSteps+1):
    warpImageData = "MetamorphosisWarpImage%02d.nhdr" % t
    (passTest, imgHtml, errHtml) = TestImages(warpDir+warpImageData, baselineDir+warpImageData, 
                                              imageDir, sliceToExtract, imageDisplaySize, imageDir)
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
            
