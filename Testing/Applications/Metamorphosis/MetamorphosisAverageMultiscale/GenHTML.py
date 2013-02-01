#!/usr/bin/python

from AtlasWerksTestUtils import *
import os.path;

imageDisplaySize = (128, 128);

warpDir = "RegressionTemp/"
imageDir = "images/"
baselineDir = "Baseline/"
parsedXML = warpDir + "MetamorphosisAverageMultiscaleParsedOutput.xml"

nImages = int(GetXMLElement(parsedXML, "NumFiles")[0]);
nTimeSteps = 4
sliceToExtract = 16

ErrorHtmlSection = "<hr><h1>Errors:</h1><br>"

TestsPass = True

print GetStartHTML("Metamorphosis Tests")

# energy graph
print "<h2>Metamorphosis Atlas Convergence</h2>"
print "<img src=\"AtlasEnergy.png\">"

# link to xml file
print "<br><a href=\"" + parsedXML + "\">Warp XML</a>"

# get original image format
inputFormat = GetXMLElement(parsedXML, "FormatString");
inputFormat = warpDir + inputFormat[0];

# get mean image
imageData = "MetamorphosisAtlasMean.nhdr"
(passTest, meanHtml, errHtml) = TestImages(warpDir+imageData, baselineDir+imageData, 
                                          imageDir, sliceToExtract, imageDisplaySize, imageDir)
if(passTest == False):
    TestsPass = False

print "<h2>Mean Image</h2>"
print meanHtml
ErrorHtmlSection = ErrorHtmlSection + errHtml

print "<h2>Metamorphosis Atlas Images</h2>"

print "<table>"
# Metamorphosis Forward-Deformed Images
for i in range(0,nImages):
    print "<tr>"
    print "<td>Metamorphosis for image %d</td>" % i
    # Print initial image
    imageData = inputFormat % i
    print "<td>" + GetImageHTML(imageData, imageDir, sliceToExtract, imageDisplaySize) + "</td>"
    # Print intermediate images
    for t in range(nTimeSteps-1,0,-1):
        warpImageData = "MetamorphosisAtlasImage%02dTime%02d.nhdr" % (i,t)
        (passTest, imgHtml, errHtml) = TestImages(warpDir+warpImageData, baselineDir+warpImageData, 
                                                  imageDir, sliceToExtract, imageDisplaySize, imageDir)
        if(passTest == False):
            TestsPass = False
        
        print "<td>" + imgHtml + "</td>"
        ErrorHtmlSection = ErrorHtmlSection + errHtml
    # Print mean image
    print "<td>",meanHtml,"</td>"
    print "</tr>"
print "</table>"

print ErrorHtmlSection

print GetEndHTML()

if(TestsPass):
    sys.exit(0)
else:
    sys.exit(-1)
            
