#!/usr/bin/python

from AtlasWerksTestUtils import *
import os.path;

imageDisplaySize = (128, 128);

warpDir = "RegressionTemp/"
imageDir = "images/"
baselineDir = "Baseline/"
parsedXML = warpDir + "LDMMAtlasParsedOutput.xml"

nImages = int(GetXMLElement(parsedXML, "NumFiles")[0]);
nTimeSteps = int(GetXMLElement(parsedXML, "NTimeSteps")[0]);
sliceToExtract = 16

ErrorHtmlSection = "<hr><h1>Errors:</h1><br>"

TestsPass = True

EPS=10.0

print GetStartHTML("LDMMShootingAtlas Tests")

# energy graph
print "<h2>LDMM Atlas Convergence</h2>"
print "<img src=\"AtlasEnergy.png\">"

# link to xml file
print "<br><a href=\"" + parsedXML + "\">Warp XML</a>"

# get original image format
inputFormat = GetXMLElement(parsedXML, "FormatString");
inputFormat = warpDir + inputFormat[0];

# get mean image
imageData = "LDMMShootingAtlasMeanImage.mha"
(passTest, meanHtml, errHtml) = TestImages(warpDir+imageData, baselineDir+imageData, 
                                          imageDir, sliceToExtract, imageDisplaySize, imageDir,
                                           eps=EPS)
if(passTest == False):
    TestsPass = False

print "<h2>Mean Image</h2>"
print meanHtml
ErrorHtmlSection = ErrorHtmlSection + errHtml

print "<h2>LDMM Atlas Images</h2>"

print "<table>"
# LDMM Atlas Images
print "<tr>"
print "<td>Timestep</td>"
for t in range(nTimeSteps,-1,-1):
    print "<td>%d</td>" % t
print "</tr>"
for i in range(0,nImages):
    print "<tr>"
    print "<td>LDMM for image %d</td>" % i
    # Print initial image
    imageData = inputFormat % i
    print "<td>" + GetImageHTML(imageData, imageDir, sliceToExtract, imageDisplaySize) + "</td>"
    # Print intermediate images
    warpImageData = "LDMMShootingAtlasBullseyeTestBlur%02dDefToMean.mha" % i
    (passTest, imgHtml, errHtml) = TestImages(warpDir+warpImageData, baselineDir+warpImageData, 
                                              imageDir, sliceToExtract, imageDisplaySize, imageDir,
                                              eps=EPS)
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
            
