#!/usr/bin/python

from AtlasWerksTestUtils import *
import os.path;

imageDisplaySize = (128, 128);

warpDir = "RegressionTemp/"
imageDir = "images/"
baselineDir = "Baseline/"
parsedXML = warpDir + "LDMMWarpParsedOutput.xml"

nTimeSteps = int(GetXMLElement(parsedXML, "NTimeSteps")[0]);
sliceToExtract = 16

ErrorHtmlSection = "<hr><h1>Errors:</h1><br>"

TestsPass = True

print GetStartHTML("LDMMShootingWarp Tests")

# energy graph
print "<h2>LDMM Warp Convergence</h2>"
print "<img src=\"WarpEnergy.png\">"

# link to xml file
print "<br><a href=\"" + parsedXML + "\">Warp XML</a>"

# get input images
inputImage = GetXMLElement(parsedXML, "MovingImage")[0];
finalImage = GetXMLElement(parsedXML, "StaticImage")[0];

print "<h3>Moving Image</h3>"
print GetImageHTML(warpDir+inputImage, imageDir, sliceToExtract, imageDisplaySize)
print "<h3>Static Image</h3>"
print GetImageHTML(warpDir+finalImage, imageDir, sliceToExtract, imageDisplaySize)

print "<h2>LDMM Warp Images</h2>"
print "<table>"
print "<tr>"
print "<td>LDMM Warp</td>"
warpImageData = "LDMMWarpBullseyeTestBlur00_to_BullseyeTestBlur01DefImage.mha"
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
            
