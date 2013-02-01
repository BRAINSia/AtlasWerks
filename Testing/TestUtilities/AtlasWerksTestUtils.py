#!/usr/bin/python

import sys
import re
import subprocess
import os.path
import time
from xml.dom import minidom

### HTML CR/LF
ENDL = "&#13;&#10"

### START EnsureDir
def EnsureCleanDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
### END EnsureDir

### START GetImageRange
def GetImageRange(image):
    try:
        output = subprocess.Popen(["PrintImageInfo", image], stdout=subprocess.PIPE).communicate()[0]
        output = output.split('\n')
        minmax = [-1, -1]
        for line in output:
            m = re.search(".*Range: \[(.*), (.*)\].*",line)
            if(m):
                minmax[0] = float(m.group(1))
                minmax[1] = float(m.group(2))
        return "Range [%f => %f]" % (minmax[0], minmax[1])
    except (), err:
        return "Unable to determine Range: " + err
### END GetImageRange

### START ExtractImageSlice
# def ExtractImageSlice(imageName, sliceName, slice):
#     try:
#         # extract a slice
#         sliceProc = subprocess.Popen(["unu", "slice", "-a", "2", "-p", str(slice), "-i", imageName], stdout=subprocess.PIPE)
#         # quantize to 8 bits
#         quantizeProc = subprocess.Popen(["unu", "quantize", "-b", "8"], stdin=sliceProc.stdout, stdout=subprocess.PIPE)
#         # save as a png
#         saveProc = subprocess.Popen(["unu", "save", "-f", "png", "-o", sliceName], stdin=quantizeProc.stdout, stdout=subprocess.PIPE)
#         output = saveProc.communicate()[0]
#     except (), err:
#         return "Unable to extract slice: " + err
### END ExtractImageSlice

### START ExtractImageSlice
def ExtractImageSlice(imageName, sliceName, slice, dim="z"):
    try:
        # extract a slice
        sliceProc = subprocess.Popen(["ImageReadSliceWrite", imageName, sliceName, "-slice", str(slice), "-dim", dim], stdout=subprocess.PIPE)
        output = sliceProc.communicate()[0]
    except (), err:
        return "Unable to extract slice: " + err
### END ExtractImageSlice

### START GetImageModifiedTime
def GetModifiedTime(file):
    try:
        modTime = os.path.getmtime(file)
        timeStr = time.ctime(modTime)
        return timeStr
    except (OSError), err:
        return "Unable to get modified time: " + str(err)
### END GetImageModifiedTime

### START Generate image ###
def GetImageHTML(file, imageDir, slice, displaySize, style="border: 2px solid transparent;", link="", titleStr="", sliceBaseName=""):
    canonicalFilePath = os.path.abspath(os.path.normpath(os.path.expanduser(file)))
    fileName = os.path.basename(canonicalFilePath)
    fileDir = os.path.dirname(canonicalFilePath)
    canonicalImageDir = os.path.abspath(os.path.normpath(os.path.expanduser(imageDir)))
    (fileBase, fileExt) = os.path.splitext(fileName)
    if(len(sliceBaseName) == 0):
        sliceBaseName = fileBase
    sliceName = imageDir + os.sep + sliceBaseName 
    imgRangeStr = GetImageRange(canonicalFilePath)
    imgModTimeStr = GetModifiedTime(canonicalFilePath)
    if(len(titleStr) > 0):
        titleStr = titleStr + ENDL
    titleStr = titleStr + canonicalFilePath + ENDL + imgRangeStr + ENDL + imgModTimeStr
    # Extract the slice
    if type(slice) != list:
        slice = [slice]
    dimlist = ["z"]
    if len(slice) == 3:
        dimlist = ["x", "y", "z"]
    elif len(slice) != 1:
        return "Error, slice must be integer or 3-element list"

    htmlStr = ""
    for i in range(len(slice)):
        curSliceName = sliceName + ("Slice%03d" % slice[i]) + dimlist[i] + ".png"
        ExtractImageSlice(canonicalFilePath, curSliceName, slice[i], dimlist[i])
        if(len(link) == 0):
            link = canonicalFilePath
        curHtmlStr = "<a href=\""+link+"\"><img style=\"" + style + "\" src=\"" + curSliceName
        if len(displaySize) == 2:
            curHtmlStr = curHtmlStr + ("\" height=%d width=%d title=\"" % displaySize)
        else:
            curHtmlStr = curHtmlStr + "\" title=\""
        curHtmlStr = curHtmlStr + titleStr + "\"></a>"
        curHtmlStr = curHtmlStr + "<br/>"
        htmlStr = htmlStr + curHtmlStr
    return htmlStr
### END Generate image ###
    
### START Generate start html ###
def GetStartHTML(pageTitle):
    return "<html><head><title>" + pageTitle + "</title></head><body>"
### END Generate start html ###

### START Generate end html ###
def GetEndHTML():
    curTimeStr = time.ctime()
    return ENDL + "Modified at " + curTimeStr + ENDL + "</body></html>"
### END Generate end html ###

### START Image Diff ###
def GetImageDiff(testIm, baseIm, diffName, tmpDir):
    testImCanonicalPath = os.path.abspath(os.path.normpath(os.path.expanduser(testIm)))
    baseImCanonicalPath = os.path.abspath(os.path.normpath(os.path.expanduser(baseIm)))
    diffPath = tmpDir + os.sep + diffName
    # get the difference image
    try:
        diffProc = subprocess.Popen(["GenDiff", testImCanonicalPath, baseImCanonicalPath, "-o", diffPath], stdout=subprocess.PIPE)
        diffOut = diffProc.communicate()[0]
        m = re.search("err:\\s*(\\S.*)",diffOut)
        err = float('inf')
        if(m):
            err = float(m.group(1))
        return err;
    except (), err:
        print "Unable to generate difference image: " + err
        return float('inf')
### END Image Diff ###

### START Image test ###
def TestImages(testIm, baseIm, imageDir, slice, displaySize, tmpDir, diffName="", eps=1.0):
    baseImCanonicalPath = os.path.abspath(os.path.normpath(os.path.expanduser(baseIm)))
    if(len(diffName) == 0):
        (testImPath, testImBase) = os.path.split(testIm)
        (testImBase, testImExt) = os.path.splitext(testImBase)
        diffName=testImBase + "_diff" + testImExt;
    diffPath = tmpDir + os.sep + diffName
    diff = GetImageDiff(testIm, baseIm, diffName, tmpDir)
    diffStr = ("err: %e against " % diff) + baseImCanonicalPath
    passTest = True
    if(diff < eps):
        imgHtml = GetImageHTML(testIm, imageDir, slice, displaySize, style="border: 2px solid blue;", titleStr=diffStr)
        errHtml = ""
    else:
        passTest = False
        anchor = "Err" + testIm
        imgHtml = GetImageHTML(testIm, imageDir, slice, displaySize, 
                               style="border: 2px solid red;", titleStr=diffStr, link=("#"+anchor))
        errHtml = "<a name=\"" + anchor + "\"><h3>Image " + testIm + "</h3></a>" + ENDL + \
            diffStr + "<br>" + \
            "<table><tr>" + ENDL + \
            "<td>" +GetImageHTML(testIm, imageDir, slice, displaySize, titleStr=diffStr) + "</td>" +  ENDL + \
            "<td>" +GetImageHTML(baseIm, imageDir, slice, displaySize, titleStr=diffStr, sliceBaseName=testImBase+"_base") + "</td>" +  ENDL + \
            "<td>" +GetImageHTML(diffPath, imageDir, slice, displaySize, titleStr=diffStr) + "</td>" +  ENDL + \
            "</tr></table>"
    return (passTest, imgHtml, errHtml)
### END Image test ###

### START search for tag in xml file ###
def GetXMLElement(xmlFile, elementName):
    # xmldoc is a Document
    xmldoc = minidom.parse(xmlFile)
    # get children of xmldoc with the given name
    elements = xmldoc.getElementsByTagName(elementName)
    elVals = [];
    for el in elements:
        val = el.getAttribute('val');
        if(len(val) != 0):
            elVals.append(val)
    return elVals
### END search for tag in xml file ###
