#!/usr/bin/env python
#
# PrepCINEforRecon.py - take CINE data and RPM *.vxp file and produce data
#  directory of *.raw files and master amplitudes.csv data file
#
# jh 2008

import sys,csv,os,shutil

def readVXP(vxp):
  """
  Read in *.vxp file and output list of tuples (time, amplitude, ttlout),
  sorted by time.
  """
  print "Reading %s..." % vxp

  output = []
  firstTTLout = 1e5000
  reader = csv.reader(open(vxp,'rb'))
  for row in reader:
    if len(row) == 7 and row[0][0] != 'D': # correct
      # row[0]=amp, row[1]=phase, row[2]=time, row[3]=valid
      # row[4]=ttlin, row[5]=mark, row[6]=ttlout
      # NOTE: RPM time is in ms in file so adjust it here
      time = float(row[2])*.001
      amp = float(row[0])
      phase = float(row[1])
      ttlout = int(row[6])
      output.append((time,amp,phase))
      if firstTTLout > time and ttlout == 1:
        firstTTLout = time

  print "First TTLOUT: %f s" % firstTTLout
  return (firstTTLout,output)

def interpAmpPhase(amps,firstTTLout,firstCT,t):
  """
  Interpolate to find amplitude and phase at time t given VXP data in amps
  """
  # Get first ttlout

  offset = firstTTLout - firstCT
  prevtime = 0
  prevamp = 0
  prevphase = 0
  for rpm in amps:
    normrpmtime = rpm[0] - firstTTLout
    normt = t-firstCT
    curramp = rpm[1]
    currphase = rpm[2]

    if normrpmtime > normt:
      # interpolate between prevt and this time
      a = ((normrpmtime-normt)*prevamp + (normt-prevtime)*curramp)/(normrpmtime-prevtime)
      p = ((normrpmtime-normt)*prevphase + (normt-prevtime)*currphase)/(normrpmtime-prevtime)
      return (a,p)
    
    prevtime = normrpmtime
    prevamp = curramp
    prevphase = currphase

  print "ERROR: slice time is out of RPM range (%f)" % t
  sys.exit(1)

if __name__ == '__main__':
  # get arguments
  if len(sys.argv) != 4:
    print "%s - prepare CINE/RPM data for 4D recon" % sys.argv[0]
    print "Usage: %s dicomdir breathing.vxp outputdir" % sys.argv[0]
    sys.exit(1)

  # Get amplitude data and times from *.vxp file
  (firstTTLout,amps) = readVXP(sys.argv[2])

  print "Reading DICOM Slices..."
  CTinfo = []
  firstCT = 1e5000
  # Loop through dicoms
  for file in os.listdir(sys.argv[1]):
    #print file

    # skip non-dicoms
    if os.path.splitext(file)[1] != '.dimg': continue

    # Read MidScanTime
    t = float(os.popen('dcmdump +P "0019,1024" ' + sys.argv[1] + '/' + file + ' | awk \'{print $3}\' | tr -d "[]"\\n').read())
    # Read SliceLocation
    z = float(os.popen('dcmdump +P "SliceLocation" ' + sys.argv[1] + '/' + file + ' | awk \'{print $3}\' | tr -d "[]"\\n').read())
    
    # Look for first slice acquired too (corresponds to first TTLOUT)
    if firstCT > t:
      firstCT = t
      
    CTinfo.append((t,z,file))
  
  print "First CT Slice: %f s" % firstCT

  print "Writing Raw Files & Index..."
  outfile = open(sys.argv[3]+'/index.csv','w')
  n = 0
  for sl in CTinfo:
    # interpolate amplitude from amps
    (a,p) = interpAmpPhase(amps,firstTTLout,firstCT,sl[0])

    # write to *.csv
    # Order is filename,slicelocation,rpmamplitude
    outfile.write("%04d,%s,%f,%f,%f\n" % (n,sl[2],sl[1],a,p))

    # write *.raw with dcmdump
    rawfile = os.popen("dcmdump +W %s %s/%s | grep 7fe0,0010 | awk '{print $3}' | tr -d '=\\n'" % (sys.argv[3],sys.argv[1],sl[2])).read()

    # move to avoid overwrites
    print "%s -> %s/%04d.raw" % (rawfile,sys.argv[3],n)
    shutil.move(rawfile,"%s/%04d.raw" % (sys.argv[3],n))

    n=n+1

  outfile.close()
  
