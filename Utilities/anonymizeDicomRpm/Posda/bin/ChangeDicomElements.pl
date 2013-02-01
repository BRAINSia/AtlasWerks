#!/usr/bin/perl -w
#$Source: /home/bbennett/pass/archive/Posda/bin/ChangeDicomElements.pl,v $
#$Date: 2008/05/14 16:19:06 $
#$Revision: 1.4 $
#
#Copyright 2008, Bill Bennett
# Part of the Posda package
# Posda may be copied only under the terms of either the Artistic License or the
# GNU General Public License, which may be found in the Posda Distribution,
# or at http://posda.com/License.html
#
use strict;
use Posda::Parser;
use Posda::Dataset;

Posda::Dataset::InitDD();

my $from = $ARGV[0];
my $to = $ARGV[1];

my($df, $ds, $size, $xfr_stx, $errors)  = Posda::Dataset::Try($ARGV[0]);
unless($ds) { die "$from didn't parse into a dataset" }
my $count = @ARGV;
unless(($count & 1) == 0){ 
  for my $i (0 .. $#ARGV){
    print "ARGV[$i] = $ARGV[$i]\n";
  }
  die "need an even number of args" 
};
my $pairs = $#ARGV/2;
if($pairs > 0){
  for my $i (1 .. $pairs){
    my $pair_id = $i * 2;
    my $sig = $ARGV[$pair_id];
    my $value = $ARGV[$pair_id + 1];
    print "$sig => $value\n";
    $ds->InsertElementBySig($sig, $value);
  }
}
if($df){
  $ds->WritePart10($to, $xfr_stx, "DICOM_TEST", undef, undef);
} else {
  $ds->WriteRawDicom($to, $xfr_stx);
}
