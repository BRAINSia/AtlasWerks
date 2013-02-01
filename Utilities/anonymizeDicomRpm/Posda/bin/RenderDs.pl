#!/usr/bin/perl -w
#$Source: /home/bbennett/pass/archive/Posda/bin/RenderDs.pl,v $
#$Date: 2008/04/30 19:17:34 $
#$Revision: 1.2 $
#
#Copyright 2008, Bill Bennett
# Part of the Posda package
# Posda may be copied only under the terms of either the Artistic License or the
# GNU General Public License, which may be found in the Posda Distribution,
# or at http://posda.com/License.html
#
use Posda::Parser;
use Posda::Dataset;
my $file = $ARGV[0];
my $xfr_stx = $ARGV[1];

Posda::Dataset::InitDD();
open FILE, "<$file" or die "can't open $file";
my $mh = Posda::Parser::ReadMetaHeader(\*FILE);
if($mh){
  my $file_xfr_stx = $mh->{xfrstx};
  if($xfr_stx eq $file_xfr_stx){
    my $start = $mh->{DataSetStart};
    seek FILE, $start, 0;
    my $buff;
    while(read(FILE, $buff, 1024)){
      print $buff;
    }
    exit;
  }
}
close FILE;
unless(
   $xfr_stx eq "1.2.840.10008.1.2" ||
   $xfr_stx eq "1.2.840.10008.1.2.1" ||
   $xfr_stx eq "1.2.840.10008.1.2.2"
){
  die "Xfr_stx $xfr_stx not currently supported for conversion (to)";
}
my($df, $ds, $size, $from_xfr_stx, $errors)  = Posda::Dataset::Try($ARGV[0]);
if($ds){
  unless(
     $from_xfr_stx eq "1.2.840.10008.1.2" ||
     $from_xfr_stx eq "1.2.840.10008.1.2.1" ||
     $from_xfr_stx eq "1.2.840.10008.1.2.2"
  ){
    die "Xfr_stx $from_xfr_stx not currently supported for conversion (from)";
  }
  if($xfr_stx eq "1.2.840.10008.1.2"){
    Posda::Dataset::WriteImpLe($ds, \*STDOUT);
  } elsif($xfr_stx eq "1.2.840.10008.1.2.1"){
    Posda::Dataset::WriteExpLe($ds, \*STDOUT);
  } elsif($xfr_stx eq "1.2.840.10008.1.2.2"){
    Posda::Dataset::WriteExpBe($ds, \*STDOUT);
  }
}
