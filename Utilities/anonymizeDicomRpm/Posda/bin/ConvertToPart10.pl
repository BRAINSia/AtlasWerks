#!/usr/bin/perl -w
#$Source: /home/bbennett/pass/archive/Posda/bin/ConvertToPart10.pl,v $
#$Date: 2008/07/31 19:29:29 $
#$Revision: 1.7 $
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

my $from = $ARGV[0];
my $to = $ARGV[1];

Posda::Dataset::InitDD();

my($df, $ds, $size, $xfr_stx, $errors)  = Posda::Dataset::Try($ARGV[0]);
unless($ds) { die "$from didn't parse into a dataset" }
$ds->MapToConvertPvt();
$ds->WritePart10($to, $xfr_stx, "DICOM_TEST", undef, undef);
