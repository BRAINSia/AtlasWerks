#!/usr/bin/perl -w
#$Source: /home/bbennett/pass/archive/Posda/bin/VaxDump.pl,v $
#$Date: 2008/04/30 19:17:34 $
#$Revision: 1.2 $
#
#Copyright 2008, Bill Bennett
# Part of the Posda package
# Posda may be copied only under the terms of either the Artistic License or the
# GNU General Public License, which may be found in the Posda Distribution,
# or at http://posda.com/License.html
#
use HexDump;
my $file = $ARGV[0];
open FILE, "<$file" or die "Can't open $file";
my $offset = 0;
my $len = read(FILE, $buff, 1024);
while($len > 0){
  HexDump::PrintVax(\*STDOUT, $buff, $offset);
  $offset += 1024;
  $len = read(FILE, $buff, 1024);
}
close FILE;
