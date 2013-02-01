#!/usr/bin/perl -w
#$Source: /home/bbennett/pass/archive/Posda/include/Posda/Find.pm,v $
#$Date: 2008/04/30 19:17:35 $
#$Revision: 1.4 $
#
#Copyright 2008, Bill Bennett
# Part of the Posda package
# Posda may be copied only under the terms of either the Artistic License or the
# GNU General Public License, which may be found in the Posda Distribution,
# or at http://posda.com/License.html
#
package Posda::Find;
use strict;
use Posda::Dataset;
use Posda::Parser;
use File::Find;

sub MakeWanted {
  my($callback) = @_;
  my $wanted = sub {
    my $f_name = $File::Find::name;
    if(-d $f_name) { return }
    unless(-r $f_name) { return }
    my($df, $ds, $size, $xfr_stx, $errors) = 
      Posda::Dataset::Try($f_name);
    unless(defined $ds){ return }
    &$callback($f_name, $df, $ds, $size, $xfr_stx, $errors);
  };
  return $wanted;
}

sub SearchDir{
  my($dir, $cb) = @_;
  find({wanted => MakeWanted($cb), follow => 1}, $dir);
}
1;
