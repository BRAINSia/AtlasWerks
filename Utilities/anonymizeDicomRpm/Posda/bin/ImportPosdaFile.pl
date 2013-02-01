#!/usr/bin/perl -w
#$Source: /home/bbennett/pass/archive/Posda/bin/ImportPosdaFile.pl,v $
#$Date: 2008/04/30 19:17:34 $
#$Revision: 1.3 $
#
#Copyright 2008, Bill Bennett
# Part of the Posda package
# Posda may be copied only under the terms of either the Artistic License or the
# GNU General Public License, which may be found in the Posda Distribution,
# or at http://posda.com/License.html
#
use Posda::DB::File;
use DBI;
my $path = $ARGV[0];
my $db = DBI->connect("dbi:Pg:dbname=$ARGV[1]", "", "");
my $comment = $ARGV[2];
unless($db) { die "couldn't connect to DB: $ARGV[1]" }
unless(-d $path) { die "$path is not a directory" }
Posda::DB::File::ScanDirectory($path, $db, $comment);
