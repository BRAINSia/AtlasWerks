#!/usr/bin/perl -w
#$Source: /home/bbennett/pass/archive/Posda/bin/contrib/OverlaysToPbm.pl,v $
#$Date: 2008/09/03 12:20:25 $
#$Revision: 1.6 $
#
#Copyright 2008, Bill Bennett
# Part of the Posda package
# Posda may be copied only under the terms of either the Artistic License or the
# GNU General Public License, which may be found in the Posda Distribution,
# or at http://posda.com/License.html
#
use strict;
use Posda::Dataset;

Posda::Dataset::InitDD;
my $file = $ARGV[0];
my $ele_lookup = {
  "0010" => {
     name => "rows",
     type => 'short',
  }, 
  "0011" => {
     name => "cols",
     type => 'short',
  }, 
  "0022" => {
     name => "description",
     type => 'text',
  }, 
  "0040" => {
     name => "type",
     type => 'text',
  }, 
  "0050" => {
     name => "origin",
     type => 'short',
  }, 
  "0100" => {
     name => "bits_allocated",
     type => 'short',
  }, 
  "0102" => {
     name => "bit_position",
     type => 'short',
  }, 
  "1500" => {
     name => "overlay_label",
     type => 'text',
  }, 
  "3000" => {
     name => "overlay_data",
     type => 'OB',
  }, 
};
my($df, $ds, $size, $xfr_stx, $errors)  = Posda::Dataset::Try($file);
my %Overlays;
my $foo = sub {
  my($ele, $sig) = @_;
  if($sig =~ /^\(60(..),(....)\)$/){
    my $ov_no = hex($1)/2;
    my $el = $2;
    if(
      $ele->{VR} eq "UN"
    ){
      unless(defined $ele_lookup->{$el}){
        print "unknown ele: $el\n";
        return;
      }
      if($ele_lookup->{$el} eq "text"){
        $Overlays{$ov_no}->{$el} =~ s/ $//;
      } elsif ($ele_lookup->{$el}->{type} eq "short"){
        my @foo = unpack("v*", $ele->{value});
        if($#foo == 0){
          $Overlays{$ov_no}->{$el} = $foo[0];
        } else {
          $Overlays{$ov_no}->{$el} = \@foo;
        }
      } else {
        $Overlays{$ov_no}->{$el} = $ele->{value};
      }
    } else {
      $Overlays{$ov_no}->{$el} = $ele->{value};
    }
  }
};
if($ds){
  $ds->MapPvt($foo);
} else {
  die "$file didn't parse";
}
for my $i (sort { $a <=> $b } keys %Overlays){
  my $decoded;
  for my $key (sort keys %{$Overlays{$i}}){
    unless($key eq "3000"){
      $decoded->{$ele_lookup->{$key}->{name}} = $Overlays{$i}->{$key};
    }
  }
  my $rows = $decoded->{rows};
  my $cols = $decoded->{cols};
  my $file_name = $Overlays{$i}->{"1500"};
  $file_name =~ s/ //g;
  my $file_name_pbm = "$file_name.pbm";
  open FILE, ">$file_name_pbm" or die "can't open $file_name_pbm";
  print FILE "P4\n";
  print FILE "$cols\n";
  print FILE "$rows\n";
  my $new_img = '\0' x ($rows * $cols);
  my $old_img = $Overlays{$i}->{"3000"};
  my $len_old = length($old_img);
  my $len_new = length($new_img);
  my @array;
  for my $i (0 .. ($rows * $cols) - 1){
    my $byte_offset = int($i / 8);
    my $bit_no = $i - ($byte_offset * 8);
    my $from_bit = ($byte_offset * 8) + $bit_no;
    my $to_bit = ($byte_offset * 8) + (7 - $bit_no);
    vec($new_img, $to_bit, 1) = vec($old_img, $from_bit, 1);
    my $bit = vec($old_img, $from_bit, 1);
    $array[($byte_offset * 8) + $bit_no] = $bit;
  }
  # Look for bare points:
  my @points;
  for my $row_i (0 .. $rows - 1){
    for my $col_i (0 .. $cols - 1){
      my $index = $col_i + ($cols * $row_i);
      my($point_above, $point_left, $point_right, $point_below);
      my($point_ul, $point_ur, $point_ll, $point_lr);
      if(
        $row_i > 0 && $row_i < $rows &&
        $col_i > 0 && $col_i < $cols
      ){
        my $pa_index_above = $col_i + ($cols * ($row_i - 1));
        my $pa_index_below = $col_i + ($cols * ($row_i + 1));
        my $pa_index_left = ($col_i - 1) + ($cols * $row_i);
        my $pa_index_right = ($col_i + 1) + ($cols * $row_i);
        my $pa_index_ul =  ($col_i - 1) + ($cols * ($row_i - 1));
        my $pa_index_ur =  ($col_i + 1) + ($cols * ($row_i - 1));
        my $pa_index_ll =  ($col_i - 1) + ($cols * ($row_i + 1));
        my $pa_index_lr =  ($col_i + 1) + ($cols * ($row_i + 1));
        $point_above = $array[$pa_index_above];
        $point_below = $array[$pa_index_below];
        $point_left = $array[$pa_index_left];
        $point_right = $array[$pa_index_right];
        $point_ul = $array[$pa_index_ul];
        $point_ur = $array[$pa_index_ur];
        $point_ll = $array[$pa_index_ll];
        $point_lr = $array[$pa_index_lr];
      } elsif($row_i == 0 && $col_i < $cols && $cols > 0){
        my $pa_index_below = $col_i + ($cols * ($row_i + 1));
        my $pa_index_left = ($col_i - 1) + ($cols * $row_i);
        my $pa_index_right = ($col_i + 1) + ($cols * $row_i);
        my $pa_index_ll =  ($col_i - 1) + ($cols * ($row_i + 1));
        my $pa_index_lr =  ($col_i + 1) + ($cols * ($row_i + 1));
        $point_below = $array[$pa_index_below];
        $point_left = $array[$pa_index_left];
        $point_right = $array[$pa_index_right];
        $point_ll = $array[$pa_index_ll];
        $point_lr = $array[$pa_index_lr];
      } elsif($row_i == ($rows - 1) && $col_i < $cols && $cols < 0){
        my $pa_index_above = $col_i + ($cols * ($row_i - 1));
        my $pa_index_left = ($col_i - 1) + ($cols * $row_i);
        my $pa_index_right = ($col_i + 1) + ($cols * $row_i);
        my $pa_index_ul =  ($col_i - 1) + ($cols * ($row_i - 1));
        my $pa_index_ur =  ($col_i + 1) + ($cols * ($row_i - 1));
        $point_above = $array[$pa_index_above];
        $point_left = $array[$pa_index_left];
        $point_right = $array[$pa_index_right];
        $point_ul = $array[$pa_index_ul];
        $point_ur = $array[$pa_index_ur];
      } elsif($row_i > 0 && $row_i < $rows && $col_i == 0){
        my $pa_index_above = $col_i + ($cols * ($row_i - 1));
        my $pa_index_below = $col_i + ($cols * ($row_i + 1));
        my $pa_index_right = ($col_i + 1) + ($cols * $row_i);
        my $pa_index_ur =  ($col_i + 1) + ($cols * ($row_i - 1));
        my $pa_index_lr =  ($col_i + 1) + ($cols * ($row_i + 1));
        $point_above = $array[$pa_index_above];
        $point_below = $array[$pa_index_below];
        $point_right = $array[$pa_index_right];
        $point_ur = $array[$pa_index_ur];
        $point_lr = $array[$pa_index_lr];
      } elsif($row_i > 0 && $row_i < $rows && $col_i == ($cols - 1)){
        my $pa_index_above = $col_i + ($cols * ($row_i - 1));
        my $pa_index_below = $col_i + ($cols * ($row_i + 1));
        my $pa_index_left = ($col_i - 1) + ($cols * $row_i);
        my $pa_index_ul =  ($col_i - 1) + ($cols * ($row_i - 1));
        my $pa_index_ll =  ($col_i - 1) + ($cols * ($row_i + 1));
        $point_above = $array[$pa_index_above];
        $point_below = $array[$pa_index_below];
        $point_left = $array[$pa_index_left];
        $point_ul = $array[$pa_index_ul];
        $point_ll = $array[$pa_index_ll];
      } elsif($row_i == 0 && $col_i == 0){
        my $pa_index_below = $col_i + ($cols * ($row_i + 1));
        my $pa_index_left = ($col_i - 1) + ($cols * $row_i);
        my $pa_index_ll =  ($col_i - 1) + ($cols * ($row_i + 1));
        $point_below = $array[$pa_index_below];
        $point_left = $array[$pa_index_left];
        $point_ll = $array[$pa_index_ll];
      } elsif($row_i == ($rows - 1) && $col_i == ($cols - 1)){
        my $pa_index_above = $col_i + ($cols * ($row_i - 1));
        my $pa_index_left = ($col_i - 1) + ($cols * $row_i);
        my $pa_index_ul =  ($col_i - 1) + ($cols * ($row_i - 1));
        $point_above = $array[$pa_index_above];
        $point_left = $array[$pa_index_left];
        $point_ul = $array[$pa_index_ul];
      } else {
        die "Invalid rows, cols: $rows, $cols ($row_i, $col_i)";
      }
      unless(defined $point_right){ $point_right = 0}
      unless(defined $point_left){ $point_left = 0}
      unless(defined $point_above){ $point_above = 0}
      unless(defined $point_below){ $point_below = 0}
      unless(defined $point_ul){ $point_ul = 0}
      unless(defined $point_ur){ $point_ur = 0}
      unless(defined $point_ll){ $point_ll = 0}
      unless(defined $point_lr){ $point_lr = 0}
      my $point = $array[$index];
      if(
        ($point == 1) &&
        ($point_above == 0) &&
        ($point_below == 0) &&
        ($point_left == 0) &&
        ($point_right == 0) &&
        ($point_ul == 0) &&
        ($point_ur == 0) &&
        ($point_ll == 0) &&
        ($point_lr == 0)
      ){
        push @points, "bare point: ($col_i, $row_i)\n";
      }
    }
  }
  print FILE $new_img;
  close FILE;
  if($#points >= 0){
    open FILE, ">$file_name.points";
    for my $i (@points){ print FILE $i };
    close FILE;
  }
#  print "Wrote file: $file_name\n";
#  print "$i|file=$file_name";
#  for my $key (keys %$decoded){
#    if(ref($decoded->{$key}) eq "ARRAY"){
#      for my $i (0 .. $#{$decoded->{$key}}){
#        print "|${key}[$i]=$decoded->{$key}->[$i]";
#      }
#    } else {
#      print "|$key=$decoded->{$key}";
#    }
#  };
#  print "|\n";
}

`convert $file $file.png`;
open VER, "convert -h |head -1|";
my $line = <VER>;
close VER;
my $ver;
if($line =~ /Version: ImageMagick (.)/){
  $ver = $1;
}

for my $file_nm (`ls *.pbm`){
  chomp $file_nm;
  unless($file_nm =~ /(.*).(pbm)/){
    next;
  }
  my $fn = $1;
  open HEAD, "head -3 $file_nm|";
  my $one = <HEAD>;
  my $cols = <HEAD>;
  my $rows = <HEAD>;
  close HEAD;
  chomp $cols;
  chomp $rows;
  my $draw = "";
  if(-r "$fn.points"){
    open POINTS, "<$fn.points";
    while(my $line = <POINTS>){
      chomp $line;
      unless($line =~ /bare point: \((.*),\s*(.*)\)/){
        print STDERR "Non-matching line: $line\n";
        next;
      }
      my $col_i = $1;
      my $row_i = $2;
      $col_i =~ s/\s*//g;
      $row_i =~ s/\s*//g;
      my $l_col = $col_i - 5;
      if($l_col < 0) { $l_col = 0 }
      my $r_col = $col_i + 5;
      if($r_col >= $cols) { $r_col = $cols - 1 }
      my $t_row = $row_i - 5;
      if($t_row < 0) { $t_row = 0 }
      my $b_row = $row_i + 5;
      if($b_row >= $rows) { $b_row = $rows - 1 }
      $draw .= "line $l_col,$row_i $r_col,$row_i " .
               "line $col_i,$t_row $col_i,$b_row ";
    }
    if($draw) {
       $draw = "-stroke green -strokewidth 3 -draw \"$draw\" " 
    }
  }
  $draw .= " -stroke white -strokewidth 2 -draw 'text 10,10 \"$fn\"' ";
  close POINTS;
  my $map_extract_temp;
  my $red_construct_temp;
  my $composite_temp;
  $map_extract_temp = "convert <source> -negate <mask>";
  my $add_points_temp = "convert <dest> <draw> <final>";
  if($ver == 6){
    $red_construct_temp = 
      "composite -compose CopyOpacity <mask> -size <cols>x<rows> xc:red " .
      "<red>";
    $composite_temp = 
      "composite -dissolve 50% <red> <image> <dest>";
  } elsif ($ver == 5){
    $red_construct_temp = 
      "composite -background transparent -size <cols>x<rows> xc:red " .
      "-compose CopyRed <mask> -negate <red>";
    $composite_temp = 
      "composite -dissolve 50% <red> <image> <mask> <dest>";
  } else {
    die "Unknown version of ImageMagick: $line";
  }
  my $source = $file_nm;
  my $mask = "$fn.mask.png";
  my $red = "$fn.red.png";
  my $dest = "$fn.comp.png";
  my $final = "$fn.final.png";
  my $image = "$file.png";
  my $map_ext_com = $map_extract_temp;
  $map_ext_com =~ s/<source>/$source/g;
  $map_ext_com =~ s/<mask>/$mask/g;
  my $red_const_com = $red_construct_temp;
  $red_const_com =~ s/<source>/$source/g;
  $red_const_com =~ s/<mask>/$mask/g;
  $red_const_com =~ s/<rows>/$rows/g;
  $red_const_com =~ s/<cols>/$cols/g;
  $red_const_com =~ s/<red>/$red/g;
  my $comp_com = $composite_temp;
  $comp_com =~ s/<source>/$source/g;
  $comp_com =~ s/<mask>/$mask/g;
  $comp_com =~ s/<rows>/$rows/g;
  $comp_com =~ s/<cols>/$cols/g;
  $comp_com =~ s/<red>/$red/g;
  $comp_com =~ s/<dest>/$dest/g;
  $comp_com =~ s/<image>/$image/g;
  my $final_com = $add_points_temp;
  $final_com =~ s/<dest>/$dest/;
  $final_com =~ s/<draw>/$draw/;
  $final_com =~ s/<final>/$final/;
  `$map_ext_com`;
  `$red_const_com`;
  `$comp_com`;
  if($draw){
    `$final_com`;
    `rm $dest`;
  } else {
    `mv $dest $final`;
  }
  `rm $red`;
  `rm $mask`;
  print "display $final\n";
}

