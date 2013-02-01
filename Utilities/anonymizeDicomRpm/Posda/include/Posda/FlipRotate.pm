#!usr/bin/perl -w
#$Source: /home/bbennett/pass/archive/Posda/include/Posda/FlipRotate.pm,v $
#$Date: 2008/04/30 19:17:35 $
#$Revision: 1.3 $
#
#Copyright 2008, Bill Bennett
# Part of the Posda package
# Posda may be copied only under the terms of either the Artistic License or the
# GNU General Public License, which may be found in the Posda Distribution,
# or at http://posda.com/License.html
#
use strict;
package Posda::FlipRotate;
use Posda::Transforms;
use VectorMath;
my $epsilon = 1.0E-13;

#use Debug;
#my $dbg = sub {print @_};
sub FlipArrayHorizontal{
  my($array, $rows, $cols, $bits_alloc) = @_;
  my $t_array;
  for my $i (0 .. $rows - 1){
    my $half_cols = int $cols/2;
    for my $j (0 .. $half_cols - 1){
      my $f_index = ($i * $cols) + $j;
      my $t_index = ($i * $cols) + ($rows - 1 - $j);
      my($from, $to);
      if($bits_alloc == 8){
        $from = vec($array, $f_index, $bits_alloc);
        $to = vec($array, $t_index, $bits_alloc);
        vec($t_array, $t_index, $bits_alloc) = $from;
        vec($t_array, $f_index, $bits_alloc) = $to;
      }elsif($bits_alloc == 16){
        $from = unpack("v", pack("n", vec($array, $f_index, $bits_alloc)));
        $to = unpack("v", pack("n", vec($array, $t_index, $bits_alloc)));
        vec($t_array, $t_index, $bits_alloc) = unpack("n", pack("v", $from));
        vec($t_array, $f_index, $bits_alloc) = unpack("n", pack("v", $to));
      } elsif($bits_alloc == 32){
        $from = unpack("V", pack("N", vec($array, $f_index, $bits_alloc)));
        $to = unpack("V", pack("N", vec($array, $t_index, $bits_alloc)));
        vec($t_array, $t_index, $bits_alloc) = unpack("N", pack("V", $from));
        vec($t_array, $f_index, $bits_alloc) = unpack("N", pack("V", $to));
      } else {
        die "bits allocated: $bits_alloc???";
      }
    }
  }
  return $t_array;
}
sub FlipArrayVertical{
  my($array, $rows, $cols, $bits_alloc) = @_;
  my $t_array;
  for my $j (0 .. $cols - 1){
    my $half_rows = int $rows/2;
    for my $i (0 .. $half_rows - 1){
      my $f_index = ($i * $cols) + $j;
      my $t_index = (($cols - 1 - $i) * $cols) + $j;
      my($from, $to);
      if($bits_alloc == 8){
        $from = vec($array, $f_index, $bits_alloc);
        $to = vec($array, $t_index, $bits_alloc);
        vec($t_array, $t_index, $bits_alloc) = $from;
        vec($t_array, $f_index, $bits_alloc) = $to;
      }elsif($bits_alloc == 16){
        $from = unpack("v", pack("n", vec($array, $f_index, $bits_alloc)));
        $to = unpack("v", pack("n", vec($array, $t_index, $bits_alloc)));
        vec($t_array, $t_index, $bits_alloc) = unpack("n", pack("v", $from));
        vec($t_array, $f_index, $bits_alloc) = unpack("n", pack("v", $to));
      } elsif($bits_alloc == 32){
        $from = unpack("V", pack("N", vec($array, $f_index, $bits_alloc)));
        $to = unpack("V", pack("N", vec($array, $t_index, $bits_alloc)));
        vec($t_array, $t_index, $bits_alloc) = unpack("N", pack("V", $from));
        vec($t_array, $f_index, $bits_alloc) = unpack("N", pack("V", $to));
      } else {
        die "bits allocated: $bits_alloc???";
      }
    }
  }
  return $t_array;
}
sub RotArray90{
  my($f_a, $rows, $cols, $bits_alloc) = @_;
  my $t_a;
  for my $i (0 .. $rows - 1){
    for my $j (0 .. $cols - 1){
      my $f_index = ($i * $cols) + $j;
      my $t_index = ($j * $rows) + ($cols - $i);
      my $from;
      if($bits_alloc == 8){
        $from = vec($f_a, $f_index, $bits_alloc);
        vec($t_a, $t_index, $bits_alloc) = $from;
      }elsif($bits_alloc == 16){
        $from = unpack("v", pack("n", vec($f_a, $f_index, $bits_alloc)));
        vec($t_a, $t_index, $bits_alloc) = unpack("n", pack("v", $from));
      } elsif($bits_alloc == 32){
        $from = unpack("V", pack("N", vec($f_a, $f_index, $bits_alloc)));
        vec($t_a, $t_index, $bits_alloc) = unpack("N", pack("V", $from));
      } else {
        die "bits allocated: $bits_alloc???";
      }
    }
  }
  return $t_a;
}
sub FlipIopIppHorizontal{
  my($iop, $ipp, $rows, $cols, $pix_sp) = @_;
  my($tlhc, $trhc, $blhc, $brhc) = 
    ToCorners($rows, $cols, $iop, $ipp, $pix_sp);
  my($n_iop, $n_ipp) = FromCorners($trhc, $tlhc, $brhc, $blhc);
  return ($n_iop, $n_ipp, $rows, $cols, $pix_sp);
}
sub FlipIopIppVertical{
  my($iop, $ipp, $rows, $cols, $pix_sp) = @_;
  my($tlhc, $trhc, $blhc, $brhc) = 
    ToCorners($rows, $cols, $iop, $ipp, $pix_sp);
  my($n_iop, $n_ipp) =  FromCorners($blhc, $brhc, $tlhc, $trhc);
  return ($n_iop, $n_ipp, $rows, $cols, $pix_sp);
}
sub RotIopIpp90{
  my($iop, $ipp, $rows, $cols, $pix_sp) = @_;
  my($tlhc, $trhc, $blhc, $brhc) = 
    ToCorners($rows, $cols, $iop, $ipp, $pix_sp);
  
  my ($n_iop, $n_ipp) = FromCorners($blhc, $tlhc, $brhc, $trhc);
  return ($n_iop, $n_ipp, $cols, $rows, [$pix_sp->[1], $pix_sp->[0]]);
}
sub ToCorners{
  my($rows, $cols, $iop, $ipp, $pix_sp) = @_;
  my $col_width = $pix_sp->[0];
  my $row_width = $pix_sp->[1];
  my $dxdc = $iop->[0];
  my $dydc = $iop->[1];
  my $dzdc = $iop->[2];
  my $dxdr = $iop->[3];
  my $dydr = $iop->[4];
  my $dzdr = $iop->[5];
  my $x = $ipp->[0];
  my $y = $ipp->[1];
  my $z = $ipp->[2];
  my $tlhc = [$x, $y, $z];
  my $trhc = [
     $x + ($dxdc * ($cols - 1) * $row_width),
     $y + ($dydc * ($cols - 1) * $row_width),
     $z + ($dzdc * ($cols - 1) * $row_width)
  ];
  my $blhc = [
     $x + ($dxdr * ($rows - 1) * $col_width),
     $y + ($dydr * ($rows - 1) * $col_width),
     $z + ($dzdr * ($rows - 1) * $col_width)
  ];
  my $brhc = [
     $x + ($dxdc * ($cols - 1) * $row_width) +
          ($dxdr * ($rows - 1) * $col_width),
     $y + ($dydc * ($cols - 1) * $row_width) +
          ($dydr * ($rows - 1) * $col_width),
     $z + ($dzdc * ($cols - 1) * $row_width) +
          ($dzdr * ($rows - 1) * $col_width)
  ];
  return ($tlhc, $trhc, $blhc, $brhc);
}
sub FromCorners{
  my($tlhc, $trhc, $blhc, $brhc) = @_;
  my $row_len = sqrt(
    ($trhc->[0] - $tlhc->[0])**2 +
    ($trhc->[1] - $tlhc->[1])**2 +
    ($trhc->[2] - $tlhc->[2])**2
  );
  my $col_len = sqrt(
    ($trhc->[0] - $brhc->[0])**2 +
    ($trhc->[1] - $brhc->[1])**2 +
    ($trhc->[2] - $brhc->[2])**2
  );
  my $dxdc = ($trhc->[0] - $tlhc->[0]) / $row_len;
  my $dydc = ($trhc->[1] - $tlhc->[1]) / $row_len;
  my $dzdc = ($trhc->[2] - $tlhc->[2]) / $row_len;
  my $dxdr = ($blhc->[0] - $tlhc->[0]) / $col_len;
  my $dydr = ($blhc->[1] - $tlhc->[1]) / $col_len;
  my $dzdr = ($blhc->[2] - $tlhc->[2]) / $col_len;
  my @iop = ($dxdc, $dydc, $dzdc, $dxdr, $dydr, $dzdr);
  my $ipp = $tlhc;
  return \@iop, $ipp;
}
sub RotateIopIpp{
  my($rows, $cols, $iop, $ipp, $pix_sp, $angle) = @_;
  unless($iop->[2] == 0 && $iop->[5] == 0){
    die "Only axial images need apply";
  }
  my($tlhc, $trhc, $blhc, $brhc) = Posda::FlipRotate::ToCorners(
    $rows, $cols, $iop, $ipp, $pix_sp
  );
  my $center_x = ($trhc->[0] + $tlhc->[0]) / 2;
  my $center_y = ($blhc->[1] + $tlhc->[1]) / 2;
  my $center_z = $trhc->[2];
  my $trans_to_org = [
    [1, 0, 0, -$center_x],
    [0, 1, 0, -$center_y],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
  ];
  my $rot = Posda::Transforms::MakeRotZ($angle);
  my $rev_rot = Posda::Transforms::MakeRotZ(-$angle);
  my $trans_from_org = [
    [1, 0, 0, $center_x],
    [0, 1, 0, $center_y],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
  ];
  my $rot_seq = [
    $trans_to_org, $rot, $trans_from_org
  ];
  my $rev_rot_seq = [
    $trans_to_org, $rev_rot, $trans_from_org
  ];
  my $r_tlhc = Posda::Transforms::ApplyTransformList($rot_seq, $tlhc);
  my $r_trhc = Posda::Transforms::ApplyTransformList($rot_seq, $trhc);
  my $r_blhc = Posda::Transforms::ApplyTransformList($rot_seq, $blhc);
  my $r_brhc = Posda::Transforms::ApplyTransformList($rot_seq, $brhc);
  
  my $b_tlhc = Posda::Transforms::ApplyTransformList($rev_rot_seq, $r_tlhc);
  my $b_trhc = Posda::Transforms::ApplyTransformList($rev_rot_seq, $r_trhc);
  my $b_blhc = Posda::Transforms::ApplyTransformList($rev_rot_seq, $r_blhc);
  my $b_brhc = Posda::Transforms::ApplyTransformList($rev_rot_seq, $r_brhc);
  
  my $d_tlhc = VectorMath::Dist($tlhc, $b_tlhc);
  my $d_trhc = VectorMath::Dist($trhc, $b_trhc);
  my $d_blhc = VectorMath::Dist($blhc, $b_blhc);
  my $d_brhc = VectorMath::Dist($brhc, $b_brhc);
  unless(
    $d_tlhc < $epsilon &&
    $d_trhc < $epsilon &&
    $d_brhc < $epsilon &&
    $d_brhc < $epsilon
  ){
    die "inverse rotation introduced unacceptable errors";
  }
  my($rot_iop, $rot_ipp) = FromCorners($r_tlhc, $r_trhc, $r_blhc, $r_brhc);
  return $rot_iop, $rot_ipp, $rev_rot_seq;
}
sub ResamplingParams{
  my($rows, $cols, $iop, $ipp, $pix_sp, $new_pix_sp) = @_;
  my $angle = acos($iop->[0]);
  my $z = $ipp->[2];
  unless($iop->[2] == 0 && $iop->[5] == 0){
    die "only axial images need apply";
  }
  my($ur_iop, $ur_ipp, $rev_rot_seq) = Posda::FlipRotate::RotateIopIpp(
    $rows, $cols, $iop, $ipp, $pix_sp, -$angle
  );
#print "rotated IOP: ";
#Debug::GenPrint($dbg, $ur_iop, 1);
#print "\n";
#print "rotated IPP: ";
#Debug::GenPrint($dbg, $ur_ipp, 1);
#print "\n";
  my @RotCorners =
    ToCorners($rows, $cols, $iop, $ipp, $pix_sp);
  my @UnRotCorners =
    ToCorners($rows, $cols, $ur_iop, $ur_ipp, $pix_sp);
  my $ur_tlhc = $UnRotCorners[0];
  my $ur_brhc = $UnRotCorners[3];
  my $tlhc = $RotCorners[0];
  my $blhc = $RotCorners[2];
  my $trhc = $RotCorners[1];
  my $brhc = $RotCorners[3];
  my $diag = [$ur_tlhc, $ur_brhc];
  my $lside = [$tlhc, $blhc];
  my $rside = [$trhc, $brhc];
  my $n_tlhc = VectorMath::LineIntersect2D($diag, $lside);
#print "diag: ";
#Debug::GenPrint($dbg, $diag, 1);
#print "\n";
#print "lside: ";
#Debug::GenPrint($dbg, $lside, 1);
#print "\n";
#print "n_tlhc: ($n_tlhc->[0], $n_tlhc->[1])\n";
  my $n_brhc = VectorMath::LineIntersect2D($diag, $rside);
  $n_tlhc->[2] = $z;
  $n_brhc->[2] = $z;
  my $n_trhc = [$n_brhc->[0], $n_tlhc->[1], $z];
  my $n_blhc = [$n_tlhc->[0], $n_brhc->[1], $z];
  my $a_row_dist = VectorMath::Dist($n_tlhc, $n_trhc);
  my $a_col_dist = VectorMath::Dist($n_tlhc, $n_blhc);
  my $new_rows = int($a_row_dist / $new_pix_sp->[1]);
  my $new_cols = int($a_col_dist / $new_pix_sp->[0]);
  my $x_offset = ($a_row_dist - ($new_cols * $new_pix_sp->[1])) / 2;
  my $y_offset = ($a_col_dist - ($new_rows * $new_pix_sp->[0])) / 2;
  $n_tlhc->[0] += $x_offset;
  $n_tlhc->[1] += $y_offset;

  #to do: #build n_trhc and nblhc
  #       #find how many pixels
  #       #transform into new_iop, new_ipp, $new_rows, $new_cols
  #       #center new square

  return ($ur_iop, $n_tlhc, $new_rows, $new_cols);
}
sub FromPixCoords{
  my($iop, $ipp, $rows, $cols, $pix_sp, $p_in_pix) = @_;
  my $col_width = $pix_sp->[0];
  my $row_width = $pix_sp->[1];
  my $dxdc = $iop->[0];
  my $dydc = $iop->[1];
  my $dzdc = $iop->[2];
  my $dxdr = $iop->[3];
  my $dydr = $iop->[4];
  my $dzdr = $iop->[5];
  my $x = $ipp->[0];
  my $y = $ipp->[1];
  my $z = $ipp->[2];
  my $c = $p_in_pix->[1];
  my $r = $p_in_pix->[0];
  
  my $new_point = [
     $x + ($dxdc * $c * $row_width) +
          ($dxdr * $r * $col_width),
     $y + ($dydc * $c * $row_width) +
          ($dydr * $r * $col_width),
     $z + ($dzdc * $c * $row_width) +
          ($dzdr * $r * $col_width)
  ];
  return $new_point;
}
sub ToPixCoords{
  my($iop, $ipp, $rows, $cols, $pix_sp, $point) = @_;
  my($c_x, $c_y, $c_z) = @$ipp;  # Corner co-ords
  my $dxdr = $iop->[0];       # dx/dr
  my $dydr = $iop->[1];       # dy/dr
  my $dzdr = $iop->[2];       # dz/dr
  my $dxdc = $iop->[3];       # dx/dc
  my $dydc = $iop->[4];       # dy/dc
  my $dzdc = $iop->[5];       # dz/dc
  my ($p_x, $p_y) = @$pix_sp;       # pixel_spacing
  my ($x, $y, $z) = @$point;
  my $normal = VectorMath::cross(
    [$dxdc, $dydc, $dzdc], [$dxdr, $dydr, $dzdr]
  ); 
  my($dxdp, $dydp, $dzdp) = @$normal;
         
  my $DistX = $x - $c_x;
  my $DistY = $y - $c_y;
  my $DistZ = $z - $c_z;
#print "DistX = $DistX, DistY = $DistY\n";
#print "dxdr = $dxdr, dydr = $dydr, dzdr = $dzdr\n";
#print "dxdc = $dxdc, dydc = $dydc, dzdc = $dzdc\n";
       
  my $deltaC = (($DistX * $dxdr) + ($DistY * $dydr) + ($DistZ * $dzdr)) / $p_x;
  my $deltaR = (($DistX * $dxdc) + ($DistY * $dydc) + ($DistZ * $dzdc)) / $p_y;
  my $deltaP = ($DistX * $dxdp) + ($DistY * $dydp) + ($DistZ * $dzdp);
  return [$deltaR, $deltaC, $deltaP];
   
  my $r = int sprintf("%.0f", $deltaR);
  my $c = int sprintf("%.0f", $deltaC);
  return ([$r, $c], $deltaP);
}
1;
