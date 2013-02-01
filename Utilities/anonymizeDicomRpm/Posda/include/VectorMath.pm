#!/usr/bin/perl
#$Source: /home/bbennett/pass/archive/Posda/include/VectorMath.pm,v $
#$Date: 2008/08/18 23:56:38 $
#$Revision: 1.7 $
#
#Copyright 2008, Bill Bennett
# Part of the Posda package
# Posda may be copied only under the terms of either the Artistic License or the
# GNU General Public License, which may be found in the Posda Distribution,
# or at http://posda.com/License.html
#
use strict;
#
package VectorMath;
sub cross {
  my($a, $b) = @_;
  my $a_x = $a->[0];
  my $a_y = $a->[1];
  my $a_z = $a->[2];
  my $b_x = $b->[0];
  my $b_y = $b->[1];
  my $b_z = $b->[2];
  # (a2b3 - a3b2)i + (a3b1 - a1b3)j + (a1b2 - a2b1)k
  my $c_x = ($a_y * $b_z) - ($a_z * $b_y);
  my $c_y = ($a_z * $b_x) - ($a_x * $b_z);
  my $c_z = ($a_x * $b_y) - ($a_y * $b_x);
  return [$c_x, $c_y, $c_z];
}
sub Dist {
  my($a, $b) = @_;
  return sqrt(
    (( $b->[0] - $a->[0]) ** 2) + 
    (( $b->[1] - $a->[1]) ** 2) + 
    (( $b->[2] - $a->[2]) ** 2)
  );
}
sub Abs {
  my($a) = @_;
  return sqrt(
    ( $a->[0] ** 2) + 
    ( $a->[1] ** 2) + 
    ( $a->[2] ** 2)
  );
}
sub Dot {
  my($a, $b) = @_;
  return ( 
    ($a->[0] * $b->[0]) +
    ($a->[1] * $b->[1]) +
    ($a->[2] * $b->[2])
  );
}
sub Sub{
  my($v1, $v2) = @_;
  my @v;
  $v[0] = $v1->[0] - $v2->[0];
  $v[1] = $v1->[1] - $v2->[1];
  $v[2] = $v1->[2] - $v2->[2];
  return \@v;
}
sub Add{
  my($v1, $v2) = @_;
  my @v;
  $v[0] = $v1->[0] + $v2->[0];
  $v[1] = $v1->[1] + $v2->[1];
  $v[2] = $v1->[2] + $v2->[2];
  return \@v;
}
sub DirVect{
  my($p1, $p2) = @_;
  my $dist = Dist($p1, $p2);
  my $d = [
    ($p2->[0] - $p1->[0]) / $dist,
    ($p2->[1] - $p1->[1]) / $dist,
    ($p2->[2] - $p1->[2]) / $dist
  ];
  return $d;
}
sub Scale{
  my($s, $v) = @_;
  return [$v->[0] * $s, $v->[1] * $s, $v->[2] * $s];
}
sub DistPointToLine{
  # point is x0, line is x1 to x2
  my($x0, $x1, $x2) = @_;
  my $d = (Abs(cross(Sub($x2, $x1), Sub($x1, $x0)))) /
    (Abs(Sub($x2, $x1)));
  return $d;
}
sub ProjPointToLine{
  # point is x0, line is x1 to x2
  my($x0, $x1, $x2) = @_;
  my $dir = DirVect($x1, $x2);
  my $shifted  = Sub($x0, $x1);
  my $mag = Dot($dir, $shifted);
  my $inc = Scale($mag, $dir);
  my $p = Add($x1, $inc);
  return $p;
}
sub Between{
  # point x0, between x1 and x2 (inclusive)
  # Assume x0 collinear with x1 and x2
  my $epsilon = .0000001;
  my($x0, $x1, $x2) = @_;
  if(
    ((Dist($x0, $x1) + Dist($x0, $x2)) - Dist($x1, $x2)) < $epsilon
  ){
    return 1;
  } else {
    return 0;
  }
}
sub Collinear{
  my ($p, $a, $b) = @_;
  my $epsilon = .0000001;
  return(Abs(cross(Sub($a, $p), Sub($b, $p))) < $epsilon);
}
sub Rot3D{
  my($rot, $v)= @_;
  my @r;
  $r[0] = ($v->[0] * $rot->[0]->[0]) + 
          ($v->[1] * $rot->[0]->[1]) +
          ($v->[2] * $rot->[0]->[2]);
  $r[1] = ($v->[0] * $rot->[1]->[0]) + 
          ($v->[1] * $rot->[1]->[1]) +
          ($v->[2] * $rot->[1]->[2]);
  $r[2] = ($v->[0] * $rot->[2]->[0]) + 
          ($v->[1] * $rot->[2]->[1]) +
          ($v->[2] * $rot->[2]->[2]);
  return \@r;
}
sub PrintRot3D{
  my($rot) = @_;
  for my $i (0 .. 2){
    printf "|%0.4f  %0.4f  %0.4f|\n", 
      $rot->[$i]->[0], $rot->[$i]->[1], $rot->[$i]->[2];
  }
}
sub ApplyTransform{
  my($x_form, $vec) = @_;
  unless(ref($x_form) eq "ARRAY" && $#{$x_form} == 15){
    die "x_form is not 4x4 array";
  }
  unless(ref($vec) eq "ARRAY" && $#{$vec} == 2){
    die "vec is not a 3D vector";
  }
  unless(
    $x_form->[12] == 0 &&
    $x_form->[13] == 0 &&
    $x_form->[14] == 0 &&
    $x_form->[15] == 1 
  ){
    print STDERR "Apply tranform: This may not be a legal DICOM transform:\n";
    print "$x_form->[0]\t$x_form->[1], $x_form->[2]\t$x_form->[3]\n";
    print "$x_form->[4]\t$x_form->[5], $x_form->[6]\t$x_form->[7]\n";
    print "$x_form->[8]\t$x_form->[9], $x_form->[10]\t$x_form->[11]\n";
    print "$x_form->[13]\t$x_form->[14], $x_form->[15]\t$x_form->[16]\n";
  }
  my $n_x = $vec->[0] * $x_form->[0] +
            $vec->[1] * $x_form->[1] +
            $vec->[2] * $x_form->[2] +
            1 * $x_form->[3];
  my $n_y = $vec->[0] * $x_form->[4] +
            $vec->[1] * $x_form->[5] +
            $vec->[2] * $x_form->[6] +
            1 * $x_form->[7];
  my $n_z = $vec->[0] * $x_form->[8] +
            $vec->[1] * $x_form->[9] +
            $vec->[2] * $x_form->[10] +
            1 * $x_form->[11];
  my $n_o = $vec->[0] * $x_form->[12] +
            $vec->[1] * $x_form->[13] +
            $vec->[2] * $x_form->[14] +
            1 * $x_form->[15];
  unless($n_o == 1){
    print STDERR "Error applying x_form: $n_o should be 1\n";
  }
  my $res = [$n_x, $n_y, $n_z];
  return $res;
}
sub ApplyTransformList{
  my($t_l, $vec) = @_;
  my $n_vec = $vec;
  for my $x (@$t_l){
    $n_vec = ApplyTransform($t_l, $vec);
  }
  return $n_vec;
}
sub LineIntersect2D{
  my($l1, $l2) = @_;
  my($p1, $p2) = @{$l1};
  my($p3, $p4) = @{$l2};
  my($x1, $y1) = @{$p1};
  my($x2, $y2) = @{$p2};
  my($x3, $y3) = @{$p3};
  my($x4, $y4) = @{$p4};
  my $ua = (
   (($x4 - $x3)*($y1 - $y3) - ($y4 - $y3)*($x1 - $x3)) /
   (($y4 - $y3)*($x2 - $x1) - ($x4 - $x3)*($y2 - $y1))
  );
  my $xr = $x1 + $ua * ($x2 - $x1);
  my $yr = $y1 + $ua * ($y2 - $y1);
  my $ret = [$xr, $yr];
  return $ret;
}
#####
# C-code generation macros
#####
# cross product
#sub cross {
#  my($a, $b) = @_;
#  my $a_x = $a->[0];
#  my $a_y = $a->[1];
#  my $a_z = $a->[2];
#  my $b_x = $b->[0];
#  my $b_y = $b->[1];
#  my $b_z = $b->[2];
#  # (a2b3 - a3b2)i + (a3b1 - a1b3)j + (a1b2 - a2b1)k
#  my $c_x = ($a_y * $b_z) - ($a_z * $b_y);
#  my $c_y = ($a_z * $b_x) - ($a_x * $b_z);
#  my $c_z = ($a_x * $b_y) - ($a_y * $b_x);
#  return [$c_x, $c_y, $c_z];
#}
# (a2b3 - a3b2)i + (a3b1 - a1b3)j + (a1b2 - a2b1)k
sub CC_crossX{
  my($ax, $ay, $az, $bx, $by, $bz) = @_;
  return "(($ay) * ($bz)) - (($az) * ($by))";
}
sub CC_crossY{
  my($ax, $ay, $az, $bx, $by, $bz) = @_;
  return "(($az) * ($bx)) - (($ax) * ($bz))";
}
sub CC_crossZ{
  my($ax, $ay, $az, $bx, $by, $bz) = @_;
  return "(($ax) * ($by)) - (($ay) * ($bx))";
}
# Dist
sub CC_Dist{
  my($x1, $y1, $z1, $x2, $y2, $z2) = @_;
  return
    "(sqrt(\n" .
    "  ((($x1) - ($x2)) * (($x1) - ($x2))) + \n" .
    "  ((($y1) - ($y2)) * (($y1) - ($y2))) + \n" .
    "  ((($z1) - ($z2)) * (($z1) - ($z2)))\n" .
    "))\n";
}
# Abs
sub CC_Abs{
  my ($x0, $y0, $z0) = @_;
  return
    "(sqrt(\n  (($x0) * ($x0)) + \n  (($y0) * ($y0)) + \n  (($z0) * ($z0))\n  )\n)";
}
# Dot
sub CC_Dot{
  my($x0, $y0, $z0, $x1, $y1, $z1) = @_;
  my $ret = "\n((($x0) * ($x1)) +\n (($y0) * ($y1)) +\n (($z0) * ($z1)))";
  #print "Dot([$x0, $y0, $z0], [$x1, $y1, $z1] = \n";
  #print "$ret\n";
  return $ret;
}
# Sub
sub CC_SubX{
  my($x0, $y0, $z0, $x1, $y1, $z1) = @_;
  return "(($x0) - ($x1))";
}
sub CC_SubY{
  my($x0, $y0, $z0, $x1, $y1, $z1) = @_;
  return "(($y0) - ($y1))";
}
sub CC_SubZ{
  my($x0, $y0, $z0, $x1, $y1, $z1) = @_;
  return "(($z0) - ($z1))";
}
# Add
sub CC_AddX{
  my($x0, $y0, $z0, $x1, $y1, $z1) = @_;
  return "(\n  ($x0) +\n  ($x1)\n)";
}
sub CC_AddY{
  my($x0, $y0, $z0, $x1, $y1, $z1) = @_;
  return "(($y0) + ($y1))";
}
sub CC_AddZ{
  my($x0, $y0, $z0, $x1, $y1, $z1) = @_;
  return "(($z0) + ($z1))";
}
# DirVect
#sub DirVect{
#  my($p1, $p2) = @_;
#  my $dist = Dist($p1, $p2);
#  my $d = [
#    ($p2->[0] - $p1->[0]) / $dist,
#    ($p2->[1] - $p1->[1]) / $dist,
#    ($p2->[2] - $p1->[2]) / $dist
#  ];
#  return $d;
#}
sub CC_DirVectX{
  my($p1x, $p1y, $p1z, $p2x, $p2y, $p2z) = @_;
  my $text = "((($p2x) - ($p1x)) / (";
  $text .= CC_Dist($p1x, $p1y, $p1z, $p2x, $p2y, $p2z);
  $text .= "))";
  return $text;
}
sub CC_DirVectY{
  my($p1x, $p1y, $p1z, $p2x, $p2y, $p2z) = @_;
  my $text = "((($p2y) - ($p1y)) / (";
  $text .= CC_Dist($p1x, $p1y, $p1z, $p2x, $p2y, $p2z);
  $text .= "))";
  return $text;
}
sub CC_DirVectZ{
  my($p1x, $p1y, $p1z, $p2x, $p2y, $p2z) = @_;
  my $text = "((($p2z) - ($p1z)) / (";
  $text .= CC_Dist($p1x, $p1y, $p1z, $p2x, $p2y, $p2z);
  $text .= "))";
  return $text;
}
# Scale
#sub Scale{
#  my($s, $v) = @_;
#  return [$v->[0] * $s, $v->[1] * $s, $v->[2] * $s];
#}
sub CC_ScaleX{
 my($s, $x0, $y0, $z0) = @_;
 return "(($s) * ($x0))";
}
sub CC_ScaleY{
 my($s, $x0, $y0, $z0) = @_;
 return "(($s) * ($y0))";
}
sub CC_ScaleZ{
 my($s, $x0, $y0, $z0) = @_;
 return "(($s) * ($z0))";
}
# DistPointToLine
#sub DistPointToLine{
#  # point is x0, line is x1 to x2
#  my($x0, $x1, $x2) = @_;
#  my $d = (Abs(cross(Sub($x2, $x1), Sub($x1, $x0)))) /
#    (Abs(Sub($x2, $x1)));
#  return $d;
#}
sub CC_DistPointToLine{
  my($x0, $y0, $z0, $x1, $y1, $z1, $x2, $y2, $z2) = @_;
  my $sub1X = CC_SubX($x2, $y2, $z2, $x1, $y1, $z1);
  my $sub1Y = CC_SubY($x2, $y2, $z2, $x1, $y1, $z1);
  my $sub1Z = CC_SubZ($x2, $y2, $z2, $x1, $y1, $z1);
  my $sub2X = CC_SubX($x1, $y1, $z1, $x0, $y0, $z0);
  my $sub2Y = CC_SubY($x1, $y1, $z1, $x0, $y0, $z0);
  my $sub2Z = CC_SubZ($x1, $y1, $z1, $x0, $y0, $z0);
  my $crossX = CC_crossX($sub1X, $sub1Y, $sub1Z, $sub2X, $sub2Y, $sub2Z);
  my $crossY = CC_crossY($sub1X, $sub1Y, $sub1Z, $sub2X, $sub2Y, $sub2Z);
  my $crossZ = CC_crossZ($sub1X, $sub1Y, $sub1Z, $sub2X, $sub2Y, $sub2Z);
  my $Abs1 = CC_Abs($crossX, $crossY, $crossZ);
  my $sub3X = CC_SubX($x2, $y2, $z2, $x1, $y1, $z1);
  my $sub3Y = CC_SubY($x2, $y2, $z2, $x1, $y1, $z1);
  my $sub3Z = CC_SubZ($x2, $y2, $z2, $x1, $y1, $z1);
  my $Abs2 = CC_Abs($sub3X, $sub3Y, $sub3Z);
  return "($Abs1 / $Abs2)";
}
#sub ProjPointToLine{
#  # point is x0, line is x1 to x2
#  my($x0, $x1, $x2) = @_;
#  my $dir = DirVect($x1, $x2);
#  my $shifted  = Sub($x0, $x1);
#  my $mag = Dot($dir, $shifted);
#  my $inc = Scale($mag, $dir);
#  my $p = Add($x1, $inc);
#  return $p;
#}
# ProjPointToLine
sub CC_ProjPointToLineX{
  my($x0, $y0, $z0, $x1, $y1, $z1, $x2, $y2, $z2) = @_;
  my $dirX = CC_DirVectX($x1, $y1, $z1, $x2, $y2, $z2);
  my $dirY = CC_DirVectY($x1, $y1, $z1, $x2, $y2, $z2);
  my $dirZ = CC_DirVectZ($x1, $y1, $z1, $x2, $y2, $z2);
  my $shiftedX = CC_SubX($x0, $y0, $z0, $x1, $y1, $z1);
  my $shiftedY = CC_SubY($x0, $y0, $z0, $x1, $y1, $z1);
  my $shiftedZ = CC_SubZ($x0, $y0, $z0, $x1, $y1, $z1);
  my $mag = CC_Dot($dirX, $dirY, $dirZ, $shiftedX, $shiftedY, $shiftedZ);
  my $incX = CC_ScaleX($mag, $dirX, $dirY, $dirZ);
  my $incY = CC_ScaleY($mag, $dirX, $dirY, $dirZ);
  my $incZ = CC_ScaleZ($mag, $dirX, $dirY, $dirZ);
  my $ret = CC_AddX($x1, $y1, $z1, $incX, $incY, $incZ);
  return $ret;
}
sub CC_ProjPointToLineY{
  my($x0, $y0, $z0, $x1, $y1, $z1, $x2, $y2, $z2) = @_;
  my $dirX = CC_DirVectX($x1, $y1, $z1, $x2, $y2, $z2);
  my $dirY = CC_DirVectY($x1, $y1, $z1, $x2, $y2, $z2);
  my $dirZ = CC_DirVectZ($x1, $y1, $z1, $x2, $y2, $z2);
  my $shiftedX = CC_SubX($x0, $y0, $z0, $x1, $y1, $z1);
  my $shiftedY = CC_SubY($x0, $y0, $z0, $x1, $y1, $z1);
  my $shiftedZ = CC_SubZ($x0, $y0, $z0, $x1, $y1, $z1);
  my $mag = CC_Dot($dirX, $dirY, $dirZ, $shiftedX, $shiftedY, $shiftedZ);
  my $incX = CC_ScaleX($mag, $dirX, $dirY, $dirZ);
  my $incY = CC_ScaleY($mag, $dirX, $dirY, $dirZ);
  my $incZ = CC_ScaleZ($mag, $dirX, $dirY, $dirZ);
  my $ret = CC_AddY($x1, $y1, $z1, $incX, $incY, $incZ);
  return $ret;
}
sub CC_ProjPointToLineZ{
  my($x0, $y0, $z0, $x1, $y1, $z1, $x2, $y2, $z2) = @_;
  my $dirX = CC_DirVectX($x1, $y1, $z1, $x2, $y2, $z2);
  my $dirY = CC_DirVectY($x1, $y1, $z1, $x2, $y2, $z2);
  my $dirZ = CC_DirVectZ($x1, $y1, $z1, $x2, $y2, $z2);
  my $shiftedX = CC_SubX($x0, $y0, $z0, $x1, $y1, $z1);
  my $shiftedY = CC_SubY($x0, $y0, $z0, $x1, $y1, $z1);
  my $shiftedZ = CC_SubZ($x0, $y0, $z0, $x1, $y1, $z1);
  my $mag = CC_Dot($dirX, $dirY, $dirZ, $shiftedX, $shiftedY, $shiftedZ);
  my $incX = CC_ScaleX($mag, $dirX, $dirY, $dirZ);
  my $incY = CC_ScaleY($mag, $dirX, $dirY, $dirZ);
  my $incZ = CC_ScaleZ($mag, $dirX, $dirY, $dirZ);
  my $ret = CC_AddZ($x1, $y1, $z1, $incX, $incY, $incZ);
  return $ret;
}
# Between
sub CC_Between{
  my($x0, $y0, $z0, $x1, $y1, $z1, $x2, $y2, $z2) = @_;
  my $dist1 = CC_Dist($x0, $y0, $z0, $x1, $y1, $z1);
  my $dist2 = CC_Dist($x0, $y0, $z0, $x2, $y2, $z2);
  my $dist3 = CC_Dist($x1, $y1, $z1, $x2, $y2, $z2);
  return "(((($dist1) + ($dist2)) - ($dist3)) < .0000001)";
};
#ApplyTransform
#  my $n_x = $vec->[0] * $x_form->[0] +
#            $vec->[1] * $x_form->[1] +
#            $vec->[2] * $x_form->[2] +
#            1 * $x_form->[3];
#  my $n_y = $vec->[0] * $x_form->[4] +
#            $vec->[1] * $x_form->[5] +
#            $vec->[2] * $x_form->[6] +
#            1 * $x_form->[7];
#  my $n_z = $vec->[0] * $x_form->[8] +
#            $vec->[1] * $x_form->[9] +
#            $vec->[2] * $x_form->[10] +
#            1 * $x_form->[11];
#  my $n_o = $vec->[0] * $x_form->[12] +
#            $vec->[1] * $x_form->[13] +
#            $vec->[2] * $x_form->[14] +
#            1 * $x_form->[15];
sub CC_ApplyTransformX{
  my($xform) = @_;
  return "((x * ($xform->[0]->[0])) +\n" .
         " (y * ($xform->[0]->[1])) +\n" .
         " (z * ($xform->[0]->[2])) +\n" .
         " (1 * ($xform->[0]->[3])) )";
}
sub CC_ApplyTransformY{
  my($xform) = @_;
  return "((x * ($xform->[1]->[0])) +\n" .
         " (y * ($xform->[1]->[1])) +\n" .
         " (z * ($xform->[1]->[2])) +\n" .
         " (1 * ($xform->[1]->[3])) )";
}
sub CC_ApplyTransformZ{
  my($xform) = @_;
  return "((x * ($xform->[2]->[0])) +\n" .
         " (y * ($xform->[2]->[1])) +\n" .
         " (z * ($xform->[2]->[2])) +\n" .
         " (1 * ($xform->[2]->[3])) )";
}
1;
