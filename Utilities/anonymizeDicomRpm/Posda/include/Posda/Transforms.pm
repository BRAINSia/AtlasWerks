#!/usr/bin/perl -w
#$Source: /home/bbennett/pass/archive/Posda/include/Posda/Transforms.pm,v $
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
use Math::Trig;
package Posda::Transforms;
sub MakeRotZ{
  my($theta) = @_;
  my $rot = [
    [ cos($theta), -sin($theta), 0, 0 ],
    [ sin($theta), cos($theta), 0, 0 ],
    [ 0, 0, 1, 0 ],
    [ 0, 0, 0, 1]
  ];
  return $rot;
}
sub MakeRotY{
  my($theta) = @_;
  my $rot = [
    [ cos($theta), 0, sin($theta), 0 ],
    [ 0, 1, 0, 0 ],
    [ -sin($theta), 0, cos($theta), 0 ],
    [ 0, 0, 0, 1]
  ];
  return $rot;
}
sub MakeRotX{
  my($theta) = @_;
  my $rot = [
    [ 1, 0, 0, 0 ],
    [ 0, cos($theta), -sin($theta), 0 ],
    [ 0, sin($theta), cos($theta), 0 ],
    [ 0, 0, 0, 1]
  ];
  return $rot;
}
sub MakeTransToOrig{
  my($vec) = @_;
  my $trans = [
    [1, 0, 0, -$vec->[0] ],
    [0, 1, 0, -$vec->[1] ],
    [0, 0, 1, -$vec->[2] ],
    [ 0, 0, 0, 1]
  ];
  return $trans;
}
sub MakeTransFromOrig{
  my($vec) = @_;
  my $trans = [
    [1, 0, 0, $vec->[0] ],
    [0, 1, 0, $vec->[1] ],
    [0, 0, 1, $vec->[2] ],
    [ 0, 0, 0, 1]
  ];
  return $trans;
}
sub NormalizeTransform{
  my($x_form) = @_;
  my @n_form;
  for my $i (0 .. 3){
    for my $j (0 ..3){
      if(abs($x_form->[$i]->[$j]) < 0.00001){
        $n_form[$i]->[$j] = 0;
      } elsif(abs($x_form->[$i]->[$j] - 1) < 0.00001){
        if($x_form->[$i]->[$j] < 0){
          $n_form[$i]->[$j] = -1;
        } else {
          $n_form[$i]->[$j] = 1;
        }
      } else {
        $n_form[$i]->[$j] = $x_form->[$i]->[$j];
      }
    }
  }
  return \@n_form;
}
sub PrintTransform{
  my($x_form) = @_;
printf "%14f\t%14f\t\%14f\t%14f\n", $x_form->[0]->[0],$x_form->[0]->[1],
  $x_form->[0]->[2],$x_form->[0]->[3];
printf "%14f\t%14f\t\%14f\t%14f\n", $x_form->[1]->[0],$x_form->[1]->[1],
  $x_form->[1]->[2],$x_form->[1]->[3];
printf "%14f\t%14f\t\%14f\t%14f\n", $x_form->[2]->[0],$x_form->[2]->[1],
  $x_form->[2]->[2],$x_form->[2]->[3];
printf "%14f\t%14f\t\%14f\t%14f\n", $x_form->[3]->[0],$x_form->[3]->[1],
  $x_form->[3]->[2],$x_form->[3]->[3];
}
sub ApplyTransform{
  my($x_form, $vec) = @_;
  unless(
    ref($x_form) eq "ARRAY" && $#{$x_form} == 3 &&
    ref($x_form->[0]) eq "ARRAY" && $#{$x_form->[0]} == 3 &&
    ref($x_form->[1]) eq "ARRAY" && $#{$x_form->[1]} == 3 &&
    ref($x_form->[2]) eq "ARRAY" && $#{$x_form->[2]} == 3 &&
    ref($x_form->[3]) eq "ARRAY" && $#{$x_form->[3]} == 3
  ){
    print STDERR "Xform:\n";
    for my $i (@$x_form){
      for my $j (@$i){
        print STDERR "$j ";
      }
      print STDERR "\n";
    }
    die "x_form is not 4x4 array";
  }
  unless(ref($vec) eq "ARRAY" && $#{$vec} == 2){
    die "vec is not a 3D vector";
  }
  unless(
    $x_form->[3]->[0] == 0 &&
    $x_form->[3]->[1] == 0 &&
    $x_form->[3]->[2] == 0 &&
    abs($x_form->[3]->[3] - 1) < .0001
  ){
    print STDERR "Apply tranform: This may not be a legal DICOM transform:\n";
    print STDERR "$x_form->[0]->[0]\t$x_form->[0]->[1]," . 
      " $x_form->[0]->[2]\t$x_form->[0]->[3]\n";
    print STDERR "$x_form->[1]->[0]\t$x_form->[1]->[1]," . 
      " $x_form->[1]->[2]\t$x_form->[1]->[3]\n";
    print STDERR "$x_form->[2]->[0]\t$x_form->[2]->[1]," . 
      " $x_form->[2]->[2]\t$x_form->[2]->[3]\n";
    print STDERR "$x_form->[3]->[0]\t$x_form->[3]->[1]," . 
      " $x_form->[3]->[2]\t$x_form->[3]->[3]\n";
  }
  my $n_x = $vec->[0] * $x_form->[0]->[0] +
            $vec->[1] * $x_form->[0]->[1] +
            $vec->[2] * $x_form->[0]->[2] +
            1 * $x_form->[0]->[3];
  my $n_y = $vec->[0] * $x_form->[1]->[0] +
            $vec->[1] * $x_form->[1]->[1] +
            $vec->[2] * $x_form->[1]->[2] +
            1 * $x_form->[1]->[3];
  my $n_z = $vec->[0] * $x_form->[2]->[0] +
            $vec->[1] * $x_form->[2]->[1] +
            $vec->[2] * $x_form->[2]->[2] +
            1 * $x_form->[2]->[3];
  my $n_o = $vec->[0] * $x_form->[3]->[0] +
            $vec->[1] * $x_form->[3]->[1] +
            $vec->[2] * $x_form->[3]->[2] +
            1 * $x_form->[3]->[3];
  unless(abs($n_o - 1) < .0001){
    print STDERR "Error applying x_form: $n_o should be 1\n";
  }
  my $res = [$n_x, $n_y, $n_z];
  return $res;
}
sub ApplyTransformList{
  my($t_l, $vec) = @_;
  my $n_vec = $vec;
  for my $x (@$t_l){
    $n_vec = ApplyTransform($x, $n_vec);
  }
  return $n_vec;
}
#sub CollapseTransformList{
#  my($vec) = @_;
#  my $trans = shift(@$vec);
#  while(my $next_xform = shift(@$vec)){
#    $trans = MatMul($trans, $next_xform);
#  }
#  return $trans;
#}
sub CollapseTransformList{
  my($vec) = @_;
  my $trans = shift(@$vec);
  while(my $next_xform = shift(@$vec)){
    $trans = MatMul($next_xform, $trans);
  }
  return $trans;
}
sub MatMul{
  my($m1, $m2) = @_;
  my @m3 = ();
  for my $i (0 .. 3) {
    my @row = ();
    my $m1i = $m1->[$i];
    for my $j (0 .. 3) {
      my $val = 0;
      for my $k (0 .. 3) {
        $val += $m1i->[$k] * $m2->[$k]->[$j];
      }
      push(@row, $val);
    }
    push(@m3, \@row);
  }
  return(\@m3);
}
sub MakeTransformPair {
  my($commands) = @_;
  my @x_form_list;
  for my $i (@$commands) {
    if($i->[0] eq "rx"){
      my $r = Math::Trig::deg2rad($i->[1]);
      push(@x_form_list, MakeRotX($r));
    }elsif($i->[0] eq "ry"){
      my $r = Math::Trig::deg2rad($i->[1]);
      push(@x_form_list, MakeRotY($r));
    }elsif($i->[0] eq "rz"){
      my $r = Math::Trig::deg2rad($i->[1]);
      push(@x_form_list, MakeRotZ($r));
    }elsif($i->[0] eq "shift"){
      unless($i->[1] =~ /^\((.*),(.*),(.*)\)$/) { die "bad shift" }
      my $x = $1;
      my $y = $2;
      my $z = $3;
      push(@x_form_list, [
        [1, 0, 0, $x],
        [0, 1, 0, $y],
        [0, 0, 1, $z],
        [0, 0, 0, 1],
      ]);
    }
  }
  my $x_form = CollapseTransformList(\@x_form_list);
  my $rev_rot = [
    [$x_form->[0]->[0], $x_form->[1]->[0], $x_form->[2]->[0], 0],
    [$x_form->[0]->[1], $x_form->[1]->[1], $x_form->[2]->[1], 0],
    [$x_form->[0]->[2], $x_form->[1]->[2], $x_form->[2]->[2], 0],
    [$x_form->[3]->[0], $x_form->[3]->[1], $x_form->[3]->[2], 1],
  ];
  my $tlhc = [0,0,0];
  my $r_tlhc = ApplyTransform($x_form, $tlhc);
  my $f_tlhc = ApplyTransform($rev_rot, $r_tlhc);
  $rev_rot->[0]->[3] = -$f_tlhc->[0];
  $rev_rot->[1]->[3] = -$f_tlhc->[1];
  $rev_rot->[2]->[3] = -$f_tlhc->[2];
  return($x_form, $rev_rot);
}

1;
