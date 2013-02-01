#$Source: /home/bbennett/pass/archive/Posda/include/Posda/Command.pm,v $
#$Date: 2008/04/30 19:17:35 $
#$Revision: 1.5 $
#
#Copyright 2008, Bill Bennett
# Part of the Posda package
# Posda may be copied only under the terms of either the Artistic License or the
# GNU General Public License, which may be found in the Posda Distribution,
# or at http://posda.com/License.html
#
use strict;
package Posda::Command;
use Posda::CmdDict;

sub new_blank{
  my($class) = @_;
  my $this = {};
  return (bless($this, $class));
}
sub new{
  my($class, $data) = @_;
  my $this = {
  };
  my $remaining = $data;
  while($remaining){
    my($grp, $ele, $len, $remain) = unpack("vvVa*", $remaining);
    my $value;
    if($len >= 0){
      ($value, $remaining) = unpack("a${len}a*", $remain);
    }
    my $sig = sprintf("(%04x,%04x)", $grp, $ele);
    if(exists $Posda::CmdDict::Dict->{$sig}){
      if($Posda::CmdDict::Dict->{$sig}->{type} eq "ulong"){
        my @values = unpack("V*", $value);
        if($#values > 0){
          $value = \@values;
        } else {
          $value = $values[0];
        }
      } elsif ($Posda::CmdDict::Dict->{$sig}->{type} eq "ushort"){
        my @values = unpack("v*", $value);
        if($#values > 0){
          $value = \@values;
        } else {
          $value = $values[0];
        }
      } else {
        if($Posda::CmdDict::Dict->{$sig}->{vr} eq "UI"){
          $value =~ s/\0+$//;
        } else {
          $value =~ s/\s+$//;
        }
      }
    }
    $this->{$sig} = $value;
  }
  return bless $this, $class;
};
sub new_store_cmd{
  my($class, $sop_cl, $sop_inst) = @_;
  my $cmd = {
    "(0000,0002)" => $sop_cl,
    "(0000,0100)" => 1,
    "(0000,0700)" => 0,
    "(0000,0800)" => 0,
    "(0000,1000)" => $sop_inst,
  };
  return bless $cmd, $class;
};
sub new_store_response{
  my($this, $status) = @_;
  my $resp = {
    "(0000,0100)" => 0x8001,
    "(0000,0800)" => 0x0101,
    "(0000,0120)" => $this->{"(0000,0110)"},
    "(0000,0900)" => $status,
    "(0000,0002)" => $this->{"(0000,0002)"},
    "(0000,1000)" => $this->{"(0000,1000)"},
  };
  return bless $resp, ref($this);
}
sub new_verif_command{
  my($class, $status) = @_;
  my $resp = {
    "(0000,0100)" => 0x30,
    "(0000,0800)" => 0x0101,
    "(0000,0002)" => '1.2.840.10008.1.1',
  };
  return bless $resp, $class;
}
sub new_verif_response{
  my($this, $status) = @_;
  my $resp = {
    "(0000,0100)" => 0x8030,
    "(0000,0800)" => 0x0101,
    "(0000,0120)" => $this->{"(0000,0110)"},
    "(0000,0002)" => $this->{"(0000,0002)"},
    "(0000,0900)" => $status,
  };
  return bless $resp, ref($this);
}
sub render{
  my($this) = @_;
  my $body = "";
  for my $i (sort keys %$this){
    my $value = $this->{$i};
    unless(exists $Posda::CmdDict::Dict->{$i}){
      die "unknown element in cmd $i";
    }
    unless($i =~ /\((....),(....)\)/){ die "invalid tag: $i" }
    my $grp = hex($1);
    my $ele = hex($2);
    my $type = $Posda::CmdDict::Dict->{$i}->{type};
    my $length;
    if($type eq "text"){
      $length = length($value);
      if($length & 1){
        if($Posda::CmdDict::Dict->{$i}->{vr} eq "UI"){
          $value .= "\0";
        } else {
          $value .= " ";
        }
      }
    } elsif ($type eq "ushort"){
      if(ref($value) eq "ARRAY"){
        $value = pack("v*", @$value);
      } else {
        $value = pack("v", $value);
      }
    } elsif ($type eq "ulong"){
      if(ref($value) eq "ARRAY"){
        $value = pack("V*", @$value);
      } else {
        $value = pack("V", $value);
      }
    } else {
    }
    $length = length($value);
    $body .= pack("vvV", $grp, $ele, $length) . $value;
  }
  my $len = length($body);
  my $head = pack("vvVV", 0, 0, 4, $len);
  my $resp = $head . $body;
  return $resp;
}
sub Debug{
  my($this) = @_;
  for my $i (sort keys %$this){
    print "Command{$i} = $this->{$i}\n";
  }
}

sub DESTROY {
  #print "Destroyed Posda::Command\n";
};
1;
