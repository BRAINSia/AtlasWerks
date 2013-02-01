#!/usr/bin/perl -w
#$Source: /home/bbennett/pass/archive/Posda/include/Dispatch/Dicom/Dataset.pm,v $
#$Date: 2008/08/28 21:41:07 $
#$Revision: 1.2 $
use Dispatch::Dicom;
my $dbg = sub {print @_};
use strict;
package Dispatch::Dicom::Dataset;
sub new{
  my($class, $file, $xfr_stx, $pdu_len) = @_;
  my $this = {
    file => $file,
    xfr_stx => $xfr_stx,
    finished => 0,
    pdv_len => $pdu_len - 6,
  };
  return bless $this, $class;
}
sub start {
  my($this) = @_;
  my $buff;
  if(exists $this->{fh}) { return }
  my $fh = FileHandle->new("RenderDs.pl $this->{file} $this->{xfr_stx}|");
  $this->{fh} = $fh;
  $this->CreateReader();
}
sub CreateReader{
  my($this) = @_;
  my $foo = sub {
    my($sock) = @_;
    if($this->{finished}){
      $sock->Remove("reader");
      return;
    }
    my $data;
    my $length_to_read = $this->{pdv_len};
    if($#{$this->{queue}} > 1){
      unless($this->{finished}){
        $this->WaitForQueueRemoval();
      }
      $sock->Remove("reader");
      return;
    }
    my $lr = read($this->{fh}, $data, $length_to_read);
    if($lr == 0){
      $this->{finished} = 1;
      $sock->Remove("reader");
    } else {
      push(@{$this->{queue}}, $data);
    }
    if(exists $this->{wait_data_in_queue}){
      $this->{wait_data_in_queue}->post_and_clear();
      delete $this->{wait_data_in_queue};
    }
  };
  my $sel = Dispatch::Select::Socket->new($foo, $this->{fh});
  $sel->Add("reader");
}
sub WaitForQueueRemoval{
  my($this) = @_;
  my $foo = sub {
    my($back) = @_;
    delete $this->{removal_event};
    $this->CreateReader();
  };
  $this->{removal_event} = Dispatch::Select::Event->new(
    Dispatch::Select::Background->new($foo)
  );
}
sub ready_out{
  # returns -1 finished, 0 not ready, 1 ready
  # not ready until > pdv_len, or finished flag set
  my($this, $size) = @_;
  if($#{$this->{queue}} < 0){
    if($this->{finished}) { return -1 }
    return 0;
  }
  if($#{$this->{queue}} > 0){
    return 1;
  }
  if($this->{finished}){
    return 1;
  }
  return 0;
}
sub get_pdv{
  # trouble unless ready for pdv_len
  my($this, $pdv_len, $pc_id) = @_;
  my $flgs = 0;
  unless(defined $this->{queue}){ die "data undefined" }
  my $data = shift(@{$this->{queue}});
  my $len = length ($data);
  if($this->{finished} && $#{$this->{queue}} < 0){
    $flgs |= 2;
  }
  if(defined($this->{removal_event})){
    unless($this->{removal_event}->can("post_and_clear")){ die "wtf!" }
    $this->{removal_event}->post_and_clear();
    delete $this->{removal_event};
  }
  my @foo;
  push(@foo, pack("NCC", length($data) + 2, $pc_id, $flgs));
  push(@foo, $data);
  return \@foo;
}
sub wait_ready_out{
  my($this, $event) = @_;
  $this->{wait_data_in_queue} = $event;
}
sub AbortPlease{
  my($this) = @_;
  if(exists $this->{wait_data_in_queue}){
    $this->{wait_data_in_queue}->post_and_clear();
    delete $this->{wait_data_in_queue};
  }
}
sub DESTROY{
  my($this) = @_;
  my $class = ref($this);
  if($ENV{POSDA_DEBUG}){
    print "Destroying $class\n";
  }
}
1;
