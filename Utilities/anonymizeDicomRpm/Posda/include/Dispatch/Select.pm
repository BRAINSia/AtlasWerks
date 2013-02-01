#!/usr/bin/perl -w
#$Source: /home/bbennett/pass/archive/Posda/include/Dispatch/Select.pm,v $
#$Date: 2008/08/28 21:41:07 $
#$Revision: 1.4 $

use strict;
use FileHandle;
use IO::Select;
use Time::HiRes;
package Dispatch::Select;

my $readers = {};
my $writers = {};
my $excepts = {};
my $background = [];
my $timer = {};

sub GetTypeArray{
  my($type) = @_;
  if($type eq "reader"){
    return $readers;
  } elsif($type eq "writer"){
    return $writers;
  } elsif($type eq "except"){
    return $excepts;
  } else {
    die "unknown type: $type";
  }
}

{
  package Dispatch::Select::Background;
  sub new{
    my($class, $closure) = @_;
    # closure($this);
    #print "Creating Dispatch::Select::Background\n";
    return bless $closure, $class;
  }
  sub queue{
    my($this) = @_;
    #print "Queueing Dispatch::Select::Background\n";
    push @$background, $this;
  }
  sub timer{
    my($this, $seconds) = @_;
    my $now = Time::HiRes::time;
    my $then = $now + $seconds;
    $timer->{$then} = $this;
  }
  sub clear{
    my($this) = @_;
    for my $k (keys %$timer){
      if($timer->{$k} eq $this){
        delete $timer->{$k};
      }
    }
    my $new_back = [];
    for my $b (@$background){
      unless($b eq $this){
        push(@$new_back, $b);
      }
    }
    $background = $new_back;
  }
  sub DESTROY{
    my($this) = @_;
    my $class = ref($this);
    if($ENV{POSDA_DEBUG}){
      print "Destroying $class\n";
    }
  }
}
{
  package Dispatch::Select::Socket;
  sub new {
    my($class, $closure, $socket) = @_;
    my $this = {
      socket => $socket,
      closure => $closure,
    };
    #print "Creating Dispatch::Select::Socket\n";
    return bless $this, $class;
    # closure($this, $socket);
  }
  sub Add {
    my($this, $type) = @_;  # type is reader, writer, or except
    my $h = Dispatch::Select::GetTypeArray($type);
    $h->{fileno($this->{socket})} = $this;
  }
  sub Remove {
    my($this, $type) = @_;  # type is reader, writer, or except
    if(defined $type){
      my $h = Dispatch::Select::GetTypeArray($type);
      delete $h->{fileno($this->{socket})};
    } else {
      $this->Remove("reader");
      $this->Remove("writer");
      $this->Remove("except");
    }
  }
  sub DESTROY{
    my($this) = @_;
    my $class = ref($this);
    if($ENV{POSDA_DEBUG}){
      print "Destroying $class\n";
    }
  }
}
{
  package Dispatch::Select::Event;
  sub new {
    my($class, $background) = @_;
    my $this = {
      background => $background,
    };
    #print "Creating Dispatch::Select::Event\n";
    return bless $this, $class;
  }
  sub post{
    my($this) = @_;
    if(exists($this->{background})){
       push(@$background, $this->{background});
    } else {
      die "Posting a cleared event";
    }
  }
  sub post_and_clear{
    my($this) = @_;
    unless(exists $this->{background}){
      die "Posting (and clearing) a cleared event";
    }
    my $bk = $this->{background};
    delete $this->{background};
    push(@$background, $bk);
  }
  sub DESTROY{
    my($this) = @_;
    my $class = ref($this);
    if($ENV{POSDA_DEBUG}){
      print "Destroying $class\n";
    }
  }
}

sub fh_list {
  my($type) = @_;
  my @list;
  my $count = 0;
  my $h = GetTypeArray($type);
  for my $i (keys %$h){
    push(@list, $h->{$i}->{socket});
    $count += 1;
  }
  return $count, \@list;
}

sub Dispatch{
  dispatch_loop:
  while(1){
    my($read_count, $read_handles) = fh_list("reader");
    my($write_count, $write_handles) = fh_list("writer");
    my($except_count, $except_handles) = fh_list("except");
    my $bk_count = @$background;
    my $timer_count = keys %$timer;
    my $timer_increment;
    if($timer_count > 0){
      $timer_increment = 
        ([sort { $a <=> $b } keys %$timer]->[0]) - Time::HiRes::time;
    }
    if(
      $read_count == 0 &&
      $write_count == 0 &&
      $except_count == 0 &&
      $timer_count == 0 &&
      $bk_count == 0
    ){
      return;
    }
    if($timer_count > 0 && defined $timer_increment){
      if($timer_increment <= 0){
        DispatchTimers();
        next dispatch_loop;
      }
    }
    if($bk_count > 0){
      $timer_increment = 0;
    }
    my $reader = IO::Select->new(@$read_handles);
    my $writer = IO::Select->new(@$write_handles);
    my $except = IO::Select->new(@$except_handles);
    my($r_dispatch, $w_dispatch, $e_dispatch);
    if(defined($timer_increment)){
      ($r_dispatch, $w_dispatch, $e_dispatch) =
        IO::Select->select($reader, $writer, $except, $timer_increment);
    } else {
      ($r_dispatch, $w_dispatch, $e_dispatch) =
        IO::Select->select($reader, $writer, $except);
    }
    DispatchFhHandlers($r_dispatch, "reader");
    DispatchFhHandlers($w_dispatch, "writer");
    DispatchFhHandlers($e_dispatch, "except");
    DispatchNextDefault();
  }
}

sub Dump{
  my($out) = @_;
  print $out "Readers:\n";
  for my $i (keys %$readers){
    print $out "\t$i: $readers->{$i}\n";
    print $out "\t\t$readers->{$i}->{closure}\n";
    print $out "\t\t$readers->{$i}->{socket}";
    my $file_no = $readers->{$i}->{socket}->fileno();
    print $out " ($file_no)\n";
  }
  print $out "Writers:\n";
  for my $i (keys %$writers){
    print $out "\t$i: $writers->{$i}\n";
    print $out "\t\t$writers->{$i}->{closure}\n";
    print $out "\t\t$writers->{$i}->{socket}";
    my $file_no = $writers->{$i}->{socket}->fileno();
    print $out " ($file_no)\n";
  }
  print $out "Excepts:\n";
  for my $i (keys %$excepts){
    print $out "\t$i: $excepts->{$i}\n";
    print $out "\t\t$excepts->{$i}->{closure}\n";
    print $out "\t\t$excepts->{$i}->{socket}";
    my $file_no = $excepts->{$i}->{socket}->fileno();
    print $out " ($file_no)\n";
  }
  print $out "Backgrounds:\n";
  for my $i (@$background){
    print $out "\t$i\n";
  }
  print $out "Timers:\n";
  for my $i (keys %$timer){
    print $out "\t$i: $timer->{$i}\n";
  }
};

sub DispatchTimers{
  my $now = Time::HiRes::time;
  for my $i (keys %$timer){
    if($i < $now){
      my $back = $timer->{$i};
      delete $timer->{$i};
      push @$background, $back;
    }
  }
}

sub DispatchFhHandlers{
  my $list = shift;
  my $type = shift;
  my $h = GetTypeArray($type);
  handle:
  for my $i (@$list){
    my $file_no = fileno($i);
    unless(defined $file_no) { next handle };
    unless(exists $h->{$file_no}) { next handle; }
    my $handler = $h->{$file_no};
    
    &{$handler->{closure}}($handler, $i);
  }
}

sub DispatchNextDefault{
  my $count = @$background;
  unless($count) { return }
  my $backgrnd = shift(@$background);
  &{$backgrnd}($backgrnd);
}


1;
