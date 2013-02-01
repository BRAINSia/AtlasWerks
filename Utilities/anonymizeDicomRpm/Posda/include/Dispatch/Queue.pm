#!/usr/bin/perl -w
#$Source: /home/bbennett/pass/archive/Posda/include/Dispatch/Queue.pm,v $
#$Date: 2008/08/28 21:41:07 $
#$Revision: 1.2 $
package Dispatch::Queue;

use vars qw( $Next );
$Next = 0;
sub new{
  my($class, $high, $low) = @_;
  my $this = {
    high => $high,
    low => $low,
    queue => [],
    label => $Next,
    finished => 0,
  };
  $Next++;
  return bless $this, $class;
}
sub ready_out {
   my($this) = @_;
   if($this->{finished}){
     return 0;
   }
   return ((scalar @{$this->{queue}}) <= $this->{high});
}
sub post_output{
  my($this) = @_;
  if(
    defined($this->{output_event})
  ){
    $this->{output_event}->post_and_clear();
    delete $this->{output_event};
  }
}
sub queue {
   my($this, $string) = @_;
  my $str_len = length($string);
  # don't queue an empty string (you'll regret it later)
  if($str_len == 0){
    return;
   }
   push(@{$this->{queue}}, $string);
   if(defined $this->{input_event}){
     $this->{input_event}->post_and_clear();
     delete $this->{input_event};
   }
}
sub dequeue{
   my($this) = @_;
   if(scalar(@{$this->{queue}}) < 1){
     return undef;
   }
  my $string = shift @{$this->{queue}};
  if(
    defined($this->{output_event}) &&
    (scalar @{$this->{queue}}) < $this->{low}
  ){
    $this->{output_event}->post_and_clear();
    delete $this->{output_event};
  }
  return $string;
}
sub wait_output{
  my($this, $event) = @_;
  $this->{output_event} = $event;
}
sub wait_input {
  my($this, $event) = @_;
  $this->{input_event} = $event;
}
sub finish{
  my($this) = @_;
  #print "In finished\n";
  $this->{finished} = 1;
  if(defined $this->{input_event}){
    $this->{input_event}->post_and_clear();
    delete $this->{input_event};
  }
  if(
    defined($this->{output_event})
  ){
    $this->{output_event}->post_and_clear();
    delete $this->{output_event};
  }
}
sub SocketWriter{
  my($queue) = @_;
  my $foo = sub {
    my($this, $sock) = @_;
    if($#{$queue->{queue}} >= 0){
      my $string = $queue->{queue}->[0];
      my $str_len = length($string);
      if($str_len == 0){
         die "string of zero length on queue";
      }
      unless(defined $queue->{current_offset}) {$queue->{current_offset} = 0 }
      my  $len = 
        syswrite $sock, $string, length($string), $queue->{current_offset};
      if($len >= 0){
        $queue->{current_offset} += $len;
      } elsif ($len < 0){
        die "error writing socket: $!";
      }
      if($len == 0){
        print STDERR "Wrote 0 (had selected true)";
        return;
      }
      if($queue->{current_offset} == length($string)){
        $queue->dequeue();
        delete $queue->{current_offset};
      } else {
      }
    } else {
      $this->Remove("writer");
      if($queue->{finished}) {
         return;
      }
      $queue->wait_input($queue->CreateQueueEmptierEvent($sock));
    }
  };
  return $foo;
}
sub CreateQueueEmptierEvent{
  my($this, $sock) = @_;
  my $foo = sub {
    my($back) = @_;
    if(defined $sock->fileno()){
      my $handler = Dispatch::Select::Socket->new($this->SocketWriter(), $sock);
      $handler->Add("writer");
    } else {
      if($#{$queue->{queue}} >= 0){
        print "socket closed when queue not empty\n";
      }
    }
  };
  $this->{input_event} = Dispatch::Select::Event->new(
    Dispatch::Select::Background->new($foo)
  );
}
sub SocketReader{
  my($queue, $sock) = @_;
  my $foo = sub {
    my($disp, $sock) = @_;
    my $string;
    if($queue->ready_out()){
      my $len = sysread($sock, $string, 1024);
      if($len <= 0){
        $queue->finish();
        $disp->Remove("reader");
        return;
      } else {
        $queue->queue($string);
      }
    }
    unless($queue->ready_out()){
      $disp->Remove("reader");
      $queue->wait_output($queue->CreateQueueFillerEvent($sock));
    }
  };
}
sub CreateQueueFillerEvent{
  my($this, $sock) = @_;
  my $foo = sub {
    my($back) = @_;
    my $handler = Dispatch::Select::Socket->new($this->SocketReader(), $sock);
    $handler->Add("reader");
  };
  $this->{output_event} = Dispatch::Select::Event->new(
    Dispatch::Select::Background->new($foo)
  );
}
sub DESTROY{
  my($this) = @_;
  my $class = ref($this);
  if($ENV{POSDA_DEBUG}){
    print "Destroying $class\n";
  }
}
1;
