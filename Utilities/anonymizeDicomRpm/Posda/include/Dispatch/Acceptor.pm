#!/usr/bin/perl -w
#$Source: /home/bbennett/pass/archive/Posda/include/Dispatch/Acceptor.pm,v $
#$Date: 2008/08/28 21:41:07 $
#$Revision: 1.2 $
use strict;
use FileHandle;
package Dispatch::Acceptor;
use Fcntl qw (F_GETFL F_SETFL O_NONBLOCK);

sub new {
  my($class, $closure) = @_;
  my $foo = sub {
    my($this, $socket) = @_;
    my $new_sock = $socket->accept();
    $new_sock->autoflush();
    &$closure($this, $new_sock);
  };
  return bless $foo, $class;
}
sub port_server{
  my($this, $port) = @_;
  my $sock = IO::Socket::INET->new(
    Listen => 1024,
    LocalPort => $port,
    Proto => 'tcp',
    Blocking => 0,
    ReuseAddr => 1,
  );
unless($sock) { die "couldn't open listener on port $port: $!" }
  my $port_server = Dispatch::Select::Socket->new($this, $sock);
  return $port_server;
}
sub DESTROY {
  my($this) = @_;
  my $class = ref($this);
  if($ENV{POSDA_DEBUG}){
    print "Destroying $class\n";
  }
}
1;
