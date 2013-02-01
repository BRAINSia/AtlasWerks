#!/usr/bin/perl -w
#$Date: 2008/09/03 17:10:16 $
#$Revision: 1.7 $
#
#Copyright 2008, Bill Bennett
# Part of the Posda package
# Posda may be copied only under the terms of either the Artistic License or the
# GNU General Public License, which may be found in the Posda Distribution,
# or at http://posda.com/License.html
#

use strict;
use Dispatch::Select;
use Dispatch::Queue;
use Dispatch::Acceptor;
use Dispatch::Dicom::Storage;
use Dispatch::Dicom::Verification;
use Dispatch::Dicom::MessageAssembler;
use Dispatch::Dicom::Dataset;
use Dispatch::Command::Basic;
use Posda::UID;
use IO::Socket::INET;
use FileHandle;

use vars qw( %Objects );

unless(
  $#ARGV == 4 ||
  ($#ARGV == 5) && ($ARGV[0] eq "UID")
){ 
  die "usage: " .
    "$0 <cmd_port> <dcm_port> <config_file> <command_dir> <rcv_dir>"
}
my($UIDRoot, $UIDSeq);
if($#ARGV == 5){
  shift @ARGV;
  my $user = `whoami`;
  chomp $user;
  my $host = `hostname`;
  chomp $host;
  $UIDRoot = Posda::UID::GetPosdaRoot({
    package => "Posda/bin/ae/Receiver.pl",
    user => $user,
    host => $host,
    purpose => "Receiver/Responder",
  });
  $UIDSeq = 1;
}

my $cmd_port = $ARGV[0];
my $dcm_port = $ARGV[1];
my $config_file = $ARGV[2];
my $cmd_dir = $ARGV[3];
my $rcv_dir = $ARGV[4];
unless(-r $config_file) { die "<config_file> is not a file" }
unless(-d $rcv_dir) { die "<rcv_dir> is not a directory" }
unless(-d $cmd_dir) { die "<command_dir> is not a directory" }

unless($config_file =~ /^\//){
  my $wd = `pwd`;
  chomp $wd;
  $config_file = "$wd/$config_file";
}
unless($config_file =~ /(.*)\/([^\/]+)$/){
  die "funny config file path: $config_file";
}
my $config_dir = $1;
my $config_file_name = $2;
 
my $cmd = Dispatch::Command::Basic::Acceptor->new($cmd_port, $cmd_dir, \%Objects);
$Objects{CmdAcceptor} = $cmd;

sub AnnounceDisconnect{
  my($name, $calling, $host, $session_info) = @_;
  my $foo = sub {
    delete $Objects{$name};
    delete $Main::ActiveConnections{"$host:$calling"};
    open FILE, "|$config_dir/$session_info->{handler}";
    print FILE "SCU|$name\n";
    print FILE "config_dir|$config_dir\n";
    if(defined $UIDRoot){
      print FILE "UIDRoot|$UIDRoot.$UIDSeq\n";
      $UIDSeq += 1;
    }
    print FILE "host|$host\n";
    print FILE "calling|$session_info->{calling}\n";
    print FILE "called|$session_info->{called}\n";
    for my $i (keys %{$session_info->{files}}){
      for my $j (keys %{$session_info->{files}->{$i}}){
        print FILE "file|$i|" .
          "$session_info->{files}->{$i}->{$j}->{sop_instance}|" .
          "$session_info->{files}->{$i}->{$j}->{xfrstx}|$j\n";
      }
    }
    close FILE;
  };
  return $foo;
}
sub AnnounceFileReceived{
  my($name, $calling, $host, $session_info) = @_;
  my $foo = sub {
    my($file_name, $sop_class, $sop_instance, $xfrstx) = @_;
    my $dir_name = "$rcv_dir/$host-$calling";
    $dir_name =~ s/ *$//;
    unless(-d "$dir_name"){
      `mkdir $dir_name`;
    }
    unless (-f $file_name){
      print "Storage interrupted (socket unexpectedly closed?)\n";
      return;
    }
    my $mcmd = "mv $file_name $dir_name";
   `$mcmd`;
    my $short_name;
    $file_name =~ /\/([^\/]*)$/;
    $short_name = $1;
    my $new_file_name = "$dir_name/$short_name";
#    print "$name: received file ($sop_class):\n\t$short_name\n" .
#      "\t$xfrstx\n";;
    $session_info->{files}->{$sop_class}->{$new_file_name} = {
      xfrstx => $xfrstx,
      sop_instance => $sop_instance,
    };
  };
  return $foo;
}
my $connection_count = 0;
sub CreateConnectionHandler{
  my($name, $rcv_dir) = @_;
  my $foo = sub {
    my($obj) = @_;
    $connection_count += 1;
    my $new = "$name" . "_" . $connection_count;
    my $called = $obj->{assoc_ac}->{called};
    my $calling = $obj->{assoc_ac}->{calling};
    my($port, $iaddr) = sockaddr_in($obj->{socket}->peername);
    my $host_ip = inet_ntoa($iaddr);
    print "$new: Connected ($calling, $called, ($port, $host_ip))\n";
    if(exists $Main::ActiveConnections{"$host_ip:$calling"}){
      print "Error: Second association $host_ip:$calling\n";
      $obj->Abort();
      return;
    }
    $Objects{$new} = $obj;
    $Main::ActiveConnections{"$host_ip:$calling"} = $new;
    my $handler = $obj->{descrip}->{assoc_normal_close};
    print "Handler: $handler\n";
    print "Calling: $calling\n";
    print "Called: $called\n";
    my $session_info = {
      handler => $handler,
      calling => $calling,
      called => $called,
    };
    if($obj->can("SetDisconnectCallback")){
      $obj->SetDisconnectCallback(AnnounceDisconnect(
        $new, $calling, $host_ip, $session_info));
    } else {
      die "Can't handle SetDisconnectCallback";
    }
    if($obj->can("SetDatasetReceivedCallback")){
      $obj->SetDatasetReceivedCallback(AnnounceFileReceived($new, 
        $calling, $host_ip, $session_info));
    } else {
      die "Can't handle SetDatasetReceivedCallback";
    }
    if($obj->can("SetStorageRoot")){
      $obj->SetStorageRoot($rcv_dir);
    } else {
      die "Can't handle SetStorageRoot";
    }
  };
  return $foo;
}
print "Calling Dispatch::Dicom::Acceptor->new\n";
my $scp = Dispatch::Dicom::Acceptor->new(
  $dcm_port, $config_file, CreateConnectionHandler("SCP", $rcv_dir)
);
$Objects{SCP} =  $scp;
$scp->Add("reader");
Dispatch::Select::Dispatch();
