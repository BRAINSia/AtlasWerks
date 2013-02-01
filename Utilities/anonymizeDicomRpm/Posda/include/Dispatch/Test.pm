#!/usr/bin/perl -w
#$Source: /home/bbennett/pass/archive/Posda/include/Dispatch/Test.pm,v $
#$Date: 2008/08/10 12:57:59 $
#$Revision: 1.1 $

use strict;
use Dispatch::Select;
use Dispatch::Queue;
use Dispatch::Acceptor;
use Dispatch::Template;
use Dispatch::Command::Basic;
use Dispatch::Http;
use Dispatch::Dicom::Storage;
use Dispatch::Dicom::Verification;
use Dispatch::Dicom::MessageAssembler;
use Dispatch::Dicom::Dataset;
use IO::Socket::INET;
use FileHandle;
use Posda::Find;

package Dispatch::Test;
use vars qw( $Commands $dbg $objs $Help);
use Debug;
$dbg = sub {print @_};
$objs = {
};
sub ProcessLines{
  my($sock, $closure) = @_;
  my $buff = "";
  my $count = 0;
  my $foo = sub {
    my($disp, $socket) = @_;
    my $inp = sysread $sock, $buff, 1, $count;
    unless($inp == 1){
      $disp->Remove("reader");
      return;
      die "socket closed prematurely"
    }
    $count += 1;
    if($buff =~ /^(.*)\n$/s){
      chomp $buff;
      $buff =~ s/\r//g;
      &$closure($buff);
      $buff = "";
      $count = 0;
    }
  };
  my $handler = Dispatch::Select::Socket->new($foo, $sock);
  $handler->Add("reader");
  return;
}
sub LineHandler{
  my($obj, $file, $back) = @_;
  my $foo = sub {
    my($line) = @_;
#    $back->queue();
    $objs->{$obj}->WaitMessageQueue(3, $back);
    my($sopcl, $xfr_stx, $sop_inst);
    if($line =~ /^Sop Class:\s*([0-9.]+)\s*Xfr Stx:\s*([0-9.]+)\s*Sop Inst:\s*([0-9.]+)$/){
      $sopcl = $1;
      $xfr_stx = $2;
      $sop_inst = $3;
    }
    unless($sopcl && $xfr_stx){
      print "couldn't get sopcl and xfr_stx from file\n";
      return;
    }
    unless(exists $objs->{$obj}->{sopcl}->{$sopcl}){
      print "connection doesn't handle sopcl $sopcl\n";
      return;
    }
    unless(exists $objs->{$obj}->{sopcl}->{$sopcl}->{$xfr_stx}){
      print "connection doesn't handle xfr_stx $xfr_stx for sopcl $sopcl\n";
      return;
    }
    my $pc_id = $objs->{$obj}->{sopcl}->{$sopcl}->{$xfr_stx};
    my $len = $objs->{$obj}->{max_length};
    my $ds = Dispatch::Dicom::Dataset->new($file, $xfr_stx, $len);
    my $cmd = Posda::Command->new_store_cmd($sopcl, $sop_inst);
    my $ma = Dispatch::Dicom::MessageAssembler->new($pc_id, 
      $cmd, $ds, CreateFileResponse($file));
    $ds->start();
    $objs->{$obj}->QueueMessage($ma);
    my $now = time();
    print "Queued $file for transmission at $now\n";
  };
  return $foo;
}
sub CreateFileSender{
  my($list, $obj) = @_;
  my $foo = sub {
    my($back) = @_;
    unless(ref($list) eq "ARRAY"){ return };
    unless($#{$list} >= 0){ return };
    my $file = shift @$list;
    my $fh =  FileHandle->new("GetSopClassXfrStx.pl $file|");
    if($fh) { 
      ProcessLines($fh, LineHandler($obj, $file, $back));
    }
    unless($#{$list} >= 0){ return };
  };
  return $foo;
}
sub CreateBetterFileSender{
  my($list, $obj) = @_;
  my $foo = sub {
    my($back) = @_;
    unless(ref($list) eq "ARRAY"){ return };
    unless($#{$list} >= 0){ return };
    my $descrip = shift @$list;
    my $file = $descrip->{path};
    my $sopcl = $descrip->{sop_class};
    my $xfr_stx = $descrip->{xfr_stx};
    my $default_xfr_stx = '1.2.840.10008.1.2';
    my $sop_inst = $descrip->{sop_instance};
    unless($sopcl && $xfr_stx){
      print "couldn't get sopcl and xfr_stx from file\n";
      return;
    }
    unless(exists $objs->{$obj}->{sopcl}->{$sopcl}){
      print "connection doesn't handle sopcl $sopcl\n";
      return;
    }
    unless(
      exists $objs->{$obj}->{sopcl}->{$sopcl}->{$xfr_stx}
    ){
      unless(
        exists $objs->{$obj}->{sopcl}->{$sopcl}->{$default_xfr_stx}
      ){
        print "connection doesn't handle xfr_stx $xfr_stx for sopcl $sopcl\n";
        return;
      }
      $xfr_stx = $default_xfr_stx;
    }
    my $pc_id = $objs->{$obj}->{sopcl}->{$sopcl}->{$xfr_stx};
    my $len = $objs->{$obj}->{max_length};
    my $ds = Dispatch::Dicom::Dataset->new($file, $xfr_stx, $len);
    my $cmd = Posda::Command->new_store_cmd($sopcl, $sop_inst);
    my $ma = Dispatch::Dicom::MessageAssembler->new($pc_id, 
      $cmd, $ds, CreateFileResponse($file));
    $objs->{$obj}->QueueMessage($ma);
    my $now = time();
#    print "Queued $file for transmission at $now\n";
    unless($#{$list} >= 0){ return };
    $back->queue();
  };
  return $foo;
}
sub SendDicomFile{
  my($obj, $file) = @_;
  unless(exists $objs->{$obj}) {
    print "$obj doesn't exist\n";
    return;
  }
  unless(ref($objs->{$obj}) eq "Dispatch::Dicom::Connection"){
    print "$obj isn't a dicom connection\n";
    return;
  }
  my $line = `GetSopClassXfrStx.pl $file`;
  chomp $line;
  my($sopcl, $xfr_stx, $sop_inst);
  if($line =~ /^Sop Class:\s*([0-9.]+)\s*Xfr Stx:\s*([0-9.]+)\s*Sop Inst:\s*([0-9.]+)$/){
    $sopcl = $1;
    $xfr_stx = $2;
    $sop_inst = $3;
  }
  unless($sopcl && $xfr_stx){
    print "couldn't get sopcl and xfr_stx from file\n";
    return;
  }
  unless(exists $objs->{$obj}->{sopcl}->{$sopcl}){
    print "connection doesn't handle sopcl $sopcl\n";
    return;
  }
  unless(exists $objs->{$obj}->{sopcl}->{$sopcl}->{$xfr_stx}){
    print "connection doesn't handle xfr_stx $xfr_stx for sopcl $sopcl\n";
    return;
  }
  my $pc_id = $objs->{$obj}->{sopcl}->{$sopcl}->{$xfr_stx};
  my $len = $objs->{$obj}->{max_length};
  my $ds = Dispatch::Dicom::Dataset->new($file, $xfr_stx, $len);
  my $cmd = Posda::Command->new_store_cmd($sopcl, $sop_inst);
  my $ma = Dispatch::Dicom::MessageAssembler->new($pc_id, 
    $cmd, $ds, CreateFileResponse($file));
  $objs->{$obj}->QueueMessage($ma);
  my $now = time();
  print "Queued $file for transmission at $now\n";
}
sub CreateHttpClosure{
  my($name, $port) = @_;
  my $foo = sub {
    my($http) = @_;
    print "in http handler $name, $port\n";
    $http->{output_queue}->queue("HTTP/1.0 200 OK\n");
    $http->{output_queue}->queue("Content-Type: text/plain\n");
    $http->{output_queue}->queue("Connection: close\n\n");
    for my $i (keys %$http){
      if(ref($http->{$i}) eq "ARRAY"){
        for my $j (0 .. $#{$http->{$i}}){
          print "\t$j: $http->{$i}->[$j]\n";
          $http->{output_queue}->queue("\t$j: $http->{$i}->[$j]\n");
        }
      } elsif(ref($http->{$i}) eq "HASH"){
        for my $j (sort keys %{$http->{$i}}){
          print "\t$j: $http->{$i}->{$j}\n";
          $http->{output_queue}->queue("\t$j: $http->{$i}->{$j}\n");
        }
      } else {
          print "$i: $http->{$i}\n";
        $http->{output_queue}->queue("$i: $http->{$i}\n");
      }
    }
  };
  return $foo;
}  
sub CreateReader{
  my $foo = sub {
    my($this, $socket) = @_;
    my $mess;
    my $count = read($socket, $mess, 1024);
    unless(defined $count){
      $this->Remove('reader');
      print "disconnected socket\n";
      return;
    }
    print "read: \"$mess\"\n";
  };
  return $foo;
}
sub CreateTrickleResponder{
  my($args) = @_;
  my $foo = sub {
    my($this, $out) = @_;
    my $queue = Dispatch::Queue->new(5, 2);
    $queue->CreateQueueEmptierEvent($out);
    my $back = Dispatch::Select::Background->new(
      CreateTrickler([undef, $args->[2], $args->[3]], $queue)
    );
    #$objs->{$args->[0]} = $back;
    $back->queue();
  }
}
sub CreateCounterResponder{
  my($count) = @_;
  my $foo = sub {
    my($this, $socket) = @_;
    my $queue = Dispatch::Queue->new(5, 2);
    $queue->CreateQueueEmptierEvent($socket);
    my $foo = Dispatch::Select::Background->new(
      CreateQueueCounter($queue, $count)
    );
    $foo->queue();
  };
  return $foo;
}
sub CreateNotifier{
  my($out, $name) = @_;
  my $foo = sub {
    print $out "Event: $name occured\n";
  };
  return $foo;
}
sub CreateCounter{
  my($out, $name, $count) = @_;
  my $foo = sub {
    my $disp = shift;
    print $out "Counter: $name $count\n";
    $count -= 1;
    if($count > 0){
      $disp->queue();
    } else {
      delete $objs->{$name};
    }
  };
  return $foo;
}
sub CreateTimer {
  my($out, $name) = @_;
  my $now = Time::HiRes::time;
  my $foo = sub {
    my $then = Time::HiRes::time;
    my $elapsed = $then - $now;
    print $out "Timer: $name timed out after $elapsed\n";
    delete $objs->{$name};
  };
  return $foo;
};
sub CreateTrickler {
  my($args, $queue) = @_;
  my $name = $args->[0];
  my $count = $args->[1];
  my $interval = $args->[2];
  my $now = Time::HiRes::time;
  my $foo = sub {
    my $back = shift;
    my $then = Time::HiRes::time;
    my $elapsed = $then - $now;
    $queue->queue("Trickler: count = $count, elapsed = $elapsed\n");
    $count--;
    if($count < 0){
      if(defined $name){
        delete $objs->{$name};
      }
      $queue->finish();
      return;
    }
    $back->timer($interval);
  };
  return $foo;
};
sub CreateQueueCounterOne{
  my($queue, $count, $name) = @_;
  my $foo = sub {
    my $back = shift;
    if($count == 0){
      $queue->finish();
      if(defined $name){
        delete $objs->{$name};
      }
    } elsif ($queue->ready_out()){
      $count--;
      $queue->queue("This is a line of output ($count remaining)\n");
      $queue->wait_output(Dispatch::Select::Event->new($back));
    }
  };
  return $foo;
}
sub CreateQueueCounter{
  my($queue, $count, $name) = @_;
  my $foo = sub {
    my $back = shift;
    while($queue->ready_out() && $count > 0){
      $count--;
      $queue->queue("This is a line of output ($count remaining)\n");
    }
    if($count > 0){
      $queue->wait_output(Dispatch::Select::Event->new($back));
      return;
    }
    $queue->finish();
    if(defined $name){
      delete $objs->{$name};
    }
  };
  return $foo;
}
sub CreateFileResponse{
  my($file_name) = @_;
  my $foo = sub {
    my($resp) = @_;
    print "response for send of file: $file_name\n";
  };
  return $foo;
}
sub CreateEchoResponse{
  my $foo = sub {
    my($resp) = @_;
    print "response to echo\n";
  };
  return $foo;
}
sub IsBlessed{
  my($name) = @_;
  my $ref = ref($objs->{$name});
  if($ref && $ref ne "ARRAY" && $ref ne "HASH" && $ref ne "SCALAR"){
    return 1;
  }
  return 0;
}
sub DeleteObj{
  my($name) = @_;
  if(defined $objs->{$name} && IsBlessed($name)){
    if($objs->{$name}->can("Remove")){
      $objs->{$name}->Remove();
    }
    if($objs->{$name}->can("Finish")){
      $objs->{$name}->Finish();
    }
    if($objs->{$name}->can("Release")){
      $objs->{$name}->Release();
    }
  }
  delete $objs->{$name};
}
sub CreateFileScanner{
  my($list) = @_;
  my $foo = sub {
   my($path, $df, $ds, $size, $xfr_stx, $errors) = @_;
   push(@$list, $path);
  };
  return $foo;
}
sub CreateBetterFileScanner{
  my($list) = @_;
  my $foo = sub {
   my($path, $df, $ds, $size, $xfr_stx, $errors) = @_;
   my $sop_class = $ds->ExtractElementBySig("(0008,0016)");
   my $sop_instance = $ds->ExtractElementBySig("(0008,0018)");
   my $descrip = {
     path => $path,
     xfr_stx => $xfr_stx,
     sop_class => $sop_class,
     sop_instance => $sop_instance,
   };
   push(@$list, $descrip);
  };
  return $foo;
}
my $seq = 1;
sub CreateObjAdder{
  my($name) = @_;
  my $foo = sub {
     my($obj) = @_;
     my $new = $name . "_" . $seq;
     $seq += 1;
     $objs->{$new} = $obj;
     if($obj->can("SetDisconnectCallback")){
       $obj->SetDisconnectCallback(CreateObjDeleter($new));
     }
  };
  return $foo;
}
sub CreateObjDeleter{
  my($name) = @_;
  my $foo = sub {
     my($obj) = @_;
     if(exists $objs->{$name}){
       delete $objs->{$name};
     }
  };
  return $foo;
}
$Help = {
  exit => "",
  foo => "<args>",
  notifier => "<name>",
  post => "<name>",
  counter => "<name> <count>",
  queue_counter => "<name> <count>",
  debug => "",
  timer => "<name> <interval>",
  accept => "<name> <port>", 
  delete => "<name>",
  list => "",
  icon => "<file_name>",
  trickler => "<name> <count> <interval>",
  queue_counter => "<name> <count>",
  queue_counter_one => "<name> <count>",
  respond => "<name> <port> <count>",
  respond_trickle => "<name> <port> <count> <interval>",
  help => "",
};
$Commands = {
  help => sub{
    my($disp, $fh, $out, $args) = @_;
    for my $i (keys %$Help){
      print "$i $Help->{$i}\n";
    }
  },
  exit => sub {
    my($disp, $fh, $out, $args) = @_;
    for my $name (keys %$objs){
      DeleteObj($name);
    }
  },
  foo => sub {
    my($disp, $fh, $out, $args) = @_;
    print $out "foo(";
    for my $i (0 .. $#{$args}){
      print "$args->[$i]";
      unless($i == $#{$args}){
        print $out ", ";
      }
    }
    print ")\n";
  },
  notifier => sub {
    my($disp, $fh, $out, $args) = @_;
    my $foo = Dispatch::Select::Event->new(
      Dispatch::Select::Background->new(CreateNotifier(\*STDOUT, "$args->[0]"))
    );
    $objs->{$args->[0]} = $foo;
  },
  post => sub {
    my($disp, $fh, $out, $args) = @_;
    if(exists $objs->{$args->[0]}){
      $objs->{$args->[0]}->post();
    }
    delete($objs->{$args->[0]});
  },
  counter => sub {
    my($disp, $fh, $out, $args) = @_;
    my $foo = Dispatch::Select::Background->new(
      CreateCounter(\*STDOUT, $args->[0], $args->[1])
    );
    $objs->{$args->[0]} = $foo;
    $foo->queue();
  },
  queue_counter_one => sub {
    my($disp, $fh, $out, $args) = @_;
    my $queue = Dispatch::Queue->new(5, 2);
    $queue->CreateQueueEmptierEvent($out);
    my $foo = Dispatch::Select::Background->new(
      CreateQueueCounterOne($queue, $args->[1], $args->[0])
    );
    $objs->{$args->[0]} = $foo;
    $foo->queue();
  },
  queue_counter => sub {
    my($disp, $fh, $out, $args) = @_;
    my $queue = Dispatch::Queue->new(5, 2);
    $queue->CreateQueueEmptierEvent($out);
    my $foo = Dispatch::Select::Background->new(
      CreateQueueCounter($queue, $args->[1], $args->[0])
    );
    $objs->{$args->[0]} = $foo;
    $foo->queue();
  },
  trickler => sub {
    my($disp, $fh, $out, $args) = @_;
    my $queue = Dispatch::Queue->new(5, 2);
    $queue->CreateQueueEmptierEvent($out);
    my $foo = Dispatch::Select::Background->new(
      CreateTrickler($args, $queue)
    );
    $objs->{$args->[0]} = $foo;
    $foo->queue();;
  },
  timer => sub {
    my($disp, $fh, $out, $args) = @_;
    my $foo = Dispatch::Select::Background->new(
      CreateTimer(\*STDOUT, $args->[0])
    );
    $objs->{$args->[0]} = $foo;
    $foo->timer($args->[1]);
  },
  debug => sub {
    my($disp, $fh, $out, $args) = @_;
    Dispatch::Select::Dump();
    for my $i (keys %$objs){
      print "obj{$i} = ";
      Debug::GenPrint($dbg, $objs->{$i}, 1, 1);
      print "\n";
    }
  },
  debug_obj => sub {
    my($disp, $fh, $out, $args) = @_;
    my $name = $args->[0];
    my $depth = $args->[1];
    unless($depth) { $depth = 3 }
    unless(exists $objs->{$name}){
      print "No object named $name exists\n";
      return;
    }
    my $ref = ref($objs->{$name});
    unless($ref){
      print "Object named $name has no type\n";
      return;
    }
    print "Object ($ref) $name: ";
    Debug::GenPrint($dbg, $objs->{$name}, 1, $depth);
    print "\n";
  },
  accept => sub {
    my($disp, $fh, $out, $args) = @_;
    my $clos = CreateReader();
    my $acceptor = Dispatch::Acceptor->new($clos);
    my $foo =  $acceptor->port_server($args->[1]);
    $foo->Add("reader");
    $objs->{$args->[0]} = $foo;
  },
  respond => sub {
    my($disp, $fh, $out, $args) = @_;
    my $clos = CreateCounterResponder($args->[2], $args->[0]);
    my $acceptor = Dispatch::Acceptor->new($clos);
    my $foo =  $acceptor->port_server($args->[1]);
    $foo->Add("reader");
    $objs->{$args->[0]} = $foo;
  },
  respond_trickle => sub {
    my($disp, $fh, $out, $args) = @_;
    my $clos = CreateTrickleResponder($args);
    my $acceptor = Dispatch::Acceptor->new($clos);
    my $foo =  $acceptor->port_server($args->[1]);
    $foo->Add("reader");
    $objs->{$args->[0]} = $foo;
  },
  delete => sub {
    my($disp, $fh, $out, $args) = @_;
    my $name = $args->[0];
    DeleteObj($name);
  },
  list => sub {
    my($disp, $fh, $out, $args) = @_;
    for my $key (keys %$objs){
      print "$key: $objs->{$key}\n";
    }
  },
  icon => sub {
    my($disp, $fh, $out, $args) = @_;
    my $file_name = $args->[0];
    my $fh1 =  FileHandle->new("<$file_name");
    unless($fh1){
      $out->print("Couldn't open $file_name\n");
      return;
    }
    my $buff;
    my $size = $fh1->read($buff, 1024);
    $out->print("read $size bytes from $file_name\n");
    $objs->{icon} = $buff;
  },
  http => sub {
    my($disp, $fh, $out, $args) = @_;
    my $name = $args->[0];
    my $port = $args->[1];
    my $http_clos = CreateHttpClosure($name, $port);
    my $foo =  Dispatch::HttpServer->new($args->[1], $http_clos);
    $foo->Add("reader");
    $objs->{$name} = $foo;
  },
  http_app => sub {
    my($disp, $fh, $out, $args) = @_;
    my $name = $args->[0];
    my $port = $args->[1];
    my $dir = $args->[2];
    my $app_dir = $args->[3];
    my $interval = $args->[4];
    unless(defined $interval && $interval > 0){ $interval = 10 }
    my $time_to_live = $args->[5];
    unless(defined $time_to_live && $time_to_live > 0){ $time_to_live = 100 }
    my $app_struct = Dispatch::Http::App->new($app_dir);
    my $App = 
      Dispatch::Http::App::Server->new(
        $dir, $app_struct, $interval, $time_to_live );
    $App->Serve($port, $interval, $time_to_live);
    $objs->{$name} = $App;
  },
  ref => sub {
    my($disp, $fh, $out, $args) = @_;
    my $name = $args->[0];
    unless(exists $objs->{$name}){
      print "Object $name doesn't exist\n";
      return;
    }
    my $ref = ref($objs->{$name});
    if($ref) {
      print "$name is of type $ref\n";
    } else {
      print "$name has no type\n";
    }
  },
  dicom_ae => sub {
    my($disp, $fh, $out, $args) = @_;
    my $name = $args->[0];
    my $port = $args->[1];
    my $file = $args->[2];
    unless(-r $file){
      print "Can't read $file\n";
      return;
    }
    my $Dicom = Dispatch::Dicom::Acceptor->new(
      $port, $file, CreateObjAdder($name)
    );
    $Dicom->Add("reader");
    $objs->{$name} = $Dicom;
  },
  dicom_ae_client => sub {
    my($disp, $fh, $out, $args) = @_;
    my $name = $args->[0];
    my $host = $args->[1];
    my $port = $args->[2];
    my $file = $args->[3];
    unless(-r $file){
      print "Can't read $file\n";
      return;
    }
    my $Dicom = Dispatch::Dicom::Connection->connect($host, $port, $file);
    unless($Dicom) {
      print "unable to connect to $host:$port\n";
      return;
    }
    $objs->{$name} = $Dicom;
  },
  send_dicom => sub {
    my($disp, $fh, $out, $args) = @_;
    my $obj = $args->[0];
    my $file = $args->[1];
    SendDicomFile($obj, $file);
  },
  release => sub {
    my($disp, $fh, $out, $args) = @_;
    my $obj = $args->[0];
    unless(exists $objs->{$obj}) {
      print "$obj doesn't exist\n";
      return;
    }
    unless(ref($objs->{$obj}) eq "Dispatch::Dicom::Connection"){
      print "$obj isn't a dicom connection\n";
      return;
    }
    $objs->{$obj}->Release();
    delete $objs->{$obj};
  },
  make_file_list => sub{
    my($disp, $fh, $out, $args) = @_;
    my $name = $args->[0];
    my $dir = $args->[1];
    $objs->{$name} = [];
    Posda::Find::SearchDir($dir, CreateFileScanner($objs->{$name}));
  },
  make_better_file_list => sub{
    my($disp, $fh, $out, $args) = @_;
    my $name = $args->[0];
    my $dir = $args->[1];
    $objs->{$name} = [];
    Posda::Find::SearchDir($dir, CreateBetterFileScanner($objs->{$name}));
  },
  send_file_list_back => sub {
    my($disp, $fh, $out, $args) = @_;
    my $name = $args->[0];
    my $obj = $args->[1];
    unless(exists $objs->{$name} && ref($objs->{$name}) eq "ARRAY"){
      print "Object $name doesn't exist or isn't a file list\n";
      return;
    }
    unless(exists $objs->{$obj}) {
      print "$obj doesn't exist\n";
      return;
    }
    unless(ref($objs->{$obj}) eq "Dispatch::Dicom::Connection"){
      print "$obj isn't a dicom connection\n";
      return;
    }
    my $back = Dispatch::Select::Background->new(
      CreateFileSender($objs->{$name}, $obj)
    );
    delete $objs->{$name};
    $back->queue();
  },
  send_better_file_list_back => sub {
    my($disp, $fh, $out, $args) = @_;
    my $name = $args->[0];
    my $obj = $args->[1];
    unless(exists $objs->{$name} && ref($objs->{$name}) eq "ARRAY"){
      print "Object $name doesn't exist or isn't a file list\n";
      return;
    }
    unless(exists $objs->{$obj}) {
      print "$obj doesn't exist\n";
      return;
    }
    unless(ref($objs->{$obj}) eq "Dispatch::Dicom::Connection"){
      print "$obj isn't a dicom connection\n";
      return;
    }
    my $back = Dispatch::Select::Background->new(
      CreateBetterFileSender($objs->{$name}, $obj)
    );
    delete $objs->{$name};
    $back->queue();
  },
  send_file_back => sub {
    my($disp, $fh, $out, $args) = @_;
    my $name = $args->[0];
    my $obj = $args->[1];
    unless(exists $objs->{$name} && ref($objs->{$name}) eq "ARRAY"){
      print "Object $name doesn't exist or isn't a file list\n";
      return;
    }
    unless(exists $objs->{$obj}) {
      print "$obj doesn't exist\n";
      return;
    }
    unless(ref($objs->{$obj}) eq "Dispatch::Dicom::Connection"){
      print "$obj isn't a dicom connection\n";
      return;
    }
    for my $file (@{$objs->{$name}}){
      SendDicomFile($obj, $file);
    }
  },
  echo => sub{
    my($disp, $fh, $out, $args) = @_;
    my $obj = $args->[0];
    unless(exists $objs->{$obj}) {
      print "$obj doesn't exist\n";
      return;
    }
    unless(ref($objs->{$obj}) eq "Dispatch::Dicom::Connection"){
      print "$obj isn't a dicom connection\n";
      return;
    }
    $objs->{$obj}->Echo(CreateEchoResponse());
  },
  print_timings => sub {
    my($disp, $fh, $out, $args) = @_;
    my $obj = $args->[0];
    unless(ref($objs->{$obj}) eq "Dispatch::Dicom::Connection"){
      print "$obj is not a Dicom Connection\n";
      return;
    }
    $objs->{$obj}->{print_timings} = $args->[1];
  },
};
sub start{
  my $CommandHandler = 
    Dispatch::Command::Basic->new(\*STDIN, \*STDOUT, $Commands);
  $objs->{CommandHandler} = $CommandHandler->handler;
  $objs->{CommandHandler}->Add("reader");
  Dispatch::Select::Dispatch();
  print "Returned from Dispatch\n";
}
1;
