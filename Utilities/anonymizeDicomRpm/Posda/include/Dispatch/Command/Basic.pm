#!/usr/bin/perl -w
#$Source: /home/bbennett/pass/archive/Posda/include/Dispatch/Command/Basic.pm,v $
#$Date: 2008/09/03 17:19:50 $
#$Revision: 1.7 $

use strict;
package Dispatch::Command::Basic;

sub CompileCommand{
  my($cmd) = @_;
  my $compile_str = "\$code = sub { $cmd };";
  my $code;
  eval $compile_str ;
  if($@){
    print STDERR "Error $@ encountered compiling:\n $compile_str\n";
    return undef;
  } else {
    return $code;
  }
}
sub IsBlessed{
  my($obj) = @_;
  my $ref = ref($obj);
  if($ref && $ref ne "ARRAY" && $ref ne "HASH" && $ref ne "SCALAR"){
    return 1;
  }
  return 0;
}

sub delete_obj{
  my($this, $name) = @_;
  if(exists $this->{objects}->{$name}){
    my $obj = $this->{objects}->{$name};
    print "Deleteing Object: $name\n";
    delete $this->{objects}->{$name};
    if(IsBlessed($obj)){
      if($obj->can("Release")){
        print "Releasing $obj\n";
        $obj->Release();
      }
      if($obj->can("Remove")){
        print "Removing $obj\n";
        $obj->Remove();
      }
      if($obj->can("Finish")){
        print "Finishing $obj\n";
        $obj->Finish();
      }
    } else {
      print "Deleting non-blessed object: $name\n";
    }
  }
}

sub handler {
  my($this) = @_;
  my $line = "";
  my $foo = sub {
    my($disp, $socket) = @_;
    my $len = sysread($socket, $line, 1, length($line));
    if($len < 1){
      $disp->Remove();
      print "disp: $disp\n";
      obj:
      for my $i (keys %{$this->{objects}}){
        if(defined $this->{objects}->{$i}){
          print "$i: $this->{objects}->{$i}\n";
        } else {
          print "$i is deleted already\n";
          next obj;
        }
        if($disp eq $this->{objects}->{$i}){
          print "$i disconnected\n";
          my $cmd_hand = $this->{objects}->{$i};
          if($i =~ /^CmdHandler(_.*)$/){
            my $hnd = "Cmd$1";
            $this->delete_obj($i);
            $this->delete_obj($hnd);
            last obj;
          }
        }
      }
    }
    unless($line =~ /\n$/s){ return };
    chomp $line;
    $line =~ s/\r//;
    my @list = split(/\s+/, $line);
    my $cmd = shift(@list);
    unless($cmd) { return }
    if(exists($this->{commands}->{$cmd})){
      &{$this->{commands}->{$cmd}}($this, $socket, $this->{out}, \@list);
    } else {
      my $out = $this->{out};
      if(defined $out){
        print $out "unknown command: $cmd\n";
      } else {
        print "unknown command (and Cmd handler finished): $cmd\n";
      }
    }
    $line = "";
  };
  return Dispatch::Select::Socket->new($foo, $this->{in});
}

my $command_debug = <<'EOF';
my($this, $fh, $out, $args) = @_;
print $out "___________________\n";
print $out "Dispatcher:\n";
Dispatch::Select::Dump($out);
print $out "___________________\n";
EOF

my $command_dump_obj = <<'EOF';
my($this, $fh, $out, $args) = @_;
my $name = $args->[0];
my $depth = $args->[1];
unless($depth) { $depth = 3 }
unless(exists $this->{objects}->{$name}){
  print $out "No object named $name exists\n";
  return;
}
my $ref = ref($this->{objects}->{$name});
unless($ref){
  print $out "Object named $name has no type\n";
  return;
}
do "Debug.pm";
print $out "Object ($ref) $name: ";
my $dbg = sub {print $out @_};
Debug::GenPrint($dbg, $this->{objects}->{$name}, 1, $depth);
print $out "\n";
EOF

my $command_exit = <<'EOF';
my($this, $fh, $out, $args) = @_;
obj:
for my $i (keys %{$this->{objects}}){
  unless(exists $this->{objects}->{$i}){
    print "$i already deleted\n";
    next obj;
  }
  if($this->{objects}->{$i} eq $this){
    print $out "Exit from $i\n";
    if($i =~ /^Cmd(_.*)$/){
      my $cmd_hand_name = "CmdHandler$1";
      $this->delete_obj($cmd_hand_name);
      $this->delete_obj($i);
      last obj;
    }
  }
}
EOF

my $command_shutdown = <<'EOF';
my($this, $fh, $out, $args) = @_;
my $mykey;
obj:
for my $i (keys %{$this->{objects}}){
  if($this eq $this->{objects}->{$i}){
    $mykey = $i;
    next obj;
  }
  print "In shutdown, deleting $i\n";
  $this->delete_obj($i);
}
print "In shutdown deleting $mykey\n";
$this->delete_obj($mykey);
EOF

my $command_delete_obj = <<'EOF';
my($this, $fh, $out, $args) = @_;
print $out "in delete_obj\n";
my $obj_name = $args->[0];
if(exists $this->{objects}->{$args->[0]}){
  print $out "deleting obj $args->[0]\n";
  my $obj_name = $args->[0];
  $this->delete_obj($obj_name);
}
print $out "leaving delete_obj\n";
EOF

my $command_list = <<'EOF';
my($this, $fh, $out, $args) = @_;
print $out "___________________\n";
print $out "Objects:\n";
for my $obj_name (sort keys %{$this->{objects}}){
  my $ref = ref $this->{objects}->{$obj_name};
  print $out "\t$obj_name: $ref\n";
}   
print $out "___________________\n";
EOF

my $default_command_txt = {
  dump_obj => $command_dump_obj,
  dispatch_debug => $command_debug,
  delete_obj => $command_delete_obj,
  exit => $command_exit,
  shutdown => $command_shutdown,
  list => $command_list,
};

sub new {
  my($class, $in, $out, $commands, $objects) = @_;
  unless (defined $objects){
    $objects = Dispatch::Command::Basic::OList->new();
  }
  my $this = {
    in => $in,
    out => $out,
    objects => $objects,
  };
  for my $cmd (keys %$default_command_txt){
    $this->{command_text}->{$cmd} = $default_command_txt->{$cmd};
    my $code = CompileCommand($this->{command_text}->{$cmd});
    if(defined $code){
      $this->{commands}->{$cmd} = $code;
    }
  }
  for my $i (keys %$commands){
    if(exists $this->{commands}->{$i}){
      print $out "Overriding command $i\n";
    }
    $this->{commands}->{$i} = $commands->{$i};
  }
  return bless $this, $class;
}
sub Finish{
  my($this) = @_;
  print "Finishing up a $this\n";
  delete $this->{objects};
  delete $this->{in};
  delete $this->{out};
}
sub new_from_dir{
  my($class, $in, $out, $dir, $objects) = @_;
  unless (defined $objects){
    $objects = Dispatch::Command::Basic::OList->new();
  }
  my $this = {
    in => $in,
    out => $out,
    objects => $objects,
  };
  for my $cmd (keys %$default_command_txt){
    $this->{command_text}->{$cmd} = $default_command_txt->{$cmd};
    my $compile_str = "\$code = sub { $this->{command_text}->{$cmd} };";
    my $code = CompileCommand($this->{command_text}->{$cmd});

    if(defined $code){
      $this->{commands}->{$cmd} = $code;
    }
  }
  unless(-d $dir) { die "$dir is not a directory" }
  opendir DIR, $dir;
  file:
  while (my $file = readdir DIR){
    unless($file =~ /^(.*)\.pc$/) {next file}
    my $command = $1;
    my $file_name = "$dir/$file";
    unless(-r $file_name) { next file}
    open FILE, "<$file_name";
    my $txt = "";
    while (my $line = <FILE>){
      $txt .= $line;
    }
    close FILE;
    my $code;
    my $compile_str = "\$code = sub { $txt };";
    eval $compile_str ;
    if($@){
      print STDERR "Error $@ encountered compiling:\n $compile_str\n";
    } else {
      if(exists $this->{command_text}->{$command}){
        print $out "overriding command $command\n";
      }
      $this->{command_text}->{$command} = $txt;
      $this->{commands}->{$command} = $code;
    }
  }
  closedir DIR;
  return bless $this, $class;
}
#sub queue{
#  my($this, $string) = @_;
#  $this->{output_queue}->queue($string);
#}
sub DESTROY{
  my($this) = @_;
  my $class = ref($this);
  if($ENV{POSDA_DEBUG}){
    print "Destroying $class\n";
  }
}
{
  package Dispatch::Command::Basic::Acceptor;
  use vars qw ( @ISA );
  @ISA = ( "Dispatch::Select::Socket" );
  sub new{
    my($class, $port, $dir, $objects) = @_;
    my $seq = 1;
    my $foo = sub {
      my($obj, $socket) = @_;
      my $cmd = 
        Dispatch::Command::Basic->new_from_dir(
          $socket, $socket,  $dir, $objects);
      
      $objects->{"Cmd_$seq"} = $cmd;
      my $handler = $cmd->handler();
      $objects->{"CmdHandler_$seq"} = $handler;
      $handler->Add("reader");
      $seq += 1;
    };
    my $serv = Dispatch::Acceptor->new($foo)->port_server($port);
    $serv->Add("reader");
    return bless $serv, $class;
  }
  sub DESTROY{
    my($this) = @_;
    my $class = ref($this);
    if($ENV{POSDA_DEBUG}){
      print "Destroying $class\n";
    }
  }
}
1;
