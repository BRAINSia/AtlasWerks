#!/usr/bin/perl -w
#$Source: /home/bbennett/pass/archive/Posda/include/Dispatch/Http.pm,v $
#$Date: 2008/08/28 21:41:07 $
#$Revision: 1.2 $

use strict;
use Dispatch::Select;
use Dispatch::Acceptor;
use File::Find;
my $dbg = sub {
  print @_;
};
#package Dispatch::Http;
{
  package Dispatch::Http::Connection;
  sub new {
    my($class, $socket, $handler) = @_;
    my $this = {
      socket => $socket,
      handler => $handler,
    };
    return bless $this, $class;
  }
  sub CreateResponseReader{
    my($this, $socket) = @_;
    my $output_queue = Dispatch::Queue->new(5, 2);
    $this->{output_queue} = $output_queue;
    $this->{output_queue}->CreateQueueEmptierEvent($socket);
  }
  sub CreateHeaderReader{
    my($this, $socket) = @_;
    my $buff = "";
    my $count = 0;
    my $foo = sub {
      my($disp, $sock) = @_;
      my $inp = sysread $sock, $buff, 1, $count;
      unless($inp == 1){ die "socket closed prematurely" }
      $count += 1;
      if($buff =~ /^(.*)\n$/s){
        chomp $buff;
        $buff =~ s/\r//g;
        if($buff eq ""){
          &{$this->{handler}}($this);
          $disp->Remove("reader");
          return;
        }
        push(@{$this->{header_lines}}, $buff);
        $buff = "";
        $count = 0;
      }
    };
    my $disp = Dispatch::Select::Socket->new($foo, $socket);
    $disp->Add("reader");
  }
  sub ParseIncomingHeader{
    my($http) = @_;
    for my $i (1 .. $#{$http->{header_lines}}){
      my $line = $http->{header_lines}->[$i];
      if($line =~ /^([^:]+):\s*(.*)\s*$/){
        my $key = $1;
        my $value = $2;
        $key =~ tr/A-Z\-/a-z_/;
        $http->{header}->{$key} = $value;
      }
    }
  }
  sub NotFound{
    my($http, $message, $uri) = @_;
      my $queue = $http->{output_queue};
      delete $http->{output_queue};
      $queue->queue("HTTP/1.0 404 Not found\n\n");
      $queue->queue("$message $uri not found");
      $queue->finish();
  }
  sub NotLoggedIn{
    my($http, $uri) = @_;
      my $queue = $http->{output_queue};
      delete $http->{output_queue};
      $queue->queue("HTTP/1.0 404 Not Found\n\n");
      $queue->queue("Object $uri not found");
      $queue->finish();
  }
  sub InternalError{
    my($http, $message) = @_;
      my $queue = $http->{output_queue};
      delete $http->{output_queue};
      $queue->queue("HTTP/1.0 500 Error\n\n");
      $queue->queue("Internal error: $message");
      $queue->finish();
  }
  sub ParseIncomingForm{
    my($http) = @_;
    my $length = $http->{header}->{content_length};
    my $buff;
    my $len_read = read $http->{socket}, $buff, $length;
    unless($len_read == $length){
      die "couldn't read form";
    }
    my @pairs = split(/&/, $buff);
    for my $p (@pairs){
      my($key, $val) = split(/=/, $p);
      $key =~ s/%(..)/pack("c",hex($1))/ge;
      $val =~ s/%(..)/pack("c",hex($1))/ge;
      if(defined $http->{form}->{$key}){
        unless(ref($http->{form}->{$key}) eq "ARRAY"){
          $http->{form}->{$key} = [ $http->{form}->{$key} ];
        }
        push(@{$http->{form}->{$key}}, $val);
      } else {
        $http->{form}->{$key} = $val;
      }
    }
  }
  sub HtmlHeader{
    my($http) = @_;
    unless(exists $http->{header_sent}){
      $http->{output_queue}->queue("HTTP/1.0 200 OK\n");
      $http->{output_queue}->queue("Content-type: text/html\n\n");
    }
    $http->{header_sent} = 1;
  }
  sub queue{
    my($http, $string) = @_;
    $http->{output_queue}->queue($string);
  }
  sub finish{
    my($http) = @_;
    if(
      defined $http->{output_queue} &&
      $http->{output_queue}->can("finish")
    ){
      $http->{output_queue}->finish();
    }
  }
  sub queuer{
    my($http) = @_;
    my $foo = sub {
      for my $string (@_){
        $http->queue($string);
      }
    };
    return $foo;
  }
  sub DESTROY {
    my($http) = @_;
    my $class = ref($this);
    if($ENV{POSDA_DEBUG}){
      print "Destroying $class\n";
    }
    if(
      defined $http->{output_queue} &&
      $http->{output_queue}->can("finish")
    ){
      $http->{output_queue}->finish();
    }
  }
}
{
  package Dispatch::HttpServer;
  use vars qw ( @ISA );
  @ISA = ( "Dispatch::Select::Socket" );
  sub new {
    my($class, $port, $handler) = @_;
    my $foo = sub {
      my($this, $socket) = @_;
      my $http = Dispatch::Http::Connection->new($socket, $handler);
      my $Response_reader = $http->CreateResponseReader($socket);
      my $header_reader = $http->CreateHeaderReader($socket);
    };
    my $serv = Dispatch::Acceptor->new($foo)->port_server($port);
    return bless $serv, $class;
  }
  sub DESTROY {
    my $class = ref($this);
    if($ENV{POSDA_DEBUG}){
      print "Destroying $class\n";
    }
  }
}
{
  package Dispatch::Http::App::Server;
  use vars qw ( @ISA $ExtToMime );
  $ExtToMime = {
    "doc" => "application/msword",
    "gif" => "image/gif",
    "htm" => "text/html",
    "html" => "text/html",
    "jpg" => "image/jpeg",
    "mpg" => "video/mpeg",
    "ppt" => "application/ppt",
    "qt" => "video/quicktime",
    "rtf" => "application/rtf",
    "tif" => "image/tiff",
    "txt" => "text/plain",
    "wav" => "audio/x-wav",
    "xls" => "application/msexcel",
    "js" => "application/x-javascript",
    "css" => "text/css",
    "ico" => "image/icon",
  };
  @ISA = ( "Dispatch::Select::Socket" );
  sub Serve{
    my($this, $port, $interval, $time_to_live) = @_;
    my $foo = sub {
      my($http) = @_;
      if($http->{header_lines}->[0] =~ /^(\S*)\s*(\S*)\s*(\S*)\s*$/){
        my $method = $1;
        my $uri = $2;
        my $prot = $3;
        my $path = "$this->{file_root}$uri";
        unless(-r $path){
          $this->AppDispatch($http, $method, $uri);
          return;
        }
        unless($method eq "GET"){
          die "Only GET supported for static content";
        }
        if(-d $path) {
          if($path =~ /\/$/){
            $path .= "index.html";
          } else {
            $path .= "/index.html";
          }
        }
        unless(-r $path){
          return $http->NotFound("No file:", $path);
        }
        my $fh = FileHandle->new("<$path");
        my $queue = $http->{output_queue};
        delete $http->{output_queue};
        my $content_type = "text/html";
        if($path =~ /\.([^\/\.]+)$/){
          my $ext = $1;
          if(exists $ExtToMime->{$ext}){
            $content_type = $ExtToMime->{$ext};
            $queue->queue("HTTP/1.0 200 OK\n");
            $queue->queue("Content-type: $content_type\n\n");
          }
        }
        $queue->CreateQueueFillerEvent($fh);
        $queue->post_output();
      } else {
        die "unable to parse first line";
      }
    };
    my $fie = sub {
      my($back) = @_;
      for my $id(keys %{$this->{Inventory}}){
        my $sess = $this->{Inventory}->{$id};
        if(
          exists($sess->{log_me_out}) ||
          time - $sess->{last_access} > $time_to_live
        ){
          $sess->TearDown();
          delete $this->{Inventory}->{$id};
        }
      }
      my $sess_count = scalar keys %{$this->{Inventory}};
      unless($this->{shutting_down} && $sess_count <= 0){
        return $back->timer($interval);
      }
    };
    my $disp = Dispatch::HttpServer->new($port, $foo);
    $this->{socket_server} = $disp;
    $disp->Add("reader");
    my $back = Dispatch::Select::Background->new($fie);
    $back->queue;
  }
  sub Remove{
    my($this) = @_;
    if(defined($this->{socket_server})){
      $this->{socket_server}->Remove();
      delete($this->{socket_server});
    }
    $this->{shutting_down} = 1;
  }
  sub new {
    my($class, $file_root, $app_root) = @_;
    my $this = {
      Inventory => {
      },
      shutting_down => 0,
      app_root => $app_root,
      file_root => $file_root,
    };
    return bless $this, $class;
  }
  sub GetSession {
    my($this, $session) = @_;
    return $this->{Inventory}->{$session};
  }
  sub NewSession {
    my($this) = @_;
    my $inst_id = int rand() * 10000;
    while(exists $this->{Inventory}->{$inst_id}){
      $inst_id = int rand() * 10000;
    }
    $this->{Inventory}->{$inst_id} = bless {
      session_id => $inst_id,
      last_access => time(),
      sess_state => {},
    }, "Dispatch::Http::Session";
    return $inst_id;
  }
  sub AppDispatch{
    my($this, $http, $method, $uri) = @_;
    $http->ParseIncomingHeader();
    my $app_root = $this->{app_root};
    $http->{method} = $method;
    my $q_string = "";
    if($uri =~ /^(.*)\?(.*)$/){
      $uri = $1;
      $q_string = $2;
    }
    $http->{uri} = $uri;
    $http->{q_string} = $q_string;
    if($uri =~ /^\/login/){
      return $this->Login($http, $uri);
    }
    unless($uri =~ /^\/([^\/]+)\/(.*)$/){
      return $http->NotFound("Can't find session_id in uri:", $uri);
    }
    my $sess_id = $1;
    my $op = $2;
    unless(defined $this->{Inventory}->{$sess_id}){
      return $http->NotLoggedIn($op);
    }
    my $sess = $this->{Inventory}->{$sess_id};
    if(exists($sess->{log_me_out})){
      $sess->TearDown();
      delete $this->{Inventory}->{$sess_id};
      return $http->NotFound("Can't find session_id in uri:", $uri);
    }
    $sess->{last_access} = time;
    unless(exists $app_root->{app}->{$op}){
      return $http->NotFound("Can't find op in app ($sess_id, $op):", $uri);
    }
    $app_root->Dispatch($http, $sess, $op, {});
  }
  sub Login{
    my($this, $http, $uri) = @_;
    my $app_root = $this->{app_root};
    unless($uri =~ /^\/login\/([^\/]+)$/){
      return $http->NotFound($uri);
    }
    my $app_name = "$1";
    unless(
      exists($app_root->{login}->{$app_name}) &&
      ref($app_root->{login}->{$app_name}) eq "CODE"
    ){
      return $http->InternalError($http, "Login to unknown app: $app_name");
    }
    &{$app_root->{login}->{$app_name}}($this, $http, $app_name);
  }
  sub DESTROY{
    my $class = ref($this);
    if($ENV{POSDA_DEBUG}){
      print "Destroying $class\n";
    }
  }
}
{
  package Dispatch::Http::Session;
  sub TearDown{
    my($sess) = @_;
    print "Tearing down session $sess->{session_id}\n";
  }
  sub logout{
    my($sess) = @_;
    $sess->{log_me_out} = 1;
  }
  sub get_state{
    my($sess) = @_;
    return $sess->{sess_state};
  }
  sub put_state{
    my($sess, $state) = @_;
    $sess->{sess_state} = $state;
  }
}
{
  package Dispatch::Http::App;
  sub new{
    my($class, $root) = @_;
    my %hash;
    my $process = sub {
      my $full_path = $File::Find::name;
      unless(-f $full_path) { return }
      unless($full_path =~ /^$root\/(.*)$/){
       print STDERR "funny path: $full_path\n";
      }
      my $rel_path = $1;
      my $type;
      my $sub_type;
      if($rel_path =~ /login\/(.*)/ ){
        $rel_path = $1;
        $type = "login";
      } elsif($rel_path =~ /app\/(.*)/){
        $rel_path = $1;
        $type = "app";
      } else {
        print STDERR "unparsable path: $rel_path\n";
        return;
      }
      my $name;
      if($rel_path =~ /(.*)\.p$/){
        $name = $1;
        $sub_type = "code";
      } elsif ($rel_path =~ /(.*)\.t/){
        $name = $1;
        $sub_type = "template";
      } else {
        print STDERR "unknown file suffix: $rel_path";
        return;
      }
      if($sub_type eq "code"){
        my @lines;
        my $code_ref;
        push (@lines, "\$code_ref = sub {");
        if($type eq "login"){
          push (@lines, "my(\$APP, \$HTTP, \$NAME) = \@_;");
        } else {
          push (@lines, "my(\$DISP, \$HTTP, \$SESS, \$DYNPARM) = \@_;");
        }
        open FILE, "<$root/$type/$rel_path";
        while (my $line = <FILE>){
          chomp $line;
          push(@lines, $line);
        }
        close FILE;
        push(@lines, "};");
        my $code = join "\n", @lines;
        #print "code:\n$code\n\n";
        eval $code;
        if($@){
          print STDERR "Error $@ in:\n$code\n";
        } else {
          #print "hash{$type}->{$rel_path} = $code_ref\n";
          $hash{$type}->{$name} = $code_ref;
        }
      } elsif($sub_type eq "template") {
        my @lines;
        open FILE, "<$root/$type/$rel_path";
        while (my $line = <FILE>){
          chomp $line;
          push(@lines, $line);
        }
        close FILE;
        my $temp = join "\n", @lines;
        $hash{$type}->{$name} = $temp;
      }
    };
    File::Find::find($process, $root);
    return bless \%hash, $class;
  }
  sub Dispatch{
    my($app_root, $http, $sess, $op, $dyn) = @_;
print "Dispatch of $op\n";
    if(ref($app_root->{app}->{$op}) eq "CODE"){
      return &{$app_root->{app}->{$op}}($app_root, $http, $sess, $dyn);
    } else {
      $http->HtmlHeader();
      return Dispatch::Template::ExpandText($http, $app_root->{app}->{$op},
        $app_root->{app}, $sess, $dyn);
    }
  }
}
1;
