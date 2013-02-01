#!/usr/bin/perl -w
#$Source: /home/bbennett/pass/archive/Posda/include/Dispatch/Dicom.pm,v $
#$Date: 2008/08/28 21:41:07 $
#$Revision: 1.6 $

use strict;
use Dispatch::Select;
use Dispatch::Acceptor;
use Dispatch::Dicom::Assoc;
use File::Find;
use Posda::Command;
use HexDump;
{
  package Dispatch::Dicom::PdataAssembler;
  sub new {
    my($class, $len) = @_;
    my $this = {
      pdu_length => $len,
      pdu_remaining => $len,
    };
    return bless $this, $class;
  }
  sub crank {
    my($this, $dcm_conn) = @_;
    unless(defined $this->{pdv_header}){
      unless(length($dcm_conn->{buff}) eq $dcm_conn->{to_read}){
        return;
      }
      $this->{pdu_remaining} -= 6;
      my $len = length($dcm_conn->{buff});
      my($pdv_len, $pc_id, $flags) =
        unpack("NCC", $dcm_conn->{buff});
      my $cmd = $flags & 1;
      my $last = $flags & 2;
      $this->{pdv_header} = {
         cmd => $cmd,
         pc_id => $pc_id,
         last  => $last,
         len => $pdv_len,
      };
      $dcm_conn->{to_read} = $pdv_len - 2;
      $dcm_conn->{buff} = "";
      unless(exists $dcm_conn->{message_being_received}){
        unless($this->{pdv_header}->{cmd}){
          return($this->Abort("ds pdv with no command"));
        }
        $dcm_conn->{message_being_received} =
          Dispatch::Dicom::Message->new(
            $this->{pdv_header}->{pc_id}, $dcm_conn);
      }
      return;
    }
    my $length_read = length($dcm_conn->{buff});
    $this->{pdu_remaining} -= $length_read;
    $dcm_conn->{to_read} -= $length_read;
    if(exists $dcm_conn->{message_being_received}){
      if($this->{pdv_header}->{cmd}){
        $dcm_conn->{message_being_received}->command_data(
          $this->{pdv_header}->{pc_id}, $dcm_conn->{buff});
        if($dcm_conn->{to_read} == 0 && $this->{pdv_header}->{last}){ 
          $dcm_conn->{message_being_received}->finalize_command($dcm_conn);
          unless($dcm_conn->{message_being_received}->has_dataset){
            delete $dcm_conn->{message_being_received};
          }
        }
      } else {
        $dcm_conn->{message_being_received}->ds_data(
          $this->{pdv_header}->{pc_id}, $dcm_conn->{buff});
        if($dcm_conn->{to_read} == 0 && $this->{pdv_header}->{last}){ 
          $dcm_conn->{message_being_received}->finalize_ds($dcm_conn);
          delete $dcm_conn->{message_being_received};
        }
      }
      $dcm_conn->{buff} = "";
      if($dcm_conn->{to_read} == 0){
        delete $this->{pdv_header};
        if($this->{pdu_remaining} == 0){
          delete $dcm_conn->{pdata_assembler};
          delete $dcm_conn->{pdu_type};
        }
      }
    }
    $dcm_conn->{buff} = "";
    return;
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
  package Dispatch::Dicom::Connection;
  sub new_accept {
    my($class, $socket, $descrip) = @_;
    my $this = {
      socket => $socket,
      state => "STA2",
      descrip => $descrip,
      incoming_message_handler => $descrip->{incoming_message_handler},
      storage_root => $descrip->{storage_root},
      message_queue => [],
      response_queue => [],
      type => "acceptor",
      mess_seq => 1,
    };
    return bless $this, $class;
  }
  sub connect {
    my($class, $host, $port, $file, $callback, $overrides) = @_;
    my $socket = IO::Socket::INET->new(
      PeerAddr => $host,
      PeerPort => $port,
      Proto => 'tcp',
    ) or return undef;
    my $this = {
      socket => $socket,
      state => "STA5",
      message_queue => [],
      response_queue => [],
      type => "initiator",
      mess_seq => 1,
    };
    if(defined $callback && ref($callback) eq "CODE"){
      $this->{connection_callback} = $callback;
    }
    my $a_assoc_rq;
    if($overrides && ref($overrides) eq "HASH"){
      $a_assoc_rq = Dispatch::Dicom::AssocRq->new_from_file(
        $file, $this, $overrides
      );
    } else {
      $a_assoc_rq = Dispatch::Dicom::AssocRq->new_from_file($file, $this);
    }
    $this->{assoc_rq} = $a_assoc_rq;
    bless $this, $class;
    $this->CreatePduReaderSta5($socket);
    $this->CreateOutputQueue($this->{socket});
    $this->{output_queue}->queue($this->{assoc_rq}->encode());
    return $this;
  }
  sub SetDisconnectCallback{
    my($this, $callback) = @_;
    $this->{disconnect_callback} = $callback;
  }
  sub SetDatasetReceivedCallback{
    my($this, $handler) = @_;
    $this->{file_handler} = $handler;
  }
  sub SetStorageRoot{
    my($this, $dir) = @_;
    $this->{storage_root} = $dir;
  }
  sub Release{
    my($this) = @_;
    $this->{ReleaseRequested} = 1;
    if(exists $this->{waiting_for_message}){
      $this->{waiting_for_message}->post_and_clear();
      delete $this->{waiting_for_message};
    }
  }
  sub Echo{
    my($this, $resp) = @_;
    my $xfr_stx = '1.2.840.10008.1.2';
    my $echo = Posda::Command->new_verif_command();
    unless(exists $this->{sopcl}->{'1.2.840.10008.1.1'}->{$xfr_stx}){
      return($this->Abort("no presentation context for echo: $xfr_stx"));
    }
    my $pc_id = $this->{sopcl}->{'1.2.840.10008.1.1'}->{$xfr_stx};
    my $ma = Dispatch::Dicom::MessageAssembler->new(
      $pc_id, $echo, undef, $resp);
    $this->QueueMessage($ma);
  }
  sub SetUpPresContexts{
    my($this) = @_;
    my $rq = $this->{assoc_rq};
    my $ac = $this->{assoc_ac};
    my %PresContexts;
    my %SopClasses;
    for my $pc_id (keys %{$ac->{presentation_contexts}}){
      my $xfr_stx = $ac->{presentation_contexts}->{$pc_id};
      my $sopcl = $rq->{presentation_contexts}->{$pc_id}->{abstract_syntax};
      if($xfr_stx){
        $PresContexts{$pc_id}->{abs_stx} = $sopcl;
        $PresContexts{$pc_id}->{xfr_stx} = $xfr_stx;
        $PresContexts{$pc_id}->{accepted} = 1;
        $SopClasses{$sopcl}->{$xfr_stx} = $pc_id;
      } else {
        $PresContexts{$pc_id}->{accepted} = 0;
      }
    }
    $this->{pres_cntx} = \%PresContexts;
    $this->{sopcl} = \%SopClasses;
    if($this->{type} eq "initiator"){
      $this->{max_length} = $ac->{max_length};
      $this->{rcv_max} = $rq->{max_length};
    } else {
      $this->{max_length} = $rq->{max_length};
      $this->{rcv_max} = $ac->{max_length};
    }
    if(defined($ac->{max_i})){
      $this->{max_outstanding} = $ac->{max_i};
    } else {
      $this->{max_outstanding} = 1;
    }
    $this->{outstanding} = 0;
  }
  sub CreatePduReaderSta5{
    my($this, $socket) = @_;
    my $foo = sub {
      my($disp, $sock) = @_;
      unless(defined $this->{to_read}){ 
        $this->{to_read} = 6;
      }
      unless(defined $this->{buff}){$this->{buff} = "" }
      my $to_read = $this->{to_read} - length($this->{buff});
      my $offset = length $this->{buff};
      my $inp = sysread $sock, $this->{buff}, $to_read, $offset;
      unless(defined $inp) {
        $disp->Remove();
        return($this->Close("error ($!) reading socket Sta5"));
      }
      if($inp == 0) {
        $disp->Remove();
        return($this->Close("read 0 bytes in Sta5"));
      }
      unless($inp == $to_read){ return }
      unless(defined $this->{pdu_size}){
        ### here we have read a pdu header
        my($pdu_type, $uk, $pdu_length) = unpack("CCN", $this->{buff});
        if($pdu_type == 3){
          print STDERR "this guy actually rejected my assoc_rq\n";
          return($this->Abort("Association rejected"));
        }
        unless($pdu_type == 2){
          $disp->Remove("reader");
        }
        $this->{pdu_size} = $pdu_length + 6;
        $this->{to_read} = $pdu_length;
        $this->{buff} = "";
        return;
      }
      $this->{assoc_ac} = Dispatch::Dicom::AssocAc->new_from_pdu(
        $this->{buff}
      );
      $disp->Remove("reader");
      $this->SetUpPresContexts();
      $this->CreateMessageQueueEmptier();
      $this->CreatePduReaderSta6($sock);
      if(defined $this->{connection_callback}){
        &{$this->{connection_callback}}($this);
      }
      delete $this->{buff};
    };
    my $disp = Dispatch::Select::Socket->new($foo, $socket);
    $disp->Add("reader");
  }
  sub queue {
    my($this, $string) = @_;
    $this->{output_queue}->queue($string);
  }
  sub QueueResponse {
    my($this, $message) = @_;
    if(exists $this->{waiting_for_message}){
      if($this->{print_timings}){
        my $elapsed = Time::HiRes::tv_interval(
          $this->{start_waiting_for_message}, 
          [Time::HiRes::gettimeofday]
        );
        print "Queued a response after a wait of $elapsed seconds\n";
      }
      $this->{waiting_for_message}->post_and_clear();
      delete $this->{waiting_for_message};
    }
    $message->{cmd_data} = $message->{cmd}->render();
    push(@{$this->{response_queue}}, $message);
  }
  sub QueueMessage {
    my($this, $message) = @_;
    if(exists $this->{waiting_for_message}){
      if($this->{print_timings}){
        my $elapsed = Time::HiRes::tv_interval(
          $this->{start_waiting_for_message}, 
          [Time::HiRes::gettimeofday]
        );
        print "Queued a message after a wait of $elapsed seconds\n";
      }
      $this->{waiting_for_message}->post_and_clear();
      delete $this->{waiting_for_message};
    }
    $message->{cmd}->{"(0000,0110)"} = $this->{mess_seq};
    $this->{mess_seq} += 1;
    $message->{cmd_data} = $message->{cmd}->render();
    push(@{$this->{message_queue}}, $message);
  }
  sub CreateMessageTransmissionEndEvent{
    my($this) = @_;
    my $foo = sub {
      my($back) = @_;
      delete $this->{outgoing_message};
      $this->CreateMessageQueueEmptier();
    };
    return Dispatch::Select::Event->new(
      Dispatch::Select::Background->new($foo)
    );
  }
  sub DecrementOutstanding{
    my($this) = @_;
    $this->{outstanding} -= 1;
    if(
      exists($this->{waiting_for_message}) && 
      $#{$this->{message_queue}} >= 0
    ){
      if($this->{print_timings}){
        my $elapsed = Time::HiRes::tv_interval(
          $this->{start_waiting_for_message}, 
          [Time::HiRes::gettimeofday]
        );
        print "Re-opened message queue after a wait of $elapsed seconds\n";
      }
      $this->{waiting_for_message}->post_and_clear();
      delete $this->{waiting_for_message};
    }
  }
  sub WaitMessageQueue{
    my($this, $count, $back) = @_;
    if($count < (scalar @{$this->{message_queue}})){
      $back->queue();
      return;
    }
    $this->{WaitingForMessageQueue} = $back;
  }
  sub CreateMessageQueueEmptier {
    my($this) = @_;
    my $foo = sub {
      my($back) = @_;
      if(exists $this->{Abort}){
        my $rq = Dispatch::Dicom::Abort->new(2, 0);
        $this->queue($rq->encode());
        $this->CreatePduReaderSta13($this->{socket});
        return;
      } elsif($#{$this->{response_queue}} >= 0){
        $this->{outgoing_message} = shift(@{$this->{response_queue}});
      } elsif(
        $#{$this->{message_queue}} >= 0 &&
        (
          $this->{max_outstanding} == 0 ||
          $this->{outstanding} < $this->{max_outstanding}
        )
      ){
        my $msg = shift(@{$this->{message_queue}});
        if(exists $this->{WaitingForMessageQueue}){
          $this->{WaitingForMessageQueue}->queue();
          delete $this->{WaitingForMessageQueue};
        }
        $this->{outgoing_message} = $msg;
        $this->{pending_messages}->{$msg->msg_id} = $msg;
        $this->{outstanding} += 1;
        if(exists($msg->{ds})){ $msg->{ds}->start }
      } elsif($this->{ReleaseRequested}) {
        my $rq = Dispatch::Dicom::ReleaseRq->new($this->{buff});
        $this->queue($rq->encode());
        $this->CreatePduReaderSta8($this->{socket});
        return;
      } else {
        if($this->{print_timings}){
          $this->{start_waiting_for_message} = [Time::HiRes::gettimeofday];
        }
        $this->{waiting_for_message} = Dispatch::Select::Event->new($back);
        return;
      }
      $this->{outgoing_message}->CreatePduAssembler($this->{output_queue},
        $this->{max_length}, $this->CreateMessageTransmissionEndEvent());
    };
    my $back = Dispatch::Select::Background->new($foo);
    $back->queue();
  }
  sub CreatePduReaderSta6{
    my($this, $socket) = @_;
    my $foo = sub {
      my($disp, $sock) = @_;
      unless(defined $this->{pdu_type} || $this->{to_read} == 6){
        $this->{to_read} = 6;
        $this->{buff} = "";
      }
      my $to_read = $this->{to_read} - length($this->{buff});
      my $offset = length $this->{buff};
      if($to_read > 0){
        my $inp = sysread $sock, $this->{buff}, $to_read, $offset;
        unless(defined $inp) {
          $disp->Remove();
          return($this->Close("error ($!) reading socket Sta6"));
        }
        if($inp == 0) { 
          $disp->Remove();
          return($this->Close("read 0 bytes in Sta6"));
        }
        unless($inp == $to_read){ return }
      }
      unless(defined $this->{pdu_type}){
        my($pdu_type, $uk, $pdu_length) = unpack("CCN", $this->{buff});
        $this->{pdu_type} = $pdu_type;
        $this->{buff} = "";
        $this->{pdu_length} = $pdu_length;
      }
      if(defined $this->{pdata_assembler}){
        $this->{pdata_assembler}->crank($this);
        return;
      }
      if($this->{finishing_release}){
        my $resp = Dispatch::Dicom::ReleaseRp->new();
        $this->queue($resp->encode());
        $disp->Remove("reader");
        $this->CreatePduReaderSta13($socket);
      }
      my $pdu_type = $this->{pdu_type};
      if($pdu_type == 1){      # Assoc-RQ
        $disp->Remove("reader");
        $this->Abort("Invalid Pdu (Assoc-RQ) in Sta6");
      } elsif($pdu_type == 2){ # Assoc-AC
        $disp->Remove("reader");
        $this->Abort("Invalid Pdu (Assoc-AC) in Sta6");
      } elsif($pdu_type == 3){ # Assoc-RJ
        $disp->Remove("reader");
        $this->Abort("Invalid Pdu (Assoc-RJ) in Sta6");
      } elsif($pdu_type == 4){ # Data-TF
        $this->{pdata_assembler} = 
          Dispatch::Dicom::PdataAssembler->new($this->{pdu_length});
        $this->{to_read} = 6;
        $this->{buff} = "";
      } elsif($pdu_type == 5){ # Release-RQ
        $this->{finishing_release} = 1;
        $this->{to_read} = 4;
      } elsif($pdu_type == 6){ # Release-RP
        $disp->Remove("reader");
        $this->Abort("Invalid Pdu (Release-RP) in Sta6");
      } elsif($pdu_type == 7){ # Abort
        $disp->Remove();
        $this->Abort("Abort Request Received");
      }
    };
    my $disp = Dispatch::Select::Socket->new($foo, $socket);
    $disp->Add("reader");
  }
  sub Close{
    my($this, $mess) = @_;
    if($mess){
      print STDERR "Closing Socket abnormally: $mess\n";
    }
    close($this->{socket});
    delete $this->{socket};
    if($this->{waiting_for_message}){
      delete $this->{waiting_for_message};
    }
    if(
      exists($this->{disconnect_callback}) && 
      ref($this->{disconnect_callback}) eq "CODE"
    ){
      &{$this->{disconnect_callback}}($this);
    }
  }
  sub Abort{
    my($this, $mess) = @_;
    $this->{Abort} = {
      mess => $mess,
    };
    if(exists $this->{waiting_for_message}){
      $this->{waiting_for_message}->post_and_clear();
      delete $this->{waiting_for_message};
    } elsif(exists $this->{outgoing_message}){
      $this->{outgoing_message}->Abort();
    }
  }
  sub CreatePduReaderSta2{
    my($this, $socket) = @_;
    my $foo = sub {
      my($disp, $sock) = @_;
      unless(defined $this->{to_read}){ 
        $this->{to_read} = 6;
      }
      unless(defined $this->{buff}){$this->{buff} = "" }
      my $to_read = $this->{to_read} - length($this->{buff});
      my $offset = length $this->{buff};
      my $inp = sysread $sock, $this->{buff}, $to_read, $offset;
      unless(defined $inp) {
        $disp->Remove();
        return($this->Close("error ($!) reading socket (Sta2)"));
      }
      if($inp == 0) {
        $disp->Remove();
        return($this->Close("read 0 bytes (Sta2)"));
      }
      unless($inp == $to_read){ return }
      unless(defined $this->{pdu_size}){
        ### here we have read a pdu header
        my($pdu_type, $uk, $pdu_length) = unpack("CCN", $this->{buff});
        unless($pdu_type == 1){
          $disp->Remove("reader");
          $this->Abort("invalid type pdu: $pdu_type in Sta2");
        }
        $this->{pdu_size} = $pdu_length + 6;
        $this->{to_read} = $pdu_length;
        $this->{buff} = "";
        return;
      }
      $this->{assoc_rq} = Dispatch::Dicom::AssocRq->new_from_pdu(
        $this->{buff}
      );
       my $resp = Dispatch::Dicom::AssocAc->new_from_rq_desc(
         $this->{assoc_rq}, $this->{descrip}
      );
      $this->{assoc_ac} = $resp;
      $this->queue($resp->encode());
      $disp->Remove("reader");
      delete $this->{pdu_type};
      delete $this->{cammand};
      delete $this->{buff};
      $this->CreatePduReaderSta6($sock);
      $this->CreateMessageQueueEmptier();
      $this->SetUpPresContexts();
      if(defined($this->{connection_callback})){
        &{$this->{connection_callback}}($this);
      }
    };
    my $disp = Dispatch::Select::Socket->new($foo, $socket);
    $disp->Add("reader");
  }
  sub CreatePduReaderSta8{
    my($this, $socket) = @_;
    my $foo = sub {
      my($disp, $sock) = @_;
      unless(defined $this->{to_read}){ 
        $this->{to_read} = 6;
      }
      unless(defined $this->{buff}){$this->{buff} = "" }
      my $to_read = $this->{to_read} - length($this->{buff});
      my $offset = length $this->{buff};
      my $inp = sysread $sock, $this->{buff}, $to_read, $offset;
      unless(defined $inp) {
         return($this->Close("error ($!) reading socket Sta8"))
      }
      if($inp == 0) {
        return($this->Close("read 0 bytes Sta8"));
      }
      unless($inp == $to_read){ return }
      unless(defined $this->{pdu_size}){
        ### here we have read a pdu header
        my($pdu_type, $uk, $pdu_length) = unpack("CCN", $this->{buff});
        $this->{pdu_type} = $pdu_type;
        $this->{buff} = "";
        $this->{pdu_length} = $pdu_length;
        $this->{to_read} = $pdu_length;
        unless($pdu_type == 6){
          $disp->Remove();
          return($this->Abort(
            "Invalid pdu ($pdu_type) when waiting for ReleaseRQ"
          ));
        }
        $this->{pdu_size} = 4;
        $this->{to_read} = 4;
        return();
      }
      # here we just read a release rsp (in response to release rq)
      $disp->Remove();
      if($this->{ReleaseRequested}){
        $this->Close();
      } else {;
        $this->Close("how did we get here? Sta8");
      }
    };
    delete $this->{to_read};
    delete $this->{buff};
    delete $this->{pdu_size};
    my $disp = Dispatch::Select::Socket->new($foo, $socket);
    $disp->Add("reader");
  }
  sub CreatePduReaderSta13{
    my($this, $socket) = @_;
    my $foo = sub {
      my($disp, $sock) = @_;
      my $buff;
      my $inp = sysread $sock, $buff, 1024;
      unless(defined $inp) { 
        $disp->Remove();
        return($this->Close("Closed socket on read (Sta13)"));
      }
      if($inp == 0) {
        # Normal place to close after sending release
        $disp->Remove();
        return($this->Close());
      } else {
        print STDERR "Extra stuff in Sta13:\n";
        HexDump::PrintVax(\*STDERR, $buff, 0);
      }
    };
    delete $this->{to_read};
    delete $this->{buff};
    delete $this->{pdu_size};
    my $disp = Dispatch::Select::Socket->new($foo, $socket);
    $disp->Add("reader");
  }
  sub CreateOutputQueue{
    my($this, $socket) = @_;
    $this->{output_queue} = Dispatch::Queue->new(5, 2);
    $this->{output_queue}->CreateQueueEmptierEvent($socket);
  }
  sub DESTROY {
    my($this) = @_;
    my $class = ref($this);
    if($ENV{POSDA_DEBUG}){
      print "Destroying $class\n";
    }
    my($dicom) = $this;
    if(
      exists($dicom->{output_queue}) &&
      defined($dicom->{output_queue}) &&
      $dicom->{output_queue}->can("finish")
    ){
      $dicom->{output_queue}->finish();
    }
  }
}
{
  package Dispatch::Dicom::Acceptor;
  use vars qw ( @ISA );
  @ISA = ( "Dispatch::Select::Socket" );
  sub new {
    my($class, $port, $file, $call_back) = @_;
    my $descrip = {};
    {
      open FILE, "<$file" or die "can't open $file";
      line:
      while(my $line = <FILE>){
        chomp $line;
        if($line =~ /^#/) { next line }
        unless($line =~ /^([a-z_]+):\s*(.*)\s*$/) { next line }
        my $type = $1;
        my $fields = $2;
        my @fields_array = split(/\|/, $fields);
        if($type eq "ae_title"){
          $descrip->{ae_title} = $fields_array[0];
        } elsif($type eq "allowed_calling_ae_titles"){
          for my $i (@fields_array){
            $descrip->{allowed_calling_ae_titles}->{$i} = 1;
          }
        } elsif($type eq "app_context"){
          $descrip->{app_context} = $fields_array[0];
        } elsif($type eq "imp_class_uid"){
          $descrip->{imp_class_uid} = $fields_array[0];
        } elsif($type eq "imp_ver_name"){
          $descrip->{imp_ver_name} = $fields_array[0];
        } elsif($type eq "protocol_version"){
          $descrip->{protocol_version} = $fields_array[0];
        } elsif($type eq "max_length"){
          $descrip->{max_length} = $fields_array[0];
        } elsif($type eq "num_invoked"){
          $descrip->{num_invoked} = $fields_array[0];
        } elsif($type eq "num_performed"){
          $descrip->{num_performed} = $fields_array[0];
        } elsif($type eq "storage_root"){
          $descrip->{storage_root} = $fields_array[0];
        } elsif($type eq "assoc_normal_close"){
          $descrip->{assoc_normal_close} = $fields_array[0];
        } elsif(
          $type eq "storage_pres_context" ||
          $type eq "delayed_storage_pres_context" ||
          $type eq "verification_pres_context"
        ){
          for my $i (1 .. $#fields_array){
            $descrip->{pres_contexts}->
             {$fields_array[0]}->{$fields_array[$i]} = 1;
          }
          if($type eq "storage_pres_context"){
            $descrip->{incoming_message_handler}->{$fields_array[0]} =
              "Dispatch::Dicom::Storage";
          }
          if($type eq "delayed_storage_pres_context"){
            $descrip->{incoming_message_handler}->{$fields_array[0]} =
              "Dispatch::Dicom::StorageWithDelay";
          }
          if($type eq "verification_pres_context"){
            $descrip->{incoming_message_handler}->{$fields_array[0]} =
              "Dispatch::Dicom::Verification";
          }
        }
      }
      close FILE;
    }
    my $foo = sub {
      my($this, $socket) = @_;
      my $dicom = Dispatch::Dicom::Connection->new_accept($socket, $descrip);
      $dicom->CreatePduReaderSta2($socket);
      $dicom->CreateOutputQueue($socket);
      #$dicom->CreateMessageQueueEmptier();
      if(defined $call_back && ref($call_back) eq "CODE"){
        $dicom->{connection_callback} = $call_back;
      }
    };
    my $serv = Dispatch::Acceptor->new($foo)->port_server($port);
    return bless $serv, $class;
  }
  sub DESTROY {
    my($this) = @_;
    my $class = ref($this);
    if($ENV{POSDA_DEBUG}){
      print "Destroying $class\n";
    }
  }
}
1;
