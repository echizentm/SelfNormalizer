#!/usr/bin/perl
use strict;
use warnings;
use JSON::XS   qw/decode_json encode_json/;
use List::Util qw/shuffle/;
use SelfNormalizer;

my $line = <STDIN>;
my $obj  = decode_json($line);
my $n    = SelfNormalizer->new($obj->{labels});

my @objs;
while (my $line = <STDIN>) { push(@objs, decode_json($line)); }

foreach (1 ... 10) {
    foreach (shuffle @objs) { $n->train($_->{data}, $_->{label}); }
}

foreach (@objs) {
    print encode_json($_)."\n";
    print "P(W) = " . $n->predict($_->{data}, 'W')."\n";
    print "P(C) = " . $n->predict($_->{data}, 'C')."\n";
    print "\n";
}

