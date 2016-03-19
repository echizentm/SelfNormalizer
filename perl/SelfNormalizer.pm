package SelfNormalizer;
use strict;
use warnings;
use parent qw/Class::Accessor::Fast/;
use constant {
    DEFAULT_LEARNING_RATE      => 0.1,
    DEFAULT_NORMALIZATION_RATE => 0.1,
};

__PACKAGE__->mk_accessors(qw/
    learning_rate
    normalization_rate
/);

sub new {
    my ($class, $labels) = @_;
    return $class->SUPER::new({
        learning_rate      => DEFAULT_LEARNING_RATE,
        normalization_rate => DEFAULT_NORMALIZATION_RATE,
        labels             => $labels,
        weight             => {},
    });
}

sub predict {
    my ($self, $data, $label) = @_;

    my $inner_product = 0.0;
    foreach my $feature (keys %$data) {
        next unless ($self->{weight}{$label}{$feature});
        $inner_product += (
            $self->{weight}{$label}{$feature} * $data->{$feature}
        );
    }
    return exp($inner_product);
}

sub train {
    my ($self, $data, $label) = @_;

    my $z = $self->get_normalizer($data);

    foreach my $a_label (@{$self->{labels}}) {
        my $prob      = $self->predict($data, $a_label) / $z;
        my $loss_diff = $prob - (($a_label eq $label) ? 1 : 0);
        foreach my $feature (keys %$data) {
            $self->{weight}{$a_label}{$feature} -= (
                $self->learning_rate * $loss_diff * $data->{$feature} +
                $self->normalization_rate * 2 * log($z) * $prob * $data->{$feature}
            );
        }
    }
}

sub get_normalizer {
    my ($self, $data) = @_;

    my $normalizer = 0.0;
    foreach my $label (@{$self->{labels}}) {
        $normalizer += $self->predict($data, $label);
    }
    return $normalizer;
}

1;
