#include "data.h"

#include <iostream>

Record::Record(int length) {
    features = new std::vector<double>(length, 0.0);
    cluster = 0;
    centroid_dist = __DBL_MAX__;
}

Record::Record(std::vector<double> *features) : 
    features(features), 
    cluster(0),
    centroid_dist(__DBL_MAX__) {}

double Record::operator[](std::size_t index) {
    return (*features)[index];
}

bool Record::operator==(Record other) {
    return features == other.get_features();
}

bool Record::operator!=(Record other) {
    return !(*this == other);
}

void Record::set_features(size_t index, double value) {
    (*features)[index] = value;
}

std::vector<double> *Record::get_features() {
    return features;
}

int Record::get_cluster() {
    return cluster;
}

void Record::set_cluster(int c) {
    cluster = c;
}

double Record::get_centroid_dist() {
    return centroid_dist;
}

void Record::set_centroid_dist(double dist) {
    centroid_dist = dist;
}

double Record::distance(Record r) {
    double sum = 0;
    for (int i = 0; i < (int)(*features).size(); i++) {
        sum += ((*features)[i] - (*r.features)[i]) * ((*features)[i] - (*r.features)[i]);
    }
    return sqrt(sum);
}

Dataset::Dataset(std::vector<Record *> *records, size_t feature_num) : 
    records(records),
    feature_num(feature_num) {}

Record *Dataset::operator[](std::size_t index) {
    return (*records)[index];
}

std::size_t Dataset::get_feature_num() {
    return feature_num;
}

std::size_t Dataset::size() {
    return records->size();
}