#include "data.h"

#include <iostream>

Record::Record(int length) {
    features = new std::vector<double>(length, 0.0);
    cluster = -1;
    centroid_dist = __DBL_MAX__;
}

Record::Record(std::vector<double> *features) : 
    features(features), 
    cluster(-1),
    centroid_dist(__DBL_MAX__) {}

double Record::operator[](std::size_t index) {
    return (*features)[index];
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

void Record::reset_centroid_dist() {
    centroid_dist = __DBL_MAX__;
}

double Record::distance(Record r) {
    double sum = 0;
    for (int i = 0; i < (int)(*features).size(); i++) {
        double num = ((*features)[i] - (*r.features)[i]);
        sum += num * num;
    }
    return sqrt(sum);
}

bool operator==(Record& lhs, Record& rhs) { 
    int lsize = lhs.get_features()->size();
    int rsize = rhs.get_features()->size();
    if (lsize != rsize) {
        return false;
    }

    std::vector<double> l = *lhs.get_features(), r = *rhs.get_features();

    for (int i = 0; i < lsize; i++) {
        double abs_l = fabs(l[i]);
        double abs_r = fabs(r[i]);
        float largest = (abs_l > abs_r) ? abs_l : abs_r;

        if (fabs(l[i] - r[i]) > largest * FLT_EPSILON) {
            return false;
        }
    } 
    return true;
}

bool operator!=(Record& lhs, Record& rhs) { 
    return !(lhs == rhs); 
}

std::ostream &operator<<(std::ostream &os, Record *r) { 
    os << "<";
    for (int i = 0; i < (int)r->get_features()->size(); i++) {
        if (i > 0) {
            os << ", ";
        }

        os << r->get_features()->at(i);
    }
    return os << ">";
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