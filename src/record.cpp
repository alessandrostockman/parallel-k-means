#include "record.h"

Record::Record(size_t length) :
    length(length) {
    features = (double *)malloc(sizeof(double) * length);
    cluster = -1;
    centroid_dist = __DBL_MAX__;

    for (int i = 0; i < (int)length; i++) {
        features[i] = 0;
    }
}

Record::Record(double *features, size_t length) : 
    features(features), 
    cluster(-1),
    centroid_dist(__DBL_MAX__),
    length(length) {}

double Record::operator[](size_t index) {
    return features[index];
}

void Record::set_features(size_t index, double value) {
    features[index] = value;
}

double *Record::get_features() {
    return features;
}

size_t Record::size() {
    return length;
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
    for (int i = 0; i < (int)r.size(); i++) {
        double num = features[i] - r.features[i];
        sum += num * num;
    }
    return sqrt(sum);
}

bool operator==(Record& lhs, Record& rhs) {
    int lsize = lhs.size();
    int rsize = rhs.size();
    if (lsize != rsize) {
        return false;
    }

    double *l = lhs.get_features(), *r = rhs.get_features();

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
    for (int i = 0; i < (int)r->size(); i++) {
        if (i > 0) {
            os << ", ";
        }

        os << (*r)[i];
    }
    return os << ">";
}