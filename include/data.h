#ifndef H_DATA
#define H_DATA

#include <math.h>
#include <vector>
#include <iostream>
#include <float.h>

class Record {
    public:
        Record(size_t length);
        Record(double *features, size_t length);
        double operator[](size_t index);
        double *get_features();
        void set_features(size_t index, double value);
        int get_cluster();
        void set_cluster(int cluster);
        double get_centroid_dist();
        void set_centroid_dist(double centroid_dist);
        void reset_centroid_dist();
        size_t get_size();
        double distance(Record r);

    private:
        double *features;
        int cluster;
        double centroid_dist;
        size_t size;

};

class Dataset {
    public:
        Dataset(std::vector<Record *> *records, size_t feature_num);
        Record *operator[](std::size_t index);
        std::size_t size();
        std::size_t get_feature_num();

    private:
        std::vector<Record *> *records;
        size_t feature_num;
        
};

std::ostream &operator<<(std::ostream &os, Record *r);

bool operator==(Record& lhs, Record& rhs);

bool operator!=(Record& lhs, Record& rhs);

#endif