#ifndef H_DATA
#define H_DATA

#include <math.h>
#include <vector>

class Record {
    public:
        Record(int length);
        Record(std::vector<double> *features);
        double operator[](std::size_t index);
        bool operator==(Record other);
        bool operator!=(Record other);
        std::vector<double> *get_features();
        void set_features(size_t index, double value);
        int get_cluster();
        void set_cluster(int cluster);
        double get_centroid_dist();
        void set_centroid_dist(double centroid_dist);
        double distance(Record r);

    private:
        std::vector<double> *features;
        int cluster;
        double centroid_dist;

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

#endif