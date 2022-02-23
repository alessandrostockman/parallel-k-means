#ifndef H_DATASET
#define H_DATASET

#include <math.h>
#include <vector>
#include <iostream>
#include <float.h>

#include "record.h"

class Dataset {
    public:
        Dataset(std::vector<Record *> records_vect, size_t feature_num);
        Record *operator[](size_t index);
        size_t size();
        size_t get_feature_num();

    private:
        Record *records;
        size_t feature_num;
        size_t length;
        
};

#endif