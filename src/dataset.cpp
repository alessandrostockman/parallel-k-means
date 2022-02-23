#include "dataset.h"

Dataset::Dataset(std::vector<Record *> records_vect, size_t features) {
        length = (size_t)records_vect.size();
        records = (Record *)malloc(sizeof(Record) * length);
        feature_num = features;

        for (int i = 0; i < (int)length; i++) {
            records[i] = *records_vect[i];
        }
    }

Record *Dataset::operator[](size_t index) {
    return &(records[index]);
}

size_t Dataset::get_feature_num() {
    return feature_num;
}

size_t Dataset::size() {
    return length;
}