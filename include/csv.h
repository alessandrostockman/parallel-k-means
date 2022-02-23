#ifndef CSV_H
#define CSV_H

#include <iostream>
#include <fstream>
#include <iterator>
#include <sstream>
#include <vector>
#include <string>

#include "dataset.h"
#include "record.h"

class CSVParser {
    public:
        Dataset *read_dataset(std::string in_file, std::vector<int> features);
        void write_cluster(Dataset d, std::string out_file);
        void write_centroids(Dataset d, Record *centroids, std::string out_file);
};

class CSVRow {
    public:
        std::string operator[](std::size_t index) const;
        std::size_t size() const;
        void readNextRow(std::istream& str);
    private:
        std::string         m_line;
        std::vector<int>    m_data;
};

std::istream& operator>>(std::istream& str, CSVRow& data);

#endif