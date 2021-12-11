#include "csv.h"

void CSVParser::write_cluster(Dataset d, std::string out_file) {
    std::ofstream cl_file(out_file);
    for (int i = 0; i < (int)d.size(); i++) {
        for (int j = 0; j < (int)d.get_feature_num(); j++) {
            cl_file << (*d[i])[j] << ",";
        }
        cl_file << d[i]->get_cluster() << "\n";
    }
    cl_file.close();
}

void CSVParser::write_centroids(Dataset d, std::vector<Record *> *centroids, std::string out_file) {
    std::ofstream ce_file(out_file);
    for (int i = 0; i < (int)centroids->size(); i++) {
        ce_file << i;
        for (int j = 0; j < (int)d.get_feature_num(); j++) {
            ce_file << "," << (*(*centroids)[i])[j];
        }
        ce_file << "\n";
    }
    ce_file.close();
}

Dataset *CSVParser::read_dataset(std::string in_file, std::vector<int> features) {
    CSVRow row;
    std::ifstream file(in_file);
    std::vector<Record *> *records = new std::vector<Record *>;
    int feature_num = features.size();

    bool first_line = true;
    while(file >> row) {
        if (first_line) {
            first_line = false;
        } else {
            std::vector<double> *f = new std::vector<double>;
            for (int i = 0; i < (int)features.size(); i++) {
                double n = std::stod(row[features[i]]);
                f->push_back(n);
            }
            records->push_back(new Record(f));
        }
    }

    return new Dataset(records, feature_num);
}

std::string CSVRow::operator[](std::size_t index) const {
    return m_line.substr(m_data[index]+1, m_data[index+1] - (m_data[index]+1));
}

std::size_t CSVRow::size() const {
    return m_data.size() - 1;
}

void CSVRow::readNextRow(std::istream& str) {
    std::getline(str, m_line);

    m_data.clear();
    m_data.emplace_back(-1);
    std::string::size_type pos = 0;
    while((pos = m_line.find(',', pos)) != std::string::npos)
    {
        m_data.emplace_back(pos);
        ++pos;
    }
    // This checks for a trailing comma with no data after it.
    pos = m_line.size();
    m_data.emplace_back(pos);
}

std::istream& operator>>(std::istream& str, CSVRow& data) {
    data.readNextRow(str);
    return str;
}