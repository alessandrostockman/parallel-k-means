#include "csv.h"

Dataset *create_dataset_from_csv(std::string in_file, std::vector<int> features) {
    CSVRow row;
    std::ifstream file(in_file);
    std::vector<Record *> *records = new std::vector<Record *>;
    int feature_num = features.size();

    bool first_line = true;
    while(file >> row) { //TODO: Check last line reading
        if (first_line) {
            first_line = false;
            std::cout << "Reading following columns: \n";
            for (int i = 0; i < (int)features.size(); i++) {
                std::cout << row[features[i]] << " ";
            }
            std::cout << "\n";
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
    //std::string_view(&m_line[m_data[index] + 1], m_data[index + 1] -  (m_data[index] + 1));
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