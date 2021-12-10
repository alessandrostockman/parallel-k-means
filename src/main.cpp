#include <iostream>
#include <fstream>
#include <string>

#include "csv.h"
#include "data.h"
#include "kmeans.h"

void write_csv(Dataset d, std::vector<Record *> *centroids, std::string out_clusters_file, std::string out_centroids_file) {
    std::ofstream cl_file(out_clusters_file);
    for (int i = 0; i < (int)d.size(); i++) {
        for (int j = 0; j < (int)d.get_feature_num(); j++) {
            cl_file << (*d[i])[j] << ",";
        }
        cl_file << d[i]->get_cluster() << "\n";
    }
    cl_file.close();

    std::ofstream ce_file(out_centroids_file);
    for (int i = 0; i < (int)centroids->size(); i++) {
        ce_file << i;
        for (int j = 0; j < (int)d.get_feature_num(); j++) {
            ce_file << "," << (*(*centroids)[i])[j];
        }
        ce_file << "\n";
    }
    ce_file.close();
}

int main(int argc, char *argv[]) {
    int max_iter = 10000, k = 4;
    std::string in_file = "data/clustering.csv", out_clusters_file = "out/clusters.csv", out_centroids_file = "out/centroids.csv";

    if (argc > 1) {
        max_iter = std::stoi(argv[1]);
    }

    if (argc > 2) {
        k = std::stoi(argv[2]);
    }

    // if (argc > 2) {
    //     in_file = argv[2];
    // }

    // if (argc > 3) {
    //     in_file = argv[3];
    // }

    // if (argc > 4) {
    //     in_file = argv[4];
    // }
    
    std::vector<int> features = {6, 8};
    int feature_num = features.size();

    CSVRow row;
    std::ifstream file(in_file);
    std::vector<Record *> *records = new std::vector<Record *>;

    bool first_line = true;
    while(file >> row) {
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
    Dataset dataset = Dataset(records, feature_num);
    std::vector<Record *> *centroids = new std::vector<Record *>;
    std::cout << "Dataset read\n";

    std::cout << "Startin algorithm execution...\n";
    bool res = kmeans(dataset, centroids, k, max_iter);
    std::cout << "Algorithm ended: " << res << "\n";
    
    std::cout << "Writing to file:\n";
    write_csv(dataset, centroids, out_clusters_file, out_centroids_file);
    std::cout << "Operation succeded!\n";
    return 0;
}