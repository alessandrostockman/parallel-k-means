#include <iostream>
#include <fstream>
#include <string>

#include "csv.h"
#include "data.h"
#include "kmeans.h"

int main(int argc, char *argv[]) {
    int max_iter = 10000, k = 4, verbose = 0;
    std::string in_file = "data/clustering.csv", out_clusters_file = "out/clusters.csv", out_centroids_file = "out/centroids.csv";
    std::vector<int> features;
    CSVParser p;

    if (argc > 1) {
        k = std::stoi(argv[1]);
        //TODO: Check > 1
    }

    if (argc > 2) {
        std::stringstream ss(argv[2]);
        for (int i; ss >> i;) {
            features.push_back(i);    
            if (ss.peek() == ',')
                ss.ignore();
        }
        //TODO: Check size > 1
    }
    if (argc > 3) {
        in_file = argv[3];
    }
    if (argc > 4) {
        out_clusters_file = argv[4];
    }
    if (argc > 5) {
        out_centroids_file = argv[5];
    }
    if (argc > 6) {
        max_iter = std::stoi(argv[6]);
    }
    if (argc > 7) {
        verbose = std::stoi(argv[7]);
    }
    
    std::cout << "Loading dataset from " << in_file << "\n";
    Dataset *dataset = p.read_dataset(in_file, features);
    std::cout << "\tDataset loaded: " << dataset->size() << " records with " << dataset->get_feature_num() << " features\n";

    std::cout << "Starting algorithm execution...\n";

    KMeans km = KMeans(k, max_iter);
    bool res = km.fit(*dataset);

    if (res) {
        std::cout << "\tAlgorithm successfully finished after " << km.get_iterations() << " iterations\n";
        std::cout << "\tInitialization time: " << km.get_timer_count(TIME_INITIALIZATION) 
            << "ms | Reclustering time: " << km.get_timer_count(TIME_RECLUSTERING) 
            << "ms | Update time: " << km.get_timer_count(TIME_UPDATE)
            << "ms | Total time: " << km.get_timer_count(TIME_TOTAL) << "ms\n";
    } else {
        std::cout << "\tAlgorithm interrupted after exceeding max limit iterations: " << max_iter << " iterations\n";
    }
    
    std::cout << "Writing output files:\n";
    p.write_cluster(*dataset, out_clusters_file);
    std::cout << "\tClusters output wrote to " << out_clusters_file << "\n";
    p.write_centroids(*dataset, km.get_centroids(), out_centroids_file);
    std::cout << "\tCentroids output wrote to " << out_clusters_file << "\n";
    std::cout << "Exiting...\n";
    return 0;
}