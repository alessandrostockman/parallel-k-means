#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <omp.h>

#include "csv.h"
#include "data.h"
#include "kmeans.h"

char* get_argument(char **begin, char **end, const std::string &option) {
    char **itr = std::find(begin, end, option);
    if (itr != end && ++itr != end) {
        return *itr;
    }
    return 0;
}

bool has_argument(char **begin, char **end, const std::string &option) {
    return std::find(begin, end, option) != end;
}

int main(int argc, char *argv[]) {
    int max_iter = 10000, log_interval = 100, k, mode;
    bool verbose, parallel;
    std::string in_file, out_clusters_file = "out/clusters.csv", out_centroids_file = "out/centroids.csv";
    std::vector<int> features;
    CSVParser p;

    if (has_argument(argv, argv+argc, "--help") || argc < 4 || !has_argument(argv, argv+argc, "--mode")) {
        std::cout << "Usage: kmeans.exe --mode M [--parallel] [--clusters-output F] [--centroids-output F] [--max-iter N] [--verbose] [--log-interval N] [--seed N] [--help]\nOptions:\n"; 
        std::cout << "\t--mode              | Selects an execution variant: [0: Standard K-Means, 1: K-Medians, 2: K-Medoids, 3: K-Means++]\n";
        std::cout << "\t--parallel          | Enables parallel execution\n";
        std::cout << "\t--clusters-output   | Output file containing the clustered data  (Default: " << out_clusters_file << ")\n";
        std::cout << "\t--centroids-output  | Output file containing the final centroids  (Default: " << out_centroids_file << ")\n";
        std::cout << "\t--max-iter          | Maximum number of iterations after which the program is stopped  (Default: " << max_iter << ")\n";
        std::cout << "\t--verbose           | If present, the program prints periodically statistics on execution times\n";
        std::cout << "\t--log-interval      | Number of iterations between logs if verbose mode is enabled (Default: " << log_interval << ")\n";
        std::cout << "\t--seed              | Sets a seed for deterministic execution\n";
        std::cout << "\t--help              | Shows this message\n";
        return 0;
    }
    
    k = std::stoi(argv[1]);
    in_file = argv[2];

    std::stringstream ss(argv[3]);
    for (int i; ss >> i;) {
        features.push_back(i);    
        if (ss.peek() == ',')
            ss.ignore();
    }

    mode = std::stoi(get_argument(argv, argv + argc, "--mode"));

    if (has_argument(argv, argv+argc, "--clusters-output")) {
        out_clusters_file = get_argument(argv, argv + argc, "--clusters-output");
    }
    if (has_argument(argv, argv+argc, "--centroids-output")) {
        out_centroids_file = get_argument(argv, argv + argc, "--centroids-output");
    }
    if (has_argument(argv, argv+argc, "--max-iter")) {
        max_iter = std::stoi(get_argument(argv, argv + argc, "--max-iter"));
    }
    if (has_argument(argv, argv+argc, "--log-interval")) {
        log_interval = std::stoi(get_argument(argv, argv + argc, "--log-interval"));
    }
    if (has_argument(argv, argv+argc, "--seed")) {
        srand(std::stoi(get_argument(argv, argv + argc, "--seed")));
    } else {
        srand((unsigned int)time(NULL));
    }
    
    parallel = has_argument(argv, argv+argc, "--parallel");
    verbose = has_argument(argv, argv+argc, "--verbose");
    
    std::cout << "Loading dataset from " << in_file << "\n";
    Dataset *dataset = p.read_dataset(in_file, features);
    std::cout << "\tDataset loaded: " << dataset->size() << " records with " << dataset->get_feature_num() << " features\n";

    std::cout << "Starting algorithm execution...\n";

    KMeans km = KMeans(k, mode, max_iter, parallel, verbose, log_interval);
    bool res = km.fit(*dataset);

    if (res) {
        std::cout << "\tAlgorithm successfully finished after " << km.get_iterations() << " iterations\n";
        std::cout << "\t" << km.get_times(TIMER_TOTAL) << "\n";
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