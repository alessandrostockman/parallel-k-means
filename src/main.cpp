#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <omp.h>

#include "csv.h"
#include "dataset.h"
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

int omp_thread_count() {
    int n = 0;
    #pragma omp parallel reduction(+:n)
    n += 1;
    return n;
}

int main(int argc, char *argv[]) {
    int max_iter = 10000, log_interval = 100, k, mode;
    bool verbose, header;
    std::string in_file, out_clusters_file = "out/clusters.csv", out_centroids_file = "out/centroids.csv";
    std::vector<int> features;
    CSVParser p;

    if (has_argument(argv, argv+argc, "--help") || !has_argument(argv, argv+argc, "--k") || !has_argument(argv, argv+argc, "--input") || !has_argument(argv, argv+argc, "--mode")) {
        std::cout << "Usage: kmeans.exe --k CLUSTERS --input INPUT_FILE --mode MODE [--cols INPUT_COLS] [--header]\n\t[--clusters-output CL_FILE] [--centroids-output CE_FILE] [--max-iter MAX_ITER] \n\t[--verbose] [--log-interval LOG_INT] [--seed SEED] [--help]\n";
        std::cout << "Options:\n"; 
        std::cout << "\t--k                 | Number of clusters\n";
        std::cout << "\t--input             | Path of the input file dataset in csv format\n";
        std::cout << "\t--mode              | Selects an execution variant: [0: Standard K-Means, 1: K-Medians, 2: K-Medoids, 3: K-Means++]\n";
        std::cout << "\t--cols              | 0-indexed columns of the input file considered for the clustering separated by comma (e.g.: `0,1,2,4`). If omitted all columns are considered\n";
        std::cout << "\t--header            | If present, the first line of the input CSV file is skipped as it's considered as a header\n";
        std::cout << "\t--clusters-output   | Output file containing the clustered data  (Default: `" << out_clusters_file << "`)\n";
        std::cout << "\t--centroids-output  | Output file containing the final centroids  (Default: `" << out_centroids_file << "`)\n";
        std::cout << "\t--max-iter          | Maximum number of iterations after which the program is stopped  (Default: `" << max_iter << "`)\n";
        std::cout << "\t--verbose           | If present, the program prints periodically statistics on execution times\n";
        std::cout << "\t--log-interval      | Number of iterations between logs if verbose mode is enabled (Default: `" << log_interval << "`)\n";
        std::cout << "\t--seed              | Sets a seed for deterministic execution\n";
        std::cout << "\t--help              | Shows this message\n";
        return 0;
    }

    // Reading of all the command line arguments
    
    k = std::stoi(get_argument(argv, argv + argc, "--k"));
    in_file = get_argument(argv, argv + argc, "--input");
    mode = std::stoi(get_argument(argv, argv + argc, "--mode"));

    if (has_argument(argv, argv+argc, "--cols")) {
        // Reads the argument integers separated by commas
        std::stringstream ss(get_argument(argv, argv + argc, "--cols"));
        for (int i; ss >> i;) {
            features.push_back(i);    
            if (ss.peek() == ',') {
                ss.ignore();
            }
        }
    } else {
        // Reads the first line of the file in order to count the values number
        std::ifstream infile(in_file);
        std::string first_line;
        getline(infile, first_line);
        int commas = std::count(first_line.begin(), first_line.end(), ',');
        for (int i = 0; i < commas + 1; i++) {
            features.push_back(i);
        }
    }

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
    
    header = has_argument(argv, argv+argc, "--header");
    verbose = has_argument(argv, argv+argc, "--verbose");

    std::cout << "Enabled parallelism using " << omp_thread_count() << " threads\n";
    
    // Dataset loading
    std::cout << "Loading dataset from " << in_file << "\n";
    Dataset *dataset = p.read_dataset(in_file, features, header);
    std::cout << "\tDataset loaded: " << dataset->size() << " records with " << dataset->get_feature_num() << " features\n";

    // Algorithm execution
    std::cout << "Starting algorithm execution...\n";
    KMeans km = KMeans(k, mode, max_iter, verbose, log_interval);
    bool res = km.fit(*dataset);

    // Prints result depending on returning value of the fit method 
    if (res) {
        std::cout << "\tAlgorithm successfully finished after " << km.get_iterations() << " iterations\n";
        std::cout << "\t" << km.get_times(TIMER_TOTAL) << "\n";
    } else {
        std::cout << "\tAlgorithm interrupted after exceeding max limit iterations: " << max_iter << " iterations\n";
    }
    
    // Output writing
    std::cout << "Writing output files:\n";
    p.write_cluster(*dataset, out_clusters_file);
    std::cout << "\tClusters output wrote to " << out_clusters_file << "\n";
    p.write_centroids(*dataset, km.get_centroids(), out_centroids_file);
    std::cout << "\tCentroids output wrote to " << out_clusters_file << "\n";
    std::cout << "Exiting...\n";
    return 0;
}