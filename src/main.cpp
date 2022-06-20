#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <omp.h>
#include <iomanip>

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

std::string value_to_str(float v) {
    std::ostringstream out;
    out.precision(3);
    out << std::fixed << v;
    return out.str();
}

int main(int argc, char *argv[]) {
    std::string out_clusters_file = "out/clusters.csv", out_centroids_file = "out/centroids.csv";
    std::vector<int> features;
    CSVParser p;
    int k = std::stoi(get_argument(argv, argv + argc, "--k"));
    std::string in_file = get_argument(argv, argv + argc, "--input");
    int mode = std::stoi(get_argument(argv, argv + argc, "--mode"));
    
    // Reads the first line of the file in order to count the values number
    std::ifstream infile(in_file);
    std::string first_line;
    getline(infile, first_line);
    int commas = std::count(first_line.begin(), first_line.end(), ',');
    for (int i = 0; i < commas + 1; i++) {
        features.push_back(i);
    }

    float init, update_clusters, update_centroids, total;
    int reps = 5;
    
    init = 0;
    update_clusters = 0;
    update_centroids = 0;
    total = 0;
    for (int x = 0; x < reps; x++) {
        Dataset *dataset = p.read_dataset(in_file, features, false);
        KMeans *km = new KMeans(k, mode, 10000, false, 100);
        km->fit(*dataset);
        init += km->get_timer_value(TIME_INITIALIZATION);
        update_clusters += km->get_timer_value(TIME_RECLUSTERING) / km->get_iterations();
        update_centroids += km->get_timer_value(TIME_UPDATE) / km->get_iterations();
        delete km;
    }
    init /= reps;
    update_clusters /= reps;
    update_centroids /= reps;
    
    total = init + update_clusters + update_centroids;
    std::cout << std::fixed << value_to_str(init) << " & " << value_to_str(update_clusters) << " & " << value_to_str(update_centroids) << " & " << value_to_str(total) << " \\\\ \n"; 
    return 0;
}