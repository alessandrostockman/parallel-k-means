#include "kmeans.h"

KMeans::KMeans(int k, int mode, int max_iter, bool verbose, int log_interval) : 
    k(k),
    mode(mode),
    max_iter(max_iter),
    verbose(verbose),
    log_interval(log_interval),
    iter(0) {
        centroids = (Record *)malloc(sizeof(Record) * k);
    }

void KMeans::start_timer(int timer) {
    start_times[timer] = omp_get_wtime();
}

void KMeans::end_timer(int timer) {
    total_times[timer] += omp_get_wtime() - start_times[timer];
}

double KMeans::get_timer_value(int timer) {
    return total_times[timer] * 1000;
}

std::string KMeans::get_times(int type) {
    // Build the string containing the times of execution of the different sections of the algorithm 
    std::string log = "Initialization time: " + std::to_string(get_timer_value(TIME_INITIALIZATION)) 
        + "ms \t| Reclustering time: " + std::to_string(get_timer_value(TIME_RECLUSTERING)) 
        + "ms \t| Update time: " + std::to_string(get_timer_value(TIME_UPDATE)) + "ms";
        if (type == TIMER_TOTAL) {
            log = log + " | Total time: " + std::to_string(get_timer_value(TIME_TOTAL)) + "ms";
        }
        return log;
}

bool KMeans::fit(Dataset& data) {
    start_timer(TIME_TOTAL);

    // Initialize the clusters based on the selected algorithm variant
    start_timer(TIME_INITIALIZATION);
    init_clusters(data);
    end_timer(TIME_INITIALIZATION);

    // Loop the algorithm until the max number of iterations is reached or the executions ends
    for (iter = 0; iter < max_iter; iter++) {
        if (iter > 0 && iter % log_interval == 0 && verbose) {
            std::cout << "Iteration #" << iter << "\t"
                << get_times(TIMER_PARTIAL) << "\n";
        }

        // Update every point based on the strategy of the selected variant
        start_timer(TIME_RECLUSTERING);
        update_clusters(data);
        end_timer(TIME_RECLUSTERING);
        
        // Recompute the centroids based on the strategy of the selected variant
        start_timer(TIME_UPDATE);
        int changed = update_centroids(data);
        end_timer(TIME_UPDATE);

        // If no centroid was changed exit the execution
        if (changed == 0) {
            end_timer(TIME_TOTAL);
            return true;
        }
    }

    end_timer(TIME_TOTAL);
    return false;
}

void KMeans::init_clusters(Dataset& data) {
    std::vector<int> random_history;
    bool duplicate;

    if (mode == MODE_K_MEANS || mode == MODE_K_MEDIANS || mode == MODE_K_MEDOIDS) {
        // Standard variant
        
        for (int i = 0; i < k; i++) {
            // For each cluster choose a random centroid
            int rand_index;
            do {
                // Search for a random point not already chosen from the dataset
                rand_index = rand() % data.size();
                duplicate = false;
                for (int j = 0; j < i && !duplicate; j++) {
                    if (rand_index == random_history[j]) {
                        duplicate = true;
                    }
                }
            } while (duplicate);

            random_history.push_back(rand_index);

            // Assign the chosen data points as centroids
            Record *r = data[rand_index];
            Record centroid = Record(data.get_feature_num());

            for (int j = 0; j < (int)data.get_feature_num(); j++) {
                centroid.set_features(j, (*r)[j]);
            }
            r->set_cluster(i);
            centroids[i] = centroid;
        }
    } else if (mode == MODE_K_MEANS_PP) {
        // Kmeans++ variant

        // Pick the first cluster centroids at random
        centroids[0] = *data[rand() % data.size()];
        double sum;
    
    
        // Select the centroids for the remaining clusters
        for (int cluster = 1; cluster < k; cluster++) {
    
            /* For each data point find the nearest centroid, save its
            distance in the distance array, then add it to the sum of
            total distance. */
            sum = 0.0;
            for (int j = 0; j < (int)data.size(); j++ ) {
                int i;
                double d, min_d;
            
                min_d = HUGE_VAL;
                for (i = 0; i < cluster; i++) {
                    d = data[j]->distance(centroids[i]);
                    if ( d < min_d ) {
                        min_d = d;
                    }
                }
            
                data[j]->set_centroid_dist(min_d);
                sum += min_d;
            }
    
            // Find a random distance within the span of the total distance
            sum = sum * rand() / (RAND_MAX - 1);
    
            // Assign the centroids. the point with the largest distance will have a greater probability of being selected
            for (int j = 0; j < (int)data.size(); j++ ) {
                sum -= data[j]->get_centroid_dist();
                if (sum <= 0) {
                    centroids[cluster] = *data[j];
                    break;
                }
            }
        }
    
        // Assign each observation the index of it's nearest cluster centroid
        update_clusters(data);
    } else {
        throw std::invalid_argument("Unknown MODE");
    }
}

void KMeans::update_clusters(Dataset& data) {
#pragma omp parallel for
    // Updates parallely every data point's cluster by choosing the one with the minimum distance
    for (int i = 0; i < (int)data.size(); i++) {
        Record *r = data[i];
        r->reset_centroid_dist();
        for (int j = 0; j < k; j++) {
            double dist = centroids[j].distance(*r);

            if (dist < r->get_centroid_dist()) {
                r->set_centroid_dist(dist);
                r->set_cluster(j);
            }
        }
    }
}

int KMeans::update_centroids(Dataset& data) {
    int changes = 0;

    if (mode == MODE_K_MEANS || mode == MODE_K_MEANS_PP) {
        // Standard variant
        int *sizes = (int *)malloc(sizeof(int) * k);
        Record *cumulatives = (Record *)malloc(sizeof(Record) * k);

        for (int i = 0; i < k; i++) {
            sizes[i] = 0;
            cumulatives[i] = Record(data.get_feature_num());
        }

        // Compute the sum of all the features for each cluster
        for (int f = 0; f < (int)data.get_feature_num(); f++) {
            double *cumulatives_f = (double *)calloc(sizeof(double), k);
#pragma omp parallel for reduction(+:cumulatives_f[:k]) reduction(+:sizes[:k])
            for (int i = 0; i < (int)data.size(); i++) {
                Record *r = data[i];
                cumulatives_f[r->get_cluster()] += (*r)[f];
                sizes[r->get_cluster()]++;
            }

            for (int i = 0; i < k; i++) {
                cumulatives[i].set_features(f, cumulatives_f[i]);
            }
        }

        // Compute the mean for each cluster
        for (int i = 0; i < k; i++) {
            Record mean_features = Record(data.get_feature_num());
            int size = sizes[i] / (int)data.get_feature_num();

            for (int f = 0; f < (int)data.get_feature_num(); f++) {
                mean_features.set_features(f, cumulatives[i][f] / size);
            }

            if (centroids[i] != mean_features) {
                changes++;
                centroids[i] = Record(mean_features);
            }
        }
    } else if (mode == MODE_K_MEDIANS) {
        // K-medians variant
        int *sizes = (int *)malloc(sizeof(int) * k);
        double ***temps = (double ***)malloc(sizeof(double **) * k);
        int *counters = (int *)malloc(sizeof(int) * k);
        for (int i = 0; i < k; i++) {
            sizes[i] = 0;
            counters[i] = 0;
        }
        
        // Count the points for each cluster
        for (int i = 0; i < (int)data.size(); i++) {
            Record *r = data[i];
            sizes[r->get_cluster()]++;
        }

        // Build a temporary array for the data points in each cluster
        for (int i = 0; i < k; i++) {
            temps[i] = (double **)malloc(sizeof(double *) * data.get_feature_num());

            for (int j = 0; j < (int)data.get_feature_num(); j++) {
                temps[i][j] = (double *)malloc(sizeof(double) * sizes[i]);
            }
        }

        // Populate the temporary array
        for (int i = 0; i < (int)data.size(); i++) {
            Record *r = data[i];
            for (int f = 0; f < (int)data.get_feature_num(); f++) {
                temps[r->get_cluster()][f][counters[r->get_cluster()]] = (*r)[f];
            }
            counters[r->get_cluster()]++;
        }

        for (int i = 0; i < k; i++) {
            // Order the temporary arrays in order to get the median values for each cluster
            Record med = Record(data.get_feature_num());
            for (int f = 0; f < (int)data.get_feature_num(); f++) {
                std::sort(temps[i][f], temps[i][f] + sizes[i]);
                med.set_features(f, sizes[i] % 2 != 0 ? temps[i][f][(int)sizes[i] / 2] : (temps[i][f][sizes[i] / 2] + temps[i][f][(sizes[i] - 1) / 2]) / 2);
            }

            if (centroids[i] != med) {
                changes++;
                centroids[i] = med;
            }
        }
    } else if (mode == MODE_K_MEDOIDS) {
        // K-medoids variant
        //TODO: Implement
    } else {
        throw std::invalid_argument("Unknown MODE");
    }
    return changes;
}

double KMeans::calculate_cost(Dataset& data) {
    double sum = 0;
    for (int i = 0; i < (int)data.size(); i++) {
        Record *r = data[i];
        sum += sqrt(r->distance(centroids[r->get_cluster()]));
    }
    return sum;
}

Record *KMeans::get_centroids() {
    return centroids;
}

int KMeans::get_iterations() {
    return iter;
}