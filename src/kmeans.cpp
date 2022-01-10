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
    std::string log = "Initialization time: " + std::to_string(get_timer_value(TIME_INITIALIZATION)) 
        + "ms | Reclustering time: " + std::to_string(get_timer_value(TIME_RECLUSTERING)) 
        + "ms | Update time: " + std::to_string(get_timer_value(TIME_UPDATE)) + "ms";
        if (type == TIMER_TOTAL) {
            log = log + " | Total time: " + std::to_string(get_timer_value(TIME_TOTAL)) + "ms";
        }
        return log;
}

bool KMeans::fit(Dataset& data) {
    start_timer(TIME_TOTAL);

    start_timer(TIME_INITIALIZATION);
    init_clusters(data);
    end_timer(TIME_INITIALIZATION);

    for (iter = 0; iter < max_iter; iter++) {
        if (iter > 0 && iter % log_interval == 0 && verbose) {
            std::cout << "Iteration #" << iter << "\t"
                << get_times(TIMER_PARTIAL) << "\n";
        }

        start_timer(TIME_RECLUSTERING);
        update_clusters(data);
        end_timer(TIME_RECLUSTERING);
        
        start_timer(TIME_UPDATE);
        int changed = update_centroids(data);
        end_timer(TIME_UPDATE);

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
    
    for (int i = 0; i < k; i++) {
        int rand_index;
        do {
            rand_index = rand() % data.size();
            duplicate = false;
            for (int j = 0; j < i && !duplicate; j++) {
                if (rand_index == random_history[j]) {
                    duplicate = true;
                }
            }
        } while (duplicate);

        random_history.push_back(rand_index);

        Record *r = data[rand_index];
        Record centroid = Record(data.get_feature_num());

        for (int j = 0; j < (int)data.get_feature_num(); j++) {
            centroid.set_features(j, (*r)[j]);
        }
        r->set_cluster(i);
        centroids[i] = centroid;
    }
}

void KMeans::update_clusters(Dataset& data) {
#pragma omp parallel for
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

// double median(int n, std::vector<int> a) {
//     // If size of the arr[] is even
//     if (n % 2 == 0) {
  
//         // Applying std::nth_element
//         // on n/2th index
//         std::nth_element(a.begin(),
//                     a.begin() + n / 2,
//                     a.end());
  
//         // Applying std::nth_element
//         // on (n-1)/2 th index
//         std::nth_element(a.begin(),
//                     a.begin() + (n - 1) / 2,
//                     a.end());
  
//         // Find the average of value at
//         // index N/2 and (N-1)/2
//         return (double)(a[(n - 1) / 2]
//                         + a[n / 2])
//                / 2.0;
//     }
  
//     // If size of the arr[] is odd
//     else {
  
//         // Applying std::nth_element
//         // on n/2
//         std::nth_element(a.begin(),
//                     a.begin() + n / 2,
//                     a.end());
  
//         // Value at index (N/2)th
//         // is the median
//         return (double)a[n / 2];
//     }
// }

int KMeans::update_centroids(Dataset& data) {
    int changes = 0;
    int *sizes = (int *)malloc(sizeof(int) * k);
    Record *cumulatives = (Record *)malloc(sizeof(Record) * k);

    for (int i = 0; i < k; i++) {
        sizes[i] = 0;
        cumulatives[i] = Record(data.get_feature_num());
    }

    for (int i = 0; i < (int)data.size(); i++) {
        Record *r = data[i];
        for (int f = 0; f < (int)data.get_feature_num(); f++) {
            cumulatives[r->get_cluster()].set_features(f, cumulatives[r->get_cluster()][f] + (*r)[f]);
        }
        sizes[r->get_cluster()]++;
    }
    
    for (int i = 0; i < k; i++) {
        Record mean_features = Record(data.get_feature_num());

        for (int f = 0; f < (int)data.get_feature_num(); f++) {
            mean_features.set_features(f, cumulatives[i][f] / sizes[i]);
        }

        if (centroids[i] != mean_features) {
            changes++;
            centroids[i] = Record(mean_features);
        }
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