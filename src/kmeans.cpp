#include "kmeans.h"

#define LOG_INTERVAL 1000 

int compute_centroids(Dataset& data, std::vector<Record *> *centroids, int k) {
    int changes = 0;
    for (int i = 0; i < k; i++) {
        Record mean_features = Record(data.get_feature_num());
        int cluster_size = 0;

        for (int j = 0; j < (int)data.size(); j++) {
            Record *r = data[j];
            if (r->get_cluster() == i) {
                cluster_size++;
                for (int f = 0; f < (int)data.get_feature_num(); f++) {
                    mean_features.set_features(f, mean_features[f] + (*r)[f]);
                }
            }
        }


        for (int f = 0; f < (int)data.get_feature_num(); f++) {
            mean_features.set_features(f, mean_features[f] / cluster_size);
        }

        if ((*(*centroids)[i]) != mean_features) {
            changes++;
            (*centroids)[i] = new Record(mean_features);
        }
    }
    return changes;
}

bool kmeans(Dataset& data, std::vector<Record *> *centroids, int k, int max_iter) {
    std::vector<int> randoms;
    bool found;

    srand((unsigned int)time(NULL));
    for (int i = 0; i < k; i++) {
        int rand_index;
        do {
            rand_index = rand() % data.size();
            found = false;
            for (int j = 0; j < i && !found; j++) {
                if (rand_index == randoms[j]) {
                    found = true;
                }
            }
        } while (found);
        randoms.push_back(rand_index);
        Record *r = data[rand_index];
        Record *centroid = new Record(data.get_feature_num());

        for (int j = 0; j < (int)data.get_feature_num(); j++) {
            centroid->set_features(j, (*r)[j]);
        }
        r->set_cluster(i);
        centroids->push_back(centroid);
    }

    for (int iter = 0; iter < max_iter; iter++) {
        if (iter % LOG_INTERVAL == 0) {
            std::cout << "iteration #" << iter << "\n";
        }

        for (int i = 0; i < (int)data.size(); i++) {
            Record *r = data[i];
            r->reset_centroid_dist();
            for (int j = 0; j < k; j++) {
                double dist = (*centroids)[j]->distance(*r);

                if (dist < r->get_centroid_dist()) {
                    r->set_centroid_dist(dist);
                    r->set_cluster(j);
                }
            }
        }

        int changed = compute_centroids(data, centroids, k);

        if (changed == 0) {
            std::cout << "Exiting on iteration " << iter << "\n";
            return true;
        }
    }

    return false;
}

double calculate_cost(Dataset& data, std::vector<Record>& centroids) {
    double sum = 0;
    for (int i = 0; i < (int)data.size(); i++) {
        Record *r = data[i];
        sum += sqrt(r->distance(centroids[r->get_cluster()]));
    }
    return sum;
}