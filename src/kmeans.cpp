#include "kmeans.h"

#define LOG_INTERVAL 1000 

KMeans::KMeans(int k, int max_iter) : 
    k(k),
    max_iter(max_iter),
    iter(0),
    centroids(new std::vector<Record *>) {}

bool KMeans::fit(Dataset& data) {
    init_centroids(data);

    for (iter = 0; iter < max_iter; iter++) {
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

        int changed = update_centroids(data);

        if (changed == 0) {
            return true;
        }
    }

    return false;
}

void KMeans::init_centroids(Dataset& data) {
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
}

int KMeans::update_centroids(Dataset& data) {
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

double KMeans::calculate_cost(Dataset& data) {
    double sum = 0;
    for (int i = 0; i < (int)data.size(); i++) {
        Record *r = data[i];
        sum += sqrt(r->distance(*(*centroids)[r->get_cluster()]));
    }
    return sum;
}

std::vector<Record *> *KMeans::get_centroids() {
    return centroids;
}

int KMeans::get_iterations() {
    return iter;
}