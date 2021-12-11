#ifndef H_KMEANS
#define H_KMEANS

#include "data.h"

#include <iostream>
#include <math.h>

#include "kmeans.h"

class KMeans {

    public:
        KMeans(int k, int max_iter);
        bool cluster(Dataset& data);
        double calculate_cost(Dataset& data);
        void init_centroids(Dataset& data);
        int update_centroids(Dataset& data);
        std::vector<Record *> *get_centroids();
        int get_iterations();
    private:
        int k;
        int max_iter;
        int iter;
        std::vector<Record *> *centroids;
};

#endif