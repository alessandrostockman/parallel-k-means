#ifndef H_KMEANS
#define H_KMEANS

#include "data.h"

#include <iostream>
#include <math.h>

#include "kmeans.h"

#define TIMES_NUMBER 4

#define TIME_INITIALIZATION 0
#define TIME_RECLUSTERING 1
#define TIME_UPDATE 2
#define TIME_TOTAL 3

class KMeans {

    public:
        KMeans(int k, int max_iter);
        bool fit(Dataset& data);
        double calculate_cost(Dataset& data);
        void init_centroids(Dataset& data);
        int update_centroids(Dataset& data);
        std::vector<Record *> *get_centroids();
        int get_iterations();
        void start_timer(int timer);
        void end_timer(int timer);
        double get_timer_count(int timer);
    private:
        int k;
        int max_iter;
        int iter;
        std::vector<Record *> *centroids;
        double total_times[TIMES_NUMBER] = {0};
        double start_times[TIMES_NUMBER] = {0};
};

#endif