#ifndef H_KMEANS
#define H_KMEANS

#include "data.h"

#include <iostream>
#include <math.h>
#include <omp.h>

#include "kmeans.h"

#define TIMES_NUMBER 4

#define TIME_INITIALIZATION 0
#define TIME_RECLUSTERING 1
#define TIME_UPDATE 2
#define TIME_TOTAL 3

#define TIMER_TOTAL 0
#define TIMER_PARTIAL 1

#define MODE_K_MEANS 0
#define MODE_K_MEDIANS 1
#define MODE_K_MEDOIDS 2
#define MODE_K_MEANS_PP 3

class KMeans {

    public:
        KMeans(int k, int mode, int max_iter, bool verbose, int log_interval);
        bool fit(Dataset& data);
        double calculate_cost(Dataset& data);
        void init_clusters(Dataset& data);
        void update_clusters(Dataset& data);
        int update_centroids(Dataset& data);
        Record *get_centroids();
        int get_iterations();
        void start_timer(int timer);
        void end_timer(int timer);
        double get_timer_value(int timer);
        std::string get_times(int type);
    private:
        int k;
        int mode;
        int max_iter;
        bool verbose;
        int log_interval;
        int iter;
        Record *centroids;
        double total_times[TIMES_NUMBER] = {0};
        double start_times[TIMES_NUMBER] = {0};
        double end_times[TIMES_NUMBER] = {0};
};

#endif