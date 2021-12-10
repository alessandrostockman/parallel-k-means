#ifndef H_KMEANS
#define H_KMEANS

#include "data.h"

#include <iostream>
#include <math.h>

#include "kmeans.h"

int compute_centroids(Dataset& data, std::vector<Record>& centroids, int k);

bool kmeans(Dataset& data, std::vector<Record *> *centroids, int k, int max_iter);

double calculate_cost(Dataset& data, std::vector<Record>& centroids);

#endif