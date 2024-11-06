#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define NUM_POINTS 253681  // Ajuste para o número total de pontos na base de dados
#define NUM_DIMENSIONS 21   // Número de variáveis ou colunas numéricas
#define K 5
#define MAX_ITERATIONS 100

// Função para calcular a distância euclidiana entre dois pontos
double euclidean_distance(double *a, double *b, int dimensions) {
    double distance = 0.0;
    for (int i = 0; i < dimensions; i++) {
        distance += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(distance);
}

// Função para o algoritmo k-means na versão paralela otimizada
void kmeans_parallel(double points[NUM_POINTS][NUM_DIMENSIONS], int labels[NUM_POINTS], double centroids[K][NUM_DIMENSIONS]) {
    int iterations = 0;
    while (iterations < MAX_ITERATIONS) {
        int changes = 0;

        // Atribui cada ponto ao centróide mais próximo
        #pragma omp parallel for reduction(+:changes)
        for (int i = 0; i < NUM_POINTS; i++) {
            int nearest_centroid = 0;
            double min_distance = euclidean_distance(points[i], centroids[0], NUM_DIMENSIONS);

            for (int j = 1; j < K; j++) {
                double distance = euclidean_distance(points[i], centroids[j], NUM_DIMENSIONS);
                if (distance < min_distance) {
                    min_distance = distance;
                    nearest_centroid = j;
                }
            }

            if (labels[i] != nearest_centroid) {
                labels[i] = nearest_centroid;
                changes++;
            }
        }

        // Acumula os novos centróides em variáveis privadas
        double new_centroids[K][NUM_DIMENSIONS] = {0};
        int counts[K] = {0};

        #pragma omp parallel
        {
            double local_centroids[K][NUM_DIMENSIONS] = {0};
            int local_counts[K] = {0};

            #pragma omp for
            for (int i = 0; i < NUM_POINTS; i++) {
                int cluster = labels[i];
                local_counts[cluster]++;
                for (int d = 0; d < NUM_DIMENSIONS; d++) {
                    local_centroids[cluster][d] += points[i][d];
                }
            }

            #pragma omp critical
            {
                for (int j = 0; j < K; j++) {
                    counts[j] += local_counts[j];
                    for (int d = 0; d < NUM_DIMENSIONS; d++) {
                        new_centroids[j][d] += local_centroids[j][d];
                    }
                }
            }
        }

        // Atualiza centróides
        for (int j = 0; j < K; j++) {
            if (counts[j] > 0) {
                for (int d = 0; d < NUM_DIMENSIONS; d++) {
                    centroids[j][d] = new_centroids[j][d] / counts[j];
                }
            }
        }

        if (changes == 0) {
            break;
        }

        iterations++;
    }
}

int main() {
    double (*points)[NUM_DIMENSIONS] = malloc(NUM_POINTS * sizeof(*points));
    int *labels = malloc(NUM_POINTS * sizeof(*labels));
    double centroids[K][NUM_DIMENSIONS];

    // Lê os dados do arquivo CSV gerado em Python
    FILE *file = fopen("processed_data_diabetes.csv", "r");
    if (!file) {
        printf("Erro ao abrir o arquivo.\n");
        return 1;
    }

    // Lê os dados e armazena em 'points'
    for (int i = 0; i < NUM_POINTS; i++) {
        for (int j = 0; j < NUM_DIMENSIONS; j++) {
            fscanf(file, "%lf,", &points[i][j]);
        }
        labels[i] = 0; // Inicializa rótulos
    }
    fclose(file);

    // Inicializa centróides com valores aleatórios
    for (int j = 0; j < K; j++) {
        for (int d = 0; d < NUM_DIMENSIONS; d++) {
            centroids[j][d] = rand() % 100; // Valor aleatório
        }
    }

    double start = omp_get_wtime();
    kmeans_parallel(points, labels, centroids);
    double end = omp_get_wtime();

    printf("Tempo de execução: %f segundos\n", end - start);

    // Libera a memória
    free(points);
    free(labels);

    return 0;
}