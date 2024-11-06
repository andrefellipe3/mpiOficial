#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

#define NUM_POINTS 253681   // Ajuste para o número total de pontos na base de dados
#define NUM_DIMENSIONS 21   // Número de variáveis ou colunas numéricas
#define K 5
#define MAX_ITERATIONS 100
#define NUM_THREADS 2

// Função para calcular a distância euclidiana entre dois pontos
double euclidean_distance(double *a, double *b, int dimensions) {
    double distance = 0.0;
    for (int i = 0; i < dimensions; i++) {
        distance += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(distance);
}

// Função para o algoritmo k-means na versão paralela otimizada
void kmeans_parallel(double points[NUM_POINTS][NUM_DIMENSIONS], int labels[NUM_POINTS], double centroids[K][NUM_DIMENSIONS], int world_rank, int world_size) {
    int iterations = 0;
    int changes;

    while (iterations < MAX_ITERATIONS) {
        changes = 0;

        // Atribui cada ponto ao centróide mais próximo
        omp_set_num_threads(NUM_THREADS);
        #pragma omp parallel for reduction(+:changes)
        for (int i = world_rank; i < NUM_POINTS; i += world_size) {
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

        // Acumula os novos centróides em variáveis locais
        double local_centroids[K][NUM_DIMENSIONS] = {0};
        int local_counts[K] = {0};

        // Paraleliza a acumulação dos centróides locais com OpenMP
        omp_set_num_threads(NUM_THREADS);
        #pragma omp parallel for
        for (int i = world_rank; i < NUM_POINTS; i += world_size) {
            int cluster = labels[i];
            local_counts[cluster]++;
            for (int d = 0; d < NUM_DIMENSIONS; d++) {
                local_centroids[cluster][d] += points[i][d];
            }
        }

        // Reduz as somas locais e contagens para os processos
        double global_centroids[K][NUM_DIMENSIONS] = {0};
        int global_counts[K] = {0};

        // MPI_Allreduce para agregar os dados de todos os processos
        MPI_Allreduce(local_centroids, global_centroids, K * NUM_DIMENSIONS, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_counts, global_counts, K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        // Atualiza centróides globais
        for (int j = 0; j < K; j++) {
            if (global_counts[j] > 0) {
                for (int d = 0; d < NUM_DIMENSIONS; d++) {
                    centroids[j][d] = global_centroids[j][d] / global_counts[j];
                }
            }
        }

        // Verifica se houve mudanças
        int global_changes;
        MPI_Allreduce(&changes, &global_changes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        if (global_changes == 0) {
            break;
        }

        iterations++;
    }
}

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    double (*points)[NUM_DIMENSIONS] = malloc(NUM_POINTS * sizeof(*points));
    int *labels = malloc(NUM_POINTS * sizeof(*labels));
    double centroids[K][NUM_DIMENSIONS];

    if (points == NULL || labels == NULL) {
        fprintf(stderr, "Erro ao alocar memória.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if(world_rank == 0){
        // Lê os dados do arquivo CSV gerado em Python
        FILE *file = fopen("processed_data_diabetes.csv", "r");
        if (!file) {
            fprintf(stderr, "Erro ao abrir o arquivo.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
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
    }

    // Broadcast dos pontos e centróides para todos os processos
    MPI_Bcast(points, NUM_POINTS * NUM_DIMENSIONS, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(centroids, K * NUM_DIMENSIONS, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Inicia a contagem de tempo
    double start_time, end_time;
    if (world_rank == 0) {
        start_time = MPI_Wtime();
    }

    // Executa o k-means paralelo
    kmeans_parallel(points, labels, centroids, world_rank, world_size);

    // Finaliza a contagem de tempo no processo 0
    if (world_rank == 0) {
        end_time = MPI_Wtime();
        printf("Tempo de execução do k-means paralelo: %f segundos\n", end_time - start_time);
    }

    // Libera a memória
    free(points);
    free(labels);
    
    MPI_Finalize();
    return 0;
}