// vram_ballast: hold GPU memory so that exactly TARGET GiB stay free (simulates a smaller card on a bigger one).
//   nvcc -O2 -o vram_ballast vram_ballast.cu && ./vram_ballast 16.5      # leaves 16.5 GiB free until killed
// It allocates in 256 MiB blocks plus a remainder, touches every byte, prints the free memory and sleeps.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <unistd.h>

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s FREE_GIB\n", argv[0]);
        return 1;
    }
    const double target = atof(argv[1]) * 1024.0 * 1024.0 * 1024.0;
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    if ((double) free_b <= target) {
        fprintf(stderr, "only %.2f GiB free, nothing to hold\n", free_b / 1073741824.0);
        return 1;
    }
    size_t to_hold = (size_t) ((double) free_b - target);
    std::vector<void *> blocks;
    const size_t block = 256ull << 20;
    while (to_hold > 0) {
        const size_t n = to_hold > block ? block : to_hold;
        void * p = nullptr;
        if (cudaMalloc(&p, n) != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed\n");
            return 1;
        }
        cudaMemset(p, 0, n);
        blocks.push_back(p);
        to_hold -= n;
        cudaMemGetInfo(&free_b, &total_b);
        if ((double) free_b <= target) {
            break;
        }
    }
    cudaDeviceSynchronize();
    cudaMemGetInfo(&free_b, &total_b);
    printf("holding: %.3f GiB free of %.3f GiB\n", free_b / 1073741824.0, total_b / 1073741824.0);
    fflush(stdout);
    for (;;) {
        sleep(3600);
    }
}
