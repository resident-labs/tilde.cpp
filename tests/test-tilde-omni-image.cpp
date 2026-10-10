// tilde omni image budget: the default 1200-cell cap stays, a request may raise it up to HARD_MAX_CELLS
#include "tilde-omni-image.h"

#include <cstdio>
#include <cstdlib>

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); exit(1); } } while (0)

int main() {
    using namespace tilde_omni;
    // a client opt-in to the training cap: the 1896x1150 window shrinks to 1408x864
    auto p = fit(1896, 1150, TRAIN_CELLS);
    CHECK(p.cells() <= TRAIN_CELLS && p.width == 1408 && p.height == 864);
    // a fit asked for more than the default is no longer clamped to 1200
    p = fit(1896, 1150, 2200);
    CHECK(p.scale == 1.0 && p.width == 1920 && p.height == 1152 && p.cells() == 60 * 36);
    // the hard ceiling still applies
    p = fit(8000, 8000, 1e9);
    CHECK(p.cells() <= HARD_MAX_CELLS && p.scale == 1.0);
    // allocate: default caps unchanged, raised caps give native, budget still shrinks
    std::vector<std::pair<int, int>> sizes = {{1896, 1150}};
    CHECK(allocate(sizes, 1 << 30)[0].width == 1920); // default: native, padded
    CHECK(allocate(sizes, 1 << 30)[0].scale == 1.0);
    CHECK(allocate(sizes, 2400, {}, 2.0, {TRAIN_CELLS})[0].cells() <= TRAIN_CELLS);
    CHECK(allocate(sizes, 2400, {}, 2.0, {2200})[0].width == 1920);
    CHECK(allocate(sizes, 1500, {}, 2.0, {2200})[0].cells() <= 1500);
    auto two = allocate({{1896, 1150}, {1896, 1150}}, 2400, {}, 2.0, {2200, 2200});
    CHECK(two[0].cells() + two[1].cells() <= 2400);
    printf("ok\n");
    return 0;
}
