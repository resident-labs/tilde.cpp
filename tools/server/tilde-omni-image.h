#pragma once

// Tilde Omni image storage: how big each image of a request is stored (budget), and the stored image itself (resize +
// pad). ref: tilda/vision/budget.py (fit, allocate, render, to_original); the resize is Pillow's LANCZOS resample
// (libImaging/Resample.c: two passes, horizontal first, 8-bit fixed-point coefficients), so stored pixels match the
// reference server's.

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace tilde_omni {

constexpr int MAX_CELLS = 1200; // per image
constexpr int MIN_SIDE  = 64;   // px, the smallest short side a shrunk image keeps

struct plan {
    double scale = 1.0; // stored px per original px
    int content_width  = 0;
    int content_height = 0;
    int width  = 0; // stored size: content padded up to multiples of 32
    int height = 0;

    int cells() const { return (width / 32) * (height / 32); }
};

// Python's round(): half to even
inline int py_round(double v) {
    return (int) std::nearbyint(v);
}

inline int padded(int n) {
    return std::max(32, (int) std::ceil(n / 32.0) * 32);
}

inline plan make_plan(int w, int h, double scale) {
    plan p;
    p.content_width  = std::max(1, py_round(w * scale));
    p.content_height = std::max(1, py_round(h * scale));
    p.scale  = (double) p.content_width / w;
    p.width  = padded(p.content_width);
    p.height = padded(p.content_height);
    return p;
}

inline plan fit(int w, int h, double max_cells) {
    if (w < 1 || h < 1) {
        throw std::invalid_argument("empty image");
    }
    max_cells = std::max(1.0, std::min(max_cells, (double) MAX_CELLS));
    double scale = std::min(1.0, std::sqrt(max_cells * 32 * 32 / ((double) w * h)));
    plan p = make_plan(w, h, scale);
    while (p.cells() > max_cells && p.scale > 1e-3) {
        scale *= 0.97;
        p = make_plan(w, h, scale);
    }
    const plan floor = make_plan(w, h, std::min(1.0, (double) MIN_SIDE / std::min(w, h)));
    return floor.scale > p.scale && floor.cells() <= max_cells ? floor : p;
}

// times: empty, or one per image (recency-weighted shares, halving every `halflife`)
inline std::vector<plan> allocate(const std::vector<std::pair<int, int>> & sizes, int budget_cells,
        const std::vector<double> & times = {}, double halflife = 2.0) {
    std::vector<plan> natural;
    int total = 0;
    for (const auto & [w, h] : sizes) {
        natural.push_back(fit(w, h, MAX_CELLS));
        total += natural.back().cells();
    }
    if (total <= budget_cells) {
        return natural;
    }
    std::vector<double> weights(sizes.size(), 1.0);
    if (!times.empty()) {
        const double newest = *std::max_element(times.begin(), times.end());
        for (size_t i = 0; i < sizes.size(); i++) {
            weights[i] = std::pow(0.5, (newest - times[i]) / halflife);
        }
    }
    // water-filling: share the budget by weight; an image needing less than its share gives the rest back
    std::vector<plan> plans(sizes.size());
    std::vector<bool> open(sizes.size(), true);
    double remaining = budget_cells;
    size_t n_open = sizes.size();
    while (n_open > 0) {
        double sum = 0;
        for (size_t i = 0; i < sizes.size(); i++) {
            if (open[i]) sum += weights[i];
        }
        std::vector<double> share(sizes.size());
        bool settled = false;
        for (size_t i = 0; i < sizes.size(); i++) {
            if (open[i]) {
                share[i] = remaining * weights[i] / sum;
                settled = settled || natural[i].cells() <= share[i];
            }
        }
        if (!settled) {
            for (size_t i = 0; i < sizes.size(); i++) {
                if (open[i]) plans[i] = fit(sizes[i].first, sizes[i].second, share[i]);
            }
            break;
        }
        for (size_t i = 0; i < sizes.size(); i++) {
            if (open[i] && natural[i].cells() <= share[i]) {
                plans[i] = natural[i];
                remaining -= natural[i].cells();
                open[i] = false;
                n_open--;
            }
        }
    }
    return plans;
}

// a stored-image point in original pixels; points in the padding clamp to the edge
inline std::pair<double, double> to_original(double x, double y, const plan & p) {
    return { std::min(x, (double) p.content_width) / p.scale, std::min(y, (double) p.content_height) / p.scale };
}

// Pillow LANCZOS resize of an RGB image (Image.resize(size, Image.LANCZOS), no box, no reducing gap)
struct resampler {
    static constexpr int PRECISION_BITS = 32 - 8 - 2;

    static double sinc(double x) {
        if (x == 0.0) return 1.0;
        x *= M_PI;
        return std::sin(x) / x;
    }
    static double lanczos(double x) {
        return (-3.0 <= x && x < 3.0) ? sinc(x) * sinc(x / 3) : 0.0;
    }

    // per output index: first input index, count, and fixed-point coefficients
    static int coeffs(int in_size, int out_size, std::vector<int> & bounds, std::vector<int32_t> & kk) {
        const double scale = (double) in_size / out_size;
        const double filterscale = std::max(1.0, scale);
        const double support = 3.0 * filterscale;
        const int ksize = (int) std::ceil(support) * 2 + 1;
        std::vector<double> k(ksize);
        bounds.assign((size_t) out_size * 2, 0);
        kk.assign((size_t) out_size * ksize, 0);
        for (int xx = 0; xx < out_size; xx++) {
            const double center = (xx + 0.5) * scale;
            const double ss = 1.0 / filterscale;
            int xmin = (int) (center - support + 0.5);
            if (xmin < 0) xmin = 0;
            int xmax = (int) (center + support + 0.5);
            if (xmax > in_size) xmax = in_size;
            xmax -= xmin;
            double ww = 0.0;
            for (int x = 0; x < xmax; x++) {
                k[x] = lanczos((x + xmin - center + 0.5) * ss);
                ww += k[x];
            }
            for (int x = 0; x < ksize; x++) {
                double v = x < xmax ? (ww != 0.0 ? k[x] / ww : k[x]) : 0.0;
                kk[(size_t) xx * ksize + x] = v < 0 ? (int32_t) (-0.5 + v * (1 << PRECISION_BITS))
                                                    : (int32_t) ( 0.5 + v * (1 << PRECISION_BITS));
            }
            bounds[xx * 2 + 0] = xmin;
            bounds[xx * 2 + 1] = xmax;
        }
        return ksize;
    }

    static uint8_t clip8(int64_t v) {
        v >>= PRECISION_BITS;
        return (uint8_t) std::min<int64_t>(255, std::max<int64_t>(0, v));
    }
};

// rgb: w x h x 3; returns ow x oh x 3
inline std::vector<uint8_t> resize_lanczos(const std::vector<uint8_t> & rgb, int w, int h, int ow, int oh) {
    if (ow == w && oh == h) {
        return rgb;
    }
    std::vector<int> bh, bv;
    std::vector<int32_t> kh, kv;
    const int ksh = resampler::coeffs(w, ow, bh, kh);
    const int ksv = resampler::coeffs(h, oh, bv, kv);
    const bool need_h = ow != w;
    const bool need_v = oh != h;
    const int y_first = bv[0];
    const int y_last  = bv[(oh - 1) * 2] + bv[(oh - 1) * 2 + 1];

    std::vector<uint8_t> tmp;
    const uint8_t * src = rgb.data();
    int src_w = w, src_y0 = 0, src_rows = h;
    if (need_h) {
        // only the rows the vertical pass reads
        src_rows = y_last - y_first;
        tmp.resize((size_t) ow * src_rows * 3);
        for (int yy = 0; yy < src_rows; yy++) {
            const uint8_t * row = rgb.data() + (size_t) (yy + y_first) * w * 3;
            for (int xx = 0; xx < ow; xx++) {
                const int xmin = bh[xx * 2], xmax = bh[xx * 2 + 1];
                const int32_t * k = &kh[(size_t) xx * ksh];
                int64_t s[3] = { 1 << (resampler::PRECISION_BITS - 1), 1 << (resampler::PRECISION_BITS - 1), 1 << (resampler::PRECISION_BITS - 1) };
                for (int x = 0; x < xmax; x++) {
                    for (int c = 0; c < 3; c++) {
                        s[c] += (int64_t) row[(x + xmin) * 3 + c] * k[x];
                    }
                }
                for (int c = 0; c < 3; c++) {
                    tmp[((size_t) yy * ow + xx) * 3 + c] = resampler::clip8(s[c]);
                }
            }
        }
        src = tmp.data();
        src_w = ow;
        src_y0 = y_first;
    }
    if (!need_v) {
        return need_h ? tmp : rgb;
    }
    std::vector<uint8_t> out((size_t) ow * oh * 3);
    for (int yy = 0; yy < oh; yy++) {
        const int ymin = bv[yy * 2] - src_y0, ymax = bv[yy * 2 + 1];
        const int32_t * k = &kv[(size_t) yy * ksv];
        for (int xx = 0; xx < src_w; xx++) {
            int64_t s[3] = { 1 << (resampler::PRECISION_BITS - 1), 1 << (resampler::PRECISION_BITS - 1), 1 << (resampler::PRECISION_BITS - 1) };
            for (int y = 0; y < ymax; y++) {
                const uint8_t * px = src + ((size_t) (y + ymin) * src_w + xx) * 3;
                for (int c = 0; c < 3; c++) {
                    s[c] += (int64_t) px[c] * k[y];
                }
            }
            for (int c = 0; c < 3; c++) {
                out[((size_t) yy * ow + xx) * 3 + c] = resampler::clip8(s[c]);
            }
        }
    }
    (void) src_rows;
    return out;
}

// budget.render: the content resized to the plan, padded right / bottom with the median edge colour of the content
inline std::vector<uint8_t> render(const std::vector<uint8_t> & rgb, int w, int h, const plan & p) {
    const int cw = p.content_width, ch = p.content_height;
    const std::vector<uint8_t> content = resize_lanczos(rgb, w, h, cw, ch);
    // right column then bottom row of the content, per channel median (the upper one for an even count)
    uint8_t fill[3];
    for (int c = 0; c < 3; c++) {
        std::vector<uint8_t> edge;
        for (int y = 0; y < ch; y++) edge.push_back(content[((size_t) y * cw + cw - 1) * 3 + c]);
        for (int x = 0; x < cw; x++) edge.push_back(content[((size_t) (ch - 1) * cw + x) * 3 + c]);
        std::nth_element(edge.begin(), edge.begin() + edge.size() / 2, edge.end());
        fill[c] = edge[edge.size() / 2];
    }
    std::vector<uint8_t> out((size_t) p.width * p.height * 3);
    for (int y = 0; y < p.height; y++) {
        for (int x = 0; x < p.width; x++) {
            uint8_t * px = &out[((size_t) y * p.width + x) * 3];
            if (x < cw && y < ch) {
                const uint8_t * s = &content[((size_t) y * cw + x) * 3];
                px[0] = s[0]; px[1] = s[1]; px[2] = s[2];
            } else {
                px[0] = fill[0]; px[1] = fill[1]; px[2] = fill[2];
            }
        }
    }
    return out;
}

} // namespace tilde_omni
