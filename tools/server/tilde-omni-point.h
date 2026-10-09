#pragma once

// Tilde Omni point fields: the image token layout (blank merging) and the decoding of the point head's outputs.
// ref: tilda/vision/model.py (token_layout), tilda/vision/point.py (point_decode), tilda/vision/common.py
// (decode_click), tilda/agents/browser/serve.py (TildaServer._point: the answer fields)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <set>
#include <vector>

namespace tilde_omni {

constexpr int TOKEN_PX = 32;
constexpr int SUB_PX   = 16;
constexpr int LOC_GRID = 3;

// floats per point field after its 1 + n token logits: 4 sub-patch logits, 4 x 9 location logits, 2 log2 sizes
constexpr int POINT_EXTRA = 4 + 36 + 2;

struct layout {
    int rows = 0;
    int cols = 0;
    std::vector<int32_t> token_of_cell; // row-major cells
    std::vector<int32_t> token_cell;    // per token: its cell, or the merged region's cell nearest its centroid
    std::vector<bool>    single;        // per token: exactly one cell

    int n_tokens() const { return (int) token_cell.size(); }

    // token_of_cell: the blank-merging map (empty: every cell is its own token)
    static layout make(int rows, int cols, std::vector<int32_t> token_of_cell) {
        layout l;
        l.rows = rows;
        l.cols = cols;
        const int n = rows * cols;
        if (token_of_cell.empty()) {
            token_of_cell.resize(n);
            for (int i = 0; i < n; i++) {
                token_of_cell[i] = i;
            }
        }
        l.token_of_cell = std::move(token_of_cell);
        int n_tok = 0;
        for (int32_t t : l.token_of_cell) {
            n_tok = std::max(n_tok, t + 1);
        }
        std::vector<std::vector<int>> members(n_tok);
        for (int c = 0; c < n; c++) {
            members[l.token_of_cell[c]].push_back(c);
        }
        l.token_cell.resize(n_tok);
        l.single.resize(n_tok);
        for (int t = 0; t < n_tok; t++) {
            const auto & cells = members[t];
            double sy = 0, sx = 0;
            for (int c : cells) {
                sy += c / cols;
                sx += c % cols;
            }
            const double cy = (sy / cells.size() + 0.5) / rows;
            const double cx = (sx / cells.size() + 0.5) / cols;
            int best = cells.empty() ? 0 : cells[0];
            double best_d = INFINITY;
            for (int c : cells) {
                const double dy = ((c / cols) + 0.5) / rows - cy;
                const double dx = ((c % cols) + 0.5) / cols - cx;
                if (dy * dy + dx * dx < best_d) {
                    best_d = dy * dy + dx * dx;
                    best = c;
                }
            }
            l.token_cell[t] = best;
            l.single[t] = cells.size() == 1;
        }
        return l;
    }
};

struct click {
    bool   none;
    double p_none;
    int    token;
    double p_token;
    double p_blob;
    double p_elsewhere;
    double x, y;          // stored-image pixels
    double width, height; // predicted target size, stored-image pixels
    std::vector<double> probs; // softmax over [none, tokens]
    std::vector<int>    blob;  // the chosen token and the tokens touching it
};

// out: 1 + n token logits, then POINT_EXTRA floats at the most likely token (see llama_model_tilde_omni::build_points)
inline click decode(const float * out, const layout & l) {
    const int n = l.n_tokens();
    click c;
    c.probs.resize(n + 1);
    double mx = -INFINITY;
    for (int i = 0; i <= n; i++) {
        mx = std::max(mx, (double) out[i]);
    }
    double sum = 0;
    for (int i = 0; i <= n; i++) {
        c.probs[i] = std::exp((double) out[i] - mx);
        sum += c.probs[i];
    }
    for (auto & p : c.probs) {
        p /= sum;
    }
    // the graph picks the token by the argmax of the logits (first of ties), as here
    int best = 0;
    for (int i = 1; i < n; i++) {
        if (out[1 + i] > out[1 + best]) {
            best = i;
        }
    }
    double p_max_tok = c.probs[1 + best];
    std::set<int> near;
    for (int cell = 0; cell < l.rows * l.cols; cell++) {
        if (l.token_of_cell[cell] != best) {
            continue;
        }
        const int r = cell / l.cols, q = cell % l.cols;
        for (int dr = -1; dr <= 1; dr++) {
            for (int dc = -1; dc <= 1; dc++) {
                if (r + dr >= 0 && r + dr < l.rows && q + dc >= 0 && q + dc < l.cols) {
                    near.insert(l.token_of_cell[(r + dr) * l.cols + q + dc]);
                }
            }
        }
    }
    c.blob.assign(near.begin(), near.end());
    c.p_blob = 0;
    for (int t : near) {
        c.p_blob += c.probs[1 + t];
    }
    c.p_elsewhere = 0;
    for (int t = 0; t < n; t++) {
        if (!near.count(t)) {
            c.p_elsewhere = std::max(c.p_elsewhere, c.probs[1 + t]);
        }
    }
    c.none    = c.probs[0] >= p_max_tok;
    c.p_none  = c.probs[0];
    c.token   = best;
    c.p_token = p_max_tok;

    const float * sub  = out + 1 + n;
    const float * loc  = sub + 4;
    const float * size = loc + 36;
    const int cell = l.token_cell[best];
    const int r = cell / l.cols, q = cell % l.cols;
    if (l.single[best]) {
        int s = 0;
        for (int i = 1; i < 4; i++) {
            if (sub[i] > sub[s]) {
                s = i;
            }
        }
        int k = 0;
        for (int i = 1; i < LOC_GRID * LOC_GRID; i++) {
            if (loc[s * 9 + i] > loc[s * 9 + k]) {
                k = i;
            }
        }
        const int sy = s / 2, sx = s % 2, ly = k / LOC_GRID, lx = k % LOC_GRID;
        const double step = (double) SUB_PX / LOC_GRID;
        c.x = q * TOKEN_PX + sx * SUB_PX + (lx + 0.5) * step;
        c.y = r * TOKEN_PX + sy * SUB_PX + (ly + 0.5) * step;
    } else {
        c.x = (q + 0.5) * TOKEN_PX;
        c.y = (r + 0.5) * TOKEN_PX;
    }
    c.width  = std::pow(2.0, (double) size[0]);
    c.height = std::pow(2.0, (double) size[1]);
    return c;
}

} // namespace tilde_omni
