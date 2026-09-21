// RKNN C API sample: ALIKE + LightGlue, two stages, image pair -> matches.
//
// Design notes that matter on the board
// -------------------------------------
// * NO float preprocessing.  The stage-1 RKNN model is configured with
//   mean=0 / std=255, so the NPU divides by 255 itself and the sample feeds raw
//   uint8 RGB.  Doing `x/255` in C++ would be both slower and wrong (it would be
//   applied a second time by the NPU).
// * Image size is FIXED at 512x512 with 512 keypoints - the models are exported
//   with static shapes and RKNN cannot resize them.  Resizing happens once, on
//   the host, before the first inference.
// * Both views go through stage 1 in ONE call: the ONNX graph was exported with
//   batch 2 for exactly this reason, halving the per-call overhead.
// * Outputs are read with `want_float = 1`.  RKNN may keep intermediate buffers
//   in fp16; letting it convert on read is one place a silent precision bug would
//   hide, so every output is requested as float32.
//
// Build: see CMakeLists.txt.  Requires librknn_api (aarch64) and OpenCV.
//
// Usage:
//   ./infer alike_stage.rknn lightglue_stage.rknn left.jpg right.jpg [--dump DIR]
//
// With --dump, every stage output is written as raw float32 (shape in a sidecar
// .txt) so the board result can be diffed against the Python parity harness.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "rknn_api.h"

#define CHECK(expr, msg)                                                    \
  do {                                                                      \
    int _r = (expr);                                                        \
    if (_r != 0) {                                                          \
      fprintf(stderr, "[E] %s failed (ret=%d)\n", (msg), _r);               \
      return 1;                                                             \
    }                                                                       \
  } while (0)

namespace {

constexpr int kSize = 512;       // network input side (must stay in sync with the export)
constexpr int kKeypoints = 512;  // fixed keypoint count (static shapes)

struct Model {
  rknn_context ctx = 0;
  std::vector<rknn_tensor_attr> in_attrs, out_attrs;

  ~Model() {
    if (ctx) rknn_destroy(ctx);
  }

  bool load(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
      fprintf(stderr, "[E] cannot open %s\n", path.c_str());
      return false;
    }
    fseek(f, 0, SEEK_END);
    size_t sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<unsigned char> buf(sz);
    if (fread(buf.data(), 1, sz, f) != sz) {
      fclose(f);
      fprintf(stderr, "[E] short read on %s\n", path.c_str());
      return false;
    }
    fclose(f);
    if (rknn_init(&ctx, buf.data(), sz, 0, nullptr) != 0) {
      fprintf(stderr, "[E] rknn_init failed for %s\n", path.c_str());
      return false;
    }
    uint32_t n_in = 0, n_out = 0;
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &n_in, sizeof(n_in));  // n_in reused below
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &n_out, sizeof(n_out));
    in_attrs.resize(n_in);
    out_attrs.resize(n_out);
    for (uint32_t i = 0; i < n_in; ++i) {
      in_attrs[i] = {};
      in_attrs[i].index = i;
      rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &in_attrs[i], sizeof(rknn_tensor_attr));
    }
    for (uint32_t i = 0; i < n_out; ++i) {
      out_attrs[i] = {};
      out_attrs[i].index = i;
      rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &out_attrs[i], sizeof(rknn_tensor_attr));
    }
    printf("[i] %s: %u input(s), %u output(s)\n", path.c_str(), n_in, n_out);
    return true;
  }
};

void write_dump(const std::string& dir, const std::string& name,
                const float* data, size_t n, const std::string& shape) {
  if (dir.empty()) return;
  std::string base = dir + "/" + name;
  FILE* f = fopen((base + ".bin").c_str(), "wb");
  if (!f) return;
  fwrite(data, sizeof(float), n, f);
  fclose(f);
  f = fopen((base + ".shape.txt").c_str(), "w");
  if (f) {
    fprintf(f, "%s\n", shape.c_str());
    fclose(f);
  }
}

// Resize to 512x512 exactly as the Python reference does (bilinear, no crop).
cv::Mat prepare(const cv::Mat& bgr) {
  cv::Mat rgb, resized;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  cv::resize(rgb, resized, cv::Size(kSize, kSize), 0, 0, cv::INTER_LINEAR);
  return resized;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    fprintf(stderr,
            "usage: %s alike_stage.rknn lightglue_stage.rknn left.jpg right.jpg "
            "[--dump DIR]\n",
            argv[0]);
    return 1;
  }
  std::string dump_dir;
  for (int i = 5; i < argc; ++i) {
    if (std::string(argv[i]) == "--dump" && i + 1 < argc) dump_dir = argv[++i];
  }

  Model stage1, stage2;
  if (!stage1.load(argv[1]) || !stage2.load(argv[2])) return 1;

  // ---- host-side image preparation (the only non-NPU step) ----------------
  cv::Mat l = cv::imread(argv[3]), r = cv::imread(argv[4]);
  if (l.empty() || r.empty()) {
    fprintf(stderr, "[E] could not read one of the images\n");
    return 1;
  }
  cv::Mat lr = prepare(l), rr = prepare(r);

  // stage 1 takes both views in one call: (2, 512, 512, 3) uint8 NHWC.
  std::vector<unsigned char> pair(2 * kSize * kSize * 3);
  memcpy(pair.data(), lr.data, kSize * kSize * 3);
  memcpy(pair.data() + kSize * kSize * 3, rr.data, kSize * kSize * 3);

  std::vector<rknn_input> inputs(1);
  inputs[0].index = 0;
  inputs[0].type = RKNN_TENSOR_UINT8;
  inputs[0].fmt = RKNN_TENSOR_NHWC;
  inputs[0].size = pair.size();
  inputs[0].buf = pair.data();
  CHECK(rknn_inputs_set(stage1.ctx, 1, inputs.data()), "rknn_inputs_set(stage1)");
  CHECK(rknn_run(stage1.ctx, nullptr), "rknn_run(stage1)");

  const uint32_t n1 = stage1.out_attrs.size();
  std::vector<rknn_output> o1(n1);
  for (uint32_t i = 0; i < n1; ++i) {
    o1[i] = {};
    o1[i].index = i;
    o1[i].want_float = 1;
  }
  CHECK(rknn_outputs_get(stage1.ctx, n1, o1.data(), nullptr),
        "rknn_outputs_get(stage1)");

  // Identify outputs by element count rather than by position: the runtime does
  // not guarantee the ONNX output order.
  const float* kpts = nullptr;
  const float* desc = nullptr;
  const float* scores = nullptr;
  size_t n_kpts = 2 * kKeypoints * 2, n_desc = 2 * kKeypoints * 128,
         n_scores = 2 * kKeypoints;
  for (uint32_t i = 0; i < n1; ++i) {
    size_t n = o1[i].size / sizeof(float);
    if (n == n_desc) desc = static_cast<float*>(o1[i].buf);
    else if (n == n_kpts) kpts = static_cast<float*>(o1[i].buf);
    else if (n == n_scores) scores = static_cast<float*>(o1[i].buf);
  }
  if (!kpts || !desc || !scores) {
    fprintf(stderr, "[E] stage-1 outputs not identified (sizes seen:");
    for (uint32_t i = 0; i < n1; ++i) fprintf(stderr, " %zu", o1[i].size / 4);
    fprintf(stderr, ")\n");
    rknn_outputs_release(stage1.ctx, n1, o1.data());
    return 1;
  }
  write_dump(dump_dir, "keypoints", kpts, n_kpts, "2,512,2");
  write_dump(dump_dir, "descriptors", desc, n_desc, "2,512,128");
  write_dump(dump_dir, "scores", scores, n_scores, "2,512");

  // ---- stage 2 ------------------------------------------------------------
  // view0 and view1 are the two halves of stage 1's batch dimension.
  const float* k0 = kpts;
  const float* k1 = kpts + kKeypoints * 2;
  const float* d0 = desc;
  const float* d1 = desc + kKeypoints * 128;

  std::vector<rknn_input> in2(4);
  const size_t kp_bytes = sizeof(float) * kKeypoints * 2;
  const size_t ds_bytes = sizeof(float) * kKeypoints * 128;
  const float* ptrs[4] = {k0, k1, d0, d1};
  const size_t sizes[4] = {kp_bytes, kp_bytes, ds_bytes, ds_bytes};
  for (int i = 0; i < 4; ++i) {
    in2[i] = {};
    in2[i].index = i;
    in2[i].type = RKNN_TENSOR_FLOAT32;
    in2[i].fmt = RKNN_TENSOR_UNDEFINED;  // keep the model's own layout
    in2[i].size = sizes[i];
    in2[i].buf = const_cast<float*>(ptrs[i]);
  }
  CHECK(rknn_inputs_set(stage2.ctx, 4, in2.data()), "rknn_inputs_set(stage2)");
  CHECK(rknn_run(stage2.ctx, nullptr), "rknn_run(stage2)");

  const uint32_t n2 = stage2.out_attrs.size();
  std::vector<rknn_output> o2(n2);
  for (uint32_t i = 0; i < n2; ++i) {
    o2[i] = {};
    o2[i].index = i;
    o2[i].want_float = 1;
  }
  CHECK(rknn_outputs_get(stage2.ctx, n2, o2.data(), nullptr),
        "rknn_outputs_get(stage2)");

  const float* matches = nullptr;
  const float* mscores = nullptr;
  for (uint32_t i = 0; i < n2; ++i) {
    size_t n = o2[i].size / sizeof(float);
    if (n == kKeypoints) {
      if (!matches) matches = static_cast<float*>(o2[i].buf);
      else mscores = static_cast<float*>(o2[i].buf);
    }
  }
  if (!matches || !mscores) {
    fprintf(stderr, "[E] stage-2 outputs not identified\n");
    rknn_outputs_release(stage1.ctx, n1, o1.data());
    rknn_outputs_release(stage2.ctx, n2, o2.data());
    return 1;
  }
  write_dump(dump_dir, "matches0", matches, kKeypoints, "1,512");
  write_dump(dump_dir, "mscores0", mscores, kKeypoints, "1,512");

  // ---- report -------------------------------------------------------------
  // matches0 is -1 where a keypoint is unmatched.
  int n_match = 0;
  double sum_score = 0.0;
  for (int i = 0; i < kKeypoints; ++i) {
    if (matches[i] >= 0.0f) {
      ++n_match;
      sum_score += mscores[i];
    }
  }
  printf("[i] matches: %d / %d\n", n_match, kKeypoints);
  if (n_match) printf("[i] mean match score: %.4f\n", sum_score / n_match);

  // stdout is the deliverable: one match per line, `idx0 idx1 score`.
  for (int i = 0; i < kKeypoints; ++i) {
    if (matches[i] >= 0.0f) {
      int j = static_cast<int>(matches[i] + 0.5f);
      printf("%d %d %.6f\n", i, j, mscores[i]);
    }
  }

  rknn_outputs_release(stage1.ctx, n1, o1.data());
  rknn_outputs_release(stage2.ctx, n2, o2.data());
  return 0;
}
