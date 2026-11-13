// Qwen3.5 vision tower (models/qwen35_vision): the host-side geometry against transformers 5.x, and, with the encoder
// file and a reference dump of tools/vision_ref/vision_conditioning.py (--dump-vision), the GPU tower against
// transformers' Qwen3_5VisionModel on the same image:
//   MOSS_VISION_TE=<qwen3.5_9b_int8_convrot.safetensors> MOSS_VISION_IMAGE=<png> MOSS_VISION_REF=<dump.safetensors>
// (the dump made with --megapixels 100000/1048576, the encoder's default cap).
#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

#include "models/qwen35_vision/vision_encoder.h"
#include "pipeline/pipeline_image.h"
#include "runtime/device_arena.h"
#include "weights/safetensors.h"

namespace {

using namespace mi::qwen35_vision;

float bf16_to_float(std::uint16_t v) {
  std::uint32_t u = static_cast<std::uint32_t>(v) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

TEST(Qwen35Vision, FitSizeKeepsTheAspectRatioOnTheMergeGridAndNeverUpscales) {
  EXPECT_EQ(fit_size(460, 500, 100'000), std::make_pair(288, 320));
  EXPECT_EQ(fit_size(1200, 864, 1'048'576), std::make_pair(1184, 864));  // snapping to 1216 would pass the cap
  EXPECT_EQ(fit_size(64, 64, 100'000), std::make_pair(64, 64));
  EXPECT_EQ(fit_size(10, 10, 100'000), std::make_pair(32, 32));  // at least one merged token
  const auto [w, h] = fit_size(4000, 1000, 100'000);
  EXPECT_LE(w * h, 100'000);
  EXPECT_EQ(w % 32, 0);
  EXPECT_EQ(h % 32, 0);
  EXPECT_THROW(fit_size(0, 10, 100), std::invalid_argument);
}

TEST(Qwen35Vision, MropePositionsMatchTransformersGetRopeIndex) {
  // Qwen3_5Model.get_rope_index for ids [t t t img*6 t t img*2 t], image grids (1, 4, 6) and (1, 2, 4).
  const auto pos = mrope_positions(14, {3, 11}, {{2, 3}, {1, 2}});
  const std::vector<std::vector<std::int32_t>> expected = {
      {0, 1, 2, 3, 3, 3, 3, 3, 3, 6, 7, 8, 8, 10},
      {0, 1, 2, 3, 3, 3, 4, 4, 4, 6, 7, 8, 8, 10},
      {0, 1, 2, 3, 4, 5, 3, 4, 5, 6, 7, 8, 9, 10},
  };
  EXPECT_EQ(pos, expected);
  const auto text = mrope_positions(4, {}, {});
  EXPECT_EQ(text[0], (std::vector<std::int32_t>{0, 1, 2, 3}));
  EXPECT_EQ(text[0], text[2]);
  EXPECT_THROW(mrope_positions(5, {3}, {{2, 2}}), std::invalid_argument);  // the image runs past the end
}

TEST(Qwen35Vision, RotaryTablesAreBlockMajorHeightThenWidth) {
  std::vector<float> cos, sin;
  rotary_tables(4, 4, cos, sin);
  ASSERT_EQ(cos.size(), 16u * 72u);
  // Patch 0 is (0, 0): no rotation.
  for (int i = 0; i < 72; ++i) {
    EXPECT_FLOAT_EQ(cos[static_cast<std::size_t>(i)], 1.0f);
    EXPECT_FLOAT_EQ(sin[static_cast<std::size_t>(i)], 0.0f);
  }
  // Patch 1 in block order is (0, 1): the height half is unrotated, the width half rotated by inv_freq[0] = 1.
  EXPECT_FLOAT_EQ(cos[72 + 0], 1.0f);
  EXPECT_FLOAT_EQ(cos[72 + 18], std::cos(1.0f));
  EXPECT_FLOAT_EQ(sin[72 + 36 + 18], std::sin(1.0f));  // the second copy of the 36
  // Patch 2 is (1, 0), the first patch of the block's second row.
  EXPECT_FLOAT_EQ(cos[2 * 72 + 0], std::cos(1.0f));
  EXPECT_FLOAT_EQ(cos[2 * 72 + 18], 1.0f);
}

TEST(Qwen35Vision, TowerMatchesTransformersOnARealImage) {
  const char* te = std::getenv("MOSS_VISION_TE");
  const char* image = std::getenv("MOSS_VISION_IMAGE");
  const char* ref = std::getenv("MOSS_VISION_REF");
  if (!te || !image || !ref) GTEST_SKIP() << "set MOSS_VISION_TE, MOSS_VISION_IMAGE and MOSS_VISION_REF";

  const Prepared prepared = prepare(mi::pipeline::load_image(image), kDefaultMaxPixels, 1.0f);
  mi::weights::SafetensorsFile f(ref);
  std::ifstream in(ref, std::ios::binary);
  auto read = [&](const char* name) {
    const auto* t = f.find(name);
    EXPECT_NE(t, nullptr) << name;
    std::vector<std::uint16_t> v(t->nbytes / 2);
    in.seekg(static_cast<std::streamoff>(t->offset));
    in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(t->nbytes));
    return v;
  };

  // The processor's patches: same resize (PIL Lanczos), normalization and order, so the same bf16 bits.
  const auto ref_patches = read("patches");
  ASSERT_EQ(ref_patches.size(), prepared.patches.size());
  EXPECT_EQ(ref_patches, prepared.patches);

  VisionEncoder encoder(te);
  mi::runtime::DeviceArena arena(encoder.workspace_bytes(static_cast<std::int64_t>(prepared.grid_h) * prepared.grid_w) +
                                 (std::size_t{64} << 20));
  cudaStream_t stream = nullptr;
  const mi::Tensor out = encoder.encode({prepared}, arena, stream);
  ASSERT_EQ(out.dim(0), prepared.tokens());
  std::vector<std::uint16_t> got(static_cast<std::size_t>(out.numel()));
  ASSERT_EQ(cudaMemcpy(got.data(), out.data, got.size() * 2, cudaMemcpyDeviceToHost), cudaSuccess);
  const auto want = read("vision");
  ASSERT_EQ(want.size(), got.size());

  // Not bit-exact (cuBLASLt vs torch GEMMs, our fused kernels). bf16 through 27 blocks drifts by itself: transformers'
  // own bf16 tower is within cosine 0.990 (worst token) and 3.2 % relative L2 of its fp32 run on this kind of image.
  // So the engine is held to that against the fp32 truth when given (MOSS_VISION_TRUTH, raw f32 [N, 4096] from the
  // fp32 tower), and to twice it against transformers' bf16 tokens.
  auto compare = [&](auto&& want_at, double& worst, double& rel) {
    double num = 0, den = 0;
    worst = 1.0;
    for (std::int64_t t = 0; t < out.dim(0); ++t) {
      double dot = 0, a2 = 0, b2 = 0;
      for (int d = 0; d < kOut; ++d) {
        const double a = bf16_to_float(got[static_cast<std::size_t>(t * kOut + d)]);
        const double b = want_at(static_cast<std::size_t>(t * kOut + d));
        dot += a * b;
        a2 += a * a;
        b2 += b * b;
        num += (a - b) * (a - b);
        den += b * b;
      }
      worst = std::min(worst, dot / std::sqrt(a2 * b2));
    }
    rel = std::sqrt(num / den);
  };
  double worst = 0, rel = 0;
  compare([&](std::size_t i) { return static_cast<double>(bf16_to_float(want[i])); }, worst, rel);
  EXPECT_GT(worst, 0.975) << "worst token cosine against transformers bf16";
  EXPECT_LT(rel, 0.065) << "relative L2 against transformers bf16";
  RecordProperty("bf16_worst_cosine", std::to_string(worst));
  RecordProperty("bf16_relative_l2", std::to_string(rel));
  if (const char* truth_path = std::getenv("MOSS_VISION_TRUTH")) {
    std::vector<float> truth(got.size());
    std::ifstream tf(truth_path, std::ios::binary);
    tf.read(reinterpret_cast<char*>(truth.data()), static_cast<std::streamsize>(truth.size() * 4));
    ASSERT_TRUE(tf.good()) << "MOSS_VISION_TRUTH must hold f32 [tokens, 4096]";
    compare([&](std::size_t i) { return static_cast<double>(truth[i]); }, worst, rel);
    EXPECT_GT(worst, 0.985) << "worst token cosine against the fp32 tower";
    EXPECT_LT(rel, 0.04) << "relative L2 against the fp32 tower";
    std::printf("vision tower vs fp32 truth: worst cosine %.4f, relative L2 %.4f\n", worst, rel);
  }
}

}  // namespace
