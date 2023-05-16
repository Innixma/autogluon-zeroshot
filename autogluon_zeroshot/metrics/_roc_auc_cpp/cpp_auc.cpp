#include <array>
#include <vector>


/*
 * This function is a faster single-threaded implementation of the binary non-weighted ROC AUC metric.
 * The primary speed-up come from a radix sort on the 23 bits of the mantissa of float values.
 * This function makes several assumptions:
 *   - y_true and y_pred are both of size len
 *   - y_pred values are in range [0.0, 1.0)
 * If these assumptions are violated, the function may crash or return incorrect results.
 */

double radix_roc_auc(bool* y_true, float* y_pred, size_t len) {

  std::vector<uint32_t> storage1(len);
  std::vector<uint32_t> storage2(len);

  // I use 3 custom sized radixes here: 5, 9, and 9, which I found to be a significant speedup over 3x 8bit radixes.
  // These were chosen based on testing various radix combinations on an m6i.x32 instance.
  const int radix0_split = 5;
  constexpr int radix0_size = 1 << radix0_split;
  constexpr int radix0_size_minus_1 = radix0_size - 1;

  const int radix1_split = 9;
  constexpr int radix1_size = 1 << radix1_split;
  constexpr int radix1_size_minus_1 = radix1_size - 1;

  const int radix2_split = 23 - radix0_split - radix1_split;
  constexpr int radix2_size = 1 << radix2_split;
  constexpr int radix2_size_minus_1 = radix2_size - 1;

  std::array<uint32_t, radix0_size> histogram0 = {0};
  std::array<uint32_t, radix1_size> histogram1 = {0};
  std::array<uint32_t, radix2_size> histogram2 = {0};

  // Add 1.0 to all floats so they are in range [1.0,2.0) to exploit IEEE-754 float memory layout
  for (size_t i = 0; i < len; ++i) {
    y_pred[i] += 1.0;
  }

  // Interpret float32 as uint32. Put bool in 8 MSBs and float mantissa in 23 LSBs of the uint32. Save to storage vector.
  for (size_t i = 0; i < len; ++i) {
    uint32_t int_entry = *reinterpret_cast<uint32_t*>(&y_pred[i]);
    storage1[i] = (static_cast<uint32_t>(y_true[i]) << 24) | (int_entry & 0x7FFFFF);
  }

  // Populate 3 histograms for radixes
  for (size_t i = 0; i < len; ++i) {
    histogram0[storage1[i] & radix0_size_minus_1]++;
    histogram1[storage1[i] >> radix0_split & radix1_size_minus_1]++;
    histogram2[storage1[i] >> (radix0_split + radix1_split) & radix2_size_minus_1]++;
  }

  // Add incremental counts to each histogram
  uint32_t radix0_cnt = 0;
  uint32_t temp0;

  for (size_t i = 0; i < radix0_size; ++i) {
    temp0 = histogram0[i] + radix0_cnt;
    histogram0[i] = radix0_cnt - 1;
    radix0_cnt = temp0;
  }

  uint32_t radix1_cnt = 0;
  uint32_t temp1;
  for (size_t i = 0; i < radix1_size; ++i) {
    temp1 = histogram1[i] + radix1_cnt;
    histogram1[i] = radix1_cnt - 1;
    radix1_cnt = temp1;
  }

  uint32_t radix2_cnt = 0;
  uint32_t temp2;
  for (size_t i = 0; i < radix2_size; ++i) {
    temp2 = histogram2[i] + radix2_cnt;
    histogram2[i] = radix2_cnt - 1;
    radix2_cnt = temp2;
  }

  // Sort radix0 (least significant bits)
  for (size_t i = 0; i < len; ++i) {
    uint32_t entry = storage1[i];
    storage2[++histogram0[entry & radix0_size_minus_1]] = entry;
  }

  // Sort radix1
  for (size_t i = 0; i < len; ++i) {
    uint32_t entry = storage2[i];
    storage1[++histogram1[entry >> radix0_split & radix1_size_minus_1]] = entry;
  }

  // Sort radix2 (most significant bits)
  for (size_t i = 0; i < len; ++i) {
    uint32_t entry = storage1[i];
    storage2[++histogram2[entry >> (radix0_split + radix1_split) & radix2_size_minus_1]] = entry;
  }

  // Perform binary non-weighted roc_auc summation on sorted entries
  uint64_t true_cnt = 0;
  uint64_t false_cnt = 0;
  uint64_t auc = 0;

  for (size_t i = 0; i < len; ++i) {
    uint32_t entry = storage2[len - 1 - i];
    bool label = entry >> 24 & 0xFF;

    true_cnt += label;
    false_cnt += !label;
    auc += !label * true_cnt;
  }

  return static_cast<double>(auc) / (true_cnt * false_cnt);
}

extern "C" {
    double cpp_auc_ext(float* ts, bool* st, size_t len) {
        return radix_roc_auc(st, ts, len);
    }
}

