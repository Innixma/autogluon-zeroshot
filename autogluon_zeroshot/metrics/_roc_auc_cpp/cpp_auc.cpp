#include <array>
#include <vector>


double radix_roc_auc(bool* y_true, float* y_pred, size_t len) {

  std::vector<unsigned long> storage1(len);
  std::vector<unsigned long> storage2(len);

  std::array<unsigned long, 256 * 3> histograms = {0};

  // Add 1 to all floats so they are in range [1.0,2.0) to exploit IEEE-754 float memory layout
  for (size_t i = 0; i < len; ++i) {
    y_pred[i] += 1.0;
  }

  // Interpret float32 as uint32. Put bool in 8 MSB and float mantissa in 23 LBS's of the uint32. Move to storage vector.
  for (size_t i = 0; i < len; ++i) {
    unsigned long int_entry = *reinterpret_cast<unsigned long*>(&y_pred[i]);
    storage1[i] = (static_cast<unsigned long>(y_true[i]) << 24) | (int_entry & 0x7FFFFF);
  }

  // Fill 3 histograms for 8-bit radix's
  for (size_t i = 0; i < len; ++i) {
    histograms[storage1[i] & 0xFF]++;
    histograms[256 + (storage1[i] >> 8 & 0xFF)]++;
    histograms[512 + (storage1[i] >> 16 & 0xFF)]++;
  }

  // Add incremental counts to each histogram entry
  unsigned long radix0_cnt = 0;
  unsigned long radix1_cnt = 0;
  unsigned long radix2_cnt = 0;
  unsigned long temp;

  for (size_t i = 0; i < 256; ++i) {
    temp = histograms[i] + radix0_cnt;
    histograms[i] = radix0_cnt - 1;
    radix0_cnt = temp;

    temp = histograms[i + 256] + radix1_cnt;
    histograms[i + 256] = radix1_cnt - 1;
    radix1_cnt = temp;

    temp = histograms[i + 512] + radix2_cnt;
    histograms[i + 512] = radix2_cnt - 1;
    radix2_cnt = temp;
  }

  // Sort radix0 (least significant bit)
  for (unsigned long i = 0; i < len; ++i) {
    unsigned long entry = storage1[i];
    storage2[++histograms[entry & 0xFF]] = entry;
  }

  // Sort radix1
  for (unsigned long i = 0; i < len; ++i) {
    unsigned long entry = storage2[i];
    storage1[++histograms[256 + (entry >> 8 & 0xFF)]] = entry;
  }

  // Sort radix2 (most significant bit)
  for (unsigned long i = 0; i < len; ++i) {
    unsigned long entry = storage1[i];
    storage2[++histograms[512 + (entry >> 16 & 0xFF)]] = entry;
  }

  // Perform binary non-weighted roc_auc calulation
  unsigned long true_cnt = 0;
  unsigned long false_cnt = 0;
  unsigned long auc = 0;

  for (unsigned long i = 0; i < len; ++i) {
    unsigned long entry = storage2[len - 1 - i];
    bool label = entry >> 24 & 0xFF;

    true_cnt += label;
    false_cnt += !label;
    auc += !label * true_cnt;
  }

  return (double) auc / (true_cnt * false_cnt);
}

extern "C" {
    double cpp_auc_ext(float* ts, bool* st, size_t len) {
        return radix_roc_auc(st, ts, len);
    }
}

