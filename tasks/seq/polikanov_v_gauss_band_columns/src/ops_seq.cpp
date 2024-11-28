#include "seq/polikanov_v_gauss_band_columns/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

bool polikanov_v_gauss_band_columns_seq::GaussBandColumnsSequential::validation() {
  internal_order_test();

  size_t val_n = *reinterpret_cast<size_t*>(taskData->inputs[1]);
  size_t val_mat_size = taskData->inputs_count[0];

  return val_n >= 2 && val_mat_size == (val_n * (val_n + 1));
}

bool polikanov_v_gauss_band_columns_seq::GaussBandColumnsSequential::pre_processing() {
  internal_order_test();

  auto* matrix_data = reinterpret_cast<double*>(taskData->inputs[0]);
  int matrix_size = taskData->inputs_count[0];
  n = *reinterpret_cast<size_t*>(taskData->inputs[1]);

  std::vector<double> matrix(n, n + 1);
  matrix.assign(matrix_data, matrix_data + matrix_size);

  mat = Matrix(matrix, n);
  answers.resize(n * (n + 1));

  return true;
}

bool polikanov_v_gauss_band_columns_seq::GaussBandColumnsSequential::run() {
  internal_order_test();

  size_t n = mat.get_rows();

  for (size_t k = 0; k < n - 1; ++k) {
    Matrix iter_mat = mat.submatrix(k, k);

    std::vector<double> factors;
    try {
      factors = iter_mat.calculate_elimination_factors();
    } catch (const std::logic_error& e) {
      std::cerr << "\nError: " << e.what() << std::endl;
      return false;
    }

    for (size_t i = 1; i < iter_mat.get_rows(); ++i) {
      double factor = factors[i - 1];
      for (size_t j = 0; j < iter_mat.get_cols(); ++j) {
        iter_mat.at(i, j) -= factor * iter_mat.at(0, j);
      }
      iter_mat.at(i, 0) = 0.0;
    }
  }

  try {
    answers = mat.backward_substitution();
  } catch (const std::exception& e) {
    std::cerr << "\nError: " << e.what() << std::endl;
    return false;
  }

  return true;
}

bool polikanov_v_gauss_band_columns_seq::GaussBandColumnsSequential::post_processing() {
  internal_order_test();

  auto* output_data = reinterpret_cast<double*>(taskData->outputs[0]);
  std::copy(answers.begin(), answers.end(), output_data);

  return true;
}
