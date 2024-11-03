#include "mpi/polikanov_v_max_of_vector_elements/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<int> polikanov_v_max_of_vector_elements::getRandomVector(int sz, int lower, int upper) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = lower + gen() % (upper - lower + 1);
  }
  vec[sz - 1] = upper;
  return vec;
}

bool polikanov_v_max_of_vector_elements::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  int count = static_cast<int>(taskData->inputs_count[0]);
  int* input = reinterpret_cast<int*>(taskData->inputs[0]);
  input_ = std::vector<int>(count);
  for (int i = 0; i < count; ++i) {
    input_[i] = input[i];
  }
  res = INT_MIN;
  return true;
}

bool polikanov_v_max_of_vector_elements::TestMPITaskSequential::validation() {
  internal_order_test();
  return (taskData->inputs_count[0] == 0 && taskData->outputs_count[0] == 0) || taskData->outputs_count[0] == 1;
}

bool polikanov_v_max_of_vector_elements::TestMPITaskSequential::run() {
  internal_order_test();
  int count = input_.size();
  for (int i = 0; i < count; i++) {
    res = std::max(res, input_[i]);
  }
  return true;
}

bool polikanov_v_max_of_vector_elements::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}

bool polikanov_v_max_of_vector_elements::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  res = INT_MIN;
  return true;
}

bool polikanov_v_max_of_vector_elements::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool polikanov_v_max_of_vector_elements::TestMPITaskParallel::run() {
  internal_order_test();
  int n = 0;

  if (world.rank() == 0) {
    n = taskData->inputs_count[0];
    auto* input_ptr = reinterpret_cast<int32_t*>(taskData->inputs[0]);
    input_.assign(input_ptr, input_ptr + n);
  }
  boost::mpi::broadcast(world, n, 0);
  int local_size = n / world.size() + (world.rank() < (n % world.size()) ? 1 : 0);
  std::vector<int> vec1(world.size(), n / world.size());
  std::vector<int> vec2(world.size(), 0);

  for (int i = 0; i < n % world.size(); ++i) {
    vec1[i]++;
  }
  for (int i = 0; i < world.size() - 1; ++i) {
    vec2[i] += vec1[i];
  }

  local_input_.resize(vec1[world.rank()]);
  boost::mpi::scatterv(world, input_.data(), vec1, vec2, local_input_.data(), local_size, 0);

  int cur_max = std::numeric_limits<int>::min();
  for (int num : local_input_) {
    if (num > cur_max) {
      cur_max = num;
    }
  }
  boost::mpi::reduce(world, cur_max, res, boost::mpi::maximum<int>(), 0);

  return true;
}

bool polikanov_v_max_of_vector_elements::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  }
  return true;
}
