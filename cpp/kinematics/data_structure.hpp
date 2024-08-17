#include <vector>

namespace tinyfk {

template <class DataT> class SizedCache {
public:
  explicit SizedCache(size_t cache_size)
      : cache_size_(cache_size), data_(std::vector<DataT>(cache_size)),
        cache_predicate_vector_(std::vector<bool>(cache_size, false)) {}

  SizedCache() : SizedCache(0) {}

  void set_cache(size_t id, const DataT &data) {
    cache_predicate_vector_[id] = true;
    data_[id] = data;
  }

  DataT const *get_cache(size_t id) const {
    const bool isAlreadyCached = (cache_predicate_vector_[id] == true);
    if (!isAlreadyCached) {
      return nullptr;
    } // the cache does not exists
    return const_cast<DataT const *>(&data_[id]);
  }

  void extend() {
    cache_size_++;
    data_.push_back(DataT());
    cache_predicate_vector_.push_back(false);
    this->clear();
  }
  void clear() { cache_predicate_vector_ = std::vector<bool>(cache_size_); }

private:
  int cache_size_;
  std::vector<DataT> data_;
  std::vector<bool> cache_predicate_vector_;
};

template <class ElementT> class SizedStack {
public:
  SizedStack() : SizedStack(0) {}
  SizedStack(size_t max_stack_size)
      : data_(std::vector<ElementT>(max_stack_size)), current_idx_(0) {}

  inline size_t size() const { return current_idx_; }
  inline bool empty() const { return current_idx_ == 0; }
  inline void reset() { current_idx_ = 0; }
  inline void push(const ElementT &elem) {
    data_[current_idx_] = elem;
    current_idx_++;
  }
  inline ElementT &top() const {
    return const_cast<ElementT &>(data_[current_idx_ - 1]);
  }
  inline void pop() { current_idx_--; }

private:
  std::vector<ElementT> data_;
  size_t current_idx_;
};

} // namespace tinyfk
