#ifndef LIB_RUNTIME_CLEARTEXTRESOURCE_H_
#define LIB_RUNTIME_CLEARTEXTRESOURCE_H_

#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

namespace heir {

// Loads the little-endian raw representation written by
// --externalize-constants into caller-owned, statically shaped storage.
template <typename T>
void loadResource(const std::string& path, T* data, std::size_t size) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Failed to open resource: " << path << '\n';
    std::abort();
  }
  file.read(reinterpret_cast<char*>(data), size * sizeof(T));
  if (!file) {
    std::cerr << "Failed to read expected bytes from resource: " << path
              << '\n';
    std::abort();
  }
}

}  // namespace heir

#endif  // LIB_RUNTIME_CLEARTEXTRESOURCE_H_
