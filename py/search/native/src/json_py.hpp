#pragma once

#include "../third_party/nlohmann/json.hpp"

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace py {

using json = nlohmann::json;

class cast_error : public std::runtime_error {
 public:
  explicit cast_error(const std::string& message) : std::runtime_error(message) {}
};

class object {
 public:
  object() : owner_(std::make_shared<json>(nullptr)), value_(owner_.get()) {}
  object(std::nullptr_t) : object() {}
  object(const char* value) : owner_(std::make_shared<json>(value == nullptr ? json(nullptr) : json(value))), value_(owner_.get()) {}
  object(const std::string& value) : owner_(std::make_shared<json>(value)), value_(owner_.get()) {}
  object(bool value) : owner_(std::make_shared<json>(value)), value_(owner_.get()) {}
  object(int value) : owner_(std::make_shared<json>(value)), value_(owner_.get()) {}
  object(long value) : owner_(std::make_shared<json>(value)), value_(owner_.get()) {}
  object(long long value) : owner_(std::make_shared<json>(value)), value_(owner_.get()) {}
  object(unsigned long value) : owner_(std::make_shared<json>(value)), value_(owner_.get()) {}
  object(unsigned long long value) : owner_(std::make_shared<json>(value)), value_(owner_.get()) {}
  object(double value) : owner_(std::make_shared<json>(value)), value_(owner_.get()) {}
  template <typename T>
  explicit object(const std::vector<T>& values) : owner_(std::make_shared<json>(json::array())), value_(owner_.get()) {
    for (const T& value : values) {
      data().push_back(value);
    }
  }
  explicit object(const json& value) : owner_(std::make_shared<json>(value)), value_(owner_.get()) {}
  object(std::shared_ptr<json> owner, json* value) : owner_(std::move(owner)), value_(value == nullptr ? owner_.get() : value) {}

  bool is_none() const { return value_ == nullptr || value_->is_null(); }
  json& data() const { return *value_; }
  std::shared_ptr<json> owner() const { return owner_; }

  object& operator=(const object& other) {
    data() = other.data();
    return *this;
  }
  object& operator=(const char* value) {
    data() = value == nullptr ? json(nullptr) : json(value);
    return *this;
  }
  object& operator=(const std::string& value) {
    data() = value;
    return *this;
  }
  object& operator=(bool value) {
    data() = value;
    return *this;
  }
  object& operator=(int value) {
    data() = value;
    return *this;
  }
  object& operator=(long value) {
    data() = value;
    return *this;
  }
  object& operator=(long long value) {
    data() = value;
    return *this;
  }
  object& operator=(unsigned long value) {
    data() = value;
    return *this;
  }
  object& operator=(unsigned long long value) {
    data() = value;
    return *this;
  }
  object& operator=(double value) {
    data() = value;
    return *this;
  }
  template <typename T>
  object& operator=(const std::vector<T>& values) {
    data() = json::array();
    for (const T& value : values) {
      data().push_back(value);
    }
    return *this;
  }

  object operator[](const char* key) const { return get_key(key == nullptr ? "" : std::string(key)); }
  object operator[](const std::string& key) const { return get_key(key); }
  object operator[](const object& key) const { return get_key(key.as_key()); }
  object operator[](int index) const { return get_index(static_cast<size_t>(index)); }
  object operator[](size_t index) const { return get_index(index); }

  bool contains(const char* key) const { return contains(std::string(key == nullptr ? "" : key)); }
  bool contains(const std::string& key) const {
    return data().is_object() && data().contains(key);
  }
  bool contains(const object& key) const { return contains(key.as_key()); }

  class attr_proxy {
   public:
    attr_proxy(std::shared_ptr<json> owner, json* target, std::string name) :
        owner_(std::move(owner)),
        target_(target),
        name_(std::move(name)) {}
    void operator()(const char* key) const {
      if (name_ == "pop" && target_ != nullptr && target_->is_object() && key != nullptr) {
        target_->erase(key);
      }
    }

   private:
    std::shared_ptr<json> owner_;
    json* target_ = nullptr;
    std::string name_;
  };

  attr_proxy attr(const char* name) const { return attr_proxy(owner_, value_, name == nullptr ? "" : std::string(name)); }

  std::string as_key() const {
    if (data().is_string()) return data().get<std::string>();
    if (data().is_number_integer()) return std::to_string(data().get<long long>());
    if (data().is_number_unsigned()) return std::to_string(data().get<unsigned long long>());
    if (data().is_number_float()) return std::to_string(data().get<double>());
    if (data().is_boolean()) return data().get<bool>() ? "true" : "false";
    if (data().is_null()) return "";
    return data().dump();
  }

 private:
  object get_key(const std::string& key) const {
    if (!data().is_object()) {
      data() = json::object();
    }
    return object(owner_, &((*value_)[key]));
  }

  object get_index(size_t index) const {
    if (!data().is_array()) {
      data() = json::array();
    }
    if (data().size() <= index) {
      while (data().size() <= index) {
        data().push_back(nullptr);
      }
    }
    return object(owner_, &((*value_)[index]));
  }

  std::shared_ptr<json> owner_;
  mutable json* value_ = nullptr;
};

using handle = object;

class str : public object {
 public:
  str() : object(std::string()) {}
  str(const char* value) : object(value == nullptr ? std::string() : std::string(value)) {}
  str(const std::string& value) : object(value) {}
  str(const object& value) : object(stringify(value)) {}

 private:
  static std::string stringify(const object& value) {
    if (value.data().is_string()) return value.data().get<std::string>();
    if (value.data().is_null()) return "None";
    if (value.data().is_boolean()) return value.data().get<bool>() ? "True" : "False";
    if (value.data().is_number_integer()) return std::to_string(value.data().get<long long>());
    if (value.data().is_number_unsigned()) return std::to_string(value.data().get<unsigned long long>());
    if (value.data().is_number_float()) return std::to_string(value.data().get<double>());
    return value.data().dump();
  }
};

class int_ : public object {
 public:
  int_(int value) : object(value) {}
  int_(long long value) : object(value) {}
  int_(unsigned long long value) : object(value) {}
};

class float_ : public object {
 public:
  float_(double value) : object(value) {}
};

class bool_ : public object {
 public:
  bool_(bool value) : object(value) {}
};

class list : public object {
 public:
  list() : object(json::array()) {}
  list(const object& value) : object(value.owner(), const_cast<json*>(&value.data())) {
    if (!data().is_array()) data() = json::array();
  }
  explicit list(const json& value) : object(value.is_array() ? value : json::array()) {}

  void append(const object& value) { data().push_back(value.data()); }
  void append(const char* value) { data().push_back(value == nullptr ? json(nullptr) : json(value)); }
  void append(int value) { data().push_back(value); }
  void append(double value) { data().push_back(value); }

  class iterator {
   public:
    iterator(std::shared_ptr<json> owner, json::iterator it) : owner_(std::move(owner)), it_(it) {}
    iterator& operator++() {
      ++it_;
      return *this;
    }
    bool operator!=(const iterator& other) const { return it_ != other.it_; }
    object operator*() const { return object(owner_, &(*it_)); }

   private:
    std::shared_ptr<json> owner_;
    json::iterator it_;
  };

  iterator begin() const { return iterator(owner(), const_cast<json&>(data()).begin()); }
  iterator end() const { return iterator(owner(), const_cast<json&>(data()).end()); }
};

class tuple : public list {
 public:
  tuple() : list() {}
  tuple(const object& value) : list(value) {}
};

class dict : public object {
 public:
  dict() : object(json::object()) {}
  dict(const object& value) : object(value.owner(), const_cast<json*>(&value.data())) {
    if (!data().is_object()) data() = json::object();
  }
  explicit dict(const json& value) : object(value.is_object() ? value : json::object()) {}

  class item {
   public:
    item(std::shared_ptr<json> owner, json::iterator it) :
        first(str(it.key())),
        second(object(std::move(owner), &(*it))) {}
    object first;
    object second;
  };

  class iterator {
   public:
    iterator(std::shared_ptr<json> owner, json::iterator it) : owner_(std::move(owner)), it_(it) {}
    iterator& operator++() {
      ++it_;
      return *this;
    }
    bool operator!=(const iterator& other) const { return it_ != other.it_; }
    item operator*() const { return item(owner_, it_); }

   private:
    std::shared_ptr<json> owner_;
    json::iterator it_;
  };

  iterator begin() const { return iterator(owner(), const_cast<json&>(data()).begin()); }
  iterator end() const { return iterator(owner(), const_cast<json&>(data()).end()); }
};

inline object none() { return object(nullptr); }

inline size_t len(const object& value) {
  return value.data().is_array() || value.data().is_object() || value.data().is_string()
      ? value.data().size()
      : 0;
}

template <typename T>
bool isinstance(const object& value) {
  if constexpr (std::is_same_v<T, dict>) return value.data().is_object();
  if constexpr (std::is_same_v<T, list>) return value.data().is_array();
  if constexpr (std::is_same_v<T, tuple>) return value.data().is_array();
  if constexpr (std::is_same_v<T, str>) return value.data().is_string();
  if constexpr (std::is_same_v<T, int_>) return value.data().is_number_integer() || value.data().is_number_unsigned();
  if constexpr (std::is_same_v<T, float_>) return value.data().is_number();
  if constexpr (std::is_same_v<T, bool_>) return value.data().is_boolean();
  return false;
}

template <typename T>
T reinterpret_borrow(const object& value) {
  if constexpr (std::is_same_v<T, dict>) return dict(value);
  if constexpr (std::is_same_v<T, list>) return list(value);
  if constexpr (std::is_same_v<T, tuple>) return tuple(value);
  return T(value);
}

template <typename T>
T cast(const object& value) {
  try {
    if constexpr (std::is_same_v<T, int>) {
      if (value.data().is_number_integer()) return value.data().get<int>();
      if (value.data().is_number()) return static_cast<int>(value.data().get<double>());
      if (value.data().is_string()) return std::stoi(value.data().get<std::string>());
    } else if constexpr (std::is_same_v<T, long long>) {
      if (value.data().is_number_integer()) return value.data().get<long long>();
      if (value.data().is_number()) return static_cast<long long>(value.data().get<double>());
      if (value.data().is_string()) return std::stoll(value.data().get<std::string>());
    } else if constexpr (std::is_same_v<T, double>) {
      if (value.data().is_number()) return value.data().get<double>();
      if (value.data().is_string()) return std::stod(value.data().get<std::string>());
    } else if constexpr (std::is_same_v<T, bool>) {
      if (value.data().is_boolean()) return value.data().get<bool>();
      if (value.data().is_number_integer()) return value.data().get<int>() != 0;
    } else if constexpr (std::is_same_v<T, std::string>) {
      if (value.data().is_string()) return value.data().get<std::string>();
      if (value.data().is_null()) return "";
      return value.data().dump();
    } else if constexpr (std::is_same_v<T, dict>) {
      return dict(value);
    } else if constexpr (std::is_same_v<T, list>) {
      return list(value);
    }
  } catch (const std::exception& exc) {
    throw cast_error(exc.what());
  }
  throw cast_error("unsupported standalone cast");
}

template <typename... Args>
tuple make_tuple(Args&&... args) {
  tuple out;
  (out.append(object(std::forward<Args>(args))), ...);
  return out;
}

inline json to_json(const object& value) {
  return value.data();
}

}  // namespace py
