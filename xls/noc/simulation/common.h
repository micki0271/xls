// Copyright 2020 The XLS Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef XLS_NOC_SIMULATION_COMMON_H_
#define XLS_NOC_SIMULATION_COMMON_H_

#include <limits>
#include <utility>

#include "absl/status/statusor.h"
#include "xls/common/integral_types.h"

namespace xls {
namespace noc {

// Used for all id components to denote an invalid id.
constexpr int kNullIdValue = -1;

// Each simulation network component has a kind defined here.
enum class NetworkComponentKind {
  kNone,    // A network component that is invalid or has not been configured.
  kRouter,  // A network router.
  kLink,    // A bundle of wires/registers between network components.
  kNISrc,   // A network interfrace (src) for ingress into the network.
  kNISink,  // A network interface (sink) for egress from the network.
};

// Simulation objects have ports and those ports can be either
// an input or output port.
enum class PortDirection {
  kInput,
  kOutput,
};

// Stores id of a single network.
class NetworkId {
 public:
  // Default constructor initializing this to an invalid id.
  constexpr NetworkId() : id_(kNullIdValue) {}

  // Constructor given an expanded id.
  constexpr explicit NetworkId(uint16 id) : id_(id) {}

  // Returns true if id is not kInvalid.
  bool IsValid();

  // Returns the local id.
  uint16 id() { return id_; }

  // Represent this as a 64-bit integer (used for logging/debug).
  uint64 AsUInt64() { return static_cast<uint64>(id_) << 48; }

  // Returns maximum index that can be encoded.
  static constexpr int64 MaxIndex() {
    return std::numeric_limits<decltype(id_)>::max() - 1;
  }

  // Validate and return id object given an unpacked id.
  static absl::StatusOr<NetworkId> ValidateAndReturnId(int64 id);

  // Hash.
  template <typename H>
  friend H AbslHashValue(H h, NetworkId id) {
    return H::combine(std::move(h), id.id_);
  }

  // Denotes an invalid network id.
  static const NetworkId kInvalid;

 private:
  uint16 id_;
};

// NetworkId equality.
inline bool operator==(NetworkId lhs, NetworkId rhs) {
  return lhs.id() == rhs.id();
}

// NetworkId inequality.
inline bool operator!=(NetworkId lhs, NetworkId rhs) {
  return lhs.id() != rhs.id();
}

inline bool NetworkId::IsValid() { return *this != kInvalid; }

// Default constructor is setup to initialize the id to invalid.
constexpr NetworkId NetworkId::kInvalid = NetworkId();

// Asserts that ids can fit within a 64-bit integer
// that way in terms of storage, an id is no worse than
// a 64-bit pointer.
static_assert(sizeof(NetworkId) <= 8 * sizeof(char));

// Id of a network component.
//
// An id is composed of the id of the network this component belongs to
// along with the component's local id.
class NetworkComponentId {
 public:
  // Default constructor initializing this to an invalid id.
  constexpr NetworkComponentId() : network_(kNullIdValue), id_(kNullIdValue) {}

  // Constructor given an non-expanded NetworkId.
  NetworkComponentId(NetworkId network, uint32 id)
      : network_(network.id()), id_(id) {}

  // Constructor given an expanded id.
  constexpr NetworkComponentId(uint16 network, uint32 id)
      : network_(network), id_(id) {}

  // Validate and return id object given an unpacked id.
  static absl::StatusOr<NetworkComponentId> ValidateAndReturnId(int64 network,
                                                                int64 id);

  // Returns true if id is not kInvalid.
  bool IsValid();

  // Returns the local id.
  uint32 id() { return id_; }

  // Returns the local network id.
  uint16 network() { return network_; }

  // Returns the NetworkId of this component.
  NetworkId GetNetworkId() { return NetworkId(network_); }

  // Represent this as a 64-bit integer (used for logging/debug).
  uint64 AsUInt64() {
    return (static_cast<uint64>(network_) << 48) |
           (static_cast<uint64>(id_) << 16);
  }

  // Returns maximum id/index that can be encoded.
  static constexpr int64 MaxIndex() {
    return std::numeric_limits<decltype(id_)>::max() - 1;
  }

  // Hash.
  template <typename H>
  friend H AbslHashValue(H h, NetworkComponentId id) {
    return H::combine(std::move(h), id.network_, id.id_);
  }

  // Denotes an invalid component id;
  static const NetworkComponentId kInvalid;

 private:
  // Store the raw unpacked network id so that these parent-id's can
  // be packed per efficiently (see PortId).
  uint16 network_;
  uint32 id_;
};

// NetworkComponentId equality.
inline bool operator==(NetworkComponentId lhs, NetworkComponentId rhs) {
  return (lhs.network() == rhs.network()) && (lhs.id() == rhs.id());
}

// NetworkComponentId inequality.
inline bool operator!=(NetworkComponentId lhs, NetworkComponentId rhs) {
  return !(lhs == rhs);
}

inline bool NetworkComponentId::IsValid() { return *this != kInvalid; }

// Default constructor is setup to initialize the id to invalid.
constexpr NetworkComponentId NetworkComponentId::kInvalid =
    NetworkComponentId();

// Asserts that ids can fit within a 64-bit integer
// that way in terms of storage, an id is no worse than
// a 64-bit pointer.
static_assert(sizeof(NetworkComponentId) <= 8 * sizeof(char));

// Id of a connection -- an edge between two ports.
//
// An id is composed of the id of the network this connectionbelongs to
// along with the connection's local id.
class ConnectionId {
 public:
  // Default constructor initializing this to an invalid id.
  constexpr ConnectionId() : network_(kNullIdValue), id_(kNullIdValue) {}

  // Constructor given an non-expanded NetworkId .
  ConnectionId(NetworkId network, uint32 id)
      : network_(network.id()), id_(id) {}

  // Constructor given an expanded id.
  constexpr ConnectionId(uint16 network, uint32 id)
      : network_(network), id_(id) {}

  // Validate and return id object given an unpacked id.
  static absl::StatusOr<ConnectionId> ValidateAndReturnId(int64 network,
                                                          int64 id);

  // Returns true if id is not kInvalid.
  bool IsValid();

  // Returns the local id.
  uint32 id() { return id_; }

  // Returns the local network id.
  uint16 network() { return network_; }

  // Returns the NetworkId of this component.
  NetworkId GetNetworkId() { return NetworkId(network_); }

  // Represent this as a 64-bit integer (used for logging/debug).
  uint64 AsUInt64() {
    return (static_cast<uint64>(network_) << 48) |
           (static_cast<uint64>(id_) << 16);
  }

  // Returns maximum id/index that can be encoded.
  static constexpr int64 MaxIndex() {
    return std::numeric_limits<decltype(id_)>::max() - 1;
  }

  // Hash.
  template <typename H>
  friend H AbslHashValue(H h, ConnectionId id) {
    return H::combine(std::move(h), id.network_, id.id_);
  }

  // Denotes an invalid connection id.
  static const ConnectionId kInvalid;

 private:
  uint16 network_;
  uint32 id_;
};

// ConnectionId equality.
inline bool operator==(ConnectionId lhs, ConnectionId rhs) {
  return (lhs.network() == rhs.network()) && (lhs.id() == rhs.id());
}

// ConnectionId inequality.
inline bool operator!=(ConnectionId lhs, ConnectionId rhs) {
  return !(lhs == rhs);
}

inline bool ConnectionId::IsValid() { return *this != kInvalid; }

// Default constructor is setup to initialize the id to invalid.
constexpr ConnectionId ConnectionId::kInvalid = ConnectionId();

// Asserts that ids can fit within a 64-bit integer
// that way in terms of storage, an id is no worse than
// a 64-bit pointer.
static_assert(sizeof(ConnectionId) <= 8 * sizeof(char));

// Stores id of a port.
//
// An id is composed of the id of the network, the component
// this port belongs to, along with this port's local id.
struct PortId {
  // Default constructor initializing this to an invalid id.
  constexpr PortId()
      : id_(kNullIdValue), network_(kNullIdValue), component_(kNullIdValue) {}

  // Constructor given an non-expanded NetworkComponentId.
  PortId(NetworkComponentId component, uint16 id)
      : id_(id),
        network_(component.GetNetworkId().id()),
        component_(component.id()) {}

  // Constructor given an expanded id.
  constexpr PortId(uint16 network, uint32 component, uint16 id)
      : id_(id), network_(network), component_(component) {}

  // Returns true if id is not kInvalid.
  bool IsValid();

  // Validate and return id object given an unpacked id.
  static absl::StatusOr<PortId> ValidateAndReturnId(int64 network,
                                                    int64 component, int64 id);

  // Returns the local id.
  uint16 id() { return id_; }

  // Returns the local network id.
  uint16 network() { return network_; }

  // Returns the local component id.
  uint32 component() { return component_; }

  // Returns the NetworkId of this port.
  NetworkId GetNetworkId() { return NetworkId(network_); }

  // Returns the NetworkComponentId of this port.
  NetworkComponentId GetNetworkComponentId() {
    return NetworkComponentId(network_, component_);
  }

  // Represent this as a 64-bit integer (used for logging/debug).
  uint64 AsUInt64() {
    return (static_cast<uint64>(network_) << 48) |
           (static_cast<uint64>(component_) << 16) | (static_cast<uint64>(id_));
  }

  // Returns maximum id/index that can be encoded.
  static constexpr int64 MaxIndex() {
    return std::numeric_limits<decltype(id_)>::max() - 1;
  }

  // Hash.
  template <typename H>
  friend H AbslHashValue(H h, PortId id) {
    return H::combine(std::move(h), id.network_, id.component_, id.id_);
  }

  // An invalid port id.
  static const PortId kInvalid;

 private:
  uint16 id_;
  uint16 network_;
  uint32 component_;
};

// PortId equality.
inline bool operator==(PortId lhs, PortId rhs) {
  return (lhs.id() == rhs.id()) && (lhs.network() == rhs.network()) &&
         (lhs.component() == rhs.component());
}

// PortId inequality.
inline bool operator!=(PortId lhs, PortId rhs) { return !(lhs == rhs); }

inline bool PortId::IsValid() { return *this != kInvalid; }

// Default constructor is setup to initialize the id to invalid.
constexpr PortId PortId::kInvalid = PortId();

// Asserts that ids can fit within a 64-bit integer
// that way in terms of storage, an id is no worse than
// a 64-bit pointer.
static_assert(sizeof(PortId) <= 8 * sizeof(char));

}  // namespace noc
}  // namespace xls

#endif  // XLS_NOC_SIMULATION_COMMON_H_
