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

#ifndef XLS_IR_CHANNEL_H_
#define XLS_IR_CHANNEL_H_

#include <iosfwd>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/integral_types.h"
#include "xls/ir/channel.pb.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"

namespace xls {

// Describes a single data element conveyed by a channel. A channel can have
// more than one data element.
struct DataElement {
  std::string name;
  Type* type;
  // The initial values (if any) in the channel for this data element. All
  // DataElements in a channel must have the same number of elements.
  std::vector<Value> initial_values;

  std::string ToString() const;
};

std::ostream& operator<<(std::ostream& os, const DataElement& data_element);

// Data structure holding a set of concrete values communicated over a channel
// in one transaction.
using ChannelData = std::vector<Value>;

// Abstraction describing a channel in XLS IR. Channels are a mechanism for
// communicating between procs or between procs and components outside of
// XLS. Send and receive nodes in procs are associated with a particular
// channel. The channel data structure carries information about how
// communication occurs over the channel.
class Channel {
 public:
  // Indicates the type(s) of operations permitted on the channel. Send-only
  // channels can only have send operations (not receive) associated with the
  // channel as might be used for communicated to a component outside of
  // XLS. Receive-only channels are similarly defined. Send-receive channels can
  // have both send and receive operations and can be used for communicated
  // between procs.
  enum class SupportedOps { kSendOnly, kReceiveOnly, kSendReceive };

  Channel(absl::string_view name, int64 id, SupportedOps supported_ops,
          absl::Span<const DataElement> data_elements,
          std::vector<ChannelData>&& initial_values,
          const ChannelMetadataProto& metadata)
      : name_(name),
        id_(id),
        supported_ops_(supported_ops),
        data_elements_(data_elements.begin(), data_elements.end()),
        initial_values_(std::move(initial_values)),
        metadata_(metadata) {}

  virtual ~Channel() = default;

  // Returns the name of the channel.
  const std::string& name() const { return name_; }

  // Returns the ID of the channel. The ID is unique within the scope of a
  // package.
  int64 id() const { return id_; }

  // Returns the suppored ops for the channel: send-only, receive-only, or
  // send-receive.
  SupportedOps supported_ops() const { return supported_ops_; }

  // Returns the data elements communicated by the channel with each
  // transaction.
  absl::Span<const DataElement> data_elements() const { return data_elements_; }

  // Returns the i-th data element.
  const DataElement& data_element(int64 i) const {
    return data_elements_.at(i);
  }

  // Returns the initial values held in the channel. The inner span holds the
  // values across data elements. The outer span holds the entries in the
  // channel FIFO.
  const std::vector<ChannelData>& initial_values() const {
    return initial_values_;
  }

  // Returns the metadata associated with this channel.
  const ChannelMetadataProto& metadata() const { return metadata_; }

  // Returns whether this channel can be used to send (receive) data.
  bool CanSend() const {
    return supported_ops() == SupportedOps::kSendOnly ||
           supported_ops() == SupportedOps::kSendReceive;
  }
  bool CanReceive() const {
    return supported_ops() == SupportedOps::kReceiveOnly ||
           supported_ops() == SupportedOps::kSendReceive;
  }

  bool IsStreaming() const;
  bool IsSingleValue() const;

  std::string ToString() const;

 protected:
  std::string name_;
  int64 id_;
  SupportedOps supported_ops_;
  std::vector<DataElement> data_elements_;
  std::vector<ChannelData> initial_values_;
  ChannelMetadataProto metadata_;
};

// A channel with FIFO semantics. Send operations add an data entry to the
// channel; receives remove an element from the channel with FIFO ordering.
class StreamingChannel : public Channel {
 public:
  static absl::StatusOr<std::unique_ptr<StreamingChannel>> Create(
      absl::string_view name, int64 id, SupportedOps supported_ops,
      absl::Span<const DataElement> data_elements,
      const ChannelMetadataProto& metadata = ChannelMetadataProto());

 private:
  StreamingChannel(absl::string_view name, int64 id, SupportedOps supported_ops,
                   absl::Span<const DataElement> data_elements,
                   std::vector<ChannelData>&& initial_values,
                   const ChannelMetadataProto& metadata)
      : Channel(name, id, supported_ops, data_elements,
                std::move(initial_values), metadata) {}
};

// A channel which can hold a single element. Send operations overwrite the
// single element; receive operations non-destructively return the last value
// sent over the channel.
class SingleValueChannel : public Channel {
 public:
  static absl::StatusOr<std::unique_ptr<SingleValueChannel>> Create(
      absl::string_view name, int64 id, SupportedOps supported_ops,
      absl::Span<const DataElement> data_elements,
      const ChannelMetadataProto& metadata = ChannelMetadataProto());

 private:
  SingleValueChannel(absl::string_view name, int64 id,
                     SupportedOps supported_ops,
                     absl::Span<const DataElement> data_elements,
                     std::vector<ChannelData>&& initial_values,
                     const ChannelMetadataProto& metadata)
      : Channel(name, id, supported_ops, data_elements,
                std::move(initial_values), metadata) {}
};

// Returns the string representation of a supported ops enum.
std::string SupportedOpsToString(Channel::SupportedOps supported_ops);

// Converts the string representation of a channel to a SupportedOps. Returns an
// error if the string is not a representation.
absl::StatusOr<Channel::SupportedOps> StringToSupportedOps(
    absl::string_view str);

std::ostream& operator<<(std::ostream& os, Channel::SupportedOps supported_ops);

}  // namespace xls

#endif  // XLS_IR_CHANNEL_H_
