// Copyright 2020 Google LLC
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

#include "xls/synthesis/yosys/yosys_util.h"

#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "xls/common/status/ret_check.h"

namespace xls {
namespace synthesis {

absl::StatusOr<int64> ParseNextpnrOutput(absl::string_view nextpnr_output) {
  bool found = false;
  double max_mhz;
  // We're looking for lines of the form:
  //
  //   Info: Max frequency for clock 'foo': 125.28 MHz (PASS at 100.00 MHz)
  //
  // And we want to extract 125.28.
  // TODO(meheff): Use regular expressions for this. Unfortunately using RE2
  // causes multiple definition link errors when building yosys_server_test.
  for (auto line : absl::StrSplit(nextpnr_output, '\n')) {
    if (absl::StartsWith(line, "Info: Max frequency for clock") &&
        absl::StrContains(line, " MHz ")) {
      std::vector<absl::string_view> tokens = absl::StrSplit(line, ' ');
      for (int64 i = 1; i < tokens.size(); ++i) {
        if (tokens[i] == "MHz") {
          if (absl::SimpleAtod(tokens[i - 1], &max_mhz)) {
            found = true;
            break;
          }
        }
      }
    }
  }
  if (!found) {
    return absl::NotFoundError(
        "Could not find maximum frequency in nextpnr output.");
  }

  return static_cast<int64>(max_mhz * 1e6);
}

absl::StatusOr<YosysSynthesisStatistics> ParseYosysOutput(absl::string_view yosys_output) {
  YosysSynthesisStatistics stats;
  std::vector<std::string> lines = absl::StrSplit(yosys_output, '\n');
  std::vector<std::string>::iterator parse_line_itr = lines.begin();

  // Advance parse_line_index until a line containing 'key' is found.
  // Return false if 'key' is not found, otherwise true.
  auto parse_until_found = [&](absl::string_view key) {
    for(; parse_line_itr != lines.end(); ++parse_line_itr) {
      if(absl::StrContains(*parse_line_itr, key)) {
        break;
      }
    }
    return (parse_line_itr != lines.end());
  };

  // This function requies the top level module to have been identified
  // in order to work correctly (however, we do not need to parse
  // the name of the top level module).
  if(!parse_until_found("Top module:")) {
    return absl::FailedPreconditionError("ParseYosysOutput could not find the term \"Top module\" in the yosys output\"");
  }

  // Find the last printed statistics - these describe the whole design rather
  // than a single module.
  std::optional<std::vector<std::string>::iterator> last_num_cell_itr;
  while(parse_until_found("Number of cells:")) {
    last_num_cell_itr = parse_line_itr;
    ++parse_line_itr;
  }
  if(!last_num_cell_itr.has_value()) {
    return absl::InternalError("ParseYosysOutput could not find the term \"Number of cells:\" in the yosys output\"");
  }

  // Process cell histogram.
  for(parse_line_itr = last_num_cell_itr.value() + 1; parse_line_itr != lines.end(); ++parse_line_itr) {
    int64 cell_count;
    char cell_name[100];
    if(sscanf(parse_line_itr->c_str(), "%s %ld", cell_name, &cell_count) == 2) {
      std::string cell_name_str(cell_name);
      XLS_RET_CHECK(!stats.cell_histogram.contains(cell_name_str));
      stats.cell_histogram[cell_name_str] = cell_count;
    } else {
      break;
    }
  }

  return stats;
}

}  // namespace synthesis
}  // namespace xls
