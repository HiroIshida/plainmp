/*
 * plainmp - library for fast motion planning
 *
 * Copyright (C) 2024 Hirokazu Ishida
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include <ompl/geometric/planners/PlannerIncludes.h>
#include <optional>

namespace plainmp::ompl_wrapper {

std::shared_ptr<ompl::base::Planner> get_algorithm(
    const std::string& name,
    const ompl::base::SpaceInformationPtr& si,
    std::optional<double> range = std::nullopt);

}  // namespace plainmp::ompl_wrapper
