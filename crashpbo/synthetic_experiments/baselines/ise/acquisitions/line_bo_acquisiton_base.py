# Copyright (c) 2022 Robert Bosch GmbH
# Author: Alessandro G. Bottero
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import time
import warnings

import torch

from ..utils import generic_utils

class LineBoAcquisitionBase:
    def __init__(
        self,
        safe_seed,
        domain,
        target_candidates=5,
        max_sampling_rounds=None,
        timeout_seconds=None,
    ):
        '''
        Constructor

        Parameters
        ----------
        safe_seed (torch.Tensor): initial safe seed
        domain (list of pairs of floats): list of the coordinates of the domain's vertices
        target_candidates (int): number of candidate points to gather before selecting the best one
        max_sampling_rounds (int or None): maximum number of sampling rounds before giving up (None disables limit)
        timeout_seconds (float or None): wall-clock time budget for the optimizer (None disables timeout)
        '''

        self._safe_seed = safe_seed
        self._last_sampled_point = safe_seed
        self._domain = domain
        self._target_candidates = target_candidates
        self._max_sampling_rounds = max_sampling_rounds
        self._timeout_seconds = timeout_seconds

    def _get_subspace_bounds(self, origin, normalized_direction):
        '''
        Compute how much one can move along normalized_direction starting from origin while still remaining
        inside the domain
        
        Parameters
        ----------
        origin (torch.Tensor): origin point from which to compute the maximum allowed displacement
        normalized_direction (torch.Tensor): vector defining units of dispacements from origin

        Returns
        -------
        (pair of torch.Tensor [(a, b)]) respectively maximum negative and positive multiples of normalized_direction
        that can be added to origin and still remaining inside the domain
        '''
        
        maximum_displacements_from_origin = []
        origin = origin.squeeze()
        normalized_direction = normalized_direction.squeeze()
        for dimension in range(len(self._domain)):
            if normalized_direction[dimension] == 0:
                continue
            maximum_left_displacement = \
                (self._domain[dimension][0] - origin[dimension]) / normalized_direction[dimension]
            maximum_right_displacement = \
                (self._domain[dimension][1] - origin[dimension]) / normalized_direction[dimension]
            maximum_displacements_from_origin.append(maximum_left_displacement)
            maximum_displacements_from_origin.append(maximum_right_displacement)
        maximum_displacements_from_origin = torch.tensor(maximum_displacements_from_origin)

        positive_displacements = maximum_displacements_from_origin[maximum_displacements_from_origin > 0]
        negative_displacements = maximum_displacements_from_origin[maximum_displacements_from_origin < 0]

        return [(torch.max(negative_displacements).item(), torch.min(positive_displacements).item())]


    def _one_d_samples_to_full_domain(self, one_d_samples, origin, direction):
        '''
        Re-embed one-dimensional point(s) along line passing through origin and with direction direction within the
        full d-dimensional domain
        
        Parameters
        ----------
        one_d_samples (torch.Tensor): 1-d coordinates of samples to re-embed in full domain. The coordinates are the
        number 'k' in 'sample = origin + k * direction'
        origin (torch.Tensor): Origin of the 1-d subspace the samples belong to
        direction (torch.Tensor): Direction of the 1-d subspace the samples belong to

        Returns
        -------
        (torch.Tensor): Re-embedded one_d_samples as d-dimensional points
        '''
        
        return origin + direction * one_d_samples


    def _get_normalized_direction(self, origin, point_for_direction):
        '''
        Computes the normalized direction of the line passing through origin and point_for_direction
        
        Parameters
        ----------
        origin (torch.Tensor): One of the two points used to calculate the direction
        point_for_direction (torch.Tensor): The other point used to calculate the direction

        Returns
        -------
        (torch.Tensor): Normalized vector representing the direction of the line passing through origin and
        point_for_direction
        '''

        direction = (point_for_direction - origin)
        return direction / torch.norm(direction)


    def _optimize(self, find_argmax_location):
        '''
        Find good candidate optimizer for the acquisition function along multiple 1d subsets of the domain
        
        Parameters
        ----------
        find_argmax_location (callable) function that optimizes a  custom objective along a 1-d line

        Returns
        -------
        (torch.Tensor) A point in the domain that maximises the acquisition function within the analyzed 1-d subspaces
        '''

        argmaxs = []
        argmaxs_values = []
        sampling_round = 0
        timeout_triggered = False
        sampling_limit_triggered = False
        start_time = time.monotonic()

        while len(argmaxs) < self._target_candidates:
            if self._timeout_seconds is not None:
                if (time.monotonic() - start_time) > self._timeout_seconds:
                    timeout_triggered = True
                    break

            if self._max_sampling_rounds is not None and sampling_round >= self._max_sampling_rounds:
                sampling_limit_triggered = True
                break

            sampling_round += 1

            origins = [self._safe_seed, self._last_sampled_point]
            for _ in range(5):
                origins.append(generic_utils.sample_uniform_in_box(self._domain, 1))
            point_for_direction = generic_utils.sample_uniform_in_box(self._domain, 1)
            for origin in origins:
                if len(argmaxs) >= self._target_candidates:
                    break
                normalized_direction = self._get_normalized_direction(origin, point_for_direction)
                if torch.isnan(normalized_direction).any():
                    continue
                argmax, value = find_argmax_location(origin, normalized_direction)
                if argmax is None:
                    continue
                argmaxs.append(argmax)
                if torch.is_tensor(value):
                    scalar_value = value.detach().reshape(-1)[0].item()
                else:
                    scalar_value = float(value)
                argmaxs_values.append(scalar_value)

        if not argmaxs:
            if timeout_triggered or sampling_limit_triggered:
                reason = "timeout" if timeout_triggered else "sampling limit"
                warnings.warn(
                    f"Line BO acquisition hit the {reason} before finding a feasible candidate; "
                    "returning the last known safe point."
                )
            fallback_value = torch.tensor(float('nan'), dtype=self._safe_seed.dtype)
            return self._last_sampled_point, fallback_value

        if (timeout_triggered or sampling_limit_triggered) and len(argmaxs) < self._target_candidates:
            reason = []
            if timeout_triggered:
                reason.append("timeout")
            if sampling_limit_triggered:
                reason.append("sampling limit")
            warnings.warn(
                f"Line BO acquisition collected only {len(argmaxs)} candidate(s) before hitting the "
                f"{' and '.join(reason)}; using the best candidate found so far."
            )

        argmaxs_values_tensor = torch.tensor(argmaxs_values, dtype=self._safe_seed.dtype)
        optimum_value_index = torch.topk(argmaxs_values_tensor, 1)[1].item()

        self._last_sampled_point = argmaxs[optimum_value_index]
        best_value = torch.tensor(argmaxs_values[optimum_value_index], dtype=self._safe_seed.dtype)

        return argmaxs[optimum_value_index], best_value
