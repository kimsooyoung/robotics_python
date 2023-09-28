# Copyright 2022 @RoadBalance
# Reference from https://pab47.github.io/legs.html
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np


def calc_rot_2d(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)],
    ])


def calc_homogeneous_2d(theta, trans):
    output = np.identity(3)
    output[:2, :2] = calc_rot_2d(theta)
    output[:2, 2] = np.transpose(np.array(trans))

    return output


if __name__ == '__main__':
    trans = [30.0, 1.0]
    print(calc_homogeneous_2d(30 * np.pi / 180, trans))
