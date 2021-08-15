""" REPTILE meta learning algorithm
Copyright (c) 2021 Robert Bosch GmbH

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

"""
from abc import ABC

import torch


# TODO: Only defines the interface
class NAS_agent(ABC):
    def __init__(self, meta_model, config):
        self.meta_model = meta_model
        self.config = config

    def test_agent(self, env, task):
        return NotImplementedError

    def act_on_episode(self) -> dict():
        """Should return a task info dictionary"""
        return {}
