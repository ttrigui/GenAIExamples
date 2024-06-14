# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import yaml
from fastapi import Request
from core.db_handler import DB_Handler


def read_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    return config

def get_db_handler(request: Request) -> DB_Handler:
    """
    Helper to grab dependencies that live in the app.state
    """
    # See application for the key name in `app`.
    return request.app.state.db_handler
