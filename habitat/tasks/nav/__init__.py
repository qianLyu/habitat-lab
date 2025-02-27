#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.core.embodied_task import EmbodiedTask
from habitat.core.registry import registry


def _try_register_nav_task():
    try:
        from habitat.tasks.nav.nav import NavigationTask  # noqa
    except ImportError as e:
        navtask_import_error = e

        @registry.register_task(name="Nav-v0")
        class NavigationTaskImportError(EmbodiedTask):
            def __init__(self, *args, **kwargs):
                raise navtask_import_error

def _try_register_social_nav_task():
    try:
        from habitat.tasks.nav.social_nav import SocialNavigationTask  # noqa
    except ImportError as e:
        socialnavtask_import_error = e

        @registry.register_task(name="SocialNav-v0")
        class NavigationTaskImportError(EmbodiedTask):
            def __init__(self, *args, **kwargs):
                raise socialnavtask_import_error
