"""Select the existing CPU and CUDA oracle arms independently for verification tiers."""

import pytest


@pytest.fixture(params=("cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)))
def device_name(request):
    """The device of one oracle arm; only the CUDA arm belongs to GPU verification."""
    return request.param
