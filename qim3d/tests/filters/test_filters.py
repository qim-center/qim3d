import re

import numpy as np
import pytest

import qim3d


def test_filter_base_initialization():
    filter_base = qim3d.filters.FilterBase(3, size=2)
    assert filter_base.args == (3,)
    assert filter_base.kwargs == {'size': 2}


def test_gaussian_filter():
    input_image = np.random.rand(50, 50)

    # Testing the function
    filtered_image_fn = qim3d.filters.gaussian(input_image, sigma=1.5)

    # Testing the class method
    gaussian_filter_cls = qim3d.filters.Gaussian(sigma=1.5)
    filtered_image_cls = gaussian_filter_cls(input_image)

    # Assertions
    assert filtered_image_cls.shape == filtered_image_fn.shape == input_image.shape
    assert np.array_equal(filtered_image_fn, filtered_image_cls)
    assert not np.array_equal(filtered_image_fn, input_image)


def test_median_filter():
    input_image = np.random.rand(50, 50)

    # Testing the function
    filtered_image_fn = qim3d.filters.median(input_image, size=3)

    # Testing the class method
    median_filter_cls = qim3d.filters.Median(size=3)
    filtered_image_cls = median_filter_cls(input_image)

    # Assertions
    assert filtered_image_cls.shape == filtered_image_fn.shape == input_image.shape
    assert np.array_equal(filtered_image_fn, filtered_image_cls)
    assert not np.array_equal(filtered_image_fn, input_image)


def test_maximum_filter():
    input_image = np.random.rand(50, 50)

    # Testing the function
    filtered_image_fn = qim3d.filters.maximum(input_image, size=3)

    # Testing the class method
    maximum_filter_cls = qim3d.filters.Maximum(size=3)
    filtered_image_cls = maximum_filter_cls(input_image)

    # Assertions
    assert filtered_image_cls.shape == filtered_image_fn.shape == input_image.shape
    assert np.array_equal(filtered_image_fn, filtered_image_cls)
    assert not np.array_equal(filtered_image_fn, input_image)


def test_minimum_filter():
    input_image = np.random.rand(50, 50)

    # Testing the function
    filtered_image_fn = qim3d.filters.minimum(input_image, size=3)

    # Testing the class method
    minimum_filter_cls = qim3d.filters.Minimum(size=3)
    filtered_image_cls = minimum_filter_cls(input_image)

    # Assertions
    assert filtered_image_cls.shape == filtered_image_fn.shape == input_image.shape
    assert np.array_equal(filtered_image_fn, filtered_image_cls)
    assert not np.array_equal(filtered_image_fn, input_image)


def test_sequential_filter_pipeline():
    input_image = np.random.rand(50, 50)

    # Individual filters
    gaussian_filter = qim3d.filters.Gaussian(sigma=1.5)
    median_filter = qim3d.filters.Median(size=3)
    maximum_filter = qim3d.filters.Maximum(size=3)

    # Testing the sequential pipeline
    sequential_pipeline = qim3d.filters.Pipeline(
        gaussian_filter, median_filter, maximum_filter
    )
    filtered_image_pipeline = sequential_pipeline(input_image)

    # Testing the equivalence to maximum(median(gaussian(input,**kwargs),**kwargs),**kwargs)
    expected_output = qim3d.filters.maximum(
        qim3d.filters.median(qim3d.filters.gaussian(input_image, sigma=1.5), size=3),
        size=3,
    )

    # Assertions
    assert filtered_image_pipeline.shape == expected_output.shape == input_image.shape
    assert not np.array_equal(filtered_image_pipeline, input_image)
    assert np.array_equal(filtered_image_pipeline, expected_output)


def test_sequential_filter_appending():
    input_image = np.random.rand(50, 50)

    # Individual filters
    gaussian_filter = qim3d.filters.Gaussian(sigma=1.5)
    median_filter = qim3d.filters.Median(size=3)
    maximum_filter = qim3d.filters.Maximum(size=3)

    # Sequential pipeline with filter initialized at the beginning
    sequential_pipeline_initial = qim3d.filters.Pipeline(
        gaussian_filter, median_filter, maximum_filter
    )
    filtered_image_initial = sequential_pipeline_initial(input_image)

    # Sequential pipeline with filter appended
    sequential_pipeline_appended = qim3d.filters.Pipeline(
        gaussian_filter, median_filter
    )
    sequential_pipeline_appended.append(maximum_filter)
    filtered_image_appended = sequential_pipeline_appended(input_image)

    # Assertions
    assert (
        filtered_image_initial.shape
        == filtered_image_appended.shape
        == input_image.shape
    )
    assert not np.array_equal(filtered_image_appended, input_image)
    assert np.array_equal(filtered_image_initial, filtered_image_appended)


def test_assertion_error_not_filterbase_subclass():
    # Get valid filter classes
    valid_filters = [
        subclass.__name__ for subclass in qim3d.filters.FilterBase.__subclasses__()
    ]

    # Create invalid object
    invalid_filter = object()  # An object that is not an instance of FilterBase

    # Construct error message
    message = (
        f'filters should be instances of one of the following classes: {valid_filters}'
    )

    # Use pytest.raises to catch the AssertionError
    with pytest.raises(AssertionError, match=re.escape(message)):
        sequential_pipeline = qim3d.filters.Pipeline(invalid_filter)
