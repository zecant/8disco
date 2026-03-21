"""
Tests for the CircularBuffer data structure.
"""
import pytest
import numpy as np
from numpy.testing import assert_array_equal
from tradsl.circular_buffer import CircularBuffer


class TestConstruction:
    """Tests for CircularBuffer initialization."""

    def test_size_must_be_positive(self):
        with pytest.raises(ValueError, match="must be positive"):
            CircularBuffer(0)

    def test_size_must_be_positive_negative(self):
        with pytest.raises(ValueError, match="must be positive"):
            CircularBuffer(-1)

    def test_initial_count_is_zero(self):
        buf = CircularBuffer(5)
        assert buf.count == 0
        assert len(buf) == 0

    def test_initial_is_ready_is_false(self):
        buf = CircularBuffer(5)
        assert buf.is_ready is False


class TestBasicOperations:
    """Tests for push, size, count, len."""

    def test_size_returns_capacity(self):
        buf = CircularBuffer(7)
        assert buf.size == 7

    def test_count_increments_until_full(self):
        buf = CircularBuffer(3)
        assert buf.count == 0
        buf.push(1.0)
        assert buf.count == 1
        buf.push(2.0)
        assert buf.count == 2
        buf.push(3.0)
        assert buf.count == 3

    def test_count_stays_at_size_when_full(self):
        buf = CircularBuffer(3)
        for i in range(10):
            buf.push(float(i))
        assert buf.count == 3

    def test_len_equals_count(self):
        buf = CircularBuffer(5)
        assert len(buf) == 0
        buf.push(1.0)
        assert len(buf) == 1
        buf.push(2.0)
        assert len(buf) == 2


class TestIsReady:
    """Tests for is_ready property."""

    def test_not_ready_before_full(self):
        buf = CircularBuffer(3)
        buf.push(1.0)
        assert buf.is_ready is False
        buf.push(2.0)
        assert buf.is_ready is False

    def test_ready_after_size_pushes(self):
        buf = CircularBuffer(3)
        buf.push(1.0)
        buf.push(2.0)
        assert buf.is_ready is False
        buf.push(3.0)
        assert buf.is_ready is True

    def test_stays_ready_after_overwriting(self):
        buf = CircularBuffer(3)
        buf.push(1.0)
        buf.push(2.0)
        buf.push(3.0)
        assert buf.is_ready is True
        buf.push(4.0)
        assert buf.is_ready is True

    def test_size_one_immediately_ready(self):
        buf = CircularBuffer(1)
        assert buf.is_ready is False
        buf.push(1.0)
        assert buf.is_ready is True


class TestLatest:
    """Tests for latest() method."""

    def test_returns_none_before_ready(self):
        buf = CircularBuffer(3)
        assert buf.latest() is None
        buf.push(1.0)
        assert buf.latest() is None

    def test_returns_newest_after_ready(self):
        buf = CircularBuffer(3)
        buf.push(1.0)
        buf.push(2.0)
        buf.push(3.0)
        assert buf.latest() == 3.0

    def test_latest_updates_correctly(self):
        buf = CircularBuffer(3)
        buf.push(1.0)
        buf.push(2.0)
        buf.push(3.0)
        assert buf.latest() == 3.0
        buf.push(4.0)
        assert buf.latest() == 4.0
        buf.push(5.0)
        assert buf.latest() == 5.0

    def test_latest_is_correct_after_wrap(self):
        buf = CircularBuffer(3)
        buf.push(1.0)
        buf.push(2.0)
        buf.push(3.0)
        buf.push(4.0)
        assert buf.latest() == 4.0
        buf.push(5.0)
        assert buf.latest() == 5.0
        buf.push(6.0)
        assert buf.latest() == 6.0


class TestContents:
    """Tests for contents() method."""

    def test_returns_padded_list_before_ready(self):
        buf = CircularBuffer(3)
        assert buf.contents() == [None, None, None]
        buf.push(1.0)
        assert buf.contents() == [None, None, 1.0]

    def test_returns_list_after_ready(self):
        buf = CircularBuffer(3)
        buf.push(1.0)
        buf.push(2.0)
        buf.push(3.0)
        arr = buf.contents()
        assert arr is not None
        assert len(arr) == 3

    def test_array_oldest_to_newest(self):
        buf = CircularBuffer(3)
        buf.push(1.0)
        buf.push(2.0)
        buf.push(3.0)
        arr = buf.contents()
        assert_array_equal(arr, [1.0, 2.0, 3.0])

    def test_array_after_wrap(self):
        buf = CircularBuffer(3)
        buf.push(1.0)
        buf.push(2.0)
        buf.push(3.0)
        buf.push(4.0)
        arr = buf.contents()
        assert_array_equal(arr, [2.0, 3.0, 4.0])

    def test_array_multiple_wraps(self):
        buf = CircularBuffer(3)
        for i in range(10):
            buf.push(float(i))
        arr = buf.contents()
        assert len(arr) == 1
        assert_array_equal(arr, [1.0, 2.0, 3.0])

    def test_array_after_wrap(self):
        buf = CircularBuffer(3)
        buf.push(1.0)
        buf.push(2.0)
        buf.push(3.0)
        buf.push(4.0)
        arr = buf.contents()
        assert_array_equal(arr, [2.0, 3.0, 4.0])

    def test_array_multiple_wraps(self):
        buf = CircularBuffer(3)
        for i in range(10):
            buf.push(float(i))
        arr = buf.contents()
        assert len(arr) == 3
        assert arr[0] == 7.0
        assert arr[1] == 8.0
        assert arr[2] == 9.0


class TestGetItem:
    """Tests for __getitem__ indexing."""

    def test_positive_index(self):
        buf = CircularBuffer(3)
        buf.push(1.0)
        buf.push(2.0)
        buf.push(3.0)
        assert buf[0] == 1.0
        assert buf[1] == 2.0
        assert buf[2] == 3.0

    def test_negative_index(self):
        buf = CircularBuffer(3)
        buf.push(1.0)
        buf.push(2.0)
        buf.push(3.0)
        assert buf[-1] == 3.0
        assert buf[-2] == 2.0
        assert buf[-3] == 1.0

    def test_index_out_of_bounds_raises(self):
        buf = CircularBuffer(3)
        buf.push(1.0)
        buf.push(2.0)
        with pytest.raises(IndexError):
            buf[2]
        with pytest.raises(IndexError):
            buf[-3]

    def test_index_during_warmup(self):
        buf = CircularBuffer(3)
        buf.push(1.0)
        assert buf[0] == 1.0
        buf.push(2.0)
        assert buf[0] == 1.0
        assert buf[1] == 2.0

    def test_index_after_wrap(self):
        buf = CircularBuffer(3)
        buf.push(1.0)
        buf.push(2.0)
        buf.push(3.0)
        buf.push(4.0)
        assert buf[0] == 2.0
        assert buf[1] == 3.0
        assert buf[2] == 4.0
        assert buf[-1] == 4.0
        assert buf[-2] == 3.0
        assert buf[-3] == 2.0


class TestWrapAround:
    """Tests for correct behavior after buffer wraps around."""

    def test_oldest_value_overwritten(self):
        buf = CircularBuffer(3)
        buf.push(1.0)
        buf.push(2.0)
        buf.push(3.0)
        assert buf[0] == 1.0
        buf.push(4.0)
        assert buf[0] == 2.0
        assert buf[2] == 4.0

    def test_order_maintained_after_multiple_wraps(self):
        buf = CircularBuffer(3)
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        for v in values:
            buf.push(v)
        arr = buf.contents()
        assert_array_equal(arr, [5.0, 6.0, 7.0])

    def test_latest_always_correct_through_wraps(self):
        buf = CircularBuffer(3)
        for expected_latest in range(1, 21):
            buf.push(float(expected_latest))
            if expected_latest >= 3:
                assert buf.latest() == float(expected_latest)

    def test_all_values_accessible_through_wraps(self):
        buf = CircularBuffer(3)
        for i in range(100):
            buf.push(float(i))
            if i >= 2:
                arr = buf.contents()
                assert arr[-1] == float(i)
                assert arr[-2] == float(i - 1)
                assert arr[-3] == float(i - 2)


class TestSingleElement:
    """Edge case tests for size=1 buffer."""

    def test_single_element_is_ready_immediately(self):
        buf = CircularBuffer(1)
        buf.push(1.0)
        assert buf.is_ready is True

    def test_single_element_latest(self):
        buf = CircularBuffer(1)
        buf.push(1.0)
        assert buf.latest() == 1.0
        buf.push(2.0)
        assert buf.latest() == 2.0

    def test_single_element_array(self):
        buf = CircularBuffer(1)
        buf.push(1.0)
        arr = buf.contents()
        assert len(arr) == 1
        assert arr[0] == 1.0

    def test_single_element_overwrites(self):
        buf = CircularBuffer(1)
        buf.push(1.0)
        buf.push(2.0)
        buf.push(3.0)
        assert buf[0] == 3.0
        assert buf.latest() == 3.0


class TestConsistency:
    """Tests ensuring consistency between different access methods."""

    def test_latest_equals_array_last(self):
        buf = CircularBuffer(5)
        for i in range(20):
            buf.push(float(i))
            if i >= 4:
                arr = buf.contents()
                assert buf.latest() == arr[-1]

    def test_contents_equals_indexed_values(self):
        buf = CircularBuffer(4)
        for i in range(15):
            buf.push(float(i))
            if i >= 3:
                arr = buf.contents()
                for j in range(len(arr)):
                    assert arr[j] == buf[j]

    def test_count_equals_len(self):
        buf = CircularBuffer(4)
        for i in range(10):
            buf.push(float(i))
            assert buf.count == len(buf)


class TestFuzz:
    """Randomized/fuzz tests to verify buffer correctness."""

    def test_random_sequence_consistency(self, seed=42):
        np.random.seed(seed)
        size = np.random.randint(2, 20)
        buf = CircularBuffer(size)

        reference = []
        for _ in range(1000):
            value = np.random.randn()
            buf.push(value)
            reference.append(value)

            if buf.is_ready:
                expected_oldest = reference[-size]
                expected_newest = reference[-1]
                assert buf.latest() == expected_newest
                arr = buf.contents()
                assert len(arr) == size
                assert arr[-1] == expected_newest
                assert arr[0] == expected_oldest

    def test_sequential_integers(self):
        buf = CircularBuffer(5)
        for i in range(200):
            buf.push(float(i))
            if i >= 4:
                arr = buf.contents()
                assert arr[-1] == float(i)
                assert arr[0] == float(max(0, i - 4))

    def test_alternating_values(self):
        buf = CircularBuffer(3)
        for i in range(100):
            buf.push(1.0 if i % 2 == 0 else -1.0)
            if i >= 2:
                arr = buf.contents()
                assert arr[-1] == (-1.0 if i % 2 == 1 else 1.0)

    def test_nan_handling(self):
        buf = CircularBuffer(3)
        buf.push(float('nan'))
        buf.push(1.0)
        buf.push(2.0)
        assert buf.is_ready is True
        assert np.isnan(buf[0])

    def test_inf_handling(self):
        buf = CircularBuffer(3)
        buf.push(float('inf'))
        buf.push(float('-inf'))
        buf.push(0.0)
        assert buf.is_ready is True
        assert buf[0] == float('inf')
        assert buf[1] == float('-inf')
        assert buf[2] == 0.0

    def test_small_floats(self):
        buf = CircularBuffer(3)
        buf.push(1e-100)
        buf.push(1e-50)
        buf.push(1e-10)
        assert buf.is_ready is True
        assert buf.latest() == 1e-10

    def test_large_floats(self):
        buf = CircularBuffer(3)
        buf.push(1e100)
        buf.push(1e150)
        buf.push(1e200)
        assert buf.is_ready is True
        assert buf.latest() == 1e200

    def test_random_sizes(self):
        np.random.seed(123)
        for _ in range(50):
            size = np.random.randint(1, 100)
            buf = CircularBuffer(size)
            for i in range(size * 3):
                buf.push(float(i))
            assert len(buf) == size
            arr = buf.contents()
            assert len(arr) == size

    @pytest.mark.parametrize("size", [1, 2, 5, 10, 50, 100])
    def test_various_sizes(self, size):
        buf = CircularBuffer(size)
        for i in range(size * 2):
            buf.push(float(i))

        arr = buf.contents()
        assert len(arr) == size
        assert buf.count == size
        assert buf.is_ready is True
        assert buf.latest() == float(size * 2 - 1)

    def test_repeated_push_same_value(self):
        buf = CircularBuffer(3)
        for _ in range(100):
            buf.push(42.0)
        arr = buf.contents()
        assert_array_equal(arr, [42.0, 42.0, 42.0])
        assert buf.latest() == 42.0

    def test_heavy_wrap_stress(self):
        buf = CircularBuffer(7)
        for i in range(10000):
            buf.push(float(i))
        arr = buf.contents()
        assert len(arr) == 7
        assert arr[-1] == 9999.0
        assert arr[0] == 9993.0

    def test_random_access_sequence(self):
        np.random.seed(456)
        size = 10
        buf = CircularBuffer(size)
        reference = []

        for _ in range(500):
            if len(reference) < size:
                value = np.random.rand()
                buf.push(value)
                reference.append(value)
            else:
                idx = np.random.randint(-size, size)
                expected = reference[idx]
                assert buf[idx] == expected

    def test_no_heap_allocation_after_init(self):
        buf = CircularBuffer(1000)
        for i in range(100):
            buf.push(float(i))
        assert buf.count == 100
