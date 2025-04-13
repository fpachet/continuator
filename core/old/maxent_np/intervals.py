from bisect import bisect_left, bisect_right


def count_strict_overlaps(intervals):
    """
    Given a list of intervals [(a1, b1), (a2, b2), ..., (an, bn)],
    returns a list 'overlaps' where overlaps[k] is the number of intervals
    that strictly overlap the interval intervals[k].

    Two intervals [a_i, b_i] and [a_j, b_j] are said to *strictly overlap* if:
        a_i < b_j  AND  a_j < b_i

    Complexity: O(n log n)
    """

    # Sort all starts and ends separately
    starts = sorted(a for (a, b) in intervals)
    ends = sorted(b for (a, b) in intervals)

    overlaps = []

    for a, b in intervals:
        # 1) Count how many intervals start before b  -> a_j < b
        #    bisect_left(starts, b) gives the index where b would be inserted
        #    to keep 'starts' sorted, i.e. the count of starts < b.
        left_count = bisect_left(starts, b)

        # 2) Count how many intervals end on or before a -> b_j <= a
        #    bisect_right(ends, a) gives the insertion index for 'a' in 'ends',
        #    i.e. the count of ends <= a.
        right_count = bisect_right(ends, a)

        # The difference gives the number of intervals J_j satisfying
        # a_j < b and b_j > a. However, this also includes the interval J_k itself
        # if it meets a < b_k (which it should, assuming a < b), so subtract 1.
        overlap_count = (left_count - right_count) - 1

        overlaps.append(overlap_count)

    return overlaps


# Example usage:
if __name__ == "__main__":
    intervals = [(1, 4), (2, 7), (5, 8)]
    print(count_strict_overlaps(intervals))
    # Expected output: [1, 2, 1]
