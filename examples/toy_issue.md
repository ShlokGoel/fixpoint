median() returns the wrong value for lists with an even number of elements

`median([1, 2, 3, 4])` returns `3`, but the median of an even-length list should be the mean of the two middle values, `2.5`. Odd-length lists work fine.
