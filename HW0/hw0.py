import numpy as np
""" Problem 6
@param a 3-valued random variable X
@param an interger n
draws n random samples from X
"""
def draw_samples(n):
    assert isinstance(n, int), "n must be an integer"
    return np.random.choice([1, 0, -1], size = n, p = [0.35, 0.45, 0.2])

print("Problem 6, testing draw_sample")
# print(draw_sample('n'))
print(draw_samples(5))
print("\n")

"""Problem 7 
@param a numpy 1-d array
@return sum of squares of this array
"""
def sum_squares(arr):
    assert isinstance(arr, np.ndarray), "input must be a np array"
    assert arr.ndim == 1, "the dimension must be 1"
    return np.dot(arr, arr)

x = np.array([1,1,1,1])
y = np.array([1,2,3,4])

print("test Problem 7", x, sum_squares(x),"\n")
print("test Problem 7", y, sum_squares(y),"\n")

"""Problem 8
@param n cycles of pouring
@return resulting respective volume
"""
def troublemakers(n):
    assert isinstance(n, int), "n must be an integer"
    assert n>=0, "valid rounds"
    results = np.array([1.0,1.0])
    for i in range(n):
        results[1] = results[1] + results[0]*0.35 
        pour = results[1] * 0.2
        results[1] = results[1] * 0.8
        results[0] = results[0] * 0.65 + pour
    return results

print("test Problem 8 \n")
print("0", troublemakers(0))
print("1", troublemakers(1))
print("5", troublemakers(5),"\n")

"""Problem 9
@param x an 1d numpy array
@param y another 1d numpy array
@return sliding weighted sum in an 1d numpy
"""
def sliding_weighted_sum(x,y):
    assert isinstance(x, np.ndarray), "x must be np arr"
    assert isinstance(y, np.ndarray), "y must be np arr"
    assert x.ndim == 1, "x must be 1d"
    assert y.ndim == 1, "y must be 1d"

    n = x.size
    k = y.size
    assert k<=n, "length y must be <= length x"

    sums = np.array([])
    for i in range (n):
        if i + k <= n:
            sums = np.append(sums, np.dot(x[i:i+k], y))
    return sums

x1 = np.array([1, 2, 3, 4, 5, 6])
y1 = np.array([7, 8, 9])
x2 = np.array([13, 17, 23, 29, 31, 37])
y2 = np.array([2, 4, 8])

print("test Problem 9\n")
print("x ", x1, "y ", y1, "s_w_s: ", sliding_weighted_sum(x1,y1))
print("x ",x2, "y ", y2, "s_w_s: ", sliding_weighted_sum(x2,y2))
