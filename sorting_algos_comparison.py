import numpy as np
import time
import matplotlib.pyplot as plt

# Define the sorting algorithms

# Bubble Sort
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                
# Cocktail Shaker Sort
def cocktail_shaker_sort(arr):
    n = len(arr)
    swapped = True
    start = 0
    end = n-1
    while swapped:
        swapped = False
        for i in range(start, end):
            if arr[i] > arr[i+1]:
                arr[i], arr[i+1] = arr[i+1], arr[i]
                swapped = True
        if not swapped:
            break
        swapped = False
        end -= 1
        for i in range(end-1, start-1, -1):
            if arr[i] > arr[i+1]:
                arr[i], arr[i+1] = arr[i+1], arr[i]
                swapped = True
        start += 1
        
# Insertion Sort
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i-1
        while j >=0 and key < arr[j]:
            arr[j+1] = arr[j]
            j -= 1
        arr[j+1] = key

def binary_insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        lo = 0
        hi = i
        while lo < hi:
            mid = (lo + hi) // 2
            if arr[mid] < key:
                lo = mid + 1
            else:
                hi = mid
        for j in range(i, lo, -1):
            arr[j] = arr[j-1]
        arr[lo] = key

# Selection Sort
def selection_sort(arr):
    for i in range(len(arr)):
        min_idx = i
        for j in range(i+1, len(arr)):
            if arr[min_idx] > arr[j]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]

# Quick Sort
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return quick_sort(left) + middle + quick_sort(right)

# Merge Sort
def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr)//2
        L = arr[:mid]
        R = arr[mid:]

        merge_sort(L)
        merge_sort(R)

        i = j = k = 0

        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1

        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1

        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1

def comb_sort(arr):
    def getNextGap(gap):
        gap = (gap * 10) // 13
        if gap < 1:
            return 1
        return gap
    n = len(arr)
    gap = n
    swapped = True
    while gap !=1 or swapped == 1:
        gap = getNextGap(gap)
        swapped = False
        for i in range(0, n-gap):
            if arr[i] > arr[i + gap]:
                arr[i], arr[i + gap] = arr[i + gap], arr[i]
                swapped = True
    
def counting_sort_for_radix(arr, exp):
    n = len(arr)
    output = [0] * n
    count = [0] * 10
    for i in range(n):
        index = arr[i] // exp
        count[index % 10] += 1
    for i in range(1, 10):
        count[i] += count[i-1]
    i = n - 1
    while i >= 0:
        index = arr[i] // exp
        output[count[index % 10] - 1] = arr[i]
        count[index % 10] -= 1
        i -= 1
    for i in range(n):
        arr[i] = output[i]

def radix_sort(arr):
    max1 = max(arr)
    exp = 1
    while max1 // exp > 0:
        counting_sort_for_radix(arr, exp)
        exp *= 10

# Generate a random array
np.random.seed(0)
arr = np.random.randint(1, 100, size=100).tolist()

# Measure the sorting times
algorithms = {'Bubble Sort': bubble_sort, 
              'Cocktail Shaker Sort': cocktail_shaker_sort,
              'Insertion Sort': insertion_sort,
              'Binary Insertion Sort': binary_insertion_sort,
              'Selection Sort': selection_sort,
              'Quick Sort': quick_sort,
              'Merge Sort': merge_sort,
              'Comb Sort': comb_sort,
              'Radix Sort': radix_sort}

times = {}
for name, func in algorithms.items():
    copy_arr = arr.copy()
    start_time = time.time()
    func(copy_arr)
    end_time = time.time()
    times[name] = end_time - start_time

# Plot the comparison
plt.bar(times.keys(), times.values(), color='skyblue')
plt.ylabel('Time (seconds)')
plt.title('Sorting Algorithm Performance')
plt.xticks(rotation=45)
plt.show()