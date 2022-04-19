import torch
print(torch.__version__)

### Merge sort
import sys
Sentinel= sys.maxsize
def merge(array, first, middle, last):
    n1 = middle -first + 1
    n2 = last - middle

    L = [None for t in range(n1 + 1)]
    R = [None for t in range(n2 + 1)]

    for i in range(n1):
        L[i] = array[first + i -1]

    for j in range(n2):
        R[j] = array[middle + j]

    L[n1]=Sentinel
    R[n2]=Sentinel

    i = 0
    j = 0
    for k in range(first-1, last):
        if L[i] <= R[j]:
            array[k] = L[i]
            i = i + 1
        else:
            array[k] = R[j]
            j = j + 1

def mergeSort(array,first,last):
    if first < last:
        middle = int((first + last)/2)
        mergeSort(array, first, middle)
        mergeSort(array, middle + 1, last)
        merge(array, first, middle, last)



array = [3,1,-1,3,0,2]
mergeSort(array, first=1,last= len(array))
print(array)


##insertion sort

def insertionSort(array):
    for j in range(1,len(array)):
        key = array[j]
        i = j-1
        while i >= 0 and array[i] > key:
            array[i + 1]= array[i]
            i = i-1
            array[i+1]= key
            print(array)

array1 = [96, 72, 61, 3, 93, 90, 38, 84]
insertionSort(array1)
print(array1)


### Maximum SubArray Sum: Kadane's Algorithm; a famous approach to solve
### problems using dynamic programming
###which solves the problem by traversing over the whole array using two
#variables to track the sum so far and maximum total

def maxSubArraySum(arr):
    max_till_now = arr[0] #initially takes first value of array but
    # this variable's value is solely dependent on max_ending variables value
    #whenever it's value gets larger, it is assigned to max_till_Now variable
    max_ending = 0
#maximum ending leh array ko partek value add gardi jhan6

    for i in range(0,len(arr)):
        max_ending = max_ending + arr[i]
        if max_ending < 0:
            max_ending = 0


        if (max_till_now < max_ending):
            max_till_now = max_ending

    return max_till_now


arr = [13,-3,-25,20,-3,-16,-23,18,20,-7,12,-5,-22,15,-4,7 ]
k = maxSubArraySum(arr)
print("Maximum Sub Array Sum Is", k)


def mergeSort1(array, first, last):

    if first < last:
        mid = int((first + last)/2)
        mergeSort1(array, first, mid)
        mergeSort1(array, mid +1, last)
        MERGE(array, first, mid, last)


def MERGE(arra1, first, mid, last):

    n1 = mid - first + 1
    n2 = last - mid

    L1 = [None for k in range(n1+1)] #0 to n1(3)
    R1 = [None for k in range(n2+1)] #0 to n2(3)

    L1[n1] = Sentinel
    R1[n2] = Sentinel

    for i in range(n1): # i = 0 1 2
        L1[i] = arra1[first + i -1]


    for j in range(n2):
        R1[j] = arra1[mid + j] # mid = 3 so, array[3+0]


    i = 0
    j = 0
    idx = 0
    print(f" ===> Start")
    for k in range(first-1, last):
        idx +=1
        if L1[i] <= R1[j]:
            arra1[k] = L1[i]
            i = i + 1
        else:
            arra1[k] = R1[j]
            j = j + 1
        print(idx, arra1)
    print("<====")
arra1 = [29, 31, 19, 72, 59, 12, 28, 63]
mergeSort1(arra1,first=1, last = len(arra1))
print("last",arra1)


def GCD(a,b):
    if(b>a):
     d = b
     b = a
     a = d
    #quotient = int(a/b)
    remainder = a%b
    print(remainder)
    if(remainder == 0):
     return b

    else:
        a = b
        b = remainder
    return GCD(a,b)
    #return j

a =56
b =15
k = GCD(a,b)
print("The ",k)


def heapify(arr, n, i):
    # Find largest among root and children
    largest = i
    l = 2 * i + 1
    r = 2 * i + 2

    if l < n and arr[ i ] > arr[ l ]:
        largest = l

    if r < n and arr[ largest ] > arr[ r ]:
        largest = r

    # If root is not largest, swap with largest and continue heapifying
    if largest != i:
        arr[ i ], arr[ largest ] = arr[ largest ], arr[ i ]
        heapify ( arr, n, largest )



def heapSort(arr):
    n = len ( arr )

    # Build max heap
    for i in range ( n //2 -1, -1, -1 ):
        #print(i)
        heapify ( arr, n, i )
        print(i,"After Build",arr)

    for i in range ( n - 1, 0, -1 ):
        # Swap
        arr[ i ], arr[ 0 ] = arr[ 0 ], arr[ i ]

        # Heapify root element
        heapify ( arr, i, 0 )
        #print(arr)
        print("After Heapify", i, arr)
print("Provided Test cases: Testcase1")
arr_test_1 = [36, 35, 29, 80, 92, 82, 39, 42, 90, 97]
print("before build", arr_test_1)
heapSort(arr_test_1)
print("After Sort",arr_test_1)


# A Divide and Conquer based program
# for maximum subarray sum problem

# Find the maximum possible sum in
# arr[] auch that arr[m] is part of it


def maxCrossingSum(arr, l, m, h):

# Include elements on left of mid.
	sm = 0
	left_sum = -10000

	for i in range(m, l-1, -1):
		sm = sm + arr[i]

		if (sm > left_sum):
			left_sum = sm

	# Include elements on right of mid
	sm = 0
	right_sum = -1000
	for i in range(m + 1, h + 1):
		sm = sm + arr[i]

		if (sm > right_sum):
			right_sum = sm

	# Return sum of elements on left and right of mid
	# returning only left_sum + right_sum will fail for [-2, 1]
	return max(left_sum + right_sum, left_sum, right_sum)


# Returns sum of maximum sum subarray in aa[l..h]
def maxSubArraySum(arr, l, h):

	# Base Case: Only one element
	if (l == h):
		return arr[l]

	# Find middle point
	m = (l + h) // 2

	# Return maximum of following three possible cases
	# a) Maximum subarray sum in left half
	# b) Maximum subarray sum in right half
	# c) Maximum subarray sum such that the
	#	 subarray crosses the midpoint
	return max(maxSubArraySum(arr, l, m),
			maxSubArraySum(arr, m+1, h),
			maxCrossingSum(arr, l, m, h))


# Driver Code
arr = [2, 3, 4, 5, 7]
n = len(arr)

max_sum = maxSubArraySum(arr, 0, n-1)
print("Maximum contiguous sum is ", max_sum)






