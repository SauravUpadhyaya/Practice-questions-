# Quick sort in Python

# function to find the partition position
def partition(array, low, high):

  # choose the rightmost element as pivot
  pivot = array[high]

  # pointer for greater element
  i = low -1

  # traverse through all elements
  # compare each element with pivot
  for j in range(low, high):
    if array[j] >= pivot:
      # if element smaller than pivot is found
      # swap it with the greater element pointed by i
      i = i + 1

      # swapping element at i with element at j
      (array[i], array[j]) = (array[j], array[i])
    print(j,array)
  # swap the pivot element with the greater element specified by i
  (array[i + 1], array[high]) = (array[high], array[i + 1])

  # return the position from where partition is done
  print("when pivot element is found", array[i+1],array)
  return i + 1

# function to perform quicksort
def quickSort(array, low, high):
  if low < high:

    # find pivot element such that
    # element smaller than pivot are on the left
    # element greater than pivot are on the right
    pi = partition(array, low, high)

    # recursive call on the left of pivot
    quickSort(array, low, pi - 1)

    # recursive call on the right of pivot
    quickSort(array, pi + 1, high)


data = [2,8,7,1,3,5,6,4]
#print("Unsorted Array")
#print(data)

size = len(data)

quickSort(data, 0, size - 1)

print('Sorted Array in Ascending Order:')
print(data)

#rodcutting
import sys


# Function to find the best way to cut a rod of length `n`
# where the rod of length `i` has a cost `price[i-1]`
def rodCut(price, n):
    # base case
    if n == 0:
        return 0

    maxValue = -sys.maxsize


    # one by one, partition the given rod of length `n` into two parts of length
    # (1, n-1), (2, n-2), (3, n-3), … ,(n-1, 1), (n, 0) and take maximum
    for i in range ( 1, n + 1 ):

        # rod of length `i` has a cost `price[i-1]`
        cost = price[ i - 1 ] + rodCut ( price, n - i )
    #print("step",i, cost)
        if cost > maxValue:
           # length = []
            maxValue = cost
        #print("step",i,cost)
            #length += price[i]
            #print(length)



    return maxValue


if __name__ == '__main__':
    price = [1,5,8,9,10,17,17,20,24,30]

    # rod length
    n = 10

    print ( 'Profit is', rodCut( price, n ) )


# for(i = 0; i < len; i++)
#	 max_price = max_of_two(max_price, prices[i] + rod_cut(prices,len – i – 1)); // subtract 1 because index starts from 0
 #     return max_price;

 #time compleiity: O(2^n); space compleity: O(1)


#for (i = 1; i <= len; i++)
  #  {
 #       int
   # tmp_max = INT_MIN; // minimum
   # value
   # an
   # integer
   # can
   # hold
   # for (j = 1; j <= i; j++)
   # {
    #    tmp_idx = i - j;
    #// subtract 1 because index of prices starts from 0
    #tmp_price =prices[j-1] + max_val[tmp_idx];
    #if (tmp_price > tmp_max)
    #tmp_max = tmp_price;
    #}
    #max_val[ i ] = tmp_max;
    #}
    #return max_val[ len ];

    #Time compleity:O(n^2), space compleity: O(n)

#bottom up approach O(n^2) best approach
def cut_rod(p, n):
    """Take a list p of prices and the rod length n and return lists r and s.
    r[i] is the maximum revenue that you can get and s[i] is the length of the
    first piece to cut from a rod of length i."""
    # r[i] is the maximum revenue for rod length i
    # r[i] = -1 means that r[i] has not been calculated yet
    r = [ -1 ] * (n + 1)
    #print("abc", r)
    r[ 0 ] = 0

    # s[i] is the length of the initial cut needed for rod length i
    # s[0] is not needed
    s = [ -1 ] * (n + 1)

    for i in range ( 1, n + 1 ):
        q = -1
        for j in range ( 1, i + 1 ):
            temp = p[ j ] + r[ i - j ]
            if q < temp:
                q = temp
                s[ i ] = j
        r[ i ] = q

    return r, s


n = int ( input ( 'Enter the length of the rod in inches: ' ) )

# p[i] is the price of a rod of length i
# p[0] is not needed, so it is set to None
p = [ None ]
for i in range ( 1, n + 1 ):
    price = input ( 'Enter the price of a rod of length {} in: '.format ( i ) )
    p.append ( int ( price ) )

r,s = cut_rod(p,n)
print ( 'The maximum revenue that can be obtained:', r )
print ( 'The maximum revenue that can be obtained:', s )
print ( 'The rod needs to be cut into length(s) of ', end='' )
while n > 0:
    print ( s[ n ], end=' ' )
    n -= s[ n ]



#coin collection
import numpy as np


# function to find maximum coins in matrix by robot
def MaxCoin(r, c, CoinMatrix):
    # 2d matrix with rows=r+1 and columns=c+1
    MaxCoin = [ [ 0 ] * (c+1 ) for row in range (r+1 ) ]
    print(MaxCoin)
    # iterate through rows
    for i in range ( 1, r+1 ):
        # iterate through columns
        for j in range ( 1, c+1):
            # find maximum number in 2d matrix and add 1 if the cell containing a coin
            MaxCoin[ i ][ j ] = max ( MaxCoin[ i - 1 ][ j ], MaxCoin[ i ][ j - 1 ] , CoinMatrix[ i - 1 ][ j - 1 ])

    # print path
    print ( 'Coins collected',',','\n', np.matrix(MaxCoin))
        # print Maximum coin collected by robot
    print ( 'Maximum Coins collected by Robot are: ', MaxCoin[ r ][ c ] )
rows = 5  # number of rows
col = 6  # number of columns
# 2d matrix containing 3 rows and 3 columns

CoinMatrix = [[0, 0, 0, 0,1,0],[0, 1, 0, 1,0,0 ],[ 0,0,0,1, 0, 1 ],[0,0,1,0,0,1],[1,0,0,0,1,0]]

#CoinMatrix = [ [ 0 ] * rows for _ in range ( col ) ]

MaxCoin ( rows, col, CoinMatrix )


#Knapsack

def restricted_knapsack(values, weights, target):
    f = [float('inf') ] * (sum( values ) + 1)
    #print((f))
    f[ 0 ] = 0

    max_valid_value = -float('inf')

    for i in range ( len ( values ) ):
        g = f[ : ]
        for v in range ( 1, sum ( values ) + 1 ):
            if v - values[ i ] >= 0:
                f[ v ] = min ( g[ v ], g[ v - values[ i ] ] + weights[ i ] )
                if f[ v ] <= target:
                    max_valid_value = max ( max_valid_value, v )

    return max_valid_value

k = restricted_knapsack([70, 80, 90, 100], [20,30,40,70], 60)
print(k)


#knapsack recurisvely

def knapSack(W, wt, val, n):
    '''
    :param W: capacity of knapsack
    :param wt: list containing weights
    :param val: list containing corresponding values
    :param n: size of lists
    :return: Integer
    '''
    # code here
    if n == 0 or W == 0:
        return 0
    if wt[n-1] <= W:
        return (max(val[n-1]+knapSack(W-wt[n-1], wt, val, n-1), knapSack(W, wt, val, n-1)))
    else:
        return (knapSack(W, wt, val, n-1))


l = knapSack(60,[20,30,40,70],[70, 80, 90, 100], 4)
print(l)


# Python Program to implement
# Optimal File Merge Pattern O(nlogn) but if list is not sorted 0(n*n)


class Heap():
    # Building own implementation of Min Heap
    def __init__(self):

        self.h = []


    def parent(self, index):
        # Returns parent index for given index

        if index > 0:
            return (index - 1) // 2

    def lchild(self, index):
        # Returns left child index for given index

        return (2 * index) + 1

    def rchild(self, index):
        # Returns right child index for given index

        return (2 * index) + 2

    def addItem(self, item):

        # Function to add an item to heap
        self.h.append(item)
        #print((self.h))
        #print(len(self.h))

        if len(self.h) == 1:

            # If heap has only one item no need to heapify
            return
        #print((self.h))
        index = len(self.h) - 1
        parent = self.parent(index)
        #print(parent)
        # Moves the item up if it is smaller than the parent
        while index > 0 and item < self.h[parent]:
            self.h[index], self.h[parent] = self.h[parent], self.h[parent]
            index = parent
            parent = self.parent(index)
            #print(index)
    def deleteItem(self):

        # Function to add an item to heap
        length = len(self.h)
        self.h[0], self.h[length-1] = self.h[length-1], self.h[0]
        deleted = self.h.pop()

        # Since root will be violating heap property
        # Call moveDownHeapify() to restore heap property
        self.moveDownHeapify(0)

        return deleted

    def moveDownHeapify(self, index):

        # Function to make the items follow Heap property
        # Compares the value with the children and moves item down

        lc, rc = self.lchild(index), self.rchild(index)
        length, smallest = len(self.h), index

        if lc < length and self.h[lc] <= self.h[smallest]:
            smallest = lc

        if rc < length and self.h[rc] <= self.h[smallest]:
            smallest = rc

        if smallest != index:
            # Swaps the parent node with the smaller child
            self.h[smallest], self.h[index] = self.h[index], self.h[smallest]

            # Recursive call to compare next subtree
            self.moveDownHeapify(smallest)

    def increaseItem(self, index, value):
        # Increase the value of 'index' to 'value'

        if value <= self.h[index]:
            return

        self.h[index] = value
        self.moveDownHeapify(index)


class OptimalMergePattern():
    def __init__(self, n, items):

        self.n = n
        self.items = items
        self.heap = Heap()

    def optimalMerge(self):

        # Corner cases if list has no more than 1 item
        if self.n <= 0:
            return 0

        if self.n == 1:
            return self.items[0]

        # Insert items into min heap
        for _ in self.items:
            self.heap.addItem(_)

        count = 0
        while len(self.heap.h) != 1:
            tmp = self.heap.deleteItem()
            count += (tmp + self.heap.h[0])
            self.heap.increaseItem(0, tmp + self.heap.h[0])
            #print(count)
        return count


# Driver Code
if __name__ == '__main__':
    array = [5, 3, 2, 7, 9, 13]
    for j in range(1,len(array)):
        key = array[j]
        i = j-1
        while i >= 0 and array[i] > key:
            array[i + 1]= array[i]
            i = i-1
            array[i+1]= key
        print(array)

    OMP = OptimalMergePattern(5, array)
    ans = OMP.optimalMerge()
    print(ans)



#coin change problem
# Dynamic Programming Python implementation of Coin
# Change problem
def count(S, m, n):
# We need n+1 rows as the table is constructed
	# in bottom up manner using the base case 0 value
	# case (n = 0)
	table = [[0 for x in range(m)] for x in range(n+1)]

	# Fill the entries for 0 value case (n = 0)
	for i in range(m):
		table[0][i] = 1

	# Fill rest of the table entries in bottom up manner
	for i in range(1, n+1):
		for j in range(m):

			# Count of solutions including S[j]
			x = table[i - S[j]][j] if i-S[j] >= 0 else 0

			# Count of solutions excluding S[j]
			y = table[i][j-1] if j >= 1 else 0

			# total count
			table[i][j] = x + y

	return table[n][m-1]

# Driver program to test above function
arr = [1, 2, 3]
m = len(arr)
n = 4
print(count(arr, m, n))



"""The following implementation assumes that the activities
are already sorted according to their finish time"""

"""Prints a maximum set of activities that can be done by a
single person, one at a time"""
# n --> Total number of activities
# s[]--> An array that contains start time of all activities
# f[] --> An array that contains finish time of all activities

def printMaxActivities(s , f ):
	n = len(f)
	print ("The following activities are selected")

	# The first activity is always selected
	i = 0
	print (i,end=' ')

	# Consider rest of the activities
	for j in range(n):

		# If this activity has start time greater than
		# or equal to the finish time of previously
		# selected activity, then select it
		if s[j] >= f[i]:
			print (j,end=' ')
			i = j

# Driver program to test above function
s = [0,  1,  5,  6, 10,  6,  8,  4, 13, 19, 14, 21, 25, 26, 29, 28]
f = [2,  3,  8, 10, 12, 15, 15, 15, 16, 21, 22, 29, 30, 33, 37, 37]
printMaxActivities(s , f)











