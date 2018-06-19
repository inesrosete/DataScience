
# Python Exercises

[Inês Rosete](http://www.linkedin.com/in/inesrosete) | January, 2018

When I started this work I felt the need to remember some python concepts. Thus, in this notebook I present some exercises of [Codility](https://app.codility.com/programmers/lessons/1-iterations/?target=_blank) and others that I did in the first weeks of work. In the first part I present some Codility exercises and in the second I present random exercises that I did with friends.

___
## Codility Exercises
___

<div class="alert alert-success">
<b> Exercise 1 </b><br>

<a href="https://app.codility.com/programmers/lessons/4-counting_elements/perm_check/" target="_blank">PermCheck</a> Codility.
</div>


A non-empty array A consisting of N integers is given.

A permutation is a sequence containing each element from 1 to N once, and only once.

For example, array A such that:

    A[0] = 4
    A[1] = 1
    A[2] = 3
    A[3] = 2
is a permutation, but array A such that:

    A[0] = 4
    A[1] = 1
    A[2] = 3
is not a permutation, because value 2 is missing.

The goal is to check whether array A is a permutation.

Write a function:

def solution(A)

that, given an array A, returns 1 if array A is a permutation and 0 if it is not.

For example, given array A such that:

    A[0] = 4
    A[1] = 1
    A[2] = 3
    A[3] = 2
the function should return 1.

Given array A such that:

    A[0] = 4
    A[1] = 1
    A[2] = 3
the function should return 0.


```python
def solution(A):

    N = len(A)
    range1 = list(range(1,N+1))
    
    if set(A) == set(range1): return 1
    else: return 0


assert solution([1,2,6]) == 0
assert solution([1,2,3]) == 1
assert solution([2,2,3]) == 0
assert solution([3,2,1]) == 1
assert solution([1]) == 1
assert solution([2]) == 0
assert solution([]) == 1
```

<div class="alert alert-success">
<b> Exercise 2 </b><br>

<a href="https://app.codility.com/programmers/lessons/4-counting_elements/max_counters/" target="_blank">MaxCounters</a>  Codility.
</div>



You are given N counters, initially set to 0, and you have two possible operations on them:

increase(X) − counter X is increased by 1,
max counter − all counters are set to the maximum value of any counter.
A non-empty array A of M integers is given. This array represents consecutive operations:

if A[K] = X, such that 1 ≤ X ≤ N, then operation K is increase(X),
if A[K] = N + 1 then operation K is max counter.
For example, given integer N = 5 and array A such that:

    A[0] = 3
    A[1] = 4
    A[2] = 4
    A[3] = 6
    A[4] = 1
    A[5] = 4
    A[6] = 4
the values of the counters after each consecutive operation will be:

    (0, 0, 1, 0, 0)
    (0, 0, 1, 1, 0)
    (0, 0, 1, 2, 0)
    (2, 2, 2, 2, 2)
    (3, 2, 2, 2, 2)
    (3, 2, 2, 3, 2)
    (3, 2, 2, 4, 2)
The goal is to calculate the value of every counter after all operations.

Write a function:

def solution(N, A)

that, given an integer N and a non-empty array A consisting of M integers, returns a sequence of integers representing the values of the counters.

The sequence should be returned as:

a structure Results (in C), or
a vector of integers (in C++), or
a record Results (in Pascal), or
an array of integers (in any other programming language).
For example, given:

    A[0] = 3
    A[1] = 4
    A[2] = 4
    A[3] = 6
    A[4] = 1
    A[5] = 4
    A[6] = 4
the function should return [3, 2, 2, 4, 2], as explained above.


```python
def solution(N,A):
    
    r = [0]*N
    
    for x in A:
        if 1 <= x <= N:
            r[x-1] += 1
        else:
            r = [max(r)]*N
    return r

assert solution(5,[3,4,4,6,1,4,4])== [3, 2, 2, 4, 2]
```

<div class="alert alert-success">
<b> Exercise 3 </b><br>

<a href="https://app.codility.com/programmers/lessons/2-arrays/odd_occurrences_in_array/" target="_blank">OddOccurrencesInArray</a> Codility.

</div>

A non-empty zero-indexed array A consisting of N integers is given. The array contains an odd number of elements, and each element of the array can be paired with another element that has the same value, except for one element that is left unpaired.

For example, in array A such that:

```python
  A[0] = 9  A[1] = 3  A[2] = 9
  A[3] = 3  A[4] = 9  A[5] = 7
  A[6] = 9
```


* the elements at indexes 0 and 2 have value 9,
* the elements at indexes 1 and 3 have value 3,
* the elements at indexes 4 and 6 have value 9,
* the element at index 5 has value 7 and is unpaired.


Write a function:

```python
def solution(A)
```

that, given an array A consisting of N integers fulfilling the above conditions, returns the value of the unpaired element.

For example, given array A such that:

```python
  A[0] = 9  A[1] = 3  A[2] = 9
  A[3] = 3  A[4] = 9  A[5] = 7
  A[6] = 9
```

the function should return 7, as explained in the example above.

Assume that:

* N is an odd integer within the range [1..1,000,000];
* each element of array A is an integer within the range [1..1,000,000,000];
* all but one of the values in A occur an even number of times.

Complexity:

* expected worst-case time complexity is O(N);
* expected worst-case space complexity is O(1), beyond input storage (not counting the storage required for input arguments).


```python
def solution(A):
    r={}
    for x in A:
        if x in r:
         r[x] += 1
        else:
         r[x] = 1
    
    for x in r:
        if r[x] % 2 != 0: return x


assert solution([9,7,9,3,9,3,9]) == 7
assert solution([1,2,3,3,2,1,2,3,1,3,1,2,9]) == 9
assert solution([6,5,9,9,6]) == 5
assert solution([1,1,1]) == 1
```

<div class="alert alert-success">
<b> Exercise 4 </b><br>

<a href="https://app.codility.com/programmers/lessons/3-time_complexity/perm_missing_elem/" target="_blank">PermMissingElem</a> Codility.

</div>

A zero-indexed array A consisting of N different integers is given. The array contains integers in the range [1..(N + 1)], which means that exactly one element is missing.

Your goal is to find that missing element.

Write a function:

```python
def solution(A)
```

that, given a zero-indexed array A, returns the value of the missing element.

For example, given array A such that:

```python
  A[0] = 2
  A[1] = 3
  A[2] = 1
  A[3] = 5
```

the function should return 4, as it is the missing element.

Assume that:

* N is an integer within the range [0..100,000];
* the elements of A are all distinct;
* each element of array A is an integer within the range [1..(N + 1)].


```python
def solution(A):
    
    N = len(A)
    rangeA = list(range(1,N+2))
    
    return set(rangeA).difference(set(A)).pop()


assert solution([2,3,1,5]) == 4
assert solution([2,3,1,4,6]) == 5
assert solution([1]) == 2
assert solution([2,3,1]) == 4
```

___
## Other Exercises
___

These are a set of challenges I made when I started learning Python with some friends. Most of the exercises are very trivial!

<div class="alert alert-success">
<b> Exercise 1 </b><br>

Considering the lists x and y, get all the elements of x that are not in y. Do two approaches: using sets and using list comprehension. <br>
<ul style="list-style-type:circle">
  <li>x = [1, 2, 3, 4, 5]</li>
  <li>y = [3, 4, 5, 6, 7]</li>
</ul>
</div>


```python
#First approach - using sets

x = [1, 2, 3, 4, 5]
y = [3, 4, 5, 6, 7]

a = set(x)
c = set(y)

list(a.difference(c))
```




    [1, 2]




```python
#Second approach - using list comprehension

[k for k in x if k not in y]
```




    [1, 2]



<div class="alert alert-success">
<b> Exercise 2 </b><br>

Browsing the dictionary to obtain, for each dictionary entry, a string with the following format: '1 is "one", and could be 1.00'. <br>
<ul style="list-style-type:circle">
  <li>d = {1: 'one', 2: 'two', 3: 'three', 4: 'four'}</li>
</ul>
</div>


```python
d = {1: 'one', 2: 'two', 3: 'three', 4: 'four'}

for k,v in d.items():
    print('{0} is "{1:^5}", and could be {0:.2f}'.format(k,v))
```

    1 is " one ", and could be 1.00
    2 is " two ", and could be 2.00
    3 is "three", and could be 3.00
    4 is "four ", and could be 4.00
    

<div class="alert alert-success">
<b> Exercise 3 </b><br>

Calculate the frequency distribution for each character in s. <br>
<ul style="list-style-type:circle">
  <li>s = 'this is a string, am I right?'</li>
</ul>
</div>


```python
#Approach - create a list with each character of the string s and then determine the frequency of each one.

s = 'this is a string, am I right?'

print(list(s))
```

    ['t', 'h', 'i', 's', ' ', 'i', 's', ' ', 'a', ' ', 's', 't', 'r', 'i', 'n', 'g', ',', ' ', 'a', 'm', ' ', 'I', ' ', 'r', 'i', 'g', 'h', 't', '?']
    


```python
r={}
for t in list(s):
    if t in r:
     r[t] += 1
    else:
     r[t] = 1
    
print(r)
```

    {'t': 3, 'h': 2, 'i': 4, 's': 3, ' ': 6, 'a': 2, 'r': 2, 'n': 1, 'g': 2, ',': 1, 'm': 1, 'I': 1, '?': 1}
    

<div class="alert alert-success">
<b> Exercise 4 </b><br>

Implement the Fibonacci Sequence

</div>


```python
#First approach - using function

import math

def fib(x):
    if x == 0: return 0
    elif x == 1:return 1
    else: return fib(x-1)+fib(x-2)
    
fib(4)
```




    3




```python
#Seconf approach - using while

a = 0
b = 1
n = int(input())

if n == 0:
    b = a
elif n > 1:
    x = 2
    while x <= n:
            last_a = a
            a = b
            b = last_a + b
            x+=1
            
print('F({0}) é {1}'.format(n, b))
```

    4
    F(4) é 3
    

<div class="alert alert-success">
<b> Exercise 5 </b><br>

Use the input () function to get a terminal number (it stays as a string) and then print the binary representation of that number <br>
    http://interactivepython.org/runestone/static/pythonds/BasicDS/ConvertingDecimalNumberstoBinaryNumbers.html <br>
<ul style="list-style-type:circle">
  <li>decimal = input()</li>
</ul>
</div>


```python
decimal = int(input())
li = []

while decimal > 0:
    resto_divisao = decimal % 2
    parte_inteira_divisao = decimal // 2
    li.append(resto_divisao)
    decimal = parte_inteira_divisao
    
print(li[::-1])
```

    45
    [1, 0, 1, 1, 0, 1]
    

Lastly I would like to thank you for reading the document to the end. Please let me know your thoughts!

___

## You may also like

* [Flight Delays Exploration](https://github.com/inesrosete/DataScience/blob/master/flight_delays_exploration.html)
* [Titanic Exploration]() (Available soon)

___

## Acknowledgments

I would like to thank Luís Silva and João Neto for the support and help they had during the execution of the first draft of this challenge in January'18.

___
[Inês Rosete](http://www.linkedin.com/in/inesrosete) | Last editing **May 2018 ** 
