Core concepts to practice:

- Array creation and reshaping: np.zeros, np.ones, reshape, transpose
- Indexing: boolean indexing, fancy indexing, slicing
- Broadcasting rules (critical — this trips up most people)
- Aggregation: mean, std, min, max along specific axes
- Stacking and concatenation: np.stack, np.concatenate, np.hstack, np.vstack
- File I/O: loading from parquet via Pandas → NumPy

### np.zeros, np.ones, reshape, transpose

```python
# The function zeros creates an array full of zeros, the function ones creates an array full of ones, and the function empty creates an array whose initial content is random and depends on the state of the memory. By default, the dtype of the created array is float64, but it can be specified via the key word argument dtype.

np.zeros((3, 4))
array([[0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.]])


np.ones((2, 3, 4), dtype=np.int16)
array([[[1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1]],

       [[1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1]]], dtype=int16)

np.empty((2, 3)) 
array([[3.73603959e-262, 6.02658058e-154, 6.55490914e-260],  # may vary
       [5.30498948e-313, 3.14673309e-307, 1.00000000e+000]])
       

# When arange is used with floating point arguments, it is generally not possible to predict the number of elements obtained, due to the finite floating point precision. For this reason, it is usually better to use the function linspace that receives as an argument the number of elements that we want, instead of the step:
np.arange(10, 30, 5) #from 10 to 30, each add 5 (seperate by 5)
array([10, 15, 20, 25])
np.arange(0, 2, 0.3)  # it accepts float arguments
array([0. , 0.3, 0.6, 0.9, 1.2, 1.5, 1.8])

#

a = np.arange(6)                    # 1d array
print(a)
# >>> [0 1 2 3 4 5]

#-> arange is the list [] from 0 to 12
#-> reshape(row,col) define how to seperate into 4 rows, and 3 columns
b = np.arange(12).reshape(4, 3)     # 2d array
print(b)
# [[ 0  1  2]
#  [ 3  4  5]
#  [ 6  7  8]
#  [ 9 10 11]]

c = np.arange(24).reshape(2, 3, 4)  # 3d array
print(c)
# [[[ 0  1  2  3]
#   [ 4  5  6  7]
#   [ 8  9 10 11]]

#  [[12 13 14 15]
#   [16 17 18 19]
#   [20 21 22 23]]]

#transpose: https://numpy.org/doc/stable/reference/generated/numpy.transpose.html#numpy.transpose
# For a 1-D array, this returns an unchanged view of the original array, as a transposed vector is simply the same vector. To convert a 1-D array into a 2-D column vector, an additional dimension must be added, e.g., np.atleast_2d(a).T achieves this, as does a[:, np.newaxis]. For a 2-D array, this is the standard matrix transpose. For an n-D array, if axes are given, their order indicates how the axes are permuted (see Examples). If axes are not provided, then transpose(a).shape == a.shape[::-1].

# example 1: 1d array, no change:
a = np.array([1, 2, 3, 4])
a
array([1, 2, 3, 4])
np.transpose(a)
array([1, 2, 3, 4])

# example 2: 2d array transpose, switch row and column
a = np.array([[1, 2], [3, 4]])
a
array([[1, 2],
       [3, 4]])
np.transpose(a)
# array([[1, 3],
#        [2, 4]])

#example 3: 3d array 当你调用 np.transpose(a, (1, 0, 2)) 时，你传入的元组定义了新数组的轴，应该由旧数组的哪些轴来填充：
# 新数组的第 1 个位置：放原本的 轴 1（长度是 2）
# 新数组的第 2 个位置：放原本的 轴 0（长度是 1）
# 新数组的第 3 个位置：放原本的 轴 2（保持不动，长度是 3）
# 所以，变换后的新顺序变成了 (轴1, 轴0, 轴2)
a = np.ones((1, 2, 3))
np.transpose(a, (1, 0, 2)).shape
(2, 1, 3)

#example 4: 为什么默认行为是完全反转？
# 这其实来源于线性代数中二维矩阵转置的自然延伸：
# 二维矩阵 M 的形状是 (row, col)，转置后变成 (col, row)。这本质上就是把轴 (0, 1) 反转成了 (1, 0)
a = np.ones((2, 3, 4, 5))
np.transpose(a).shape
(5, 4, 3, 2)

#example 5: 
# 1. 原始状态 (Shape: 3, 4, 5) 
#             >>> 轴 0; 轴 1; 轴 2
# 2. 映射变换 (-1, 0, -2) >>> -1 实际指向的是 轴 2（长度 5）; 0 实际指向的是 轴 0（长度 3）; -2 实际指向的是 轴 1（长度 4）
# 3. 结果形状 >>> 新排列: (轴 2,轴 0,轴 1)⟶(5,3,4)
a = np.arange(3*4*5).reshape((3, 4, 5))
np.transpose(a, (-1, 0, -2)).shape
(5, 3, 4)

```


### Indexing: boolean indexing, fancy indexing, slicing
```python
# One-dimensional

# 间隔改数
# equivalent to a[0:6:2] = 1000;
# from start to position 6, exclusive, set every 2nd element to 1000
a[:6:2] = 1000
a
array([1000,    1, 1000,   27, 1000,  125,  216,  343,  512,  729])

#直接reverse array
a[::-1]  # reversed a
array([ 729,  512,  343,  216,  125, 1000,   27, 1000,    1, 1000])

```

Multidimensional arrays can have one index per axis. These indices are given in a tuple separated by commas:
```python
# 1. 数组的生成：np.fromfunction
def f(x, y):
    return 10 * x + y
b = np.fromfunction(f, (5, 4), dtype=int)
```
这一段代码展示了 NumPy 中两个非常核心且强大的概念：**通过函数构建数组** 以及 **多维数组的切片（Slicing）与索引**。

 1. 数组的生成：`np.fromfunction`

```python
def f(x, y):
    return 10 * x + y
b = np.fromfunction(f, (5, 4), dtype=int)

```

`np.fromfunction` 的核心逻辑是：**它会把数组中每个位置的“坐标（索引）”作为参数传给函数，然后用函数的返回值填充该位置。**

因为指定的形状是 `(5, 4)`（5行4列），NumPy 会对这 20 个格子进行遍历：

* 对于第 0 行第 0 列的格子：传入 `x=0, y=0` -> function 10*0 + 0 = 0
* 对于第 1 行第 2 列的格子：传入 `x=1, y=2` -> function 10*1 + 2 = 12
* 对于第 4 行第 3 列的格子：传入 `x=4, y=3` -> function 10*4 + 3 = 43

这就巧妙地生成了一个十位数代表**行号**、个位数代表**列号**的矩阵。
```python
b
array([[ 0,  1,  2,  3],
       [10, 11, 12, 13],
       [20, 21, 22, 23],
       [30, 31, 32, 33],
       [40, 41, 42, 43]])
```
2. 数组的切片与索引

在 NumPy 中，多维切片的语法是 `b[行切片row, 列切片col]`。如果某一个维度只写一个冒号 `:`，就代表**选取该维度的所有元素**。

#### 示例 1：单点索引 `b[2, 3]`

* **含义**：精准定位到第 2 行（第三行）、第 3 列（第四大列）相交的元素。
* **结果**：`23`

#### 示例 2：获取特定列 `b[0:5, 1]` 与 `b[:, 1]`

* `b[0:5, 1]` 的意思是：row/行索引(0:5) = 从 `0` 选到 `4`（左闭右开，不包含5），col/列索引(1)固定选 `1`。
表示 each row in the second column of b

* `b[:, 1]` 中的 `:` 是简写，表示**所有行**。
* **结果**：它们都把第二列的元素全部抽了出来，合并成了一个一维数组 `[1, 11, 21, 31, 41]`。

#### 示例 3：获取特定行区域 `b[1:3, :]`

* **含义**：行索引选择 `1` 和 `2`（即第二行和第三行），列索引选择 `:`（所有列）。
* **结果**：切出了一个 `(2, 4)` 的二维子矩阵：
```python
[[10, 11, 12, 13], 
 [20, 21, 22, 23]]

>> 1:3 代表1-2 rows
>> : 代表所有col 
所以return >> 第一行[], 第二行[]
```
#### When fewer indices are provided than the number of axes, the missing indices are considered complete slices:
```python
b[-1]   # the last row. Equivalent to b[-1, :]
array([40, 41, 42, 43])
```
#### 快捷用法: The dots (...) represent as many colons as needed to produce a complete indexing tuple. For example, if x is an array with 5 axes, then
```python
x[1, 2, ...] is equivalent to x[1, 2, :, :, :],
x[..., 3] to x[:, :, :, :, 3] and
x[4, ..., 5, :] to x[4, :, :, 5, :].
```

#### If the number of objects in the selection tuple is less than N, then : is assumed for any subsequent dimensions. For example:
```python
x = np.array([[[1],[2],[3]], [[4],[5],[6]]])
x.shape
(2, 3, 1)
x[1:2]
array([[[4],[5],[6]]])
x.shape(1,3,1)

x[1]
array([[4],[5],[6]]) 
x.shape(3,1)
```
---

#### 💡 一个容易踩坑的细节：维度的变化

细心的你可能会发现：

* 当你运行 `b[:, 1]` 时，结果变成了一个**一维数组**（Shape 从 `(5, 4)` 降级成了 `(5,)`）。这是因为列那一轨你传的是一个**标量数字 `1**`。
* 如果你希望切完之后**依然保持二维矩阵的形状**（即具有列的属性），你可以在列的地方传入一个列表或切片，例如 `b[:, 1:2]`，这样它的 Shape 就会是 `(5, 1)`。

NumPy slicing creates a view instead of a copy as in the case of built-in Python sequences such as string, tuple and list. Care must be taken when extracting a small portion from a large array which becomes useless after the extraction, because the small portion extracted contains a reference to the large original array whose memory will not be released until all arrays derived from it are garbage-collected. In such cases an explicit copy() is recommended.

---
#### Advanced indexing / 高级索引
当选择对象 `obj` 为非元组序列对象、`ndarray`（数据类型为整数或布尔值），或者包含至少一个序列对象或 `ndarray`（数据类型为整数或布尔值）的元组时，即触发高级索引。高级索引主要分为两种类型：整数索引和布尔索引。

高级索引总是返回数据的副本（这一点与返回视图的基础切片操作截然不同）。

警告
根据高级索引的定义，表达式 `x[(1, 2, 3),]` 与 `x[(1, 2, 3)]` 在本质上有着根本区别。后者等价于 `x[1, 2, 3]`，将触发基础选择操作；而前者则会触发高级索引。请务必理解产生这一差异的原因。

1. Integer array indexing
1d array
```python
x = np.arange(10, 1, -1)
x
array([10,  9,  8,  7,  6,  5,  4,  3,  2])
# idx: 0.   1.  2.  3.  4.  5.  6.  7.  8
# -idx 0.   -8  -7  -6  -5  -4  -3  -2  -1

# exampe 1, 想取idx 3, 3, 1, 8
x[np.array([3, 3, 1, 8])]
array([7, 7, 9, 2])

#exampe 2
x[np.array([3, 3, -3, 8])]
array([7, 7, 4, 2])
```
2d array
```python
y = np.arange(35).reshape(5, 7)
y
array([[ 0,  1,  2,  3,  4,  5,  6],
       [ 7,  8,  9, 10, 11, 12, 13],
       [14, 15, 16, 17, 18, 19, 20],
       [21, 22, 23, 24, 25, 26, 27],
       [28, 29, 30, 31, 32, 33, 34]])
y[np.array([0, 2, 4]), np.array([0, 1, 2])]
#y[row,col]
array([ 0, 15, 30])
```
简单来说, 如果正好np-row和np-col长度一致, 那么正好可以一组一组找到具体的idx>>> y[0, 0], y[2, 1], y[4, 2]
The result would be a 1D array. 直接call y[row,col]

In this case, if the index arrays have a matching shape, and there is an index array for each dimension of the array being indexed, the resultant array has the same shape as the index arrays, and the values correspond to the index set for each position in the index arrays. In this example, the first index value is 0 for both index arrays, and thus the first value of the resultant array is y[0, 0]. The next value is y[2, 1], and the last is y[4, 2].

如果不一致: 可以broadcast:
```python
y[np.array([0, 2, 4]), 1]
#  --> same as:
y[np.array([0, 2, 4]), np.array([1, 1, 1])]
```
更难的例子 (如果return不是1d而是一个subgrid, 这个时候需要broadcast), 在此之前, 下面是一个broadcast非常完善的例子:

`[1,2,3]`和`[a,b,c,d,e]`本身在**1D下没法对齐**(3和5不兼容)。但用`None`(= `np.newaxis`)加一个新轴,就能在**2D下对齐**——只不过这时做的运算就不再是"一对一elementwise"了,而是**pairwise(两两组合)**。

#### 怎么做

```python
a = np.array([1, 2, 3])              # shape (3,)
b = np.array([10, 20, 30, 40, 50])   # shape (5,)

a + b              # ❌ ValueError: (3,) 和 (5,) 不兼容

a[:, None] + b[None, :]   # ✅ (3,1) + (1,5) → (3,5)
# 或者更简洁(右对齐自动补1):
a[:, None] + b            # ✅ (3,1) + (5,) → (3,5)
```

#### 发生了什么?用图看

```
a[:, None] shape (3,1):       b[None, :] shape (1,5):
[[1],                          [[10, 20, 30, 40, 50]]
 [2],
 [3]]

按broadcast规则各自虚拟拉伸到 (3,5):

a拉伸成:                       b拉伸成:
[[1, 1, 1, 1, 1],              [[10, 20, 30, 40, 50],
 [2, 2, 2, 2, 2],               [10, 20, 30, 40, 50],
 [3, 3, 3, 3, 3]]               [10, 20, 30, 40, 50]]

逐元素相加 → (3, 5) 矩阵:
[[11, 21, 31, 41, 51],
 [12, 22, 32, 42, 52],
 [13, 23, 33, 43, 53]]
```

结果的`[i, j]`位置 = `a[i] + b[j]` —— 所有3×5=15种组合。

#### 关键:意义变了

| 操作 | shape | 含义 |
|---|---|---|
| `a + b` (1D对1D,长度相同) | `(N,)` | 一对一elementwise |
| `a[:,None] + b[None,:]` | `(N, M)` | **pairwise**,所有两两组合 |

这正是**outer-product / pairwise距离 / attention score**的常用套路:

```python
# pairwise欧氏距离
A  # (N, D)
B  # (M, D)
diff = A[:, None, :] - B[None, :, :]   # (N, 1, D) - (1, M, D) → (N, M, D)
dist = (diff**2).sum(-1)               # (N, M)

# attention score (q和所有k的点积)
Q  # (seq_q, d)
K  # (seq_k, d)
scores = Q @ K.T   # 本质也是pairwise组合
```

#### 一句话

> `None`不是"让3和5在1D里对齐",而是**升维**到2D,让`(3,)`和`(5,)`变成正交的两个轴`(3,1)`和`(1,5)`,结果是`(3,5)`的pairwise组合矩阵——**含义从elementwise变成了outer/pairwise**。

现在来看-> return sub-grid 而不是1darray
```python
```