Core concepts to practice:

- Array creation and reshaping: np.zeros, np.ones, reshape, transpose
- Indexing: boolean indexing, fancy indexing, slicing
- Broadcasting rules (critical — this trips up most people)
- Aggregation: mean, std, min, max along specific axes
- Stacking and concatenation: np.stack, np.concatenate, np.hstack, np.vstack
- File I/O: loading from parquet via Pandas → NumPy

# 1 np.zeros, np.ones, reshape, transpose

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

# 2 Indexing: boolean indexing, fancy indexing, slicing wiz Broadcasting

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

- 对于第 0 行第 0 列的格子：传入 `x=0, y=0` -> function 10*0 + 0 = 0
- 对于第 1 行第 2 列的格子：传入 `x=1, y=2` -> function 10*1 + 2 = 12
- 对于第 4 行第 3 列的格子：传入 `x=4, y=3` -> function 10*4 + 3 = 43

这就巧妙地生成了一个十位数代表**行号**、个位数代表**列号**的矩阵。

```python
b
array([[ 0,  1,  2,  3],
       [10, 11, 12, 13],
       [20, 21, 22, 23],
       [30, 31, 32, 33],
       [40, 41, 42, 43]])
```

1. 数组的切片与索引

在 NumPy 中，多维切片的语法是 `b[行切片row, 列切片col]`。如果某一个维度只写一个冒号 `:`，就代表**选取该维度的所有元素**。

#### 示例 1：单点索引 `b[2, 3]`

- **含义**：精准定位到第 2 行（第三行）、第 3 列（第四大列）相交的元素。
- **结果**：`23`

#### 示例 2：获取特定列 `b[0:5, 1]` 与 `b[:, 1]`

- `b[0:5, 1]` 的意思是：row/行索引(0:5) = 从 `0` 选到 `4`（左闭右开，不包含5），col/列索引(1)固定选 `1`。
表示 each row in the second column of b
- `b[:, 1]` 中的 `:` 是简写，表示**所有行**。
- **结果**：它们都把第二列的元素全部抽了出来，合并成了一个一维数组 `[1, 11, 21, 31, 41]`。

#### 示例 3：获取特定行区域 `b[1:3, :]`

- **含义**：行索引选择 `1` 和 `2`（即第二行和第三行），列索引选择 `:`（所有列）。
- **结果**：切出了一个 `(2, 4)` 的二维子矩阵：

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

- 当你运行 `b[:, 1]` 时，结果变成了一个**一维数组**（Shape 从 `(5, 4)` 降级成了 `(5,)`）。这是因为列那一轨你传的是一个**标量数字 `1`**。
- 如果你希望切完之后**依然保持二维矩阵的形状**（即具有列的属性），你可以在列的地方传入一个列表或切片，例如 `b[:, 1:2]`，这样它的 Shape 就会是 `(5, 1)`。

NumPy slicing creates a view instead of a copy as in the case of built-in Python sequences such as string, tuple and list. Care must be taken when extracting a small portion from a large array which becomes useless after the extraction, because the small portion extracted contains a reference to the large original array whose memory will not be released until all arrays derived from it are garbage-collected. In such cases an explicit copy() is recommended.

---

### Advanced indexing / 高级索引

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

# 3 broadcasting (within 2)

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


| 操作                      | shape    | 含义                  |
| ----------------------- | -------- | ------------------- |
| `a + b` (1D对1D,长度相同)    | `(N,)`   | 一对一elementwise      |
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
x = np.array([[ 0,  1,  2],
              [ 3,  4,  5],
              [ 6,  7,  8],
              [ 9, 10, 11]])
rows = np.array([0, 3], dtype=np.intp)
columns = np.array([0, 2], dtype=np.intp)
x[rows, columns] 不对: 因为输入的是全部一维: (0,0), (3,2)
正确:x[rows[:, np.newaxis], columns] 升维, 第二个也自动升维(broadcasting)
x[np.ix_(rows, columns)] 也是一样的
结果
array([[ 0,  2],
       [ 9, 11]])
```

### boolean array index

1. filter and select all not numpy.nan

```python
x = np.array([[1., 2.], [np.nan, 3.], [np.nan, np.nan]])
x[~np.isnan(x)]
array([1., 2., 3.])
```

1. add a contant to all negative elements:

```python
x = np.array([1., -1., -2., 3])
x[x < 0] += 20
x
array([ 1., 19., 18., 3.])
```

1. 复杂多维

```python
x = np.arange(35).reshape(5, 7)
b = x > 20
b[:, 5]
array([False, False, False,  True,  True])
x[b[:, 5]]
array([[21, 22, 23, 24, 25, 26, 27],
      [28, 29, 30, 31, 32, 33, 34]])
```

逐步拆:

`x` 是 5×7 的矩阵,值是 0~34。

`b = x > 20` 得到一个 shape 完全相同 (5,7) 的 boolean mask。

`b[:, 5]` 取第 5 列(也就是 `x` 的第 5 列 `[5, 12, 19, 26, 33]` 与 20 比较的结果):

```
[False, False, False, True, True]

```

这是一个长度为 5 的 **1D** boolean 数组。

`x[b[:, 5]]` —— 关键在这里。索引数组是 1D,长度 5,正好等于 `x` 的第 0 轴(行数)。numpy 的规则是:**1D boolean mask 会沿着第 0 轴对齐**,True 的位置保留对应的整行。

所以拿到第 3、4 两行:

```
[[21, 22, 23, 24, 25, 26, 27],
 [28, 29, 30, 31, 32, 33, 34]]
```

**语义可以这么读**:"挑出第 5 列大于 20 的那些行"。等价于 `x[x[:, 5] > 20]`,这是更常见的写法。

对比一下容易混的几种:

- `x[b]` —— 用 **2D** mask,shape (5,7),返回 1D 扁平数组,只保留所有 `>20` 的元素。
- `x[:, b[:, 5]]` —— 长度 5 的 mask 试图沿 axis=1 (有 7 列) 对齐,**shape 不匹配会报错**。
- `x[b[:, 5], :]` —— 和 `x[b[:, 5]]` 完全等价,只是显式写出来。

🙋问题:

为什么`b = x > 20` 得到一个 shape 完全相同 (5,7) 的 boolean mask, 又为什么b[:,5]取的是column不是row

两个问题分别答:

#### 1. 为什么 `x > 20` 是 (5,7)

这是 **element-wise broadcasting**。`x > 20` 中,左边是 (5,7) 的数组,右边是标量 20。numpy 会把 20 "广播" 成一个 (5,7) 的虚拟数组(每个位置都是 20),然后**逐元素**比较:

```
x:           20 (broadcast):       x > 20:
0  1 ... 5             20 20 ... 20             F F ... F
7  8 ...12             20 20 ... 20             F F ... F
14 ... 20              20 20 ... 20             F F ... F
21 ... 27              20 20 ... 20             T T ... T
28 ... 34              20 20 ... 20             T T ... T

```

所有 numpy 的比较运算符 (`>`, `<`, `==`, `!=`)、算术运算符 (`+`, `*`) 都是这样工作的 —— **输出 shape = 输入 shape**(广播后)。所以 35 个元素进去,35 个 boolean 出来,自然还是 (5,7)。

#### 2. 为什么 `b[:, 5]` 取的是 column

这是 numpy/Python 索引的语法约定:**逗号分隔不同维度,从 axis 0 开始**。

对一个 2D 数组 `b[i, j]`:

- 第一个位置 `i` → axis 0(行)
- 第二个位置 `j` → axis 1(列)

`:` 表示"这个维度全要",具体数字表示"这个维度取这一个 index"。所以:


| 写法          | 含义                | 结果 shape  |
| ----------- | ----------------- | --------- |
| `b[:, 5]`   | 所有行,第 5 列         | (5,) ← 一列 |
| `b[5, :]`   | 第 5 行,所有列         | (7,) ← 一行 |
| `b[5]`      | 简写,等价 `b[5, :]`   | (7,) ← 一行 |
| `b[:, 5:6]` | 所有行,第 5 列(但保留 2D) | (5, 1)    |


记忆方法:**位置 = 轴**。`b[:, 5]` 的 5 出现在**第二个位置**,所以是在 axis 1(列)上取 index=5,也就是"第 5 列"。第一个位置是 `:`,所以行全要 → 拿到的是一整列里的 5 个元素。

回到原题:`b[:, 5]` 取的是 `b` 第 5 列上每一行的值,正好对应"`x` 第 5 列每个元素是否 > 20"。

🙋为什么完全等价

`x[b[:, 5], :]` —— 和 `x[b[:, 5]]` 完全等价,只是显式写出来。

因为 numpy 对**省略的尾部维度**有一条默认规则:**没写的维度自动补** `:`**(全要)**。

具体来说,`x` 是 2D 的 (5,7),完整索引应该提供 2 个位置(axis 0 和 axis 1):

```
x[ <axis 0 索引> , <axis 1 索引> ]

```

当你只写一个位置 `x[b[:, 5]]`,numpy 把它解读为:

```
x[ b[:, 5] , : ]    ← 尾部自动补 :

```

也就是说,**第一个位置作用在 axis 0,剩下的维度默认全要**。这和普通 Python list 的 `arr[3]` 行为是一致的(取一行/一个元素,后面的维度整个保留)。

##### 推广到任意维度

这条规则对任何 ndim 都成立。比如 `y` 是 4D 的:

```
y[2]              ≡ y[2, :, :, :]
y[2, 3]           ≡ y[2, 3, :, :]
y[2, 3, 1]        ≡ y[2, 3, 1, :]

```

如果想"跳过前面的轴、只指定后面的轴",得用 `...`(Ellipsis):

```
y[..., 0]         ≡ y[:, :, :, 0]   ← 最后一个轴取 0

```

##### 容易踩的坑

**省略只对尾部生效**,不能跳着省。

```
x[, 5]            ← 语法错误
x[:, 5]           ← 正确写法,axis 0 显式写 :

```

如果你想要"所有行的第 5 列",必须显式写 `:`,因为你要指定的是 **axis 1**,不能跳过 axis 0。

##### 回到原题

```
x[b[:, 5]]        ≡  x[b[:, 5], :]
```

两边作用在 axis 0 的索引完全相同(都是那个长度 5 的 boolean mask),axis 1 一边显式 `:`、一边隐式 `:`,效果一致。所以结果完全一样:挑出 mask 为 True 的那些行,每行的 7 列全部保留。

#### example: From an array, select all rows which sum up to less or equal two:

```
x = np.array([[0, 1], [1, 1], [2, 2]])
```

```python
x[x.sum(axis=1) <= 2] 
```

结果:

```
array([[0, 1],
       [1, 1]])
```

**Step 1:**`x.sum(axis=1)` —— 沿 axis 1(列方向)求和,等价于"对每一行求和":

```
[[0, 1],     →  0+1 = 1
 [1, 1],     →  1+1 = 2
 [2, 2]]     →  2+2 = 4

```

得到 1D 数组 `[1, 2, 4]`,shape `(3,)`。

**Step 2:**`<= 2` —— element-wise 比较,得到 boolean mask:

```
[1, 2, 4] <= 2  →  [True, True, False]

```

**Step 3:**`x[mask]` —— 长度 3 的 1D boolean mask 沿 axis 0 对齐 `x` 的 3 行,保留 True 的那些行。

## `axis` 的方向记忆

新手最容易卡在"行求和到底用 axis=0 还是 axis=1"。规则:

`axis=k` **表示"沿着第 k 轴塌缩掉这一维"**。


| 操作              | 塌缩的轴              | 保留的轴         | 直觉                        |
| --------------- | ----------------- | ------------ | ------------------------- |
| `x.sum(axis=0)` | axis 0 (3 行) → 消失 | axis 1 (2 列) | "每一**列**的和" → `[3, 4]`    |
| `x.sum(axis=1)` | axis 1 (2 列) → 消失 | axis 0 (3 行) | "每一**行**的和" → `[1, 2, 4]` |


口诀:**axis=k 消掉哪个维度,结果就是"按那个维度方向加起来"**。要"每行求和"就是把列那一维加掉,所以 `axis=1`。

#### example 2

Use boolean indexing to select all rows adding up to an even number. At the same time columns 0 and 2 should be selected with an advanced integer index. Using the `ix_` function this can be done with:

```
x = np.array([[ 0,  1,  2],
              [ 3,  4,  5],
              [ 6,  7,  8],
              [ 9, 10, 11]])
```

注意区分:  
rows = x.sum(axis=1) % 2 == 0          # [False, True, False, True]
cols = np.array([0, 2])
x[np.ix_(rows, cols)]

结果:

```
array([[ 3,  5],
       [ 9, 11]])

各步解释:
```

**行 mask**:每行求和 `[3, 12, 21, 30]`,取模 2 等于 0 的位置 → `[F, T, F, T]`,选中 row 1 和 row 3。

**列 index**:`[0, 2]` 是 fancy integer indexing,要哪几列直接列出来。

**为什么必须用** `ix_`:这才是关键。如果你直接写

```python
x[rows, cols]            # ❌ 报错或结果错误
x[rows, [0, 2]]          # ❌ 同上

```

numpy 会把两个 advanced index **配对**(broadcast 到一起),而不是做"行 × 列"的笛卡尔积。它会试图把长度 2 的 `rows`(True 的个数)和长度 2 的 `cols` 一一对应,变成"取 (row1, col0), (row3, col2)"两个元素,返回 1D 数组 `[3, 11]`。

`np.ix_` 的作用是把多个 1D 索引数组"撑开"到不同的轴上,实现笛卡尔积的语义:

```python
np.ix_([F, T, F, T], [0, 2])
# →  ( array([[1], [3]]),   shape (2, 1)  ← 沿 axis 0
#      array([[0, 2]]) )    shape (1, 2)  ← 沿 axis 1

```

(注意 `ix_` 会先把 boolean 转成它对应的整数 index `[1, 3]`。)

这两个数组送进 `x[...]` 时,numpy broadcast 出一个 (2, 2) 的索引网格:

```
(行, 列):
(1,0) (1,2)
(3,0) (3,2)

```

正好就是"选 1、3 两行 × 选 0、2 两列"。

对比记忆


| 写法                          | 语义                   | 结果                   |
| --------------------------- | -------------------- | -------------------- |
| `x[[1, 3], [0, 2]]`         | **配对**:(1,0) 和 (3,2) | `[3, 11]`,shape (2,) |
| `x[np.ix_([1, 3], [0, 2])]` | **笛卡尔积**:2×2 网格      | shape (2, 2)         |
| `x[[1, 3]][:, [0, 2]]`      | 链式索引,等价网格            | shape (2, 2)         |


最后一种写法不用 `ix_`,但要做两次索引(可能产生中间拷贝)。`ix_` 一次到位,语义最清晰,推荐用在"我要在多个轴上独立挑 index"的场景。

好问题,这是在区分"**索引**"和"**索引的结果**"两个概念。

```python
rows = x.sum(axis=1) % 2 == 0     # [F, T, F, T]   ← 索引(mask)
x[rows]                            # [[3,4,5],[9,10,11]]  ← 索引的结果(数据)

```

`rows` 本身是个 boolean 数组,告诉 numpy "哪些行要选";它不包含 `x` 的任何值,只是一个"选择器"。

`np.ix_` 需要的是**选择器**,不是数据。它的工作是:接收"轴 0 的选择器"和"轴 1 的选择器",把它们撑开成网格,然后让 `x[...]` 一次性按这个网格取数。所以:

```python
x[np.ix_(rows, cols)]
#         ↑     ↑
#      行选择器  列选择器

```

如果换成 `x[np.ix_(x[rows], cols)]`,就荒谬了 —— 你把"已经选出来的数据"当成"选择器"传进去,numpy 会以为 `[[3,4,5],[9,10,11]]` 是一组要在 `x` 的 axis 0 上取的 index,值 9、10、11 会越界(`x` 只有 4 行)。

一句话总结

> `rows` 是"谁被选中"的描述,`x[rows]` 是"被选中的人"。`ix_` 要的是描述,因为它的任务是协调多个轴上的选择动作。

# 4 Aggregation: mean, std, min, max along specific axes

下面按"1D → 2D → 3D → 接 broadcast"的顺序展开,每段都有可立刻在 REPL 验证的例子,最后给两道练习题(附答案)。

### 0. 一句话回顾 axis

> `arr.op(axis=k)` 把 axis k **塌缩消失**,其它维度保留。如果传 tuple `(i,j)` 就同时消掉那几个轴。`axis=None`(默认)把所有轴消光,返回标量。

四种聚合用法完全一致 —— `mean`、`std`、`min`、`max`、`sum` 都吃 `axis` 和 `keepdims` 两个参数。

### 1. 1D:基线

```python
a = np.array([1, 2, 3, 4, 5])
a.mean()    # 3.0
a.std()     # 1.4142...  (默认 ddof=0,除以 N)
a.std(ddof=1)  # 1.5811   (除以 N-1,样本标准差)
a.min()     # 1
a.max()     # 5
a.argmax()  # 4   ← 最大值的位置(很常用)

```

注意:`np.std` 默认是**总体标准差**(`ddof=0`),和 pandas 默认 `ddof=1` 不一样,容易踩坑。

### 2. 2D:axis=0 vs axis=1

```python
x = np.arange(1, 13).reshape(3, 4)
# [[ 1,  2,  3,  4],
#  [ 5,  6,  7,  8],
#  [ 9, 10, 11, 12]]

```

**视觉化记忆**(箭头表示"沿这个方向加"):

```
axis=0 ↓ (塌缩行)              axis=1 → (塌缩列)
┌─ 1  2  3  4 ─┐               1  2  3  4 → 10/4=2.5
│  5  6  7  8  │               5  6  7  8 → 26/4=6.5
└─ 9 10 11 12 ─┘               9 10 11 12 → 42/4=10.5
   ↓  ↓  ↓  ↓
   5  6  7  8     ← 每列的均值       结果是 [2.5, 6.5, 10.5] ← 每行的均值
   shape (4,)                       shape (3,)

```

```python
x.mean(axis=0)   # [5., 6., 7., 8.]      shape (4,)
x.mean(axis=1)   # [2.5, 6.5, 10.5]      shape (3,)
x.max(axis=0)    # [9, 10, 11, 12]
x.min(axis=1)    # [1, 5, 9]
x.std(axis=0)    # [3.27, 3.27, 3.27, 3.27]
x.sum()          # 78        axis=None,全部塌缩
```

### 3. 3D:多种 axis 组合

```python
x = np.arange(24).reshape(2, 3, 4)
# shape (2, 3, 4) → 想象成 2 张 3×4 的"sheet"
```

```
sheet 0:                sheet 1:
[[ 0,  1,  2,  3],       [[12, 13, 14, 15],
 [ 4,  5,  6,  7],        [16, 17, 18, 19],
 [ 8,  9, 10, 11]]        [20, 21, 22, 23]]

```

不同 axis 的几何直觉:


| 操作                  | 塌缩的轴    | 结果 shape | 含义                   |
| ------------------- | ------- | -------- | -------------------- |
| `x.sum(axis=0)`     | sheet 维 | `(3, 4)` | 两张 sheet **逐元素相加**   |
| `x.sum(axis=1)`     | 行维      | `(2, 4)` | 每张 sheet 内**按列求和**   |
| `x.sum(axis=2)`     | 列维      | `(2, 3)` | 每张 sheet 内**按行求和**   |
| `x.sum(axis=(0,1))` | sheet+行 | `(4,)`   | 同一列位置上 6 个数加起来       |
| `x.sum(axis=(1,2))` | 行+列     | `(2,)`   | 每张 sheet 内**所有元素求和** |
| `x.sum()`           | 全部      | `()` 标量  | 0+1+...+23 = 276     |


具体数值(挑两个验证):

```python
x.sum(axis=0)
# [[12, 14, 16, 18],     ← 0+12, 1+13, 2+14, 3+15
#  [20, 22, 24, 26],
#  [28, 30, 32, 34]]

x.sum(axis=(1,2))
# [66, 210]   ← sheet 0 的 12 个数加起来 / sheet 1 的 12 个数加起来

```

### 4. `keepdims=True`:接住 broadcast 的关键

这是 axis 聚合 **最重要的实际应用** —— 用聚合结果回去和原数组做运算(去均值、归一化等)。

```python
x = np.arange(1, 13).reshape(3, 4).astype(float)

```

#### 场景 A:列方向标准化(z-score)

```python
mu  = x.mean(axis=0, keepdims=True)   # shape (1, 4),不是 (4,)
sig = x.std(axis=0, keepdims=True)    # shape (1, 4)
z = (x - mu) / sig                    # (3,4) - (1,4) → broadcast OK

```

如果没加 `keepdims`,这里**也能跑**:`(3,4) - (4,)` 时,numpy 把 `(4,)` 当成 `(1,4)` 处理,broadcast 成功。但下一个场景就会出问题。

#### 场景 B:行方向去均值(常见坑)

```python
row_mean = x.mean(axis=1)             # shape (3,)
x - row_mean                          # ❌ ERROR
# operands could not be broadcast together with shapes (3,4) (3,)

```

为什么?broadcast 规则是**从尾部对齐**:

```
x:          (3, 4)
row_mean:      (3,)        ← 长度 3 想对齐到 axis 1(长度 4)→ 失败

```

修法两种,推荐第二种:

```python
x - row_mean[:, None]                       # 手动加维度:(3,) → (3,1)
x - x.mean(axis=1, keepdims=True)           # 推荐:从一开始就 (3,1)

```

```
现在能广播:
x:          (3, 4)
row_mean:   (3, 1)   ← 沿 axis 1 复制 4 次

```

**口诀**:**聚合后还要喂回原数组的,一律加** `keepdims=True`。形状对齐零成本,代码可读性也高。

### 5. argmin / argmax:返回位置而非值

```python
x = np.array([[3, 1, 4],
              [1, 5, 9],
              [2, 6, 5]])

x.argmax()              # 5   (沿扁平化后的 index)
x.argmax(axis=0)        # [0, 2, 1]    每列最大值在哪一行
x.argmax(axis=1)        # [2, 2, 1]    每行最大值在哪一列
```

实战用法:用 `argmax` 拿到位置后,用 fancy indexing 取对应另一个数组的值(比如 trajectory prediction 里多模态预测取 best mode)。

### 练习题

### 题 1

```python
x = np.array([[ 2.,  4.,  6.],
              [ 1.,  3.,  5.],
              [ 4.,  8., 12.]])

```

写一行代码,把 `x` 的**每一行**缩放到该行的 max 等于 1(行最大归一化)。

### 题 2

```python
x = np.arange(24).reshape(2, 3, 4)
```

求 `x` 在每个 `(i, j)` 位置上、沿最后一个轴的 **(max - min)** 范围,结果 shape 应为 `(2, 3)`。然后再算"两个 sheet 在每个列位置上的均值之差",结果 shape 应为 `(4,)`。

---

### **题 1 - 答案**

```python
x / x.max(axis=1, keepdims=True)
# row max: [[6],[5],[12]]  shape (3,1)
# 结果:
# [[0.333, 0.667, 1.0],
#  [0.2,   0.6,   1.0],
#  [0.333, 0.667, 1.0]]
```

不加 `keepdims` 就会爆形状,正是上面场景 B 的坑。

两个层面都不对,正好是个值得拆开讲的点。

1. numpy 本身没有 `.softmax()` 方法

```python
x.softmax(axis=1)   # AttributeError: 'numpy.ndarray' object has no attribute 'softmax'
```

要用得从 scipy 拿,或者手写:

```python
from scipy.special import softmax
softmax(x, axis=1)

# 或者手动(还要做数值稳定化,减 max):
e = np.exp(x - x.max(axis=1, keepdims=True))
e / e.sum(axis=1, keepdims=True)

```

PyTorch tensor 才有 `.softmax(dim=...)` 方法,你大概是把两个 API 混了。

1. 关键:softmax 解决的不是同一个问题

题目要求是"**每行的 max = 1**",而 softmax 保证的是"**每行的 sum = 1**"。两件事完全不同。

拿 row 0 `[2, 4, 6]` 实测一下:


| 方法                                 | 结果                      | 行内 max    | 行内 sum  |
| ---------------------------------- | ----------------------- | --------- | ------- |
| `x / x.max(axis=1, keepdims=True)` | `[0.333, 0.667, 1.0]`   | **1.0** ✓ | 2.0     |
| `softmax(x, axis=1)`               | `[0.016, 0.117, 0.867]` | 0.867 ✗   | **1.0** |


而且 softmax 还做了一个 `exp(·)` 的**非线性变换**,会把数值大的元素相对放大、小的相对压扁。题目只想做线性缩放,softmax 改变了数据的相对关系。

```
原始 row 0:        [2, 4, 6]       ratios 2:4:6 = 1:2:3
max-normalize:     [0.33, 0.67, 1] ratios 同上,仍是 1:2:3   ← 线性
softmax:           [0.02, 0.12, 0.87] ratios 1:7.4:54.6     ← 指数放大
```

###### 3. 各种"归一化"对照速记

每个都是"沿某个 axis 把数据约束到某种范围",目的不同:


| 方法                | 公式(每行)                    | 保证              | 用途              |
| ----------------- | ------------------------- | --------------- | --------------- |
| **max-norm**      | `x / max(x)`              | max = 1         | 题目要求            |
| **sum-norm / L1** | `x / sum(x)`              | sum = 1,非负前提    | 概率分布(线性)        |
| **L2-norm**       | `x / ‖x‖₂`                | 向量模长 = 1        | 余弦相似度、embedding |
| **min-max**       | `(x - min) / (max - min)` | 范围 [0, 1]       | 图像、特征缩放         |
| **z-score**       | `(x - μ) / σ`             | 均值 0, 标准差 1     | 标准化             |
| **softmax**       | `exp(x) / Σ exp(x)`       | sum = 1,**非线性** | logits → 概率     |


题目要的就是第一种。

1. 一个有趣的关联

虽然 softmax 不是答案,但 softmax 的**数值稳定实现**里**恰好用了 max-normalization 的思想**:

```python
softmax_stable = np.exp(x - x.max(axis=1, keepdims=True))  # 先减行 max
softmax_stable /= softmax_stable.sum(axis=1, keepdims=True)
```

减去行 max 不改变 softmax 的结果(因为分子分母同乘 `exp(-max)`),但能避免 `exp(大数)` 溢出。这一步就是你写的 `x.max(axis=1, keepdims=True)` 用法,只是放在了 softmax 公式里面。

你做轨迹预测的多模态 head 输出 mode probability 时,框架里几乎一定就是这么实现的 —— 这条 `keepdims=True` 的肌肉记忆,从 numpy 练习直接迁移到 PyTorch 的 logits 处理。

### **题 2 - 答案**

```python
rng = x.max(axis=2) - x.min(axis=2)         # shape (2, 3),每行内的 max-min,值全是 3
diff = x.mean(axis=1)[0] - x.mean(axis=1)[1]   # shape (4,)
# 或者更优雅:
diff = -np.diff(x.mean(axis=1), axis=0).squeeze()

```

第一题练 `keepdims`,第二题练"先聚合一个轴、再在结果上继续操作"的链式思维。

完全合理的困惑 —— 这两行其实是同一件事的两种写法,我直接拆开走一遍数值。

```python
x = np.arange(24).reshape(2, 3, 4)
```

`x` 是两张 sheet:

```
sheet 0:                sheet 1:
[[ 0,  1,  2,  3],       [[12, 13, 14, 15],
 [ 4,  5,  6,  7],        [16, 17, 18, 19],
 [ 8,  9, 10, 11]]        [20, 21, 22, 23]]

```

##### Step 1:`x.mean(axis=1)` 在做什么

塌缩 axis 1(行维,长度 3),得到 shape `(2, 4)`。也就是**每张 sheet 内,每一列那 3 个数取平均**。

```
sheet 0 的 4 个列均值: [(0+4+8)/3, (1+5+9)/3, (2+6+10)/3, (3+7+11)/3]
                    = [4, 5, 6, 7]

sheet 1 的 4 个列均值: [(12+16+20)/3, (13+17+21)/3, (14+18+22)/3, (15+19+23)/3]
                    = [16, 17, 18, 19]

```

所以:

```python
x.mean(axis=1)
# array([[ 4.,  5.,  6.,  7.],     ← sheet 0 的列均值
#        [16., 17., 18., 19.]])    ← sheet 1 的列均值
# shape (2, 4)

```

##### Step 2:`[0]` 和 `[1]` 分别取哪一行

这就是普通的 2D 索引,从刚才的 (2,4) 数组里取出两行:

```python
x.mean(axis=1)[0]    # [4., 5., 6., 7.]      ← sheet 0 的列均值,shape (4,)
x.mean(axis=1)[1]    # [16., 17., 18., 19.]  ← sheet 1 的列均值,shape (4,)

```

##### Step 3:相减

```python
diff = x.mean(axis=1)[0] - x.mean(axis=1)[1]
#    = [4-16, 5-17, 6-18, 7-19]
#    = [-12., -12., -12., -12.]    shape (4,)

```

题目就是要这个:**对每个列位置,sheet 0 的均值 减 sheet 1 的均值**。题目语境里两张 sheet 可能代表"实验组 vs 对照组"或者"两个时间点",所以这种"对位相减"是常见操作。

##### 那个 `np.diff` 写法只是上面的同义改写

```python
np.diff(x.mean(axis=1), axis=0)

```

`np.diff` 的语义是:沿指定 axis 算**相邻元素之差**(后面的 - 前面的)。

输入 shape `(2, 4)`,沿 axis 0 算相邻之差,只有"row 1 - row 0"这一个差:

```
[[ 4,  5,  6,  7],      
 [16, 17, 18, 19]]
↓ 后 - 前
[[12, 12, 12, 12]]    shape (1, 4)

```

注意 shape 是 `(1, 4)` 不是 `(4,)` —— `np.diff` 会保留原来的维度数,只是该轴长度从 2 变成 1。所以需要:

- `.squeeze()`:把长度为 1 的轴去掉 → `(1, 4)` 变 `(4,)`
- **前面加** `-`:因为 `np.diff` 算的是"后 - 前"(`sheet1 - sheet0`),而题目要的是"sheet 0 - sheet 1",刚好反号。

合起来:

```python
diff = -np.diff(x.mean(axis=1), axis=0).squeeze()
# = -[12, 12, 12, 12] = [-12, -12, -12, -12]

```

##### 两种写法对照


| 写法                              | 思路         | 何时适合                                     |
| ------------------------------- | ---------- | ---------------------------------------- |
| `m[0] - m[1]`                   | 显式拿两行相减    | **只有两个的时候**,直观易读                         |
| `-np.diff(m, axis=0).squeeze()` | 通用"相邻差分"工具 | 多个元素要求两两相邻差,比如时间序列 `[t1-t0, t2-t1, ...]` |


如果 sheet 数变成 5,要"相邻 sheet 的列均值差",第一种就不优雅了(得手写 4 个减法),`np.diff` 一行就出 `(4, 4)`。但本题只有两个 sheet,第一种其实更清楚 —— 我标"更优雅"有点过度炫技,**实战里** `m[0] - m[1]` **才是更好的选择**。

### np.diff()是什么?

`n` 是**差分的阶数** —— 把 `np.diff` 反复套几次。

## `n=1`(默认):一阶差分

相邻两个元素之差:

```
x:        [ 1,  2,  4,  7,  0]
            ↘  ↘  ↘  ↘
diff(x):  [   1,  2,  3, -7]     ← [2-1, 4-2, 7-4, 0-7]

```

长度从 5 变成 4(少 1 个)。

## `n=2`:二阶差分 = 对一阶结果再做一次 diff

```
x:               [ 1,  2,  4,  7,  0]
diff(x):         [    1,  2,  3, -7]      ← 一阶
diff(diff(x)):   [       1,  1, -10]      ← 再做一次:[2-1, 3-2, -7-3]

```

长度从 5 变成 3(少 2 个)。所以 `np.diff(x, n=2)` 完全等价于 `np.diff(np.diff(x))`。

推广:`np.diff(x, n=k)` 让长度从 `N` 变成 `N - k`。

## 直觉:这就是离散版的导数

连续函数有 1 阶导、2 阶导;离散数组对应 1 阶差分、2 阶差分。物理含义:


| 数组            | 一阶差分          | 二阶差分               |
| ------------- | ------------- | ------------------ |
| 位置 (position) | 速度 (velocity) | 加速度 (acceleration) |
| 累积量           | 增量            | 增量的变化              |
| 信号            | 趋势            | 拐点 / 曲率            |


这正是你 trajectory prediction 里天天用的:`(B, T, 2)` 的位置序列,沿 `axis=1` 做 `np.diff` 就拿到速度,再 diff 一次就拿到加速度。物理常识在这条 API 上完全对应。

```python
traj = np.array([[0, 0], [1, 0], [3, 1], [6, 3], [10, 6]])  # (5, 2) 位置
vel  = np.diff(traj, axis=0)         # (4, 2) 速度
acc  = np.diff(traj, n=2, axis=0)    # (3, 2) 加速度

```

## 另外几个有用参数

```python
np.diff(x, prepend=0)   # 前面塞一个 0,长度不变
# x = [1, 2, 4, 7, 0],prepend=0:
# 等价于先变成 [0, 1, 2, 4, 7, 0],再 diff → [1, 1, 2, 3, -7]

```

`prepend` / `append` 解决一阶差分"长度少 1"的常见痛点 —— 比如想保留时间序列原长度方便和其它特征拼接时。

`axis=`:多维数组指定沿哪条轴差分(默认 `-1`,最后一条轴)。

## 反操作:`np.cumsum`

差分的逆运算是累积求和:`cumsum(diff(x))` 能恢复 `x`(差一个起点常数)。

```python
np.cumsum(np.diff(x))  # [1, 3, 6, -1]  → 加上 x[0]=1 就是 [2, 4, 7, 0] 即 x[1:]

```

物理上对应:从加速度积分回速度,从速度积分回位置。trajectory generation 时反向用这个把预测的速度还原成位置轨迹,也很常见。

## 5 Stacking and concatenation: np.stack, np.concatenate, np.hstack, np.vstack

按"概念 → 2D 例子 → 高维 → 陷阱 → 练习"展开。这一块最大的混乱点是 `stack` 和 `concatenate` 到底差什么,先把它锁死。

### 0. 核心区别(一句话)

> `np.concatenate` **沿已存在的轴拼接,ndim 不变。** `np.stack` **创建一个新的轴,ndim +1。**

记住这条,其它都是变体。

### 1. 1D 例子建立直觉

```python
a = np.array([1, 2, 3])     # shape (3,)
b = np.array([4, 5, 6])     # shape (3,)

np.concatenate([a, b])      # [1, 2, 3, 4, 5, 6]    shape (6,)    ← 还是 1D
np.stack([a, b])            # [[1,2,3],[4,5,6]]     shape (2, 3)  ← 升 2D
np.stack([a, b], axis=1)    # [[1,4],[2,5],[3,6]]   shape (3, 2)  ← 新轴在末尾

```

视觉上:

```
concatenate                  stack (axis=0)
 [a a a b b b]                [a a a]    ← 第 0 行
  线性首尾相接                  [b b b]    ← 第 1 行
                              形成新维度

```

### 2. 2D 例子(最常用)

```python
A = np.array([[1, 2],       B = np.array([[5, 6],
              [3, 4]])                    [7, 8]])
# A, B 都是 (2, 2)

```

四种典型操作:

```
concat axis=0 (上下拼)        concat axis=1 (左右拼)
┌───┬───┐                     ┌───┬───┬───┬───┐
│ 1 │ 2 │                     │ 1 │ 2 │ 5 │ 6 │
│ 3 │ 4 │                     │ 3 │ 4 │ 7 │ 8 │
│ 5 │ 6 │                     └───┴───┴───┴───┘
│ 7 │ 8 │                      shape (2, 4)
└───┴───┘
 shape (4, 2)

stack axis=0                  stack axis=-1
shape (2, 2, 2)               shape (2, 2, 2)
两张 sheet:A 和 B 各占一张      最后一维区分 A 和 B
                              结果[i,j,0]=A[i,j], 结果[i,j,1]=B[i,j]
```

代码:

```python
np.concatenate([A, B], axis=0)   # (4, 2)
np.concatenate([A, B], axis=1)   # (2, 4)
np.stack([A, B])                 # (2, 2, 2)  new axis at 0 (默认)
np.stack([A, B], axis=-1)        # (2, 2, 2)  new axis at end
```

详细🔎讲解:

虽然两者 shape 都是 `(2, 2, 2)`,但**数据在内存里的排布完全不同**,导致后续操作(索引、reduce、reshape、broadcast)都有差异。直接拿数据看最清楚。

```python
A = [[1, 2],     B = [[5, 6],
     [3, 4]]         [7, 8]]
```

## `np.stack([A, B], axis=0)`:A、B 各占一片 sheet

```
shape (2, 2, 2),三个轴语义: (which_input, row, col)

result[0] = A          result[1] = B
[[1, 2],               [[5, 6],
 [3, 4]]                [7, 8]]

```

完整张开:

```
result[i, j, k]
i=0 (A):              i=1 (B):
  j=0: [1, 2]            j=0: [5, 6]
  j=1: [3, 4]            j=1: [7, 8]

```

## `np.stack([A, B], axis=-1)`:A、B 在最后一维交错

```
shape (2, 2, 2),三个轴语义: (row, col, which_input)

result[:, :, 0] = A       result[:, :, 1] = B
[[1, 2],                  [[5, 6],
 [3, 4]]                   [7, 8]]

```

完整张开(注意"sheet"现在沿 axis 0 切了,不是按 A/B 切):

```
result[i, j, k]
i=0:                  i=1:
  j=0: [1, 5]            j=0: [3, 7]    ← 每个最内层 [A值, B值]
  j=1: [2, 6]            j=1: [4, 8]

```

##### 1. 取出 A、B 的方式不同


| 操作          | `axis=0`          | `axis=-1`         |
| ----------- | ----------------- | ----------------- |
| 取 A         | `result[0]`       | `result[..., 0]`  |
| 取 B         | `result[1]`       | `result[..., 1]`  |
| 取 A 的 (i,j) | `result[0, i, j]` | `result[i, j, 0]` |


`...` 是 Ellipsis,代替"前面所有维度都要"。

#### 2. 内存排布 / flatten 顺序不同

numpy 默认 C order(行优先),flatten 时按最后一维变化最快:

```python
# axis=0:
result.ravel()  → [1, 2, 3, 4, 5, 6, 7, 8]   ← A 整个连续,再 B 整个
                                                因为 A/B 是最外层维度

# axis=-1:
result.ravel()  → [1, 5, 2, 6, 3, 7, 4, 8]   ← A、B 交错
                                                因为 A/B 是最内层维度

```

这对**写文件、和其它库(C/Fortran/PyTorch)交换内存**时有影响。

##### 3. Reduce 操作的语义不同

聚合"A 和 B 的均值"时,两种 stack 走的 axis 也不同:

```python
# axis=0 版:对 A 和 B 求 element-wise 均值
mean_AB_0 = result_axis0.mean(axis=0)   # shape (2, 2)
# = (A + B) / 2 = [[3, 4], [5, 6]]

# axis=-1 版:同样目的,但走 axis=-1
mean_AB_1 = result_axisN.mean(axis=-1)  # shape (2, 2)
# = (A + B) / 2 = [[3, 4], [5, 6]]

```

数值一样,但**调用 API 时要写对 axis 编号**。错位是常见 bug。

##### 4. Broadcast 行为不同

假设你要把每个数组各自乘一个权重 `w = [0.5, 1.5]`(A 乘 0.5,B 乘 1.5):

```python
# axis=0 版:w 形状 (2,) 要对齐到 axis 0
w = np.array([0.5, 1.5])
result_axis0 * w[:, None, None]    # w shape (2,1,1) 广播到 (2,2,2)

# axis=-1 版:w 直接对齐到 axis -1
result_axisN * w                   # w shape (2,) 自动对齐到最后一维,直接广播

```

第二种代码更简洁,因为 numpy broadcast 是**从尾部对齐**的,小维度放在 axis=-1 时最省心。

##### 5. 实战:什么时候用哪个

这其实是个**惯例**问题,看维度的语义角色:


| Stack 方向  | 适合"X 维度是什么角色"             | 例子                                                             |
| --------- | ------------------------- | -------------------------------------------------------------- |
| `axis=0`  | batch 维、"样本"维、"sheet"维    | 多个 trajectory 凑成 batch:`stack(trajs, axis=0)` → `(B, T, A, D)` |
| `axis=-1` | feature 维、channel 维、"通道"维 | R/G/B 三通道凑成图:`stack([R,G,B], axis=-1)` → `(H, W, 3)`           |


**口诀**:**"哪些样本"放前面,"哪些特征"放后面**。

具体到你 trajectory prediction 的常见模式:

```python
# 多个 agent 的轨迹 (T, D) 凑成 (N_agents, T, D):用 axis=0
np.stack(agent_trajs, axis=0)

# x, y 坐标分别是 (T,) 的数组,凑成 (T, 2) 的位置序列:用 axis=-1
np.stack([xs, ys], axis=-1)

```

两种用法都极其常见,关键是你想让"哪些"这个维度坐在最自然的位置。一般 batch / sample 在前,channel / feature 在后,跟主流深度学习框架的 `(N, C, H, W)` 或 `(N, H, W, C)` 约定一致。

### 一图总结

```
A = [[1,2],[3,4]]     B = [[5,6],[7,8]]

stack axis=0:                  stack axis=-1:
"输入" 在最外层                  "输入" 在最内层
─────────────                  ──────────────
sheet 0 = A 整张                每个 (i,j) 格子里:[A值, B值]
sheet 1 = B 整张                
                                
访问:result[0/1, ...]           访问:result[..., 0/1]
flatten:A整段 || B整段          flatten:A、B 交错
最适合:批量样本                  最适合:特征通道
```

```
  stack axis=-1:
```

```python
result = np.stack([A, B], axis=-1)

# print(result):
[[[1, 5],
  [2, 6]],

 [[3, 7],
  [4, 8]]]
```

##### 怎么读这个三维数组

shape `(2, 2, 2)`,轴语义是 `(row, col, which_input)`。numpy 打印 3D 数组时:**最外层 axis 是 sheet,中间是行,最内层是列**。所以这里 print 出来的"sheet/行/列"和你脑子里 A、B 的"行/列"已经错位了。

```
result:
┌─ sheet 0 (row=0) ─┐    ┌─ sheet 1 (row=1) ─┐
│  [1, 5]   (col=0)  │    │  [3, 7]   (col=0)  │
│  [2, 6]   (col=1)  │    │  [4, 8]   (col=1)  │
└────────────────────┘    └────────────────────┘
   ↑     ↑                   ↑     ↑
   A 的值  B 的值              A 的值  B 的值

```

每个最内层的二元组 `[x, y]` 就是 `[A[i,j], B[i,j]]` —— 把同一位置上 A 和 B 的值并排放在最内层。

##### 验证三种切片视角

```python
# 切 axis 0(行):
result[0]           # = [[1, 5], [2, 6]]    A 和 B 的第 0 行交错
result[1]           # = [[3, 7], [4, 8]]    A 和 B 的第 1 行交错

# 切 axis 1(列):
result[:, 0]        # = [[1, 5], [3, 7]]    A 和 B 的第 0 列交错
result[:, 1]        # = [[2, 6], [4, 8]]    A 和 B 的第 1 列交错

# 切 axis 2(which_input):
result[:, :, 0]     # = [[1, 2], [3, 4]] = A   完整取出
result[:, :, 1]     # = [[5, 6], [7, 8]] = B   完整取出

```

**只有沿 axis 2 切才能完整还原 A 或 B**,因为 A/B 这个语义维度就在 axis 2 上。沿其它轴切,A 和 B 的值会混在一起出现。

##### 对比 axis=0 版本的 print

```python
result_0 = np.stack([A, B], axis=0)

# print(result_0):
[[[1, 2],          ← sheet 0 = A 完整
  [3, 4]],

 [[5, 6],          ← sheet 1 = B 完整
  [7, 8]]]
```

这种版本 print 出来"两张 sheet 各是 A 和 B 的完整副本",视觉上和 `axis=-1` 版完全不同 —— 虽然 shape 都是 `(2, 2, 2)`,数据也都是同样 8 个数,但排列方式不同。

## 3. `hstack` / `vstack` / `dstack`:别名 + 陷阱

它们看起来是 `concatenate` 的语义快捷方式,但**对 1D 的行为不一致**,这是经典坑。

### 对 2D 数组(直觉一致)


| 函数                  | 等价于                   | 含义                 |
| ------------------- | --------------------- | ------------------ |
| `np.vstack([A, B])` | `concatenate(axis=0)` | vertical,加行        |
| `np.hstack([A, B])` | `concatenate(axis=1)` | horizontal,加列      |
| `np.dstack([A, B])` | 沿第 3 轴 stack          | depth,产出 (H, W, 2) |


### 对 1D 数组(踩坑警告)

```python
a = np.array([1, 2, 3])     # (3,)
b = np.array([4, 5, 6])     # (3,)

np.hstack([a, b])    # [1,2,3,4,5,6]      shape (6,)    ← 还是 1D
np.vstack([a, b])    # [[1,2,3],[4,5,6]]  shape (2, 3)  ← 升 2D!
np.dstack([a, b])    # shape (1, 3, 2)    ← 升 3D!!

```

`vstack` 偷偷把 1D 视为 "1 行" 再拼,所以"行数"加 1 时 ndim 升了一档。`hstack` 不升维。这种"看场景换语义"的行为最容易制造 bug。

### 安全建议

> **写代码时优先用显式的** `np.concatenate(axis=...)` **或** `np.stack(axis=...)`,只在脚本里偶尔用 `vstack/hstack`。生产代码碰到 1D vs 2D 自动行为切换是常见 bug 源。

## 4. 还有两个有用的:`column_stack` 和 `np.r_` / `np.c_`

```python
# 把多个 1D 向量摆成矩阵的列
np.column_stack([a, b])
# [[1, 4],
#  [2, 5],
#  [3, 6]]      shape (3, 2)

```

这正是 `hstack` 对 1D 做不了的事(它会扁平拼接而不是变成列)。

`np.c_` 和 `np.r_` 是 indexing-style 语法糖(不是函数调用):

```python
np.r_[a, b]            # row-wise 拼 → [1,2,3,4,5,6]
np.c_[a, b]            # column-wise → 等价 column_stack

```

## 5. 形状要求

不管哪种,都有形状约束:


| 操作                    | 输入形状要求                |
| --------------------- | --------------------- |
| `concatenate(axis=k)` | 除 axis k 外,其它所有维度必须相同 |
| `stack`               | 所有输入 shape **完全相同**   |


```python
A: (3, 4)
B: (5, 4)
np.concatenate([A, B], axis=0)   # OK → (8, 4)
np.concatenate([A, B], axis=1)   # ERROR,axis 1 之外不同
np.stack([A, B])                 # ERROR,shape 不同

```

## 6. 决策树:该用哪个

```
我要把多个数组合并 ……
│
├─ 它们 shape 完全相同 + 我想多一个新维度 (比如"batch", "frame")
│   → np.stack(arrays, axis=...)
│
├─ 它们已经有同一个轴,我要把那个轴"接长"
│   → np.concatenate(arrays, axis=...)
│
└─ 1D 向量我要按列拼成矩阵
    → np.column_stack(arrays)

```

## 7. ML 场景里的具体对应

直接对应到你 trajectory prediction 的工作场景:

```python
# 场景 A:list of N 个样本,每个 (T, A, D),要堆成 batch
samples = [s1, s2, ..., sN]            # 每个 (T, A, D)
batch = np.stack(samples, axis=0)      # (N, T, A, D)    ← stack 新轴

# 场景 B:两个 batch 合并成一个大 batch
batch_combined = np.concatenate([batch1, batch2], axis=0)   # (N1+N2, T, A, D)

# 场景 C:多模态预测,K 个 mode 的预测拼到一起
modes = [pred_mode_k for k in range(K)]   # 每个 (B, T, D)
multi_modal = np.stack(modes, axis=1)     # (B, K, T, D)    ← K 作为新轴

```

简单口诀:**"凑 batch / 凑 mode / 凑 frame" 用 stack,"已有的 batch 接长" 用 concatenate。**

## 练习题

### 题 1(基础)

你有三个 1D 数组,各长度 4:

```python
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])
c = np.array([9, 10, 11, 12])
```

要得到 shape `(3, 4)` 的矩阵(每个数组占一行)。写两种写法。

### 题 2(中等)

特征矩阵 `X` shape `(5, 3)`,标签 `y` shape `(5,)`,要做一个 `(5, 4)` 的矩阵,前 3 列是 `X`,最后一列是 `y`。给出至少 2 种写法。

### 题 3(综合)

你有 10 帧 RGB 图像,每帧 shape `(H, W, 3)`,放在一个 Python list 里。

- (a) 想要 video tensor shape `(10, H, W, 3)`,用什么?
- (b) 如果错用 `np.concatenate(frames)`(默认 `axis=0`),结果 shape 是什么?这样做有什么物理上的(错误)含义?

---

答案

**题 1**

```python
np.stack([a, b, c])              # axis=0 默认,(3, 4)  ✓
np.vstack([a, b, c])             # 对 1D 自动当行处理,(3, 4)  ✓
# 注意 np.hstack([a, b, c]) → (12,) 错的!

```

**题 2**

```python
np.column_stack([X, y])                              # 最优雅
np.concatenate([X, y[:, None]], axis=1)              # y 先升 (5,1) 再拼
np.hstack([X, y.reshape(-1, 1)])                     # 同上
np.c_[X, y]                                          # 速记法

```

`y` 必须先升成 `(5, 1)` 才能在 `axis=1` 上和 `X` 对齐;`column_stack` 和 `np.c_` 自动帮你做了这一步。

**题 3**

(a) `np.stack(frames, axis=0)` 或简写 `np.stack(frames)` → `(10, H, W, 3)`,frame 是新维度。

(b) `np.concatenate(frames)` 默认沿 axis=0 拼接,结果 shape `(10*H, W, 3)`。物理含义:**把 10 张图沿高度方向上下首尾粘成一张超高的图**,完全没有"时间/帧"的概念了。这正是 stack vs concatenate 最直观的区别 —— **stack 增加"哪一个"的维度,concatenate 把"哪一个"压回到原维度里**。

# 6 File I/O: loading from parquet via Pandas → NumPy

#### 思维模型

```

parquet (磁盘表格)
   ↓ 用 pandas 解析、清洗、筛选
DataFrame (内存表格,带元数据)
   ↓ 用 .to_numpy() 扔掉元数据
numpy ndarray (纯张量,模型输入)
```

经典 ML / DL 训练前的数据准备:

```

原始数据(各种来源)

  ↓ 预处理(Spark / Polars / Python 脚本)

parquet 文件(版本化、可复现)        ← ★ 你接手的起点

  ↓ [pd.read](http://pd.read)_parquet

DataFrame

  ↓ 筛选、转 dtype、嵌套列展开
  ↓ .to_numpy()

numpy 数组

  ↓ torch.from_numpy()

torch tensor

  ↓ DataLoader 喂入模型

训练 / 推理
```

---

### 1. Parquet 自带的"魔法"

#### Schema 内嵌 (文件里有完整的"表结构"):

```

agent_id   : int64
class      : string
timestamp  : int64
xy         : list<float>
heading    : float32
```

读出来 dtype 准确无误,不像 CSV 还得 `pd.read_csv(dtype={'age': int})` 猜。

#### 统计信息(Min/Max/Null count)

每个"列块"(column chunk)都存了 min、max、null 数。这让 **predicate pushdown** 成为可能:

```python

[pd.read](http://pd.read)_parquet('data.parquet', filters=[('age', '>', 50)])

```

引擎一看某个块的 max=40,就整个块跳过不读。

### 2. Pandas(为什么走这一层?)

pandas 的 `DataFrame` 是表格数据的"瑞士军刀":支持混合 dtype、命名列、缺失值、过滤、聚合、JOIN。从 parquet 读出来天然就是 DataFrame,因为:

- parquet 本身就是表格(有列名 + 每列 dtype)
- pandas 提供了最成熟的 `read_parquet` API

什么时候你需要 pandas 这一步,而不是直接读到 numpy:

- 需要先按列名筛选`df[['x', 'y']]`)
- 需要按条件过滤行`df[df['valid_len'] > 20]`)
- 需要处理缺失值`df.dropna()df.fillna()`)
- 需要类型转换 / 单位换算

如果你的数据已经是干净的纯数值矩阵,可以跳过 pandas,用 `pyarrow` 直接读到 numpy(性能更好,但灵活性差)。

### 3. NumPy(为什么是终点?)

因为下游训练 / 计算用的就是 numpy(或 torch tensor,而 torch tensor 跟 numpy 零拷贝互转)。**模型只认张量,不认 DataFrame**。

pandas 的 DataFrame 虽然好用,但有性能开销:

- 每列是独立对象,缓存不友好
- 支持混合 dtype 导致没法 SIMD 向量化
- 索引、列名等元数据占内存

#### 4. 三大坑(实战必看)

#### 坑 1:混合 dtype → object

数值 + 字符串列一起 `.to_numpy()`,dtype 退化为 `object`,失去向量化,运算慢几十倍。

```python

[df.select](http://df.select)_dtypes(include=[np.number]).to_numpy()   # 只挑数值列再转

```

#### 坑 2:NaN 让 int 变 float

整数列有一个 NaN 就被强升 float64(因为 int 无法表示 NaN)。

```python

df['a'].to_numpy(dtype=np.float32, na_value=0.0)   # 显式控制 dtype + NaN 填充

```

#### 坑 3:嵌套列(list-valued cells)

trajectory 等数据每行存一个 `(T, 2)` 数组,长度不一时不能直接 stack。

解决方案:**padding + valid_length mask**(见下面工作流模板 Step 3)。padding 的 0 在模型眼里和真实坐标无区别,**必须配 mask 给 loss 和 attention 用**。

---

### 实战工作流模板

把上面的内容拼成一个 trajectory prediction 数据加载常见模式:

```python
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

PATH = 'nuscenes_processed.parquet'

# Step 1: 探查
pf = pq.ParquetFile(PATH)
print(pf.schema_arrow)

# Step 2: 选择性读取
df = pd.read_parquet(
    PATH,
    columns=['agent_id', 'trajectory', 'valid_len', 'scene_token'],
    filters=[('valid_len', '>=', 20)],     # 只要 20 帧以上的样本
)

# Step 3: 嵌套列 → tensor
T_max = df['valid_len'].max()
N = len(df)
trajs = np.zeros((N, T_max, 2), dtype=np.float32)
for i, (t, L) in enumerate(zip(df['trajectory'], df['valid_len'])):
    trajs[i, :L] = np.asarray(t, dtype=np.float32)

# Step 4: 元数据另存
agent_ids = df['agent_id'].to_numpy()           # (N,)
valid_lens = df['valid_len'].to_numpy()         # (N,)

print(trajs.shape, trajs.dtype)                 # (N, T_max, 2) float32
```

这个模板覆盖了:**列存优势**(只读 4 列)、**预过滤**(行级过滤)、**嵌套展开**(list → tensor)、**dtype 控制**(float32 省内存)。生产 pipeline 基本就是这几步的变体。