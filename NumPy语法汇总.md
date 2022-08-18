NumPy的最重要特点是其N维数组对象ndarray<br />Start your data scientist career with NumPy.
```python
import numpy as np
```
<a name="HC0NY"></a>
# 数组创建

---

```python
np.array([1, 2, 3] )#使用array函数从Python列表或元组中创建
np.array([1, 2], [3, 4])#创建多维数组
np.array([1, 2], [3, 4],dtype=complex)#创建时显式指定数组类型

#特殊数组，默认情况下数组的dtype类型是float64
np.zeros((3, 4))#零数组
np.ones((2, 3, 4), dtype=np.int16)#1数组，特别指定dtype类型
np.empty((2, 3))#空数组
np.full((3,4),3)#4行3列全为3的数组

#按一定取值范围创建数组
np.arange(start, stop, step, dtype)#不包含终止值，创建一维数组
np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None))#等差数列，num样本数量，endpoint是否包含终止值（默认包含），retstep是否显示样本间距
np.logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None)#等比数列，base对数log的底数，默认为10
    np.logspace(1, 2, num=10)
    运行结果：[ 10.           12.91549665     16.68100537      21.5443469  27.82559402      
              35.93813664   46.41588834     59.94842503      77.42636827    100.    ]

#np.random创建数组
np.random.random((3,2))#数字取值范围[0,1)随机，3行2列
np.random.rand(3,2)#数字取值范围[0,1)随机，3行2列
    #random.ramndom和random.rand效果相同，不同点在于参数传递，random.ramndom接受的参数必须是元组
np.random.randint(low, high=None, size=None, dtype=int)#数字取值范围[low,high)
np.random.randn(8)#8个数，正态分布

#利用函数创建数组
np.fromfunction(func,size,dtype)  #自定义函数func，构建的数组在坐标(a,b)处元素值为func(a,b)
```
<a name="ZCOgg"></a>
# 数组属性

---

```python
a.ndim                         # 数组的维度的数量，也叫秩、轴的数量（中括号有几层）
a.shape                        # 数组的维度，对于矩阵而言是n行m列
a.size                         # 数组元素的总个数
a.dtype                        # ndarray对象的元素类型
```
<a name="ceWol"></a>
# 形状操纵

---

数组的形状由每个轴的元素数量决定
```python
#数组形状改变
a.ravel                        # 多维数组变一维
a.reshape((2,3))               # 数组转为2行3列不改变原数据
    #reshape中如果将size定为-1，会自动计算这一轴的size
    >>> a.reshape(3,-1)
        array([[ 2.,  8.,  0.,  6.],
               [ 4.,  5.,  1.,  1.],
               [ 8.,  9.,  3.,  6.]])
a.resize((2,3))                # 数组转换为2行3列不改变原数据
np.resize(a,(2,3))             # 数组转换为2行3列改变原数据
a.T                            # 转置，原位置在(m1,m2,...mn)的元素位置变为(mn,...m2,m1)
a.transpose((0,1,2))           # a不变
a.transpose((1,0,2))           # a 0轴和1轴的元素交换

#数组堆叠
#针对2D数组
>>> a = np.floor(10*np.random.random((2,2)))
>>> a
array([[ 8.,  8.],
       [ 0.,  0.]])
>>> b = np.floor(10*np.random.random((2,2)))
>>> b
array([[ 1.,  8.],
       [ 0.,  4.]])
np.vstack((a,b))               # 纵向堆叠
    #输出结果
    #array([[ 8.,  8.],
            [ 0.,  0.],
            [ 1.,  8.],
            [ 0.,  4.]])
np.hstack((a,b))               # 横向堆叠
    #输出结果
    #array([[ 8.,  8.,  1.,  8.],
            [ 0.,  0.,  0.,  4.]])
#针对1D数组
a = np.array([4.,2.])
b = np.array([3.,8.])
np.column_stack((a,b))         # 1D数组作为列堆叠
    #输出结果
    #array([[ 4., 3.],
            [ 2., 8.]])
np.hsatck((a,b))               # 1D数组直接拼接
    #输出结果
    #array([ 4., 2., 3., 8.])
    
#数组拆分
np.hsplit(a,3)                 # a拆分为3个数组，沿横轴拆分
np.hsplit(a,(3,4))             # a拆分为3个数组，第4列单独一组
np.vsplit(a,3)                 # 沿纵轴拆分
```
<a name="Bb7uZ"></a>
# 基本操作

---

数组的算数运算符会应用到 元素 级别
```python
#乘积运算
>>> A = np.array( [[1,1],
...             [0,1]] )
>>> B = np.array( [[2,0],
...             [3,4]] )
>>> A * B                       # *乘积运算符按元素进行
array([[2, 0],
       [0, 4]])
>>> A @ B                       # 矩阵乘积用@
array([[5, 4],
       [3, 4]])
>>> A.dot(B)                    # 矩阵乘积也可以用dot函数
array([[5, 4],
       [3, 4]])

#沿着axis的操作
#中括号由外向内，对应axis=0到逐渐增大，对于二维数组而言，axis=0沿列操作，axis=1沿行操作

#矩阵运算
a*b                             # 矩阵点乘
np.vdot(a,b)                    # 矩阵点积
np.linalg.inv(a)                # 逆矩阵
np.dot(a,b)                     # 矩阵乘法
np.linalg,det(a)                # 矩阵标量
np.linalg.solve(A,B)            # 矩阵求解方程
q, r = linalg.qr(m3)            # 矩阵二维分解
```
 
<a name="uQUbc"></a>
# 通函数

---

在NumPy中，这些函数在数组上按元素进行计算，产生一个数组作为输出
```python
(a == b).all()                  # 矩阵是否所有对应元素相等
(a == b).any()                  # 矩阵是否有一个对应元素相等
np.apply_along_axis(func,axis,arr)      # 自定义函数应用到数组中
np.argmax(a, axis=0)            # axis=0最大的元素的索引值
np.argmin(a, axis=0)
np.argsort(a, axis=0)           # 按axis=0排序后返回元素索引值
np.cumsum(a, axis=0)            # 沿着axis=0累加
np.cumprod(a, axis=0)           # 沿着axis=0累乘
np.sort(a, axis, kind, order)   # kind：排序方式。默认为"quicksort"快速排序
                                # order：如果数组中包含字段，指定字段排序
```
<a name="zxlYl"></a>
# 索引、切片和迭代

---

```python
#一维数组的索引、切片和迭代与Python列表等序列类型相同
a[start:stop:step]           # 切片[start,stop)之间的元素，步长为step
a[1:5]                       # 切割[2:6)的
a[1:]                        # 切割2以后的

#多维数组切片
#中括号中的冒号从axis=0递增
a[1:5,:]                     # 切片[2,6)行的所有数据
a[1:5,2:4]                   # 切片[2,6)行[3,5)列的所有数据

#多维数组迭代
for raw in b:
    print(raw)               # 相对于第一个轴进行迭代，二维数组就是输出每行数据
for element in b.flat:
    print(element)           # 迭代每一个元素
```
