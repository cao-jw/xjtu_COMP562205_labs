import cupy as cp

# 矩阵尺寸
width = 256

# 生成随机矩阵
a = cp.random.randint(0, 10, (width, width))
b = cp.random.randint(0, 10, (width, width))

# 将矩阵移动到GPU设备上
d_a = cp.asarray(a)
d_b = cp.asarray(b)

# 执行矩阵相加
d_c = d_a + d_b

# 将结果从GPU设备移动回主机内存


# 打印结果
print(d_c)
