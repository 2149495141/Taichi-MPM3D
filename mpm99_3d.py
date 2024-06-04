import numpy as np
import taichi as ti

ti.init(arch=ti.gpu)  # 尝试在 GPU 上运行
 
dim, n_grid, steps, dt, R, res = 3, 64, 29, 2e-4, 1.2, 360 
#dim, n_grid, steps, dt, R, res = 3, 128, 16, 1e-4, 1, 480  # 维度, 网格数, 模拟帧率, 时间步长, 半径, 分辨率

n_particles = n_grid**dim // 2**(dim - 1)  # 粒子数量
dx = 1 / n_grid  # 网格间距
inv_dx =  float(n_grid)  # 格点（单个网格的中心）
p_vol= (dx * 0.5) ** 2  # 粒子体积
p_rho = 1  # 粒子密度
p_mass = p_vol * p_rho  # 粒子质量
gravity = 9.8  # 重力
bound = 3  # 边界值
E = 400  # 杨氏模量
nu = 0.2  # 泊松比
mu_0 = E / (2 * (1 + nu))  # 剪切模量
lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu))  # 拉梅参数

X = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles)  # 位置向量数组
V = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles)  # 速度向量数组
C = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)  # 仿射速度矩阵数组
F = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)  # 变形梯度矩阵数组
J = ti.field(dtype=ti.f32, shape=n_particles)  # 塑性变形数组
Grid_v = ti.Vector.field(dim, dtype=ti.f32, shape=(n_grid, ) * dim)  # 网格节点动量数组
Grid_m = ti.field(dtype=ti.f32, shape=(n_grid, ) * dim)  # 网格节点质量数组
material = ti.field(dtype=ti.uint32, shape=n_particles)  # 粒子材质数组
color = ti.field(dtype=ti.uint32, shape=n_particles)  # 粒子颜色数组

neighbour = (3, ) * dim
group_size = n_particles // 3  # 每个材质块的粒子数


@ti.kernel
def substep():
    for i in ti.grouped(Grid_m):
        Grid_v[i] = ti.zero(Grid_v[i])  # 重置每个网格节点的速度
        Grid_m[i] = 0  # 重置每个网格节点的质量

    ti.loop_config(block_dim=n_grid)
    for p in X:  # 把粒子状态更新到网格节点 (P2G)
        xp = X[p] / dx
        base = int(xp - 0.5)
        fx = xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        F[p] = (ti.Matrix.identity(float, 3) + dt * C[p]) @ F[p]  # 变形梯度更新
        h = 0.3 if (material[p] == 1) else ti.exp(10 * (1 - J[p]))  # h是硬化系数，如果材质是果冻就限制硬化强度
        la = lambda_0 * h
        mu = 0.0 if (material[p] == 0) else mu_0 * h  # 如果材质是液体就设剪切模量为0
        u, sig, v = ti.svd(F[p])
        j = 1.0  # J是粒子的体积变化
        for d in ti.static(range(3)):
            new_sig = ti.min(ti.max(sig[d, d], 1 - 2.5e-2), 1 + 5.4e-3) if (material[p] == 2) else sig[d, d]  # 如果材质是雪就限制塑性范围
            J[p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            j *= new_sig
        if material[p] == 0:
            new_F = ti.Matrix.identity(float, 3)
            new_F[0, 0] = j
            F[p] = new_F  # 重置变形梯度以避免数值不稳定
        elif material[p] == 2:
            F[p] = u @ sig @ v.transpose()  # 塑性后重建弹性变形梯度
        stress = 2 * mu * (F[p] - u @ v.transpose()) @ F[p].transpose() + ti.Matrix.identity(float, 3) * la * j * (j - 1)
        stress = (-dt * p_vol * 4) * stress / dx**2
        affine = stress + p_mass * C[p]
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            Grid_v[base + offset] += weight * (p_mass * V[p] + affine @ dpos)
            Grid_m[base + offset] += weight * p_mass

    for i in ti.grouped(Grid_m):  # (P2G) 这一步是为了把网格内粒子的质量加权平均赋加到网格上
        if Grid_m[i] > 0:
            Grid_v[i] /= Grid_m[i]
        Grid_v[i][1] -= dt * gravity
        cond = (i < bound) & (Grid_v[i] < 0) | (i > n_grid - bound) & (Grid_v[i] > 0)  # 边界条件
        Grid_v[i] = ti.select(cond, 0, Grid_v[i])

    ti.loop_config(block_dim=n_grid)
    for p in X:  # 从网格节点获取状态到粒子 (G2P)
        xp = X[p] / dx
        base = int(xp - 0.5)
        fx = xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.zero(V[p])
        new_C = ti.zero(C[p])
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            g_v = Grid_v[base + offset]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) / dx ** 2
        V[p] = new_v
        C[p] = new_C
        X[p] += dt * V[p]

@ti.kernel
def initialize():
    mat_colors = ti.Vector([0x068599, 0xFF8888, 0xEEEEF0])
    for i in range(n_particles):  # 初始化粒子的位置
        X[i] = [ti.random() * 0.1 + 0.3 + 0.03 * (i // group_size),  # 材质块x长度(随机*x偏移量） + 所有粒子的在x轴位置 + 材质块x轴的相对间隔(粒子i//材质块粒子数)
                ti.random() * 0.2 + 0.1 + 0.3 * (i // group_size),  # 材质块y长度(随机*y偏移量） + 所有粒子的在y轴位置 + 材质块y轴的相对间隔(例如:材质块粒子数=21845, 21845//21845=1, 粒子在第二个材质块偏移量为1)
                ti.random() * 0.2 + 0.3 + 0.1 * (i // group_size)]  # 材质块z长度(随机*z偏移量） + 所有粒子的在z轴位置 + 材质块z轴的相对间隔(z偏移量*(材质块ID))
        material[i] = i // group_size  # 材质块ID 0:流体; 1:果冻; 2:雪;
        color[i] = mat_colors[i]
        V[i] = ti.Matrix([0, 0, 0])  # 初始化速度
        F[i] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # 初始化形变梯度
        J[i] = 1  # 初始化塑性变形

def T(a):  # 视角投影变换
    phi, theta = np.radians(30), np.radians(30)  # 左右角度， 上下角度

    a = a - 0.5
    x, y, z = a[:, 0], a[:, 1], a[:, 2]
    c, s = np.cos(phi), np.sin(phi)
    C, S = np.cos(theta), np.sin(theta)
    x, z = x * c + z * s, z * c - x * s
    u, v = x, y * C + z * S
    return np.array([u, v]).swapaxes(0, 1) + 0.55


initialize()
gui = ti.GUI("MLS-MPM-99-3D", res, background_color=0x112F41)
colors = color.to_numpy()
materials = material.to_numpy()
while gui.running and not gui.get_event(gui.ESCAPE):
    for s in range(steps):
        substep()  # 更新模拟帧
    gui.circles(T(X.to_numpy()), radius=R, color=colors[materials])
    gui.show()
