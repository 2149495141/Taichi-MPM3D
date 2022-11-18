
import numpy as np

import taichi as ti

ti.init(arch=ti.gpu)  # 尝试在 GPU 上运行


#dim, n_grid, steps, dt, res = 3, 64, 24, 3e-4, 500  # 维度, 网格数, 模拟帧率, 时间步长, 分辨率
dim, n_grid, steps, dt, res = 3, 64, int(2e-3 // 4e-4), 4e-4, 400  # 更友好的配置
#dim, n_grid, steps, dt, res = 3, 128, int(2e-3 // 2e-4), 2e-4, 720  # 更好的效果

n_particles = n_grid**dim // 2**(dim - 1)  # 粒子数
dx, inv_dx = 1 / n_grid, float(n_grid)
p_vol, p_rho = (dx * 0.5) ** 2, 1
p_mass = p_vol * p_rho
gravity = 9.8
bound = 3
E, nu = 400, 0.2  # 杨氏模量和泊松比
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # 拉梅参数

x = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles)  # 位置
v = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles)  # 速度
C = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)  # 仿射速度场
F = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)  # 变形梯度
Jp = ti.field(dtype=ti.f32, shape=n_particles)  # 塑性变形
grid_v = ti.Vector.field(dim, dtype=ti.f32, shape=(n_grid, )*dim)  # 网格节点动量/速度
grid_m = ti.field(dtype=ti.f32, shape=(n_grid,)*3)  # 网格节点质量
material = ti.field(dtype=ti.int32, shape=n_particles)  # 材质 id
neighbour = (3, ) * dim

@ti.kernel
def substep():
    for I in ti.grouped(grid_m):
        grid_v[I] = ti.zero(grid_v[I])
        grid_m[I] = 0
    ti.block_dim(n_grid)
    for p in x:  # 粒子状态更新和散布到网格 (P2G)
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        F[p] = (ti.Matrix.identity(float, 3) + dt * C[p]) @ F[p]  # 变形梯度更新
        h = ti.exp(4 * (1 - Jp[p]))  # 硬化系数：雪被压缩时变硬

        if material[p] == 1:  # 果冻，让它变软
            h = 0.3
        mu, la = mu_0 * h, lambda_0 * h

        if material[p] == 0:  # 液体
            mu = 0.0

        U, sig, V = ti.svd(F[p])
        J = 1.0
        for d in ti.static(range(3)):
            new_sig = sig[d, d]
            if material[p] == 2:  # 雪
                new_sig = min(max(sig[d, d], 1 - 2.5e-2), 1 + 5.4e-3)  # 可塑性
            Jp[p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig
        if material[p] == 0:  # 重置变形梯度以避免数值不稳定
            F[p] = ti.Matrix.identity(float, 3) * ti.sqrt(J)
            new_F = ti.Matrix.identity(float, 3)
            new_F[0, 0] = J
            F[p] = new_F
        elif material[p] == 2:
            F[p] = U @ sig @ V.transpose()  # 塑性后重建弹性变形梯度

        stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + ti.Matrix.identity(float, 3) * la * J * (J - 1)
        stress = (-dt * p_vol * 4) * stress / dx**2
        affine = stress + p_mass * C[p]

        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass
    for I in ti.grouped(grid_m):
        if grid_m[I] > 0:
            grid_v[I] /= grid_m[I]
        grid_v[I][1] -= dt * gravity
        cond = I < bound and grid_v[I] < 0 or I > n_grid - bound and grid_v[I] > 0  # 边界条件
        grid_v[I] = 0 if cond else grid_v[I]
    ti.block_dim(n_grid)
    for p in x:
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.zero(v[p])
        new_C = ti.zero(C[p])
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            g_v = grid_v[base + offset]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) / dx ** 2
        v[p], C[p] = new_v, new_C
        x[p] += dt * v[p]


group_size = n_particles // 3  # 材质块粒子数


@ti.kernel
def copy_material(np_x: ti.ext_arr(), input_x: ti.template()):
    for i in x:
        np_x[i] = input_x[i]

@ti.kernel
def copy_color(np_c: ti.ext_arr(), input_c: ti.ext_arr()):
    for i in x:
        np_c[i] = input_c[i]

@ti.kernel
def initialize():
    for i in range(n_particles):  # 初始化粒子的位置
        x[i] = [ti.random() * 0.2 + 0.3 + 0.08 * (i // group_size),  # 材质块x长度(随机*x偏移量） + 所有粒子的在x轴位置 + 材质块x轴的相对间隔(x偏移量*(粒子i//材质块粒子数，粒子i超过材质块粒子数时说明在下一材质块内，偏移一个整数量))
                ti.random() * 0.2 + 0.5 + 0.32 * (i // group_size),  # 材质块y长度(随机*y偏移量） + 所有粒子的在y轴位置 + 材质块y轴的相对间隔(y偏移量*(例如:材质块粒子数=21845, 21845//21845=1,粒子在第二个材质块偏移量为1))
                ti.random() * 0.2 + 0.3 + 0.1 * (i // group_size)]  # 材质块z长度(随机*z偏移量） + 所有粒子的在z轴位置 + 材质块z轴的相对间隔(z偏移量*(材质块ID))
        material[i] = i // group_size  # 材质块ID 0:流体; 1:果冻; 2:雪;
        v[i] = ti.Matrix([0, 0, 0])  # 初始化速度
        F[i] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # 初始化形变梯度
        Jp[i] = 1

def T(a):  # 投影变换

    phi, theta = np.radians(28), np.radians(32)

    a = a - 0.5
    x, y, z = a[:, 0], a[:, 1], a[:, 2]
    c, s = np.cos(phi), np.sin(phi)
    C, S = np.cos(theta), np.sin(theta)
    x, z = x * c + z * s, z * c - x * s
    u, v = x, y * C + z * S
    return np.array([u, v]).swapaxes(0, 1) + 0.5


initialize()
gui = ti.GUI("Taichi MLS-MPM-99-3D", res, background_color=0x112F41)
while gui.running and not gui.get_event(gui.ESCAPE):

    for s in range(steps):
        substep()  # 更新模拟帧
    pos = x.to_numpy()  # 转换位置的类型

    colors = np.array([0x068599, 0xFF8888, 0xEEEEF0], dtype=np.uint32)  # 材质颜色
    np_color = np.ndarray((n_particles,), dtype=np.uint32)  # 粒子颜色
    copy_color(np_color, colors)  # 把颜色一一对应赋给粒子

    np_material = np.ndarray((n_particles,), dtype=np.uint32)  # 粒子材质属性
    copy_material(np_material, material)  # 把材质属性一一对应赋给粒子

    gui.circles(T(pos), radius=1.4, color=np_color[np_material])
    gui.show()
