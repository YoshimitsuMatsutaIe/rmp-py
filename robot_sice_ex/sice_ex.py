import numpy as np
from math import sin, cos, pi, sqrt
np.set_printoptions(precision=1)
import copy

from numba import njit

import sys
sys.path.append(".")

import mappings


@njit("f8[:,:](f8, f8)", cache=True)
def HTM(theta, l):
    return np.array([
        [cos(theta), -sin(theta), l],
        [sin(theta), cos(theta), 0.],
        [0., 0., 1.]
    ])

@njit("f8[:,:](f8)", cache=True)
def HTM_dot_by_q(theta):
    return np.array([
        [-sin(theta), -cos(theta), 0.],
        [cos(theta), -sin(theta), 0.],
        [0., 0., 0.]
    ])

@njit("f8[:,:](f8, f8)", cache=True)
def HTM_dot_by_t(theta, theta_dot):
    return HTM_dot_by_q(theta) * theta_dot

@njit("f8[:,:](f8, f8)", cache=True)
def HTM_dot_by_q_dot_by_t(theta, theta_dot):
    return np.array([
        [-cos(theta)*theta_dot, sin(theta)*theta_dot, 0.],
        [-sin(theta)*theta_dot, -cos(theta)*theta_dot, 0.],
        [0., 0., 0.]
    ])

# @njit("Tuple((f8[:,:], f8[:,:], f8[:,:], f8[:,:], f8[:,:], f8[:,:]))(i8, f8[:,:], f8[:,:], f8[:], i8)", cache=True)
# def calc_all_kinematics(n, q, dq, l, c_dim):
#     t_dim = 2
    
#     # local同時変換行列
#     T_local = np.empty((t_dim+1, (c_dim+1)*3))
#     for i in range(c_dim):
#         if i == 0:
#             T_local[:, 0:3] = HTM(q[0, 0], 0)
#         else:
#             T_local[:, 3*i:3*(i+1)] = HTM(q[i, 0], l[i-1])
#     T_local[:, c_dim*3:] = HTM(0, l[c_dim-1])


#     # global同時変換行列
#     T_global = np.empty((t_dim+1, (c_dim+1)*3))
#     for i in range(c_dim+1):
#         if i == 0:
#             T_global[:, 0:3] = T_local[:, 0:3]
#         else:
#             T_global[:, 3*i:3*(i+1)] = T_global[:, 3*(i-1):3*i] @ T_local[:, 3*i:3*(i+1)]


#     # local同時変換行列の角度微分
#     dTdq_local = np.empty((t_dim+1, (c_dim+1)*3))
#     for i in range(c_dim):
#         if i == 0:
#             dTdq_local[:, 0:3] = HTM_dot_by_q(q[0, 0])
#         else:
#             dTdq_local[:, 3*i:3*(i+1)] = HTM_dot_by_q(q[i, 0])
#     dTdq_local[:, 3*c_dim:3*(c_dim+1)] = HTM_dot_by_q(0)


#     dTdq_s = np.empty((c_dim*(t_dim+1), (c_dim+1)*3))
#     for i in range(c_dim):  #joint val
#         tmp = np.empty((t_dim+1, (c_dim+1)*3))
#         for j in range(c_dim+1):  #T val
#             if j == 0:
#                 if i == 0:
#                     tmp[:, 0:3] = dTdq_local[:, 0:3]
#                 else:
#                     tmp[:, 0:3] = T_local[:, 0:3]
#             else:
#                 if j == i:
#                     tmp[:, 3*j:3*(j+1)] = tmp[:, 3*(j-1):3*j] @ dTdq_local[:, 3*j:3*(j+1)]
#                 else:
#                     tmp[:, 3*j:3*(j+1)] = tmp[:, 3*(j-1):3*j] @ T_local[:, 3*j:3*(j+1)]
        
#         dTdq_s[3*i:3*(i+1), :] = tmp


#     Jrx = np.empty((t_dim, c_dim*(c_dim+1)))
#     Jry = np.empty((t_dim, c_dim*(c_dim+1)))
#     Joo = np.empty((t_dim, c_dim*(c_dim+1)))

#     for i in range(c_dim+1):
#         Jrx_ = np.zeros((t_dim, c_dim))
#         Jry_ = np.zeros((t_dim, c_dim))
#         Joo_ = np.zeros((t_dim, c_dim))
        
#         for j in range(i):
#             T_ = dTdq_s[3*j:3*(j+1), 3*i:3*(i+1)]
#             Jrx_[:, j:j+1] = T_[0:t_dim, 0:1]
#             Jry_[:, j:j+1] = T_[0:t_dim, 1:2]
#             Joo_[:, j:j+1] = T_[0:t_dim, 2:3]
        
#         Jrx[:, c_dim*i:c_dim*(i+1)] = Jrx_
#         Jry[:, c_dim*i:c_dim*(i+1)] = Jry_
#         Joo[:, c_dim*i:c_dim*(i+1)] = Joo_
    
#     rx = T_global[0:2, 3*n+0:3*n+1]
#     ry = T_global[0:2, 3*n+1:3*n+2]
#     oo = T_global[0:2, 3*n+2:3*n+3]
#     jrx = Jrx[:, c_dim*n:c_dim*(n+1)]
#     jry = Jry[:, c_dim*n:c_dim*(n+1)]
#     joo = Joo[:, c_dim*n:c_dim*(n+1)]
    
    
    
    
#     return rx, ry, oo, jrx, jry, joo


@njit("Tuple((f8[:,:], f8[:,:], f8[:,:], f8[:,:], f8[:,:], f8[:,:], f8[:,:], f8[:,:], f8[:,:]))(i8, f8[:,:], f8[:,:], f8[:], i8)", cache=True)
def calc_all_kinematics2(n, q, dq, l, c_dim):
    """1と変わらん"""
    t_dim = 2
    
    # local同時変換行列
    T_local = []
    for i in range(c_dim):
        if i == 0:
            T_local.append(HTM(q[0,0], 0))
        else:
            T_local.append(HTM(q[i,0], l[i-1]))
    T_local.append(HTM(0, l[-1]))


    # global同時変換行列
    T_global = []
    for i, T in enumerate(T_local):
        if i == 0:
            T_global.append(T_local[0])
        else:
            T_global.append(T_global[i-1] @ T)



    # local同時変換行列の角度微分
    dTdq_local = []
    for i in range(c_dim):
        dTdq_local.append(HTM_dot_by_q(q[i, 0]))
    dTdq_local.append(np.zeros((3,3)))


    dTdq_s = []
    for i in range(c_dim):  #joint val
        tmp_dTdq_s = []
        for j, T in enumerate(T_local):  #T val
            if j == 0:
                if i == 0:
                    #print("ho =", dTdq_local[0])
                    tmp_dTdq_s.append(dTdq_local[0])
                else:
                    tmp_dTdq_s.append(T_local[0])
            else:
                if j == i:
                    tmp_dTdq_s.append(tmp_dTdq_s[j-1] @ dTdq_local[j])
                else:
                    tmp_dTdq_s.append(tmp_dTdq_s[j-1] @ T)
        dTdq_s.append(tmp_dTdq_s)

    #print("dTdq_s = ", len(dTdq_s))

    Jrx = []
    Jry = []
    Joo = []

    for i in range(c_dim+1):
        Jrx_ = np.zeros((t_dim, c_dim))
        Jry_ = np.zeros((t_dim, c_dim))
        Joo_ = np.zeros((t_dim, c_dim))

        #print(len(dTdq_s[i]))
        for j in range(i+1):
            #print("i, j = ", i, j)
            if j == c_dim:
                T_ = dTdq_s[j-1][i]
            else:
                T_ = dTdq_s[j][i]
            Jrx_[:, j:j+1] = T_[0:t_dim, 0:1]
            Jry_[:, j:j+1] = T_[0:t_dim, 1:2]
            Joo_[:, j:j+1] = T_[0:t_dim, 2:3]

        Jrx.append(Jrx_)
        Jry.append(Jry_)
        Joo.append(Joo_)
    
    
    # local同時変換行列の時間微分
    T_local_dot = []
    for i in range(c_dim):
        T_local_dot.append(HTM_dot_by_t(q[i,0], dq[i,0]))
    T_local_dot.append(np.zeros((3,3)))
    
    # local同時変換行列の角度微分の時間便
    dTdq_local_dot = []
    for i in range(c_dim):
        dTdq_local_dot.append(HTM_dot_by_q_dot_by_t(q[i,0], dq[i,0]))
    dTdq_local_dot.append(np.zeros((3,3)))
    
    dTdq_dot_s = []
    for i in range(c_dim):  #joint val
        tmp = []
        for j in range(c_dim+1):  #T val
            if j == 0:
                if i == 0:
                    tmp.append(dTdq_local_dot[0])
                else:
                    tmp.append(T_local_dot[0])
            else:
                if j == i:
                    tmp.append(tmp[j-1] @ dTdq_local[j] + dTdq_s[i][j-1] @ dTdq_local_dot[j])
                else:
                    tmp.append(tmp[j-1] @ T_local[j] + dTdq_s[i][j-1] @ T_local_dot[j])
        dTdq_dot_s.append(tmp)


    Jrx_dot = []
    Jry_dot = []
    Joo_dot = []

    for i in range(c_dim+1):
        Jrx_dot_ = np.zeros((t_dim, c_dim))
        Jry_dot_ = np.zeros((t_dim, c_dim))
        Joo_dot_ = np.zeros((t_dim, c_dim))

        for j in range(i+1):
            if j == c_dim:
                T_ = dTdq_dot_s[j-1][i]
            else:
                T_ = dTdq_dot_s[j][i]
            Jrx_dot_[:, j:j+1] = T_[0:t_dim, 0:1]
            Jry_dot_[:, j:j+1] = T_[0:t_dim, 1:2]
            Joo_dot_[:, j:j+1] = T_[0:t_dim, 2:3]

        Jrx_dot.append(Jrx_dot_)
        Jry_dot.append(Jry_dot_)
        Joo_dot.append(Joo_dot_)
    
    rx = T_global[n][0:2, 0:1]
    ry = T_global[n][0:2, 1:2]
    oo = T_global[n][0:2, 2:3]
    
    return rx, ry, oo, Jrx[n], Jry[n], Joo[n], Jrx_dot[n], Jry_dot[n], Joo_dot[n]


def calc_kinematics(n, q, dq, l, c_dim):
    jrx, jry, joo = np.zeros((2,c_dim)), np.zeros((2,c_dim)), np.zeros((2,c_dim))
    jrx_dot, jry_dot, joo_dot = np.zeros((2,c_dim)), np.zeros((2,c_dim)), np.zeros((2,c_dim))
    
    if n == 0:
        rx, ry, oo = np.zeros((2,1)), np.zeros((2,1)), np.zeros((2,1))
    else:
        rx, ry, oo, jrx_, jry_, joo_, jrx_dot_, jry_dot_, joo_dot_ = calc_all_kinematics2(n, q, dq, l, n)
        jrx[:, :n] = jrx_
        jry[:, :n] = jry_
        joo[:, :n] = joo_
        jrx_dot[:, :n] = jrx_dot_
        jry_dot[:, :n] = jry_dot_
        joo_dot[:, :n] = joo_dot_
        
    return rx, ry, oo, jrx, jry, joo, jrx_dot, jry_dot, joo_dot



class CPoint(mappings.Identity):
    t_dim = 2
    
    def __init__(self, frame_num, position_num, **kwargs):
        self.c_dim = kwargs.pop('c_dim')
        self.ee_id = (self.c_dim, 0)
        self.RS_ALL = [((0, 0),) for _ in range(self.c_dim+1)]
        self.frame_num = frame_num
        total_length = kwargs.pop('total_length')
        self.ls = np.array([total_length/self.c_dim] * self.c_dim)
        
        self.r = self.RS_ALL[frame_num][position_num]
        
        self.q_neutral = np.zeros((self.c_dim, 1))  # ニュートラルの姿勢
        self.q_min = np.array([[-90] * self.c_dim]).T * np.pi/180
        self.q_max = -self.q_min

    def phi(self, q):
        return self.calc_o(self.c_dim, q)
    
    def calc_all(self, q, dq):
        rx, ry, oo, jrx, jry, joo, jrx_dot, jry_dot, joo_dot = calc_kinematics(self.frame_num, q, dq, self.ls, self.c_dim)
        #print("dq = ", dq.T)
        #print("Joo_dot_norm = ", np.linalg.norm(joo_dot))
        x = rx * self.r[0] + ry * self.r[1] + oo
        J = jrx*self.r[0] + jry*self.r[1] + joo
        x_dot = J @ dq
        
        J_dot = jrx_dot*self.r[0] + jry_dot*self.r[1] + joo_dot
        #J_dot = np.zeros(J.shape)
        return x, x_dot, J, J_dot


    def calc_o(self, n, q):
        #print(self.ls)
        _, _, out, _, _, _,_,_,_ = calc_kinematics(n, q, np.zeros(q.shape), self.ls, self.c_dim)
        return out


    def calc_joint_position_all(self, q):
        return [
            self.calc_o(i, q) for i in range(self.c_dim+1)
        ]



if __name__ == "__main__":
    import numpy as np
    from numpy import linalg as LA
    import robot_sice.sice as sice
    q = np.array([[1.1, 2.4, 3.3, 4.]]).T
    dq = q + 1
    #dq = np.zeros((4,1))
    l = np.array([1., 1., 1., 1.])
    for i in range(5):
        rx, ry, oo, jrx, jry, joo, jrx_dot, jry_dot, joo_dot = calc_all_kinematics2(i, q, dq, l, 4)
        rx_ = sice.rx(q, i)
        ry_ = sice.ry(q, i)
        oo_ = sice.o(q, i, *l)
        jrx_ = sice.jrx(q, i)
        jry_ = sice.jry(q, i)
        joo_ = sice.jo(q, i, *l)
        jrx_dot_ = sice.jrx_dot(q, dq, i)
        jry_dot_ = sice.jry_dot(q, dq, i)
        joo_dot_ = sice.jo_dot(q, dq, i, *l)
        print("\ni = ", i)
        # print("Jrx_dot = \n", jrx_dot - jrx_dot_)
        # print("Jry_dot = \n", jry_dot - jry_dot_)
        # print("Jroo_dot = \n", joo_dot - joo_dot_)
        # print("rx      = ", LA.norm(rx - rx_))
        # print("ry      = ", LA.norm(ry - ry_))
        # print("oo      = ", LA.norm(oo - oo_))
        # print("jrx     = ", LA.norm(jrx - jrx_))
        # #print("ori = \n", jrx_, "\ngen = \n", jrx, "\n")
        # print("jry     = ", LA.norm(jry - jry_))
        # #print("ori = \n", jry_, "\ngen = \n", jry, "\n")
        # print("joo     = ", LA.norm(joo - joo_))
        # #print("ori = \n", joo_, "\ngen = \n", joo, "\n")
        
        
        print("Jrx_dot = ", LA.norm(jrx_dot - jrx_dot_))
        print("ori = \n", jrx_dot_, "\ngen = \n", jrx_dot, "\n")
        print("Jry_dot = ", LA.norm(jry_dot - jry_dot_))
        print("ori = \n", jry_dot_, "\ngen = \n", jry_dot, "\n")
        print("Joo_dot = ", LA.norm(joo_dot - joo_dot_))
        print("ori = \n", joo_dot_, "\ngen = \n", joo_dot, "\n")
    
    # ss = hoge.JOINT_PHI()
    # for s in ss:
    #     print(s(np.random.rand(4,1)))
    
    
    
    
    
    # import robot_sice.sice as sice

    # hoge = JOINT_PHI()
    
    # q = np.random.rand(4, 1)
    # for i, f in enumerate(hoge):
    #     print("\n", f(q))
    #     print(sice.o(q, i, 1, 1, 1, 1))
    
    
    
    
    # ### チェック
    # # print("\nchecking rx...")
    # # for i in range(5):
    # #     print("i = ", i)
    # #     print(np.linalg.norm(T_global[i][0:2, 0:1] - sice.rx(q, i)))

    # # print("\nchecking ry...")
    # # for i in range(5):
    # #     print("i = ", i)
    # #     print(np.linalg.norm(T_global[i][0:2, 1:2] - sice.ry(q, i)))

    # # print("\nchecking o...")
    # # for i in range(5):
    # #     print("i = ", i)
    # #     print(np.linalg.norm((T_global[i][0:2, 2:3] - sice.o(q, i, l[0], l[1], l[2], l[3]))))

    # print("\nchecking jrx...")
    # for i in range(5):
    #     print("i = ", i)
    #     print("tmp = \n", Jrx[i])
    #     print("true = \n", sice.jrx(q, i))
    #     print(np.linalg.norm(Jrx[i] - sice.jrx(q, i)))

    # print("\nchecking jry...")
    # for i in range(5):
    #     print("i = ", i)
    #     print("tmp = \n", Jry[i])
    #     print("true = \n", sice.jry(q, i))
    #     print(np.linalg.norm(Jry[i] - sice.jry(q, i)))

    # print("\nchecking jo...")
    # for i in range(5):
    #     print("i = ", i)
    #     print("tmp = \n", Joo[i])
    #     print("true = \n", sice.jo(q, i, l[0], l[1], l[2], l[3]))
    #     print(np.linalg.norm(Joo[i] - sice.jo(q, i, l[0], l[1], l[2], l[3])))