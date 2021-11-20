
import os
import sys
import math
import heapq
from heapdict import heapdict
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.kdtree as kd
import astar
import draw
import reeds_shepp as rs
import random



class C:  # Parameter config
    PI = math.pi

    XY_RESO = 2.0  # [m]
    YAW_RESO = np.deg2rad(15.0)  # [rad]
    MOVE_STEP = 0.4  # [m] path interporate resolution
    N_STEER = 20.0  # 20steer command number
    COLLISION_CHECK_STEP = 5  # skip number for collision check
    EXTEND_BOUND = 1  # collision check range extended

    GEAR_COST = 100.0  # switch back penalty cost
    BACKWARD_COST = 5.0  # backward penalty cost
    STEER_CHANGE_COST = 5.0  # steer angle change penalty cost
    STEER_ANGLE_COST = 2.0  # steer angle penalty cost
    H_COST = 15.0  # Heuristic cost penalty cost

    RF = 3.9  # [m] distance from rear to vehicle front end of vehicle
    RB = 1.0  # [m] distance from rear to vehicle back end of vehicle
    W = 2.2  # [m] width of vehicle
    WD = 0.7 * W  # [m] distance between left-right wheels
    WB = 3  # [m] Wheel base
    TR = 0.5  # [m] Tyre radius
    TW = 1  # [m] Tyre width
    MAX_STEER = 0.4  # [rad] maximum steering angle


class Node:
    def __init__(self, xind, yind, yawind, direction, x, y,
                 yaw, directions, steer, cost, pind):
        self.xind = xind
        self.yind = yind
        self.yawind = yawind
        self.direction = direction
        self.x = x
        self.y = y
        self.yaw = yaw
        self.directions = directions
        self.steer = steer
        self.cost = cost
        self.pind = pind


class Para:
    def __init__(self, minx, miny, minyaw, maxx, maxy, maxyaw,
                 xw, yw, yaww, xyreso, yawreso, ox, oy, kdtree):
        self.minx = minx
        self.miny = miny
        self.minyaw = minyaw
        self.maxx = maxx
        self.maxy = maxy
        self.maxyaw = maxyaw
        self.xw = xw
        self.yw = yw
        self.yaww = yaww
        self.xyreso = xyreso
        self.yawreso = yawreso
        self.ox = ox
        self.oy = oy
        self.kdtree = kdtree


class Path:
    def __init__(self, x, y, yaw, direction, cost):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.direction = direction
        self.cost = cost


class QueuePrior:
    def __init__(self):
        self.queue = heapdict()

    def empty(self):
        return len(self.queue) == 0  # if Q is empty

    def put(self, item, priority):
        self.queue[item] = priority  # push

    def get(self):
        return self.queue.popitem()[0]  # pop out element with smallest priority


def hybrid_astar_planning(sx, sy, syaw, gx, gy, gyaw, ox, oy, xyreso, yawreso):
    sxr, syr = round(sx / xyreso), round(sy / xyreso)
    gxr, gyr = round(gx / xyreso), round(gy / xyreso)
    syawr = round(rs.pi_2_pi(syaw) / yawreso)
    gyawr = round(rs.pi_2_pi(gyaw) / yawreso)

    nstart = Node(sxr, syr, syawr, 1, [sx], [sy], [syaw], [1], 0.0, 0.0, -1)
    ngoal = Node(gxr, gyr, gyawr, 1, [gx], [gy], [gyaw], [1], 0.0, 0.0, -1)

    kdtree = kd.KDTree([[x, y] for x, y in zip(ox, oy)])
    P = calc_parameters(ox, oy, xyreso, yawreso, kdtree)

    hmap = astar.calc_holonomic_heuristic_with_obstacle(ngoal, P.ox, P.oy, P.xyreso, 1.0)
    steer_set, direc_set = calc_motion_set()
    open_set, closed_set = {calc_index(nstart, P): nstart}, {}

    qp = QueuePrior()
    qp.put(calc_index(nstart, P), calc_hybrid_cost(nstart, hmap, P))

    while True:
        if not open_set:
            return None

        ind = qp.get()
        n_curr = open_set[ind]
        closed_set[ind] = n_curr
        open_set.pop(ind)

        # update, fpath = update_node_with_analystic_expantion(n_curr, ngoal, P)

        if math.sqrt((n_curr.xind - ngoal.xind)**2 + (n_curr.yind - ngoal.yind)**2) < 5: #Terminal Reeds Shepp Check
            fnode = n_curr
            print(n_curr.x, n_curr.y, ngoal.x, ngoal.y)
            break

        for i in range(len(steer_set)):
            node = calc_next_node(n_curr, ind, steer_set[i], direc_set[i], P)

            if not node:
                continue

            node_ind = calc_index(node, P)

            if node_ind in closed_set:
                continue

            if node_ind not in open_set:
                open_set[node_ind] = node
                qp.put(node_ind, calc_hybrid_cost(node, hmap, P))
            else:
                if open_set[node_ind].cost > node.cost:
                    open_set[node_ind] = node
                    qp.put(node_ind, calc_hybrid_cost(node, hmap, P))

    return extract_path(closed_set, fnode, nstart)


def extract_path(closed, ngoal, nstart):
    rx, ry, ryaw, direc = [], [], [], []
    cost = 0.0
    node = ngoal

    while True:
        rx += node.x[::-1]
        ry += node.y[::-1]
        ryaw += node.yaw[::-1]
        direc += node.directions[::-1]
        cost += node.cost

        if is_same_grid(node, nstart):
            break

        node = closed[node.pind]

    rx = rx[::-1]
    ry = ry[::-1]
    ryaw = ryaw[::-1]
    direc = direc[::-1]

    direc[0] = direc[1]
    path = Path(rx, ry, ryaw, direc, cost)

    return path


def calc_next_node(n_curr, c_id, u, d, P):
    step = C.XY_RESO * 2
    d = d*2
    nlist = math.ceil(step / C.MOVE_STEP)
    xlist = [n_curr.x[-1] + d * C.MOVE_STEP * math.cos(n_curr.yaw[-1])]
    ylist = [n_curr.y[-1] + d * C.MOVE_STEP * math.sin(n_curr.yaw[-1])]
    yawlist = [rs.pi_2_pi(n_curr.yaw[-1] + d * C.MOVE_STEP / C.WB * math.tan(u))]

    for i in range(nlist - 1):
        xlist.append(xlist[i] + d * C.MOVE_STEP * math.cos(yawlist[i]))
        ylist.append(ylist[i] + d * C.MOVE_STEP * math.sin(yawlist[i]))
        yawlist.append(rs.pi_2_pi(yawlist[i] + d * C.MOVE_STEP / C.WB * math.tan(u)))

    xind = round(xlist[-1] / P.xyreso)
    yind = round(ylist[-1] / P.xyreso)
    yawind = round(yawlist[-1] / P.yawreso)

    if not is_index_ok(xind, yind, xlist, ylist, yawlist, P):
        return None

    cost = 0.0

    if d > 0:
        direction = 1
        cost += abs(step)
    else:
        direction = -1
        cost += abs(step) * C.BACKWARD_COST

    if direction != n_curr.direction:  # switch back penalty
        cost += C.GEAR_COST

    cost += C.STEER_ANGLE_COST * abs(u)  # steer angle penalyty
    cost += C.STEER_CHANGE_COST * abs(n_curr.steer - u)  # steer change penalty
    cost = n_curr.cost + cost

    directions = [direction for _ in range(len(xlist))]

    node = Node(xind, yind, yawind, direction, xlist, ylist,
                yawlist, directions, u, cost, c_id)

    return node


def is_index_ok(xind, yind, xlist, ylist, yawlist, P):
    if xind <= P.minx or \
            xind >= P.maxx or \
            yind <= P.miny or \
            yind >= P.maxy:
        return False

    ind = range(0, len(xlist), C.COLLISION_CHECK_STEP)

    nodex = [xlist[k] for k in ind]
    nodey = [ylist[k] for k in ind]
    nodeyaw = [yawlist[k] for k in ind]

    if is_collision(nodex, nodey, nodeyaw, P):
        return False

    return True




def is_collision(x, y, yaw, P):
    for ix, iy, iyaw in zip(x, y, yaw):
        d = 1
        dl = (C.RF - C.RB) / 2.0
        r = (C.RF + C.RB) / 2.0 + d

        cx = ix + dl * math.cos(iyaw)
        cy = iy + dl * math.sin(iyaw)

        ids = P.kdtree.query_ball_point([cx, cy], r)

        if not ids:
            continue

        for i in ids:
            xo = P.ox[i] - cx
            yo = P.oy[i] - cy
            dx = xo * math.cos(iyaw) + yo * math.sin(iyaw)
            dy = -xo * math.sin(iyaw) + yo * math.cos(iyaw)

            if abs(dx) < r and abs(dy) < C.W / 2 + d:
                return True

    return False


def calc_rs_path_cost(rspath):
    cost = 0.0

    for lr in rspath.lengths:
        if lr >= 0:
            cost += 1
        else:
            cost += abs(lr) * C.BACKWARD_COST

    for i in range(len(rspath.lengths) - 1):
        if rspath.lengths[i] * rspath.lengths[i + 1] < 0.0:
            cost += C.GEAR_COST

    for ctype in rspath.ctypes:
        if ctype != "S":
            cost += C.STEER_ANGLE_COST * abs(C.MAX_STEER)

    nctypes = len(rspath.ctypes)
    ulist = [0.0 for _ in range(nctypes)]

    for i in range(nctypes):
        if rspath.ctypes[i] == "R":
            ulist[i] = -C.MAX_STEER
        elif rspath.ctypes[i] == "WB":
            ulist[i] = C.MAX_STEER

    for i in range(nctypes - 1):
        cost += C.STEER_CHANGE_COST * abs(ulist[i + 1] - ulist[i])

    return cost


def calc_hybrid_cost(node, hmap, P):
    cost = node.cost + \
           C.H_COST * hmap[node.xind - P.minx][node.yind - P.miny]

    return cost


def calc_motion_set():
    s = np.arange(C.MAX_STEER / C.N_STEER,
                  C.MAX_STEER, C.MAX_STEER / C.N_STEER)

    steer = list(s) + [0.0] + list(-s)
    direc = [1.0 for _ in range(len(steer))] + [-1.0 for _ in range(len(steer))]
    steer = steer + steer

    return steer, direc


def is_same_grid(node1, node2):
    if node1.xind != node2.xind or \
            node1.yind != node2.yind or \
            node1.yawind != node2.yawind:
        return False

    return True


def calc_index(node, P):
    ind = (node.yawind - P.minyaw) * P.xw * P.yw + \
          (node.yind - P.miny) * P.xw + \
          (node.xind - P.minx)

    return ind


def calc_parameters(ox, oy, xyreso, yawreso, kdtree):
    minx = round(min(ox) / xyreso)
    miny = round(min(oy) / xyreso)
    maxx = round(max(ox) / xyreso)
    maxy = round(max(oy) / xyreso)

    xw, yw = maxx - minx, maxy - miny

    minyaw = round(-C.PI / yawreso) - 1
    maxyaw = round(C.PI / yawreso)
    yaww = maxyaw - minyaw

    return Para(minx, miny, minyaw, maxx, maxy, maxyaw,
                xw, yw, yaww, xyreso, yawreso, ox, oy, kdtree)


def draw_car(x, y, yaw, steer, color='black'):
    car = np.array([[-C.RB, -C.RB, C.RF, C.RF, -C.RB],
                    [C.W / 2, -C.W / 2, -C.W / 2, C.W / 2, C.W / 2]])

    wheel = np.array([[-C.TR, -C.TR, C.TR, C.TR, -C.TR],
                      [C.TW / 4, -C.TW / 4, -C.TW / 4, C.TW / 4, C.TW / 4]])

    rlWheel = wheel.copy()
    rrWheel = wheel.copy()
    frWheel = wheel.copy()
    flWheel = wheel.copy()

    Rot1 = np.array([[math.cos(yaw), -math.sin(yaw)],
                     [math.sin(yaw), math.cos(yaw)]])

    Rot2 = np.array([[math.cos(steer), math.sin(steer)],
                     [-math.sin(steer), math.cos(steer)]])

    frWheel = np.dot(Rot2, frWheel)
    flWheel = np.dot(Rot2, flWheel)

    frWheel += np.array([[C.WB], [-C.WD / 2]])
    flWheel += np.array([[C.WB], [C.WD / 2]])
    rrWheel[1, :] -= C.WD / 2
    rlWheel[1, :] += C.WD / 2

    frWheel = np.dot(Rot1, frWheel)
    flWheel = np.dot(Rot1, flWheel)

    rrWheel = np.dot(Rot1, rrWheel)
    rlWheel = np.dot(Rot1, rlWheel)
    car = np.dot(Rot1, car)

    frWheel += np.array([[x], [y]])
    flWheel += np.array([[x], [y]])
    rrWheel += np.array([[x], [y]])
    rlWheel += np.array([[x], [y]])
    car += np.array([[x], [y]])

    plt.plot(car[0, :], car[1, :], color)
    plt.plot(frWheel[0, :], frWheel[1, :], color)
    plt.plot(rrWheel[0, :], rrWheel[1, :], color)
    plt.plot(flWheel[0, :], flWheel[1, :], color)
    plt.plot(rlWheel[0, :], rlWheel[1, :], color)
    draw.Arrow(x, y, yaw, C.WB * 0.8, color)



latestfire = 0
n_ob = 10;
ob_or = [];
ob_type = [];
iter1 = 0
while iter1 < n_ob:
    a = random.randrange(30, 210, 45)
    b = random.randrange(30, 210, 60)
    if (a,b) in ob_or:
        continue
    ob_or.append((a,b))
    iter1+=1

burning = np.zeros(n_ob)

side = 15
otype = []
otype.append([(side,0), (side,side), (side,2*side), (0,2*side)])
otype.append([(0,0), (0,side), (0,2*side), (side,side)])
otype.append([(side,0), (side,side), (0,side), (0,2*side)])
otype.append([(side,0), (0,side), (side,2*side), (side,side)])

def design_obstacles(x, y):
    ox, oy = [], []

    for i in range(x):
        ox.append(i)
        oy.append(0)
    for i in range(x):
        ox.append(i)
        oy.append(y - 1)
    for i in range(y):
        ox.append(0)
        oy.append(i)
    for i in range(y):
        ox.append(x - 1)
        oy.append(i)


    for org in ob_or:
        r = random.randrange(0, 3, 1)
        ob_type.append(r)
        for ocoor in otype[r]:
            (x,y) = org
            x = x + ocoor[0]
            y = y + ocoor[1]
            add_square(15,x,y, ox, oy)

    return ox, oy

def burning_obstacles():
    bx, by = [], []
    for i in range(n_ob):
        if burning[i] == 1:
            r = ob_type[i]
            for ocoor in otype[r]:
                (x,y) = ob_or[i]
                x = x + ocoor[0]
                y = y + ocoor[1]
                add_square(15,x,y, bx, by)

    return bx, by


def add_square(side, x, y, ox, oy):
    ctr=0
    for i in range(side):
        for j in range(side):
            ox.append(x+j)
            oy.append(y+ctr)
        ctr+=1


    #     ox.append(x)
    #     oy.append(y+i)
    #
    #     ox.append(x+i)
    #     oy.append(y+side)
    #
    #     ox.append(x+side)
    #     oy.append(y+i)
    # ctr=0
    # for i in range(1,side-1):
    #     ox.append(x+i)
    #     oy.append(ctr+=1)



def burn_nearby(index):
    (latestx, latesty) = ob_or[index]
    for i in range(n_ob):
        (x,y) = ob_or[i]
        if x>latestx-59 and x<latestx+59 and y>latesty-74 and y<latesty+74:
            burning[i] = 1


def main():
    print("start!")
    # x, y = 51, 31
    gsize = 251

    sx, sy, syaw0 = 10.0, 10.0, np.deg2rad(0.0)
    carx, cary, caryaw = sx, sy, syaw0

    ox, oy = design_obstacles(gsize, gsize)
    x1=[]
    y1=[]
    yaw1=[]
    direction1=[]
    burning1=[]
    n_min = 2
    arson = random.sample(range(0, n_ob-1), n_min)
    stime = 0
    water = [0]
    bx0=[]
    by0=[]
    bx1=[]
    by1=[]
    gx0=[]
    gy0=[]
    gyaw1=[]
    gx1=[]
    gy1=[]
    gyaw2=[]

    while stime < n_min * 60:
        if stime % 60 == 0:
            burning[arson[int(stime/60)]] = 1
            latestfire = arson[int(stime/60)]

        if burning.sum() == 0:
            stime += 1
            continue


        bdist = np.inf
        goal = [0]
        for i in range(n_ob):
            if burning[i] == 0:
                continue
            dist = math.sqrt((carx - ob_or[i][0])**2 + (cary - ob_or[i][1])**2)
            if dist < bdist:
                bdist = dist
                goal[0] = ob_or[i]
                water[0] = i

        gx, gy, gyaw0 = goal[0][0]-4 +side, goal[0][1]-4, np.deg2rad(0.0)

        t0 = time.time()
        path = hybrid_astar_planning(carx, cary, caryaw, gx, gy, gyaw0,
                                     ox, oy, C.XY_RESO, C.YAW_RESO)
        t1 = time.time()
        print("running T: ", t1 - t0)

        if not path:
            print("Searching failed!")
            return

        x = path.x
        y = path.y
        yaw = path.yaw
        direction = path.direction
        x1.extend(x)
        y1.extend(y)
        yaw1.extend(yaw)
        direction1.extend(direction)
        burning1.extend(burning)
        bx1.extend(bx0)
        by1.extend(by0)
        bx, by = burning_obstacles()
        pathlen = len(x)
        for k in range(len(x)):

            stime += 0.1
            stime = round(stime,1)
            if stime % 60 == 0:
                burning[arson[int(stime/60)]] = 1
                latestfire = arson[int(stime/60)]
                bx, by = burning_obstacles()

            if stime % 60 == 20:
                burn_nearby(latestfire)
                bx, by = burning_obstacles()

            bx0.append(bx)
            by0.append(by)
            gx0.append(gx)
            gy0.append(gy)
            gyaw1.append(gyaw0)

        carx, cary, caryaw = x[pathlen-1], y[pathlen-1], yaw[pathlen-1]
        print("Done!")
        burning[water[0]] = 0

        stime = stime//1
        stime += 1
        gx1.extend(gx0)
        gy1.extend(gy0)
        gyaw2.extend(gyaw1)
    for k in range(len(x1)):
        plt.cla()
        plt.plot(ox, oy, "bo", markersize=1)
        plt.plot(bx1[k], by1[k], "bo", markersize=1, color = 'r')
        plt.plot(x1, y1, linewidth=1.5, color='r')

        if k < len(x1) - 2:
            dy = (yaw1[k + 1] - yaw1[k]) / C.MOVE_STEP
            steer = rs.pi_2_pi(math.atan(-C.WB * dy / direction1[k]))
        else:
            steer = 0.0

        draw_car(gx1[k], gy1[k], gyaw2[k], 0.0, 'dimgray')
        draw_car(x1[k], y1[k], yaw1[k], steer)
        # print(stime)
        plt.axis("equal")
        plt.pause(0.0001)
    plt.show()

if __name__ == '__main__':
    main()
