import numpy as np
import matplotlib.pyplot as plt
from random import randrange
import random

def coverage(f,grid):
    random.seed(10)
    ctr=0
    ans=0
    while(ans<f):
        r=randrange(127*127-127)
        obs=randrange(4)
        if obs==0: #horizontal obstacle
            for i in range(4):
                grid[r+i]=1
        elif obs==1: #vertical obstacle
            for i in range(4):
                grid[r+128*i]=1
        elif obs==2: # vertical Z obstacle
            for i in range(4):
                if i<2:
                    grid[r+128*i]=1
                else:
                    grid[r+128*i-1]=1
        else:         # horizontal T obstacle
            for i in range(4):
                if i==3:
                    grid[r+1+128*(i-2)]=1
                else:
                    grid[r+128*i]=1

        ctr+=1
        ans=int(4*ctr/(128*128)*100)
    return grid

def generate_grid(f):
    grid=[0]*(128*128)
    grid=coverage(10,grid)
    grid=np.array(grid).reshape(128,128)
    return grid

def main():
    # grid = [ [0] * 128 for _ in range(128)]
    # print(grid)
    f=20
    grid=generate_grid(f)
    plt.figure(num="10 percent coverage")
    plt.imshow(grid,interpolation='nearest')
    plt.show()

main()
