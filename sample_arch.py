import sys
import os
import argparse
import numpy as np
import genotypes

# randomly sample an arch from DARTS Search Space
def sample_arch():
    # 4 steps in each cell
    steps=4
    k = sum(1 for i in range(steps) for n in range(2+i))  # possible connections
    num_ops = len(genotypes.PRIMITIVES)
    n_nodes = steps

    normal = []
    reduction = []

    for i in range(n_nodes):
        # randomly choose an operation for each node
        ops = np.random.choice(range(num_ops), 4)
        nodes_in_normal = np.random.choice(range(i+2), 2, replace=False)
        nodes_in_reduce = np.random.choice(range(i+2), 2, replace=False)
        normal.extend([(genotypes.PRIMITIVES[ops[0]],nodes_in_normal[0]), (genotypes.PRIMITIVES[ops[1]],nodes_in_normal[1])])
        reduction.extend([(genotypes.PRIMITIVES[ops[2]],nodes_in_reduce[0]), (genotypes.PRIMITIVES[ops[3]],nodes_in_reduce[1])])

    
    concat = range(2, steps+2)
    genotype = genotypes.Genotype(
      normal=normal, normal_concat=concat,
      reduce=reduction, reduce_concat=concat
    )
    return genotype


def main(args):
    num=args.num
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    for i in range(num):
        genotype=sample_arch()
        save_path=os.path.join(args.save_dir,str(i+1))
        with open(save_path,'w') as f:
            f.write(str(genotype))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args for sampling random archs')
    parser.add_argument('--num', dest='num', type=int, default=400)
    parser.add_argument('--save_dir', dest='save_dir', type=str, default="archs")
    args = parser.parse_args() 
    main(args)
