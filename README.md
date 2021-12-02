# BPart

**BPart** is a graph partition scheme, which support two-dimensional balance partition. The core idea of **BPart** is to partition the graph into small subgraphs then combine some small ones into bigger one. **BPart** using 2-phases to achieve above idea, and can describle as fellows:

- The first phase consider both vertices and edges into balance condition, and using this weighted balance condition to get reduced skewness distribution for both vertices and edges.
- The second phase combine the subgraph with bigger number of vertces and the subgraph with smaller number of vertices to get a bigger subgraph.


**BPart** is implemented on the top of KnightKing[SOSP'19], which is the state-of-art distributed graph system that support random walk based application. We just replace the partition module of KnightKing, and do not modify the computation module. So the useage of this code is similarity with KnightKing (detail you can visit: https://github.com/KnightKingWalk/KnightKing.git). Next we using BPart to represent the code we modified on the top of KnightKing.

## Quick Start

This section gives a guide about how to compile BPart.

### Tested Environment

- Centos 7.9.2009
- gcc 4.8.5
- cmake 3.14.5
- OpenMPI 3.1.2

### Compile

Firstly you need to download this repository from GitHub,
then compile it with CMake:

```
cd BPart

mkdir build && cd build

cmake ..

make
```

The compiled executable files will be installed at the "bin" directory:

```
ls ./bin
```

### Run Built-in Applications

Here we take deepwalk as an example to show how to run the built-in applications. The usage of other four applications is quite similar.

Use "-h" option to print out deepwalk's options

```
~BPart/build$./bin/deepwalk -h
 ./bin/deepwalk {OPTIONS}


  OPTIONS:
      -h, --help                        Display this help menu
      -v[vertex]                        vertex number
      -g[graph]                         graph data path
      --make-undirected                 load graph and treat each edge as undirected edge
      -x[partition-type]                Choose the partition algorithm
      -w[walker]                        walker number
      -o[output]                        [optional] the output path. Omit this option for pure random walk performance testing without output.
      -r[rate]                          Set this option will break random walk into multiple iterations to save memory. Each iteration has rate% walkers.
      -l[length]                        walk length
      -s[static_comp]                   [weighted | unweighted] a weighted graph usually indicates a non-trivial static component.
      -c[cache_rate]                    Integer number, cache c% fraction of edges.
```

A graph random walk application usually takes a graph as input, then setups a group of walkers to wander around the graph. So we need to specify the path of the graph file, the number of vertices, and the number of walkers.

"-v" option specifies how many vertices the graph have. The range of vertex ID is \[0, vertex_num).

"-g" option specifies the path of input graph file.

"--make-undirected" option will treat each edge as undirected edge. For each edge (u, v) in the file, a new edge of (v, u) will be created.

"x" decide the partition algorithm we use. To be specific, when x equal to:

 - 0: Chunk-V partition algorithm.
 - 1: CHunk-E partition algorithm.
 - 2: BPart-C partition algorithm.
 - 3: Fennel partition algorithm.
 - 5: BPart-S partition algorithm.

"-w" option specifies how many walkers there are.

"-o" option specifies the prefix of the name of output files. This is an optional paramemter. If this parameter is set, then after random walk, the walking paths will be dumped to the disk.

"-r" option will break random walk into multiple epochs. Each iteration has rate% walkers. Use this option if memory is not enough to hold all walkers and their paths.

"-l" option specifies the walk length. Node2vec is a truncated random walk, which means each walker walks a pre-defined length.

"-s" option specifies whether the graph is a weighted graph.

"-c" option specifies how much edges we should cache in this application.

There is a text file containing a sample graph, but deepwalk takes a binary file as input. So first we convert the text file to binary format:

```
./bin/gconverter -i ../dataset/karate.txt -o ./karate.data -s weighted
```

Then we can invoke deepwalk:

```
mkdir out
./bin/node2vec -g ./karate.data -v 34 -w 5 -s weighted -x 0 -l 10 -p 2 -q 0.5 -c 0 -o ./out/walks.txt
```

See the random walk output:

```
~/graph/KnightKing/build$ cat ./out/walks.txt.0
0 5 16 6 4 6 16 6 4 6 16
1 21 0 2 8 30 33 15 32 8 32
2 13 0 10 0 31 0 6 16 5 0
3 12 0 11 0 17 1 2 9 33 30
4 0 1 21 1 17 1 2 8 32 2
```

There are 5 lines, each representing a path for one walker.

Note that you may have an output different from above example. Since this is a random walk, so the output is also random.

### Run in Distributed Environment

First, copy the graph file to the same path of each node, or simply place it to a shared file system. Second, write each node's IP address to a text file (e.g. ./hosts). Then use MPI to run the application. Suppose the graph file is placed at ./karate.data. For OpenMPI:

```
mpiexec -npernode 1 -hostfile ./hosts ./bin/deepwalk -g ./karate.data -v 34 -w 34 -s weighted -x 0 -l 10 -c 0 -o ./out/walks.txt
```
```

The "-npernode 1" setting is recommended, which tells MPI to instantiate one instance per node. The code will automatically handle the concurrency within each node. Instantiating more than one instances per node may make the graph more fragmented and thus hinge the performance.

See the random walk output:

```
cat ./out/walks.txt.*

### Creat your own applications

You can see the detail at https://github.com/KnightKingWalk/KnightKing.git, we do not talk about it here.