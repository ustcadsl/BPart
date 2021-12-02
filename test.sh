#node_array = ('node1' 'node2' 'node3' 'node4' 'node5' 'node6' 'node7')
for((i=1;i<=1;i++)); 
do
#   mpiexec -npernode 1 -hostfile ./hosts --prefix /usr/local/openmpi --mca btl_tcp_if_include 10.0.0.51/24 ./bin/deepwalk -g ../../data/friendster.data -v 68349467 -w 1000000000 -s unweighted -l 10  >> result_8/result_8_deepwalk_friendster_directed.txt
mpiexec -npernode 1 -hostfile ./hosts --prefix /usr/local/openmpi --mca btl_tcp_if_include 10.0.0.51/24 ./bin/deepwalk -g ../../data/friendster_new.data -v 65608367 -w 1000000000 -s unweighted -l 10 # >> result_8_fennel/result_8_deepwalk_friendster_double.txt
#   mpiexec -npernode 1 -hostfile ./hosts --prefix /usr/local/openmpi --mca btl_tcp_if_include 10.0.0.51/24 ./bin/deepwalk -g ../../data/twitter_double_reorder.data -v 41391963 -w 1000000000 -s unweighted -l 10 #>> result_8_fennel/result_8_deepwalk_twitter_double.txt
#   mpiexec -npernode 1 -hostfile ./hosts --prefix /usr/local/openmpi --mca btl_tcp_if_include 192.168.1.107/24 ./bin/deepwalk -g ../../data/friendster.data -v 68349467 -w 1000000000 -s unweighted -l 10 >> result_back_walker_count_1gbps/result4/result_4_deepwalk_friendster_directed.txt
#   mpiexec -npernode 1 -hostfile ./hosts --prefix /usr/local/openmpi --mca btl_tcp_if_include 192.168.1.107/24 ./bin/deepwalk -g ../../data/friendster_new.data -v 65608367 -w 1000000000 -s unweighted -l 10 >> result_back_walker_count_1gbps/result4/result_4_deepwalk_friendster_double.txt
#   mpiexec -npernode 1 -hostfile ./hosts --prefix /usr/local/openmpi --mca btl_tcp_if_include 192.168.1.107/24 ./bin/deepwalk -g ../../data/twitter_double_reorder.data -v 41391963 -w 1000000000 -s unweighted -l 10 >> result_back_walker_count_1gbps/result4/result_4_deepwalk_twitter_double.txt
#   mpiexec -npernode 1 -hostfile ./hosts --prefix /usr/local/openmpi --mca btl_tcp_if_include 10.0.0.51/24 ./bin/deepwalk -g ../../data/twitter_double_reorder.data -v 41391963 -w 1000000000 -s unweighted -l 10
#   mpiexec -npernode 1 -hostfile ./hosts --prefix /usr/local/openmpi --mca btl_tcp_if_include 10.0.0.51/24 ./bin/deepwalk -g ../../data/twitter_double_reorder.data -v 41391963 -w 1000000000 -s unweighted -l 10
#   mpiexec -npernode 1 -hostfile ./hosts --prefix /usr/local/openmpi --mca btl_tcp_if_include 192.168.1.107/24 ./bin/deepwalk -g ../../data/gen_graph_400K.data -v 400000 -w 10000000 -s unweighted -l 10 >> result_gendata/result_4_deepwalk_gendata_20badcut.txt
done
