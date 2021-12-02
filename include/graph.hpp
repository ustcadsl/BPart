/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2019 Ke Yang, Tsinghua University 
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <type_traits>
#include <thread>
#include <random>
#include <functional>
#include <unistd.h>
#include <sys/mman.h>
#include <limits.h>
#include <numa.h>

#include <omp.h>

#include "type.hpp"
#include "util.hpp"
#include "constants.hpp"
#include "storage.hpp"
#include "mpi_helper.hpp"

template<typename edge_data_t>
struct AdjUnit
{
    vertex_id_t neighbour;
    edge_data_t data;
};

template<>
struct AdjUnit<EmptyData>
{
    union
    {
        vertex_id_t neighbour;
        EmptyData data;
    };
};

template<typename edge_data_t>
struct AdjList
{
    AdjUnit<edge_data_t> *begin;
    AdjUnit<edge_data_t> *end;
//    bool local_vertex;
    void init()
    {
        begin = nullptr;
        end = nullptr;
//        local_vertex = false;
    }
};

//comprised column row
template<typename edge_data_t>
struct EdgeContainer
{
    AdjList<edge_data_t> *adj_lists;
    AdjUnit<edge_data_t> *adj_units;
    EdgeContainer() : adj_lists(nullptr), adj_units(nullptr) {}
    ~EdgeContainer()
    {
        if (adj_lists != nullptr)
        {
            delete []adj_lists;
        }
        if (adj_units != nullptr)
        {
            delete []adj_units;
        }
    }
};

enum MPIMessageTag {
    Tag_ShuffleGraph,
    Tag_Msg,
    Tag_graph_vertex,
    Tag_graphdata,
    Tag_graphdata_adjlists,
    Tag_Fennel
};

template<typename T>
class Message
{
public:
    vertex_id_t dst_vertex_id;
//    partition_id_t last_partition_id;
    T data;
};

struct DistributedExecutionCtx
{
    std::mutex phase_locks[DISTRIBUTEDEXECUTIONCTX_PHASENUM];
    int unlocked_phase;
    size_t **progress;
public:
    DistributedExecutionCtx()
    {
        progress = nullptr;
    }
};

enum GraphFormat
{
    GF_Binary,
    GF_Edgelist
};

template<typename edge_data_t>
class GraphEngine
{
protected: 
    vertex_id_t v_num;
    edge_id_t e_num;
    int worker_num;
    edge_id_t local_e_num;
    double cache_rate;

    vertex_id_t *vertex_partition_begin;
    vertex_id_t *vertex_partition_end;

    partition_id_t local_partition_id;
    partition_id_t partition_num;

    bool *vertex_cached_partition_id;

    MessageBuffer **thread_local_msg_buffer; 
    MessageBuffer **msg_send_buffer;
    MessageBuffer **msg_recv_buffer;
    std::mutex *send_locks;
    std::mutex *recv_locks;
    std::mutex *send_msg_locks;
    int threads;
    int sockets;
    int threads_per_socket;

    DistributedExecutionCtx dist_exec_ctx;
public:
    vertex_id_t *vertex_in_degree;
    vertex_id_t *vertex_out_degree;
    partition_id_t *vertex_partition_id;
    partition_id_t *this_is_local_vertex;

    EdgeContainer<edge_data_t> *csr;
    EdgeContainer<edge_data_t> *csr_cut;

protected:
    void set_graph_engine_concurrency(int worker_num_param)
    {
        this->worker_num = worker_num_param;
        omp_set_dynamic(0);
        omp_set_num_threads(worker_num);
        //message buffer depends on worker number
        free_msg_buffer();
    }

public:
    inline bool is_local_vertex(vertex_id_t v_id)
    {
        //return v_id >= vertex_partition_begin[local_partition_id]
        //    && v_id < vertex_partition_end[local_partition_id];

        //return vertex_partition_id[v_id] == local_partition_id;
        return this_is_local_vertex[v_id] == this->local_partition_id;
    }
    inline bool is_valid_edge(Edge<edge_data_t> e)
    {
        return e.src < v_num && e.dst < v_num;
    }
    inline vertex_id_t get_vertex_num()
    {
        return v_num;
    }
    inline edge_id_t get_edge_num()
    {
        return e_num;
    }
    inline int get_worker_num()
    {
        return worker_num;
    }
    inline vertex_id_t get_local_vertex_begin()
    {
        return vertex_partition_begin[local_partition_id];
    }
    inline vertex_id_t get_local_vertex_end()
    {
        return vertex_partition_end[local_partition_id];
    }
    inline vertex_id_t get_vertex_begin(partition_id_t p)
    {
        return vertex_partition_begin[p];
    }
    inline vertex_id_t get_vertex_end(partition_id_t p)
    {
        return vertex_partition_end[p];
    }

public:
    // deallocate a vertex array
    template<typename T>
    void dealloc_vertex_array(T * array)
    {
        dealloc_array(array, v_num);
    }

    template<typename T>
    void dealloc_array(T * array, size_t num)
    {
        munmap(array, sizeof(T) * num);
    }

    // allocate a vertex array
    template<typename T>
    T * alloc_vertex_array()
    {
        return alloc_array<T>(v_num);
    }

    template<typename T>
    T * alloc_array(size_t num)
    {
        T* array = (T*) mmap(NULL, sizeof(T) * num, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        assert(array != nullptr);
        return array;
    }

    GraphEngine()
    {
        vertex_partition_begin = nullptr;
        vertex_partition_end = nullptr;
        cache_rate = 0.0;

        thread_local_msg_buffer = nullptr;
        vertex_cached_partition_id = nullptr;
        msg_send_buffer = nullptr;
        msg_recv_buffer = nullptr;

        send_locks = nullptr;
        recv_locks = nullptr;

        vertex_in_degree = nullptr;
        vertex_out_degree = nullptr;
        vertex_partition_id = nullptr;

        send_msg_locks = nullptr;
        this_is_local_vertex = nullptr;

        csr = nullptr;
        csr_cut = nullptr;
//	threads = numa_num_configured_cpus();
//	sockets = numa_num_configured_nodes();
//	threads_per_socket = threads / sockets;
		

        this->worker_num = std::max(1, ((int)std::thread::hardware_concurrency()) - 1);
        //this->worker_num = std::max(1, ((int)std::thread::hardware_concurrency())) * 2 - 1;
//        this->worker_num = threads;
	//printf("worker num %d, %d\n", this->worker_num, this->local_partition_id);
        //this->worker_num = 6;
        omp_set_dynamic(0);
        omp_set_num_threads(worker_num);
//	#pragma omp parallel for
//        for (int t_i=0;t_i<threads;t_i++) 
//        {
//            int s_i = t_i / threads_per_socket;
//            assert(numa_run_on_node(s_i)==0);
//        }
    }

    virtual ~GraphEngine()
    {
        if (vertex_partition_begin != nullptr)
        {
            delete []vertex_partition_begin;
        }
        if (vertex_partition_end != nullptr)
        {
            delete []vertex_partition_end;
        }

        if (send_locks != nullptr)
        {
            delete []send_locks;
        }
        if (recv_locks != nullptr)
        {
            delete []recv_locks;
        }

        if (vertex_in_degree != nullptr)
        {
            dealloc_vertex_array<vertex_id_t>(vertex_in_degree);
        }
        if (vertex_out_degree != nullptr)
        {
            dealloc_vertex_array<vertex_id_t>(vertex_out_degree);
        }
        if (vertex_partition_id != nullptr)
        {
            dealloc_vertex_array<partition_id_t>(vertex_partition_id);
        }

        if (csr != nullptr)
        {
            delete csr;
        }

        if (dist_exec_ctx.progress != nullptr)
        {
            for (partition_id_t t_i = 0; t_i < worker_num; t_i++)
            {
                delete []dist_exec_ctx.progress[t_i];
            }
            delete []dist_exec_ctx.progress;
        }

        free_msg_buffer();
    }

        void build_edge_container(Edge<edge_data_t> *edges, edge_id_t local_edge_num, EdgeContainer<edge_data_t> *ec, vertex_id_t* vertex_out_degree)
    {
        ec->adj_lists = new AdjList<edge_data_t>[v_num];
        ec->adj_units = new AdjUnit<edge_data_t>[local_edge_num];
        edge_id_t chunk_edge_idx = 0;
        for (vertex_id_t v_i = 0; v_i < v_num; v_i++)
        {
            if(vertex_partition_id[v_i] == local_partition_id)
            {
                ec->adj_lists[v_i].begin = ec->adj_units + chunk_edge_idx;
                ec->adj_lists[v_i].end = ec->adj_lists[v_i].begin;
                chunk_edge_idx += vertex_out_degree[v_i];
            }
        }
        for (edge_id_t e_i = 0; e_i < local_edge_num; e_i++)
        {
            auto e = edges[e_i];
            auto ep = ec->adj_lists[e.src].end ++;
            
            ep->neighbour = e.dst;
            
            if (!std::is_same<edge_data_t, EmptyData>::value)
            {
                ep->data = e.data;
            }
        }

	    delete []edges;
    }

    void shuffle_edges(Edge<edge_data_t> *misc_edges, edge_id_t misc_e_num, Edge<edge_data_t> *local_edges, edge_id_t local_e_num)
    {
        std::vector<edge_id_t> e_count(partition_num, 0);
        for (edge_id_t e_i = 0; e_i < misc_e_num; e_i++)
        {
            e_count[vertex_partition_id[misc_edges[e_i].src]]++;
        }
        Edge<edge_data_t> *tmp_es  = new Edge<edge_data_t>[misc_e_num];
        std::vector<edge_id_t> e_p(partition_num, 0);
        for (partition_id_t p_i = 1; p_i < partition_num; p_i++)
        {
            e_p[p_i] = e_p[p_i - 1] + e_count[p_i - 1];
        }
        auto e_begin = e_p;
        for (edge_id_t e_i = 0; e_i < misc_e_num; e_i++)
        {
            auto pt = vertex_partition_id[misc_edges[e_i].src];
            tmp_es[e_p[pt] ++] = misc_edges[e_i];
        }
        edge_id_t local_edge_p = 0;
        std::thread send_thread([&](){
            for (partition_id_t step = 0; step < partition_num; step++)
            {
                partition_id_t dst = (local_partition_id + step) % partition_num;
                size_t tot_send_sz = e_count[dst] * sizeof(Edge<edge_data_t>);
#ifdef UNIT_TEST
                const int max_single_send_sz = (1 << 8) / sizeof(Edge<edge_data_t>) * sizeof(Edge<edge_data_t>);
#else
                const int max_single_send_sz = (1 << 28) / sizeof(Edge<edge_data_t>) * sizeof(Edge<edge_data_t>);
#endif
                void* send_data = tmp_es + e_begin[dst];
                while (true)
                {
                    MPI_Send(&tot_send_sz, 1, get_mpi_data_type<size_t>(), dst, Tag_ShuffleGraph, MPI_COMM_WORLD);
                    if (tot_send_sz == 0)
                    {
                        break;
                    }
                    int send_sz = std::min((size_t)max_single_send_sz, tot_send_sz);
                    tot_send_sz -= send_sz;
                    MPI_Send(send_data, send_sz, get_mpi_data_type<char>(), dst, Tag_ShuffleGraph, MPI_COMM_WORLD);
                    send_data = (char*)send_data + send_sz;
                }
                usleep(10000);
            }
        });
        std::thread recv_thread([&](){
            for (partition_id_t step = 0; step < partition_num; step++)
            {
                partition_id_t src = (local_partition_id + partition_num - step) % partition_num;
                while (true)
                {
                    size_t remained_sz;
                    MPI_Recv(&remained_sz, 1, get_mpi_data_type<size_t>(), src, Tag_ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    if (remained_sz == 0)
                    {
                        break;
                    }

                    MPI_Status recv_status;
                    MPI_Probe(src, Tag_ShuffleGraph, MPI_COMM_WORLD, &recv_status);
                    int sz;
                    MPI_Get_count(&recv_status, get_mpi_data_type<char>(), &sz);

                    MPI_Recv(local_edges + local_edge_p, sz, get_mpi_data_type<char>(), src, Tag_ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    local_edge_p += sz / sizeof(Edge<edge_data_t>);
                }
                usleep(100000);
            }
        });
        send_thread.join();
        recv_thread.join();
        delete []tmp_es;
	//delete []misc_edges;
	if(local_e_num != local_edge_p)
	{
		printf("local_e_num != local_edge_p(%lu %lu) \n", local_e_num, local_edge_p);
	}
        assert(local_e_num == local_edge_p);
    }

    void Send_graph_data(vertex_id_t *local_cut_vertex, vertex_id_t count, EdgeContainer<edge_data_t> *csr_build, EdgeContainer<edge_data_t> *input_csr, edge_id_t *total_csr_edge)
    {
        
        vertex_id_t **local_cut_partition;
        vertex_id_t **vertex_buffer = new vertex_id_t*[this->partition_num];;
        local_cut_partition = new vertex_id_t*[partition_num];
        //**************Send needed graph data*********************

        vertex_id_t *local_cut_partition_number = new vertex_id_t[partition_num]();
        vertex_id_t **local_cut_partition_number_thread = new vertex_id_t*[partition_num];

        vertex_id_t partition_id;
        for(partition_id_t p_i = 0; p_i < partition_num; p_i++)
        {
            local_cut_partition_number_thread[p_i] = new vertex_id_t[worker_num]();
        }


        #pragma omp parallel for
        for(vertex_id_t v_i = 0; v_i < count; v_i++)
        {
            int worker_id = omp_get_thread_num();
            partition_id = vertex_partition_id[local_cut_vertex[v_i]];
            local_cut_partition_number_thread[partition_id][worker_id] += 1;

        }

        for(partition_id_t p_i = 0; p_i < partition_num ; p_i++)
        {
            for(partition_id_t p_j = 0; p_j < worker_num; p_j++)
            {
                local_cut_partition_number[p_i] += local_cut_partition_number_thread[p_i][p_j];
            }

            local_cut_partition[p_i] = new vertex_id_t[local_cut_partition_number[p_i]];
        }

        vertex_id_t *local_cut_partition_number_pos = new vertex_id_t[partition_num]();
        //printf("%d Now Send graph data3\n", this->local_partition_id);

        for(vertex_id_t begin_i = 0; begin_i < count; begin_i++)
        {
            partition_id = vertex_partition_id[local_cut_vertex[begin_i]];

            assert(local_cut_partition_number_pos[partition_id] < local_cut_partition_number[partition_id]);
            
            local_cut_partition[partition_id][local_cut_partition_number_pos[partition_id]] = local_cut_vertex[begin_i];
            local_cut_partition_number_pos[partition_id] += 1;
        }
        vertex_id_t *vertex_buffer_count = new vertex_id_t[partition_num]();

        std::thread send_thread([&](){
            for(partition_id_t step=0; step < partition_num; step++)
            {
                partition_id_t dst=(local_partition_id + step) % partition_num;
                size_t total_send_size=local_cut_partition_number[dst] *sizeof(vertex_id_t);
                MPI_Send((vertex_id_t*)local_cut_partition[dst], local_cut_partition_number[dst], get_mpi_data_type<vertex_id_t>(), dst, Tag_graph_vertex, MPI_COMM_WORLD);
                usleep(10000);
            }
        });
        std::thread recv_thread([&](){
            for(partition_id_t step=0; step<partition_num; step++)
            {
                
                partition_id_t src=(local_partition_id - step + partition_num) % partition_num;
                MPI_Status recv_status;
                MPI_Probe(src, Tag_graph_vertex, MPI_COMM_WORLD, &recv_status);
                int sz;
                MPI_Get_count(&recv_status, get_mpi_data_type<vertex_id_t>(), &sz);
                vertex_buffer_count[src] = sz;
                vertex_buffer[src] = new vertex_id_t[sz + 1];
                MPI_Recv((vertex_id_t*)vertex_buffer[src], sz, get_mpi_data_type<vertex_id_t>(), src, Tag_graph_vertex, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                usleep(10000);
            }
        });
        send_thread.join();
        recv_thread.join();
        edge_id_t *maxedge = new edge_id_t[partition_num]();
        edge_id_t maxedges = 0;
        for(partition_id_t begin_i=0; begin_i < partition_num; begin_i++)
        {
            for(vertex_id_t begin_j=0; begin_j < vertex_buffer_count[begin_i]; begin_j++)
            {
                vertex_id_t tmpnode = vertex_buffer[begin_i][begin_j];
                maxedge[begin_i] += vertex_out_degree[tmpnode];
            }
            if(maxedges < maxedge[begin_i])
            {
                maxedges = maxedge[begin_i];
            }
        }
        AdjUnit<edge_data_t> **graph_data_send;
        graph_data_send = new AdjUnit<edge_data_t>*[partition_num];
        for(partition_id_t begin_i = 0; begin_i < partition_num; begin_i++)
        {
            graph_data_send[begin_i] = new AdjUnit<edge_data_t>[maxedge[begin_i]];
        }

        #pragma omp parallel for
        for(partition_id_t begin_i = 0; begin_i < partition_num; begin_i++)
        {
            vertex_id_t e_m = 0;
            for(vertex_id_t begin_j = 0; begin_j < vertex_buffer_count[begin_i]; begin_j++)
            {
                vertex_id_t tmpnode11 = vertex_buffer[begin_i][begin_j];
                AdjList<edge_data_t> *adj = input_csr->adj_lists + tmpnode11;
                AdjUnit<edge_data_t> *begin;
                begin = adj->begin;
                AdjUnit<edge_data_t> *end;
                end = adj->end;
                while(begin != end)
                {
                    graph_data_send[begin_i][e_m] = *begin;
                    begin++;
                    e_m++;
                }
            }
        }
        MPI_Allreduce(maxedge, total_csr_edge, partition_num, get_mpi_data_type<edge_id_t>(), MPI_SUM, MPI_COMM_WORLD);
        csr_build->adj_lists = new AdjList<edge_data_t>[v_num];
        csr_build->adj_units = new AdjUnit<edge_data_t>[total_csr_edge[local_partition_id]];
        //**********************Send Graph Data ********************
        std::thread Send_graph_thread([&](){
            for(partition_id_t step = 0; step < partition_num; step++)
            {
                partition_id_t dst = (local_partition_id + step) % partition_num;
                size_t tot_send_sz = maxedge[dst] * sizeof(AdjUnit<edge_data_t>);
                edge_id_t Send_count = 0;
#ifdef UNIT_TEST
                const int max_single_send_sz = (1 << 8) / sizeof(AdjUnit<edge_data_t>) * sizeof(AdjUnit<edge_data_t>);
#else
                const int max_single_send_sz = (1 << 28) / sizeof(AdjUnit<edge_data_t>) * sizeof(AdjUnit<edge_data_t>);
#endif
                while(true)
                {
                    MPI_Send(&tot_send_sz, 1, get_mpi_data_type<size_t>(), dst, Tag_graphdata, MPI_COMM_WORLD);
                    if(tot_send_sz == 0)
                    {
                        break;
                    }
                    int send_sz = std::min((size_t)max_single_send_sz, tot_send_sz);
                    tot_send_sz -= send_sz;
                    MPI_Send((AdjUnit<edge_data_t>*)graph_data_send[dst] + Send_count, send_sz, get_mpi_data_type<char>(), dst, Tag_graphdata, MPI_COMM_WORLD);
                    Send_count += send_sz / sizeof(AdjUnit<edge_data_t>);
                }
                usleep(1000);            
            }
        });

        std::thread Recv_graph_thread([&](){
            vertex_id_t units_number=0;
            vertex_id_t units_number_back;
            for(partition_id_t step=0; step<partition_num; step++)
            {
                Timer send_time;
                units_number_back=units_number;
                partition_id_t src=(local_partition_id - step + partition_num) % partition_num;
                while(true)
                {
                    size_t remained_sz;
                    MPI_Recv(&remained_sz, 1, get_mpi_data_type<size_t>(), src, Tag_graphdata, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    if(remained_sz==0)
                    {
                        break;
                    }
                    MPI_Status recv_adjunits_status;
                    MPI_Probe(src, Tag_graphdata, MPI_COMM_WORLD, &recv_adjunits_status);
                    int sz;
                    MPI_Get_count(&recv_adjunits_status, get_mpi_data_type<char>(), &sz);
                    MPI_Recv(((AdjUnit<edge_data_t>*)csr_build->adj_units) + units_number, sz, get_mpi_data_type<char>() , src, Tag_graphdata, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    units_number += sz / sizeof(AdjUnit<edge_data_t>);
                }
                //printf("%d send is %lf\n", src, send_time.duration());
                usleep(1000) ;
                vertex_id_t sub_number = 0;
                for(vertex_id_t begin_i=0; begin_i<local_cut_partition_number[src]; begin_i++)
                {
                    vertex_id_t tmpnode = local_cut_partition[src][begin_i];
                    AdjList<edge_data_t> *tmplist = csr_build->adj_lists + tmpnode;
                    tmplist->begin = csr_build->adj_units + units_number_back + sub_number;
                    sub_number += vertex_out_degree[tmpnode];
                    tmplist->end=csr_build->adj_units + units_number_back + sub_number;
                }
            }
        });
        Send_graph_thread.join();
        Recv_graph_thread.join();
    }

    partition_id_t not_in_vector(partition_id_t *input_partition, partition_id_t input_count, partition_id_t value)
    {
        for(partition_id_t p_i = 0; p_i < input_count; p_i++)
        {
            if(input_partition[p_i] == value)
            {
                return 0;
            }
        }
        return 1;
    }


    partition_id_t find_id(vertex_id_t *input_vector, vertex_id_t input_value, partition_id_t *input_partition, partition_id_t input_count)
    {
        for(partition_id_t p_i = 0; p_i < partition_num; p_i++)
        {
            if(input_value == input_vector[p_i] && not_in_vector(input_partition, input_count, p_i) == 1)
            {
                //printf("output is %d\n", p_i);
                return p_i;
            }
        }
        return partition_num;
    }

    void Cached_subgraph_sorted(vertex_id_t *partition_vertex_number, partition_id_t *sorted_partition)
    {
        vertex_id_t *partition_vertex_number_sort = new vertex_id_t[partition_num]();
        for(partition_id_t p_i = 0; p_i < partition_num; p_i++)
        {
            sorted_partition[p_i] = find_id(partition_vertex_number, partition_vertex_number_sort[p_i], sorted_partition, p_i);
        }
        for(partition_id_t p_i = 0; p_i < partition_num; p_i++)
        {
            sorted_partition[p_i] = sorted_partition[p_i] - 1;
        }
    }

    template<typename T>
    T balanced_partition_id(T value, T* input_vector, T vector_num)
    {
        for(T i = 0; i < vector_num; i++)
        {
            if(value == input_vector[i])
            {
                return i / 2;
            }
        }
        return vector_num;
    }



    void chunk_partition(EdgeContainer<edge_data_t> *csr_tmp1, int *my_vertex_partition_id, int under_partition_num, vertex_id_t *partition_vertex_number, double *partition_edge_number, partition_id_t partition_num_plan, partition_id_t iter, int partition_type)
    {
       if(my_vertex_partition_id == nullptr)
        {
            my_vertex_partition_id = new int[this->v_num]();
        }
        double average_goal = double(v_num) / partition_num_plan / pow(2, iter);
	    double average_v_num = double(v_num) / partition_num_plan / pow(2, iter) ;
        double average_degree = double(e_num)/double(v_num);
        int current_partition = 1;
        double *score_sum = new double[under_partition_num + 1]();
        double score_tmp;
        for(vertex_id_t v_i = 0; v_i < v_num; v_i++)
        {
            if(vertex_partition_id[v_i] == partition_num_plan)
            {
                switch(partition_type % 3)
                {
                    case 0:
                        score_tmp = double(partition_vertex_number[current_partition]);
                        break;
                    case 1:
                        score_tmp = partition_edge_number[current_partition];
                        break;
                    case 2:
                        score_tmp = score_sum[current_partition] / 2;
                        break;
                }

                if(score_tmp < average_goal)
	            {
	                my_vertex_partition_id[v_i] = current_partition;
		            partition_vertex_number[current_partition]++;
		            partition_edge_number[current_partition] += double(this->vertex_out_degree[v_i]) / average_degree;
		            score_sum[current_partition] = double(partition_vertex_number[current_partition]) + partition_edge_number[current_partition];
	            }
    	        else
	            {
		            current_partition++;
        		    my_vertex_partition_id[v_i] = current_partition;
		            partition_vertex_number[current_partition]++;
        		    partition_edge_number[current_partition] += double(this->vertex_out_degree[v_i]) / average_degree;
		            score_sum[current_partition] = double(partition_vertex_number[current_partition]) + partition_edge_number[current_partition];
                }
	        }
	    }
    }

    void fennel_partition(EdgeContainer<edge_data_t> *csr_tmp1, int *my_vertex_partition_id, int under_partition_num, vertex_id_t *partition_vertex_number, double *partition_edge_number, partition_id_t partition_num_plan, partition_id_t iter, int partition_type)
    {
        if(my_vertex_partition_id == nullptr)
        {
            my_vertex_partition_id = new int[this->v_num]();
        }
        double average_degree = double(e_num)/double(v_num);
        double average_balance = double(v_num) / partition_num_plan / pow(2, iter) * 2;

        double alpha = 1.5;
        double giamma = 1.5;
        double *common_vertex = new double[under_partition_num + 1]();
        double *score = new double[under_partition_num + 1]();
        double *score_sum = new double[under_partition_num + 1]();
        double change_score = v_num / 4;
        double maxscore;

        vertex_id_t **partition_vertex_number_thread = new vertex_id_t*[worker_num];
        double **partition_edge_number_thread = new double*[worker_num];
        for(int w_i = 0; w_i < worker_num; w_i++)
        {
            partition_vertex_number_thread[w_i] = new vertex_id_t[under_partition_num + 1]();
            partition_edge_number_thread[w_i] = new double[under_partition_num + 1]();
        }

        //std::mutex fennel_metadata;

        partition_type = partition_type % 3;

        for(partition_id_t p_i = 0; p_i < partition_num; p_i++)
        {
            if(p_i == local_partition_id)
            {
                vertex_id_t local_vertex_partition_begin = vertex_partition_begin[local_partition_id];
                vertex_id_t local_vertex_partition_end = vertex_partition_end[local_partition_id];

                for(vertex_id_t v_i = local_vertex_partition_begin; v_i < local_vertex_partition_end; v_i++)
                {
                    if(vertex_partition_id[v_i] == partition_num_plan)
                    {
                        for(int i = 0; i <= under_partition_num; i++)
                        {   
                            common_vertex[i] = 0;
                            score[i] = 0.0;
                        }
                        AdjList<edge_data_t> *adjlist = csr_tmp1->adj_lists + v_i;
                        AdjUnit<edge_data_t> *adj_begin = adjlist->begin;
                        AdjUnit<edge_data_t> *adj_end = adjlist->end;
                        maxscore = -100000000000.0;
                        partition_id_t maxid = 0;
                        while(adj_begin != adj_end)
                        {
                            vertex_id_t tmp_vertex = adj_begin->neighbour;
                            if(my_vertex_partition_id[tmp_vertex] > 0)
                            {
                                assert(my_vertex_partition_id[tmp_vertex] <= under_partition_num);
                                common_vertex[my_vertex_partition_id[tmp_vertex]] += 1;
                            }    
                            adj_begin++;
                        }
                        for(partition_id_t p_j = 1; p_j <= under_partition_num; p_j++)
                        {

                            double frac_balance = average_balance - score_sum[p_j];//control 
                            if(frac_balance < 0 && (partition_type == 2))
                            {
                                score[p_j] = -100000000001.0;
                                //score[p_j] = common_vertex[p_j] * ((average_balance - (double(partition_vertex_number[p_j]) + partition_edge_number[p_j])) / average_balance);
                            }
                            else
                            {
                                switch(partition_type)
                                {
                                    case 0:
                                        score[p_j] = common_vertex[p_j] - alpha * giamma * pow(partition_vertex_number[p_j], giamma - 1);// +  0.5 * pow(partition_edge_number[p_j], giamma - 1));
                                        break;
                                    case 1:
                                        score[p_j] = common_vertex[p_j] - alpha * giamma * pow(partition_edge_number[p_j], giamma - 1);// +  0.5 * pow(partition_edge_number[p_j], giamma - 1));
                                        break;
                                    case 2:
                                        score[p_j] = common_vertex[p_j] - alpha * giamma * pow(0.5 * score_sum[p_j], giamma - 1);// +  0.5 * pow(partition_edge_number[p_j], giamma - 1));
                                        break;
                                }
                            }
                            if(maxscore < score[p_j])
                            {
                                maxscore = score[p_j];
                                maxid = p_j;
                            }
                        }
                        if(maxid == 0)
                        {
                            double tmp_max_score = v_num * 2;
                            partition_id_t tmp_max_id = 0;
                            for(partition_id_t p_j = 1; p_j <= under_partition_num; p_j++)
                            {
                                if(tmp_max_score >= (double(partition_vertex_number[p_j]) + partition_edge_number[p_j]))
                                {
                                    tmp_max_score = double(partition_vertex_number[p_j]) + partition_edge_number[p_j];
                                    tmp_max_id = p_j;
                                }
                            }
                            maxid = tmp_max_id;
                        }
                        my_vertex_partition_id[v_i] = maxid;
                        assert(maxid <= under_partition_num);

                        partition_vertex_number[maxid]++;
                        partition_edge_number[maxid] += double(vertex_out_degree[v_i])/average_degree;
                        
                        score_sum[maxid] = partition_vertex_number[maxid] + partition_edge_number[maxid];
                    }
                }

                if(p_i < partition_num - 1)
                {  
                    partition_id_t tmp_dst = local_partition_id + 1;
                    MPI_Send(my_vertex_partition_id, v_num, get_mpi_data_type<int>(), tmp_dst, Tag_Fennel, MPI_COMM_WORLD);
                }
            }

            if((p_i + 1) % partition_num == local_partition_id && p_i < (partition_num - 1))
            {
                //接收这个数组
                MPI_Status recv_status;
                partition_id_t src = p_i;
                assert(p_i == ((local_partition_id - 1 + partition_num) % partition_num));
                MPI_Probe(src, Tag_Fennel, MPI_COMM_WORLD, &recv_status);
                int sz;
                MPI_Get_count(&recv_status, get_mpi_data_type<int>(), &sz);
                MPI_Recv(my_vertex_partition_id, sz, get_mpi_data_type<int>(), src, Tag_Fennel, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                #pragma omp parallel for
                for(vertex_id_t v_i = 0; v_i < vertex_partition_begin[local_partition_id]; v_i++)
                {
                    int worker_id = omp_get_thread_num();
                    partition_id_t tmp_partition = my_vertex_partition_id[v_i];
                    if(tmp_partition > 0)
                    {
                        partition_vertex_number_thread[worker_id][tmp_partition] += 1;
                        partition_edge_number_thread[worker_id][tmp_partition] += double(vertex_out_degree[v_i]) / average_degree;
                    }
                }
                for(partition_id_t p_i = 1; p_i <= under_partition_num; p_i++)
                {
                    partition_edge_number[p_i] = 0.0;
                    partition_vertex_number[p_i] = 0;
                    for(int w_i = 0; w_i < worker_num; w_i++)
                    {
                        partition_vertex_number[p_i] += partition_vertex_number_thread[w_i][p_i];
                        partition_edge_number[p_i] += partition_edge_number_thread[w_i][p_i];
                    }
                }
            }
            if(p_i == partition_num - 1)
            {
                //广播这个数组
                MPI_Bcast(my_vertex_partition_id, v_num, get_mpi_data_type<int>(), p_i, MPI_COMM_WORLD);
                MPI_Bcast(partition_vertex_number, under_partition_num + 1, get_mpi_data_type<vertex_id_t>(), p_i, MPI_COMM_WORLD);
                MPI_Bcast(partition_edge_number, under_partition_num + 1, get_mpi_data_type<double>(), p_i, MPI_COMM_WORLD);
            }
            //MPI_Barrer
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    void Generate_partition_id(Edge<edge_data_t> *read_edges, edge_id_t read_e_num, int partition_type)
    {
        switch(partition_type)
        {
            case 0:
                if(this->local_partition_id == 0)
                    printf("Start to generate partition file with Chunk-V partition algorithm.\n");
                break;
            case 1:
                if(this->local_partition_id == 0)
                    printf("Start to generate partition file with Chunk-E partition algorithm.\n");
                break;
            case 2:
                if(this->local_partition_id == 0)
                    printf("Start to generate partition file with BPart-Chunk partition algorithm.\n");
                break;
            case 3:
                if(this->local_partition_id == 0)
                    printf("Start to generate partition file with Fennel algorithm.\n");
                break;
            case 5:
                if(this->local_partition_id == 0)
                    printf("Start to generate partition file with BPart-Fennel algorithm.\n");
                break;
            default:
                if(this->local_partition_id == 0)
                {
                    printf("Bad partition algorithm choice!\n");
                    printf("0: Chunk vertex balance\n");
                    printf("1: Chunk edge balance\n");
                    printf("2: BPart Chunk\n");
                    printf("3: Fennel balance\n");
                    printf("5: BPart fennel\n");
                    printf("Exit now!\n");
                }
                exit(1);
            
        }
        partition_id_t current_partition_id = 0;
        for(vertex_id_t v_i = 0; v_i < v_num; v_i++)
        {
            if(v_i < vertex_partition_end[current_partition_id])
            {
                vertex_partition_id[v_i] = current_partition_id;
                //printf("%u %d %u\n", v_i, vertex_partition_id[v_i], vertex_out_degree[v_i]);
            }
            else
            {
                current_partition_id += 1;
                vertex_partition_id[v_i] = current_partition_id;
                    //printf("%u %d %u\n", v_i, vertex_partition_id[v_i], vertex_out_degree[v_i]);
            }
        }

        edge_id_t tmp_local_e_num = 0;
        edge_id_t *tmp_local_e_num_thread = new edge_id_t[worker_num]();
        #pragma omp parallel for
        for (vertex_id_t v_i = 0; v_i < v_num; v_i++)
        {
            int worker_id = omp_get_thread_num();
            if(vertex_partition_id[v_i] == local_partition_id)
            {
                tmp_local_e_num_thread[worker_id] += vertex_out_degree[v_i];
            }
        }

        for (int w_i = 0; w_i < worker_num; w_i++)
        {
            tmp_local_e_num += tmp_local_e_num_thread[w_i];
        }   

        Edge<edge_data_t> *tmp_local_edges = nullptr;
        tmp_local_edges = new Edge<edge_data_t>[tmp_local_e_num];

        shuffle_edges(read_edges, read_e_num, tmp_local_edges, tmp_local_e_num);

        EdgeContainer<edge_data_t> *csr_tmp1 = nullptr;
        csr_tmp1 = new EdgeContainer<edge_data_t>();
        build_edge_container(tmp_local_edges, tmp_local_e_num, csr_tmp1, vertex_out_degree);

        // Start to balance partition
        partition_id_t iter;

        if(partition_type % 3 == 2)
        {
            iter = 1;
        }
        else
        {
            iter = 0;
        }
        partition_id_t finished_partition_num = 0;
        partition_id_t partition_num_plan = partition_num;
        vertex_id_t aim_subgraph_vertex = v_num / partition_num_plan;
        edge_id_t aim_subgraph_edges = e_num / partition_num_plan;
        //vertex_id_t *vertex_partition_id_tmp = new partition_id_t[this->v_num]();
        //#pragma omp parallel for
        for(vertex_id_t v_i = 0; v_i < v_num; v_i++)
        {
            vertex_partition_id[v_i] = partition_num_plan;
        }

        while(finished_partition_num < partition_num_plan)
        { 
            partition_id_t pow_2_iter = pow(2, iter);
            int *tmp_partition_id_vector = new int[v_num]();
            int remain_partition_num = partition_num_plan - finished_partition_num;
            int under_partition_num = pow(2, iter) * (remain_partition_num);
            vertex_id_t *partition_vertex_number = new vertex_id_t[under_partition_num + 1]();
            double *partition_edge_number = new double[under_partition_num + 1]();
            double average_degree = double(e_num)/double(v_num);

            switch(partition_type / 3)
            {
                case 0:
                    chunk_partition(csr_tmp1, tmp_partition_id_vector, under_partition_num, partition_vertex_number, partition_edge_number, partition_num_plan, iter, partition_type);
                    break;
                case 1:
                    fennel_partition(csr_tmp1, tmp_partition_id_vector, under_partition_num, partition_vertex_number, partition_edge_number, partition_num_plan, iter, partition_type);
                    break;
                default:
                    printf("wrong type of partition algorithm\n");
                    printf("0: Chunk vertex balance\n");
                    printf("1: Chunk edge balance\n");
                    printf("2: BPart Chunk\n");
                    printf("3: Fennel balance\n");
                    printf("5: BPart fennel\n");
            }

            int combined_subgraph_num = pow(2, iter);
            vertex_id_t *partition_vertex_number_tmp = new vertex_id_t[under_partition_num]();

            for(int p_i = 0; p_i < under_partition_num; p_i++)
            {
                partition_vertex_number_tmp[p_i] = partition_vertex_number[p_i + 1];
            }

            int *combined_subgraph = new int[under_partition_num];
            for(int p_i = 0; p_i < under_partition_num; p_i++)
            {
                combined_subgraph[p_i] = p_i;
            }
            
            for(partition_id_t p_i = 1; p_i <= iter; p_i++)
            {
                int pow_2_p_i = pow(2, p_i);
                int pow_2_p_i_1 = pow(2, p_i - 1);
                int *combined_subgraph_tmp = new int[under_partition_num];
                
                int combined_number = under_partition_num / pow(2, p_i);//number of vectors
                vertex_id_t *combined_partition_vertex_num = new vertex_id_t[combined_number]();
                double *combined_partition_edge_num = new double[combined_number]();
                int *sorted_by_vertex_num = new int[combined_number * 2 ]();
                
                for(int p_j = 0; p_j < combined_number * 2; p_j++)
                {
                    vertex_id_t max_number = 0;
                    int max_id = combined_number * 2;
                    for(int p_z = 0; p_z < combined_number * 2; p_z++)
                    {
                        if(max_number <= partition_vertex_number_tmp[p_z] && partition_vertex_number_tmp[p_z] < v_num + 1)
                        {
                            max_number = partition_vertex_number_tmp[p_z];
                            max_id = p_z;
                        }
                    }
                    sorted_by_vertex_num[p_j] = max_id;
                    //printf("%d ", sorted_by_vertex_num[p_j]);
                    partition_vertex_number_tmp[max_id] = v_num + 1;
                }


                for(int p_j = 0; p_j < combined_number; p_j++)
                {
                    int tmp_count = 0;
                    int tmp_id = sorted_by_vertex_num[p_j];
                    for(int p_z = 0; p_z < pow(2, p_i - 1); p_z++)
                    {
                        combined_subgraph_tmp[p_j * pow_2_p_i + tmp_count] = combined_subgraph[tmp_id * pow_2_p_i_1 + p_z];
                        tmp_count++;
                    }
                    tmp_id = sorted_by_vertex_num[combined_number * 2 - 1 - p_j];
                    for(int p_z = 0; p_z < pow(2, p_i - 1); p_z++)
                    {
                        combined_subgraph_tmp[p_j * pow_2_p_i + tmp_count] = combined_subgraph[tmp_id * pow_2_p_i_1 + p_z];
                        tmp_count++;
                    }
                }

                for(int p_j = 0; p_j < under_partition_num; p_j++)
                {
                    combined_subgraph[p_j] = combined_subgraph_tmp[p_j];
                }

                for(int p_j = 0; p_j < combined_number; p_j++)
                {
                    int sort_id_1 = sorted_by_vertex_num[p_j] + 1;
                    int sort_id_2 = sorted_by_vertex_num[combined_number * 2 - 1 - p_j] + 1;
                    combined_partition_vertex_num[p_j] = partition_vertex_number[sort_id_1] + partition_vertex_number[sort_id_2];
                    combined_partition_edge_num[p_j] = partition_edge_number[sort_id_1] + partition_edge_number[sort_id_2];
                }

                for(int p_j = 0; p_j < combined_number; p_j++)
                {
                    partition_vertex_number_tmp[p_j] = combined_partition_vertex_num[p_j];
                    partition_vertex_number[p_j + 1] = combined_partition_vertex_num[p_j];
                    partition_edge_number[p_j + 1] = combined_partition_edge_num[p_j];
                }
            }

            for(int p_i = 0; p_i < remain_partition_num; p_i++)
            {
                double min_num = double(partition_vertex_number[p_i + 1]) - double(aim_subgraph_vertex);
                double min_edge = partition_edge_number[p_i + 1] * average_degree - double(aim_subgraph_edges);
                
                if(((std::abs(min_num) < ((double)aim_subgraph_vertex * 0.05 + 2)) && (std::abs(min_edge) < (double)aim_subgraph_edges * 0.05 + 2)) || iter == 5 || finished_partition_num >= (partition_num_plan - 2) || partition_type % 3 != 2)
                {
                    #pragma omp parallel for
                    for(vertex_id_t v_i = 0; v_i < v_num; v_i++)
                    {
                        bool is_true = false;
                        if(tmp_partition_id_vector[v_i] > 0)
                        {
                            int tmp_id = tmp_partition_id_vector[v_i] - 1;

                            for(partition_id_t p_j = 0; p_j < pow(2, iter); p_j++)
                            {
                                int combined_subgraph_id = combined_subgraph[p_i * pow_2_iter + p_j];
                                if(tmp_id == combined_subgraph_id)
                                {
                                    is_true = true;
                                    break;
                                }
                            }
                        }
                        if(is_true == true)
                        {
                            assert(vertex_partition_id[v_i] == partition_num_plan);
                            vertex_partition_id[v_i] = finished_partition_num;
                        }
                    }
                    finished_partition_num++;
                }
            }
            delete[] tmp_partition_id_vector;

            if(local_partition_id == 0)
                printf("Iter %d, finished subgraph %d, \n", iter, finished_partition_num);
            iter++;
        }

        delete []csr_tmp1->adj_lists;
        delete []csr_tmp1->adj_units;
    }

  void Generate_cache_list(EdgeContainer<edge_data_t> *csr_tmp, vertex_id_t *cache_id_vector)
    {
        vertex_id_t Work_load_per_time = vertex_id_t(v_num / partition_num) + 1;
        vertex_id_t Neighb_dis_num = Work_load_per_time * partition_num;

        vertex_id_t init_vertex = 0;
        std::vector<std::vector<vertex_id_t> > cache_classification;


        for(int i = 0; i < 12; i++)
        {
            cache_classification.push_back(std::vector<vertex_id_t>() );
        }

        vertex_id_t *local_out_degree = new vertex_id_t[v_num]();

        for(vertex_id_t v_i = 0; v_i < v_num; v_i++)
        {
            if(vertex_partition_id[v_i] == local_partition_id)
            {
                AdjList<edge_data_t> *adjlist = csr_tmp->adj_lists + v_i;
                AdjUnit<edge_data_t> *adj_begin = adjlist->begin;
                AdjUnit<edge_data_t> *adj_end = adjlist->end;
                while(adj_begin < adj_end)
                {
                    local_out_degree[adj_begin->neighbour] += 1;
                    adj_begin++;
                }
            }
        }

        for(partition_id_t p_i = 0; p_i < partition_num; p_i++)
        {
            vertex_id_t *Neighb_dis = new vertex_id_t[Neighb_dis_num]();
            vertex_id_t *Neighb_dis_final = new vertex_id_t[Neighb_dis_num]();
            vertex_id_t work_begin = Work_load_per_time * p_i;
            vertex_id_t work_end = std::min(Work_load_per_time * (p_i + 1), v_num);

            for(vertex_id_t v_i = work_begin; v_i < work_end; v_i++)
            {
                if(vertex_partition_id[v_i] == local_partition_id)
                {
                    AdjList<edge_data_t> *adjlist = csr_tmp->adj_lists + v_i;
                    AdjUnit<edge_data_t> *adj_begin = adjlist->begin;
                    AdjUnit<edge_data_t> *adj_end = adjlist->end;
                    while(adj_begin < adj_end)
                    {
                        Neighb_dis[(v_i - work_begin) * partition_num + vertex_partition_id[adj_begin->neighbour]] += 1;
                        adj_begin++;
                    }
                }
            }

            

            MPI_Allreduce(Neighb_dis, Neighb_dis_final, Neighb_dis_num, get_mpi_data_type<partition_id_t>(), MPI_SUM, MPI_COMM_WORLD);

            double second_score = 1.0 / double(partition_num);

            for(vertex_id_t v_i = work_begin; v_i < work_end; v_i++)
            {
                if(vertex_partition_id[v_i] != local_partition_id)
                {
                    vertex_id_t vertex_pos = v_i - work_begin;
                    partition_id_t current_partition_id = vertex_partition_id[v_i];
                    vertex_id_t in_local_num = Neighb_dis_final[vertex_pos * partition_num + local_partition_id];
                    vertex_id_t not_local_num = vertex_out_degree[v_i] - in_local_num - Neighb_dis_final[vertex_pos * partition_num + current_partition_id];
                    
                    double cache_score = 0;
                    if(vertex_out_degree[v_i] > 0)
                    {
                        cache_score = double(local_out_degree[v_i]) * (double(in_local_num * 2 + not_local_num) / double(vertex_out_degree[v_i]))/ double(vertex_out_degree[v_i]);
                    }
                    if(cache_score >= 1)
                    {
                        cache_classification[11].push_back(v_i);
                    }
                    else
                    {
                        if(cache_score >= second_score)
                        {
                            cache_classification[10].push_back(v_i);
                        }
                        else
                        {
                            int class_id = int(cache_score * double(partition_num) * 10.0);

                            cache_classification[class_id].push_back(v_i);
                        }
                    }
                }
            }

        }

        vertex_id_t tot_cache_num = 0;
        for(int i = 0; i < 12; i++)
        {
            for(vertex_id_t v_j = 0; v_j < cache_classification[11 - i].size(); v_j++)
            {
                cache_id_vector[tot_cache_num + v_j] = cache_classification[11-i][v_j];
            }
            tot_cache_num += cache_classification[11 - i].size();
        }
    }

    void load_graph(vertex_id_t v_num_param, const char* graph_path, int partition_type, bool load_as_undirected = false, double cache_rate_param = 0.0, GraphFormat graph_format = GF_Binary)
    {
        Timer timer;

        this->v_num = v_num_param;
        this->partition_num = get_mpi_size();
        this->local_partition_id = get_mpi_rank();
        this->local_e_num = 0;
        this->cache_rate = cache_rate_param;

        Edge<edge_data_t> *read_edges;
        edge_id_t read_e_num;
        if (graph_format == GF_Binary)
        {
            Timer time_read_graph;
            read_graph(graph_path, local_partition_id, partition_num, read_edges, read_e_num);
        } else if (graph_format == GF_Edgelist)
        {
            read_edgelist(graph_path, local_partition_id, partition_num, read_edges, read_e_num);
        } else
        {
            fprintf(stderr, "Unsupported graph formant");
            exit(1);
        }
        if (load_as_undirected)
        {
            Edge<edge_data_t> *undirected_edges = new Edge<edge_data_t>[read_e_num * 2];
#pragma omp parallel for
            for (edge_id_t e_i = 0; e_i < read_e_num; e_i++)
            {
                undirected_edges[e_i * 2] = read_edges[e_i];
                std::swap(read_edges[e_i].src, read_edges[e_i].dst);
                undirected_edges[e_i * 2 + 1] = read_edges[e_i];
            }
            delete []read_edges;
            read_edges = undirected_edges;
            read_e_num *= 2;
        }

        this->vertex_out_degree = alloc_vertex_array<vertex_id_t>();
        std::vector<vertex_id_t> local_vertex_degree(v_num, 0);
        for (edge_id_t e_i = 0; e_i < read_e_num; e_i++) 
        {
            local_vertex_degree[read_edges[e_i].src]++;
        }
        MPI_Allreduce(local_vertex_degree.data(),  vertex_out_degree, v_num, get_mpi_data_type<vertex_id_t>(), MPI_SUM, MPI_COMM_WORLD);

        std::fill(local_vertex_degree.begin(), local_vertex_degree.end(), 0);
        for (edge_id_t e_i = 0; e_i < read_e_num; e_i++) 
        {
            local_vertex_degree[read_edges[e_i].dst] ++;
        }

        std::vector<vertex_id_t>().swap(local_vertex_degree);

        edge_id_t *edge_number = new edge_id_t[partition_num]();
        vertex_id_t average_vertex = (v_num / partition_num) + 1;
        partition_id_t tmp_partition = 0;
        for(vertex_id_t p_i = 0; p_i < v_num; p_i++)
        {
            if(p_i < average_vertex)
            {
                edge_number[tmp_partition] += vertex_out_degree[p_i];
            }
            else
            {
                tmp_partition = tmp_partition + 1;
                average_vertex = ((v_num / partition_num) + 1) * (tmp_partition + 1);
            }
        }

        vertex_partition_begin = new vertex_id_t[partition_num];
        vertex_partition_end = new vertex_id_t[partition_num];
        edge_id_t total_workload = 0;
        e_num = 0;
        for (vertex_id_t v_i = 0; v_i < v_num; v_i++)
        {
            total_workload +=  vertex_out_degree[v_i];
            e_num += vertex_out_degree[v_i];
        }
        
        edge_id_t workload_per_node = (total_workload + partition_num - 1) / partition_num;
        for (partition_id_t p_i = 0; p_i < partition_num; p_i++)
        {
            if (p_i == 0)
            {
                vertex_partition_begin[p_i] = 0;
            } 
            else
            {
                vertex_partition_begin[p_i] = vertex_partition_end[p_i - 1];
            }
            vertex_partition_end[p_i] = vertex_partition_begin[p_i];
            edge_id_t workload = 0;
            for (vertex_id_t v_i = vertex_partition_begin[p_i]; v_i < v_num && workload < workload_per_node; v_i++)
            {
                workload += vertex_out_degree[v_i];
                vertex_partition_end[p_i]++;
            }
#ifdef PERF_PROF
            if (local_partition_id == 0)
            {
                printf("partition %d: %u %u (%zu %zu)\n", p_i, vertex_partition_begin[p_i], vertex_partition_end[p_i], workload, workload_per_node);
            }
#endif
        }
        
        assert(vertex_partition_end[partition_num - 1] == v_num);

        vertex_partition_id = alloc_vertex_array<partition_id_t>();
        vertex_cached_partition_id = new bool[v_num];

        for(vertex_id_t p_i = 0; p_i < v_num; p_i++)
        {
            vertex_cached_partition_id[p_i] = false;
        }

        

        std::string read_file = graph_path;
        read_file = read_file + "_partition_type" + std::to_string(partition_type) + "_subgraph" + std::to_string(partition_num) + ".data";

        FILE *read_PF = fopen(read_file.c_str(), "r");

        partition_id_t read_pf_flag = 0;
        partition_id_t read_pf_flag_tot = 0;

        if(read_PF == NULL)
        {
            read_pf_flag = 1;
        }//Check if read file fail

        MPI_Allreduce(&read_pf_flag, &read_pf_flag_tot, 1, get_mpi_data_type<partition_id_t>(), MPI_SUM, MPI_COMM_WORLD);

        if(read_pf_flag_tot != 0)// If exist some machine fail to load partition file, then generating the file again.
        {
            Timer time_gen_id;
            Generate_partition_id(read_edges, read_e_num, partition_type);
            write_graph(read_file.c_str(), vertex_partition_id, v_num);
        }
        else//Successfully read partition file!
        {
            if(this->local_partition_id == 0)
            {
                switch(partition_type)
                {
                    case 0:
                        if(this->local_partition_id == 0)
                            printf("Start to generate partition file with Chunk-V partition algorithm.\n");
                        break;
                    case 1:
                        if(this->local_partition_id == 0)
                            printf("Start to generate partition file with Chunk-E partition algorithm.\n");
                        break;
                    case 2:
                        if(this->local_partition_id == 0)
                            printf("Start to generate partition file with BPart-Chunk partition algorithm.\n");
                        break;
                    case 3:
                        if(this->local_partition_id == 0)
                            printf("Start to generate partition file with Fennel algorithm.\n");
                        break;
                    case 5:
                        if(this->local_partition_id == 0)
                            printf("Start to generate partition file with BPart-Fennel algorithm.\n");
                        break;
                    default:
                        if(this->local_partition_id == 0)
                        {
                            printf("Bad partition algorithm choice!\n");
                            printf("0: Chunk vertex balance\n");
                            printf("1: Chunk edge balance\n");
                            printf("2: BPart Chunk\n");
                            printf("3: Fennel balance\n");
                            printf("5: BPart fennel\n");
                            printf("Exit now!\n");
                        }
                        exit(1);
                }
                printf("Succesfully read partition file\n");

            }
            auto ret = fread(vertex_partition_id, sizeof(uint8_t), v_num, read_PF);
		    assert(ret == v_num);
            fclose(read_PF);
        }

        local_e_num = 0;
        vertex_id_t local_v_num = 0;
        edge_id_t *local_e_num_thread = new edge_id_t[worker_num]();
        vertex_id_t *local_v_num_thread = new vertex_id_t[worker_num]();

#pragma omp parallel for
        for (vertex_id_t v_i = 0; v_i < v_num; v_i++)
        {
            int worker_id = omp_get_thread_num();
            if(vertex_partition_id[v_i] == local_partition_id)
            {
                local_e_num_thread[worker_id] += vertex_out_degree[v_i];
                local_v_num_thread[worker_id] += 1;
            }
        }

        for (int w_i = 0; w_i < worker_num; w_i++)
        {
            local_e_num += local_e_num_thread[w_i];
            local_v_num += local_v_num_thread[w_i];
        }

        

        delete []local_e_num_thread;
        delete []local_v_num_thread;

        Edge<edge_data_t> *local_edges = new Edge<edge_data_t>[local_e_num];
            
        printf("Subgraph %d, local vertex number %u, local edge number %lu\n",local_partition_id, local_v_num ,local_e_num);

        Timer shuffle_edge_time;
        shuffle_edges(read_edges, read_e_num, local_edges, local_e_num);
        delete []read_edges;

        EdgeContainer<edge_data_t> *csr_tmp = nullptr;
        csr_tmp = new EdgeContainer<edge_data_t>();
        build_edge_container(local_edges, local_e_num, csr_tmp, vertex_out_degree);

        std::string read_cache_file = graph_path;
        read_cache_file = read_file + "_cache_" + std::to_string(local_partition_id) + ".data";

        FILE *read_CF = fopen(read_cache_file.c_str(),"r");

        vertex_id_t max_cache_num = v_num - local_v_num;

        vertex_id_t *cache_id_vector = new vertex_id_t[max_cache_num]();

        partition_id_t read_cache_flag = 0;
        partition_id_t tot_read_cache_flag = 0;
        if(read_CF == NULL)
        {
            read_cache_flag = 1;
        }

        MPI_Allreduce(&read_cache_flag, &tot_read_cache_flag, 1, get_mpi_data_type<partition_id_t>(), MPI_SUM, MPI_COMM_WORLD);


        if(partition_type % 3 == 2)
        {
            if(tot_read_cache_flag != 0)
            {//failed to read cache file, generate now!
                Timer time_gen_cache_file;
                if(this->local_partition_id == 0)
                    printf("generate cache file\n");
                Generate_cache_list(csr_tmp, cache_id_vector);
                write_graph(read_cache_file.c_str(), cache_id_vector, max_cache_num);
                if(this->local_partition_id == 0)
                    printf("finish generate cache file, total gen cache file time %lf s\n", time_gen_cache_file.duration());
            }
            else
            {
                if(this->local_partition_id == 0)
                    printf("Successfully found cache file, and read now!\n");
                auto ret = fread(cache_id_vector, sizeof(vertex_id_t), max_cache_num, read_CF);

    		    assert(ret == max_cache_num);
                fclose(read_CF);
            }
        }


        
        vertex_id_t cached_vertex_buffer_num = 0;
        edge_id_t local_cached_edge = 0;
        edge_id_t max_cache_edge = e_num / partition_num * cache_rate;

        if(partition_type % 3 != 2)
        {
            max_cache_edge = 0;
        }

        this_is_local_vertex = new partition_id_t[this->v_num]();

        #pragma omp parallel for
        for(vertex_id_t v_i = 0; v_i < this->v_num; v_i++)
        {
            this_is_local_vertex[v_i] = vertex_partition_id[v_i];
        }

        vertex_id_t *cached_vertex_buffer = new vertex_id_t[v_num]();

        for(vertex_id_t v_i = 0; v_i < max_cache_num; v_i++)
        {
            if(local_cached_edge >= max_cache_edge)
            {
                break;
            }
            cached_vertex_buffer[cached_vertex_buffer_num] = cache_id_vector[v_i];
            cached_vertex_buffer_num += 1;
            local_cached_edge += vertex_out_degree[cache_id_vector[v_i]];
            this_is_local_vertex[cache_id_vector[v_i]] = this->local_partition_id;
        }

        for(vertex_id_t v_i = 0; v_i < cached_vertex_buffer_num; v_i++)
        {
            cached_vertex_buffer[cached_vertex_buffer_num] = cache_id_vector[v_i];
        }

        delete []cache_id_vector;


        csr = new EdgeContainer<edge_data_t>();

        printf("Subgraph %d, cached %d vertex and %lu edges\n", local_partition_id, cached_vertex_buffer_num, local_cached_edge);

        for(vertex_id_t v_i = 0; v_i < v_num; v_i++)
        {
            if(vertex_partition_id[v_i] == local_partition_id)
            {
                cached_vertex_buffer[cached_vertex_buffer_num] = v_i;
                cached_vertex_buffer_num++;
                //this_is_local_vertex[v_i] = 1;
            }
            
        }

        edge_id_t *total_csr_edge = new edge_id_t[partition_num]();
        Timer time_send_graph;
        Send_graph_data(cached_vertex_buffer, cached_vertex_buffer_num, csr, csr_tmp, total_csr_edge);
//        printf("at %d, finish load graph send graph %lf s\n", local_partition_id, time_send_graph.duration());
        delete []csr_tmp->adj_lists;
        delete []csr_tmp->adj_units;
        
        delete []cached_vertex_buffer;

//******************end*************************************


        recv_locks = new std::mutex[partition_num];

        send_msg_locks = new std::mutex[partition_num];

        dist_exec_ctx.progress = new size_t*[worker_num];
        for (partition_id_t t_i = 0; t_i < worker_num; t_i++)
        {
            dist_exec_ctx.progress[t_i] = new size_t[partition_num];
        }

#ifdef PERF_PROF
        printf("finish build graph, time %.3lfs\n", timer.duration());
#endif
    }



    void set_msg_buffer(size_t max_msg_num, size_t max_msg_size)
    {
        if (thread_local_msg_buffer == nullptr)
        {
            thread_local_msg_buffer = new MessageBuffer*[worker_num];
            #pragma omp parallel
            {
                int worker_id = omp_get_thread_num();
                thread_local_msg_buffer[worker_id] = new MessageBuffer();
            }
        }
        if (msg_send_buffer == nullptr)
        {
            msg_send_buffer = new MessageBuffer*[partition_num];
            for (partition_id_t p_i = 0; p_i < partition_num; p_i++)
            {
                msg_send_buffer[p_i] = new MessageBuffer();
            }
        }
        if (msg_recv_buffer == nullptr)
        {
            msg_recv_buffer = new MessageBuffer*[partition_num];
            for (partition_id_t p_i = 0; p_i < partition_num; p_i++)
            {
                msg_recv_buffer[p_i] = new MessageBuffer();
            }
        }

        size_t local_buf_size = max_msg_size * THREAD_LOCAL_BUF_CAPACITY;
        #pragma omp parallel
        {
            int worker_id = omp_get_thread_num();
            if (thread_local_msg_buffer[worker_id]->sz < local_buf_size)
            {
                thread_local_msg_buffer[worker_id]->alloc(local_buf_size);
            }
        }
        size_t comm_buf_size = max_msg_size * max_msg_num;
        for (partition_id_t p_i = 0; p_i < partition_num; p_i++)
        {
            if (msg_send_buffer[p_i]->sz < comm_buf_size)
            {
                msg_send_buffer[p_i]->alloc(comm_buf_size);
            }
            if (msg_recv_buffer[p_i]->sz < comm_buf_size)
            {
                msg_recv_buffer[p_i]->alloc(comm_buf_size);
            }
        }
    }

    void free_msg_buffer()
    {
        if (thread_local_msg_buffer != nullptr)
        {
            for (partition_id_t t_i = 0; t_i < worker_num; t_i ++)
            {
                delete thread_local_msg_buffer[t_i];
            }
            delete []thread_local_msg_buffer;
        }
        if (msg_send_buffer != nullptr)
        {
            for (partition_id_t p_i = 0; p_i < partition_num; p_i++)
            {
                delete msg_send_buffer[p_i];
            }
            delete []msg_send_buffer;
        }
        if (msg_recv_buffer != nullptr)
        {
            for (partition_id_t p_i = 0; p_i < partition_num; p_i++)
            {
                delete msg_recv_buffer[p_i];
            }
            delete []msg_recv_buffer;
        }
    }

    template<typename msg_data_t>
    void emit(vertex_id_t dst_id, msg_data_t data, int worker_id)
    {
        typedef Message<msg_data_t> msg_t;
        msg_t* buf_data = (msg_t*)thread_local_msg_buffer[worker_id]->data;
        auto &count = thread_local_msg_buffer[worker_id]->count;
        buf_data[count].dst_vertex_id = dst_id;
        buf_data[count].data = data;
//        buf_data[count].last_partition_id = this->local_partition_id;
        count++;
#ifdef UNIT_TEST
        thread_local_msg_buffer[worker_id]->self_check<msg_t>();
        assert(dst_id < v_num);
#endif
        if (count == THREAD_LOCAL_BUF_CAPACITY)
        {
            flush_thread_local_msg_buffer<msg_t>(worker_id);
        }
    }


    template<typename msg_data_t>
    void emit(vertex_id_t dst_id, msg_data_t data)
    {
        emit(dst_id, data, omp_get_thread_num());
    }

    template<typename msg_t>
    void flush_thread_local_msg_buffer(partition_id_t worker_id)
    {
        auto local_buf = thread_local_msg_buffer[worker_id];
        msg_t *local_data = (msg_t*)local_buf->data;
        auto &local_msg_count = local_buf->count;
        if (local_msg_count != 0)
        {
            vertex_id_t dst_count[partition_num];
            std::fill(dst_count, dst_count + partition_num, 0);
            for (vertex_id_t m_i = 0; m_i < local_msg_count; m_i++)
            {
                dst_count[this_is_local_vertex[local_data[m_i].dst_vertex_id]] ++;
            }
            msg_t *dst_data_pos[partition_num];
            size_t end_data_pos[partition_num];
            for (partition_id_t p_i = 0; p_i < partition_num; p_i++)
            {
                vertex_id_t start_pos = __sync_fetch_and_add(&msg_send_buffer[p_i]->count, dst_count[p_i]);
#ifdef UNIT_TEST
                msg_send_buffer[p_i]->self_check<msg_t>();
#endif
                dst_data_pos[p_i] = (msg_t*)(msg_send_buffer[p_i]->data) + start_pos;
                end_data_pos[p_i] = start_pos + dst_count[p_i];
            }
            for (vertex_id_t m_i = 0; m_i < local_msg_count; m_i++)
            {
                *(dst_data_pos[this_is_local_vertex[local_data[m_i].dst_vertex_id]]++) =  local_data[m_i];
            }
            for (partition_id_t p_i = 0; p_i < partition_num; p_i++)
            {
                dist_exec_ctx.progress[worker_id][p_i] = end_data_pos[p_i];
            }
            local_msg_count = 0;
        }
    }

    void notify_progress(vertex_id_t progress_begin, vertex_id_t progress_end, vertex_id_t workload, bool phased_exec)
    {
        int phase_num = phased_exec ? DISTRIBUTEDEXECUTIONCTX_PHASENUM : 1;
        if (phase_num > 1)
        {
            vertex_id_t work_per_phase = workload / phase_num + 1;
            int phase_begin = 0;
            while (progress_begin >= work_per_phase)
            {
                phase_begin ++;
                progress_begin -= work_per_phase;
            }
            int phase_end = 0;
            while (progress_end >= work_per_phase)
            {
                phase_end ++;
                progress_end -= work_per_phase;
            }
            if (phase_end == phase_num)
            {
                phase_end --;
            }
            for (int phase_i = phase_begin; phase_i < phase_end; phase_i++)
            {
                dist_exec_ctx.phase_locks[phase_i].unlock();
                __sync_fetch_and_add(&dist_exec_ctx.unlocked_phase, 1);
            }
        }
    }

    template<typename msg_data_t>
    size_t distributed_execute(
        std::function<void(void)> msg_producer,
        std::function<void(Message<msg_data_t> *, Message<msg_data_t> *)> msg_consumer,
        Message<msg_data_t> *zero_copy_data = nullptr,
        bool phased_exec = false
    )
    {
        typedef Message<msg_data_t> msg_t;
        int phase_num = phased_exec ? DISTRIBUTEDEXECUTIONCTX_PHASENUM : 1;
        for (int phase_i = 0; phase_i < phase_num; phase_i++)
        {
            dist_exec_ctx.phase_locks[phase_i].lock();
        }
        for (partition_id_t t_i = 0; t_i < worker_num; t_i++)
        {
            for (partition_id_t p_i = 0; p_i < partition_num; p_i++)
            {
                dist_exec_ctx.progress[t_i][p_i] = 0;
            }
        }
        dist_exec_ctx.unlocked_phase = 0;
        for (partition_id_t p_i = 0; p_i < partition_num; p_i++)
        {
            recv_locks[p_i].lock();
        }
        volatile size_t zero_copy_recv_count = 0;
        std::thread recv_thread([&](){
            for (partition_id_t p_i = 0; p_i < partition_num; p_i++)
            {
                msg_recv_buffer[p_i]->count = 0;
            }
            std::vector<MPI_Request*> requests[partition_num];
            auto recv_func = [&] (partition_id_t src)
            {
                MPI_Status prob_status;
                MPI_Probe(src, Tag_Msg, MPI_COMM_WORLD, &prob_status);
                int sz;
                MPI_Get_count(&prob_status, get_mpi_data_type<char>(), &sz);
                //printf("recv %u <- %u: %zu\n", local_partition_id, src, sz / sizeof(msg_t));
                MPI_Request *recv_req = new MPI_Request();
                requests[src].push_back(recv_req);
                if (zero_copy_data == nullptr)
                {
                    MPI_Irecv(((msg_t*)msg_recv_buffer[src]->data) + msg_recv_buffer[src]->count, sz, get_mpi_data_type<char>(), src, Tag_Msg, MPI_COMM_WORLD, recv_req);
                    msg_recv_buffer[src]->count += sz / sizeof(msg_t);
                    msg_recv_buffer[src]->template self_check<msg_t>();
                } else
                {
                    MPI_Irecv(zero_copy_data + zero_copy_recv_count, sz, get_mpi_data_type<char>(), src, Tag_Msg, MPI_COMM_WORLD, recv_req);
                    zero_copy_recv_count += sz / sizeof(msg_t);
                }
            };
            for (int phase_i = 0; phase_i < phase_num; phase_i ++)
            {
                if (phase_i + 1 == phase_num)
                {
                    for (partition_id_t step = 0; step < partition_num; step++)
                    {
                        partition_id_t src = (partition_num + local_partition_id - step) % partition_num;
                        recv_func(src);
                    }
                } else
                {
                    partition_id_t src = (partition_num + local_partition_id - phase_i % partition_num) % partition_num;
                    recv_func(src);
                }
            }
            for (partition_id_t step = 0; step < partition_num; step++)
            {
                partition_id_t src = (partition_num + local_partition_id - step) % partition_num;
                for (auto req : requests[src])
                {
                    MPI_Status status;
                    MPI_Wait(req, &status);
                    delete req;
                }
                recv_locks[src].unlock();
            }
        });

        std::thread send_thread([&](){
            size_t send_progress[partition_num];
            for (partition_id_t p_i = 0; p_i < partition_num; p_i++)
            {
                send_progress[p_i] = 0;
            }
            std::vector<MPI_Request*> requests;
            auto send_func = [&] (partition_id_t dst, size_t diff)
            {
                msg_send_buffer[dst]->template self_check<msg_t>();
                MPI_Request* req = new MPI_Request();
                requests.push_back(req);
                MPI_Isend(((msg_t*)msg_send_buffer[dst]->data) + send_progress[dst], diff * sizeof(msg_t), get_mpi_data_type<char>(), dst, Tag_Msg, MPI_COMM_WORLD, req);
#ifdef PERF_PROF
                if (local_partition_id == 0)
                {
                    printf("end send %u -> %u: %zu time %lf\n", local_partition_id, dst, diff, timer.duration());
                }
#endif
                send_progress[dst] += diff;
            };
            for (int phase_i = 0; phase_i < phase_num; phase_i++)
            {
                dist_exec_ctx.phase_locks[phase_i].lock();
                if (phase_i + 1 == phase_num)
                {
                    for (partition_id_t step = 0; step < partition_num; step++)
                    {
                        partition_id_t dst = (local_partition_id + step) % partition_num;
                        size_t max_progress = 0;
                        for (partition_id_t t_i = 0; t_i < worker_num; t_i++)
                        {
                            volatile size_t temp_val = dist_exec_ctx.progress[t_i][dst];
                            if (temp_val > max_progress)
                            {
                                max_progress = temp_val;
                            }
                        }
                        size_t diff = max_progress - send_progress[dst];
                        send_func(dst, diff);
                    }
                } else
                {
                    partition_id_t dst = (local_partition_id + phase_i) % partition_num;
                    size_t min_progress = UINT_MAX;
                    for (partition_id_t t_i = 0; t_i < worker_num; t_i++)
                    {
                        volatile size_t temp_val = dist_exec_ctx.progress[t_i][dst];
                        if (temp_val < min_progress)
                        {
                            min_progress = temp_val;
                        }
                    }
                    size_t diff = min_progress - send_progress[dst];
                    send_func(dst, diff);
                }
                dist_exec_ctx.phase_locks[phase_i].unlock();
            }
            for (partition_id_t p_i = 0; p_i < partition_num; p_i++)
            {
                msg_send_buffer[p_i]->count = 0;
            }
            for (auto req : requests)
            {
                MPI_Status status;
                MPI_Wait(req, &status);
                delete req;
            }
        });

        Timer timer;
        msg_producer();
//        printf("this is %d, and finish msg producer at %.3lf\n",this->local_partition_id, timer.duration());

        size_t flush_workload = 0;
        for (partition_id_t t_i = 0; t_i < worker_num; t_i++)
        {
            flush_workload += thread_local_msg_buffer[t_i]->count;
        }
#pragma omp parallel for if (flush_workload * 2 >= OMP_PARALLEL_THRESHOLD)
        for (partition_id_t t_i = 0; t_i < worker_num; t_i++)
        {
            flush_thread_local_msg_buffer<msg_t>(t_i);
        }

        for (int phase_i = dist_exec_ctx.unlocked_phase; phase_i < phase_num; phase_i++)
        {
            dist_exec_ctx.phase_locks[phase_i].unlock();
        }
#ifdef PERF_PROF
        if (local_partition_id == 0)
        {
            printf("%u: finish msg_producer in %lfs\n", local_partition_id, timer.duration());
        }
#endif

        size_t msg_num = 0;
        for (int step = 0; step < partition_num; step++)
        {
            partition_id_t src_partition_id = (partition_num + local_partition_id - step) % partition_num;
            recv_locks[src_partition_id].lock();
            if (zero_copy_data == nullptr)
            {
                size_t data_amount = msg_recv_buffer[src_partition_id]->count;
                msg_num += data_amount;
                msg_t* data_begin = (msg_t*)(msg_recv_buffer[src_partition_id]->data);
                msg_t* data_end = data_begin + data_amount;
                msg_consumer(data_begin, data_end);
            }
            recv_locks[src_partition_id].unlock();
        }
        if (zero_copy_data != nullptr)
        {
            msg_consumer(zero_copy_data, zero_copy_data + zero_copy_recv_count);
            msg_num = zero_copy_recv_count;
        }
#ifdef PERF_PROF
        if (local_partition_id == 0)
        {
            printf("%u: finish msg_consumer in %lfs\n", local_partition_id, timer.duration());
        }
#endif
        recv_thread.join();
        send_thread.join();
#ifdef PERF_PROF
        if (local_partition_id == 0)
        {
            printf("%u: finish transmission in %lfs\n", local_partition_id, timer.duration());
        }
#endif
        size_t glb_msg_num;
        MPI_Allreduce(&msg_num, &glb_msg_num, 1, get_mpi_data_type<size_t>(), MPI_SUM, MPI_COMM_WORLD);
        return glb_msg_num;
    }

    template<typename reducer_data_t>
    reducer_data_t process_vertices(std::function<reducer_data_t(vertex_id_t)> process)
    {
        vertex_id_t progress = 0;//vertex_partition_begin[local_partition_id];
        vertex_id_t step_length = PARALLEL_CHUNK_SIZE;
        reducer_data_t reducer = 0;
#pragma omp parallel reduction(+:reducer)
        {
            vertex_id_t work_begin, work_end;
            while ((work_begin = __sync_fetch_and_add(&progress, step_length)) < this->v_num)//vertex_partition_end[local_partition_id])
            {
                vertex_id_t work_end = std::min(work_begin + step_length, this->v_num);//vertex_partition_end[local_partition_id]);
                for (vertex_id_t v_i = work_begin; v_i != work_end; v_i++)
                {
                    if(this->this_is_local_vertex[v_i] == local_partition_id)
                    {
                        reducer += process(v_i);
                    }
                }
            }
        }
        reducer_data_t glb_reducer;
        MPI_Allreduce(&reducer, &glb_reducer, 1, get_mpi_data_type<reducer_data_t>(), MPI_SUM, MPI_COMM_WORLD);
        return glb_reducer;
    }
};
