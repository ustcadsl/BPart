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
#include <cstdlib>

#include "walk.hpp"
#include "static_comp.hpp"

template <typename edge_data_t>
void deepwalk(WalkEngine<edge_data_t, EmptyData> *graph, walker_id_t walker_num, step_t walk_length, vertex_id_t v_num_tmp, WalkConfig *walk_conf = nullptr)
{
    MPI_Barrier(MPI_COMM_WORLD);
    Timer timer;
//    StdRandNumGenerator *gen = get_thread_local_rand_gen();
    srand((unsigned)time(NULL));
    WalkerConfig<edge_data_t, EmptyData> walker_conf(walker_num);
    auto extension_comp = [&] (Walker<EmptyData>& walker, vertex_id_t current_v) {
	    if(graph->get_thread_local_rand_gen()->gen(5) == 1 && walker.step < walk_length)
	    {
		walker.step++;
		//printf("true\n");
		vertex_id_t next_vertex = graph->get_thread_local_rand_gen()->gen(v_num_tmp);
//		printf("true current %u, nex %u\n", current_v, next_vertex);
//		assert(next_vertex < graph->get_vertex_num());
//		#ifdef COLLECT_WALK_SEQUENCE
//			int worker_id_1 = omp_get_thread_num();
  //              	graph->footprints[worker_id_1].push_back(Footprint(walker.id, next_vertex, walker.step));
//                #endif
		graph->emit(next_vertex, walker);
		return 0.0;
	    }
	
            return walker.step >= walk_length ? 0.0 : 1.0;
    };
    auto static_comp = get_trivial_static_comp(graph);
    TransitionConfig<edge_data_t, EmptyData> tr_conf(extension_comp, static_comp);
    graph->random_walk(&walker_conf, &tr_conf, walk_conf);

#ifndef UNIT_TEST
    printf("total time %lfs\n", timer.duration());
#endif
}
