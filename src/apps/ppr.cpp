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

#include "walk.hpp"
#include "option_helper.hpp"
#include "ppr.hpp"

class PPROptionHelper : public SRandomWalkOptionHelper
{
private:
    args::ValueFlag<real_t> terminate_prob_flag;
    args::ValueFlag<real_t> rate_flag;
public:
    real_t terminate_prob;
    double cache_rate;
    PPROptionHelper():
        terminate_prob_flag(parser, "terminate", "terminate probabiility", {'t'}),
        rate_flag(parser, "rate", "Set this option will break random walk into multiple iterations to save memory. Each iteration has rate% walkers.", {'r'})
    {}
    virtual void parse(int argc, char** argv)
    {
        SRandomWalkOptionHelper::parse(argc, argv);

        assert(terminate_prob_flag);
        terminate_prob = args::get(terminate_prob_flag);

        cache_rate = double(args::get(rate_flag))/100.0;
    }
};


template<typename edge_data_t>
void run(WalkEngine<edge_data_t, EmptyData> *graph, PPROptionHelper *opt)
{
    graph->load_graph(opt->v_num, opt->graph_path.c_str(), opt->partition_type, opt->make_undirected);
    WalkConfig walk_conf;
    if (!opt->output_path.empty())
    {
        walk_conf.set_output_file(opt->output_path.c_str());
    }
    if (opt->set_rate)
    {
        walk_conf.set_walk_rate(opt->rate);
    }
    ppr(graph, opt->walker_num, opt->terminate_prob, nullptr, &walk_conf);
}

int main(int argc, char** argv)
{
    MPI_Instance mpi_instance(&argc, &argv);

    PPROptionHelper opt;
    opt.parse(argc, argv);

    if (opt.static_comp.compare("weighted") == 0)
    {
        WalkEngine<real_t, EmptyData> graph;
        run(&graph, &opt);
    } else if(opt.static_comp.compare("unweighted") == 0)
    {
        WalkEngine<EmptyData, EmptyData> graph;
        run(&graph, &opt);
    } else
    {
        exit(1);
    }
    return 0;
}
