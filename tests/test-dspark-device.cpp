#include "../src/llama-dspark-device.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <stdexcept>

static void require(bool ok, const char * message) { if (!ok) { throw std::runtime_error(message); } }
static void supported(ggml_backend_t backend, ggml_cgraph * graph) {
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto * node = ggml_graph_node(graph, i);
        if (!ggml_backend_supports_op(backend, node)) {
            fprintf(stderr, "unsupported op: %s\n", ggml_op_name(node->op));
            throw std::runtime_error("test cannot use CPU fallback");
        }
    }
}
static void check_chain(ggml_backend_t backend, int vocab, int rank) {
    const int count = 7;
    ggml_context_ptr ctx(ggml_init({ 512*ggml_tensor_overhead() + ggml_graph_overhead_custom(512, false), nullptr, true }));
    auto * w1 = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, rank, vocab);
    auto * w2 = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q8_0, rank, vocab);
    auto * base = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, vocab, count);
    auto * uniforms = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, count);
    auto * anchor = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);
    auto * graph = ggml_new_graph_custom(ctx.get(), 512, false);
    ggml_tensor * prev = anchor;
    std::vector<ggml_tensor *> probabilities, ids;
    for (int i = 0; i < count; ++i) {
        auto * row = ggml_get_rows(ctx.get(), w1, prev);
        auto * bias = ggml_mul_mat(ctx.get(), w2, row);
        auto * logits = ggml_add(ctx.get(), ggml_view_2d(ctx.get(), base, vocab, 1, base->nb[1], i*base->nb[1]), bias);
        auto * probs = ggml_soft_max(ctx.get(), logits);
        prev = llama_dspark_sample_probs(ctx.get(), probs, ggml_view_1d(ctx.get(), uniforms, 1, i*sizeof(float)));
        probabilities.push_back(probs); ids.push_back(prev);
        ggml_set_output(probs); ggml_set_output(prev);
        ggml_build_forward_expand(graph, probs); ggml_build_forward_expand(graph, prev);
    }
    supported(backend, graph);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    require(bool(buffer), "allocation failed");
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);
    std::vector<float> a((size_t) rank*vocab), b(a.size()), logits((size_t) vocab*count);
    for (auto & x:a) { x=dist(rng); } for (auto & x:b) { x=dist(rng); }
    for (auto & x:logits) { x=dist(rng)*4.0f; }
    const size_t row_size = ggml_row_size(GGML_TYPE_Q8_0, rank);
    std::vector<unsigned char> quantized(row_size*vocab);
    ggml_quantize_chunk(GGML_TYPE_Q8_0,b.data(),quantized.data(),0,vocab,rank,nullptr);
    for (int v=0;v<vocab;++v) { ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(quantized.data()+v*row_size,b.data()+(size_t)v*rank,rank); }
    ggml_backend_tensor_set(w1,a.data(),0,ggml_nbytes(w1));
    ggml_backend_tensor_set(w2,quantized.data(),0,quantized.size());
    ggml_backend_tensor_set(base,logits.data(),0,ggml_nbytes(base));
    float max_abs = 0.0f;
    double max_l1 = 0.0;
    for (int repeat=0;repeat<3;++repeat) {
        int32_t last=3+repeat;
        float u[count]={0.02f,0.19f,0.39f,0.51f,0.71f,0.88f,0.99f};
        u[2]+=repeat*0.013f;
        ggml_backend_tensor_set(anchor,&last,0,sizeof(last));
        ggml_backend_tensor_set(uniforms,u,0,sizeof(u));
        require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"graph failed");
        std::vector<double> ref(vocab); std::vector<float> actual(vocab);
        for (int i=0;i<count;++i) {
            double maximum=-INFINITY;
            for (int v=0;v<vocab;++v) {
                double value=logits[(size_t)i*vocab+v];
                for(int k=0;k<rank;++k) { value+=(double)a[(size_t)last*rank+k]*b[(size_t)v*rank+k]; }
                maximum=std::max(maximum,value);ref[v]=value;
            }
            double sum=0;for(auto & x:ref) { x=std::exp(x-maximum);sum+=x; }
            for(auto & x:ref) { x/=sum; }
            ggml_backend_tensor_get(probabilities[i],actual.data(),0,vocab*sizeof(float));
            int32_t got=-1;ggml_backend_tensor_get(ids[i],&got,0,sizeof(got));
            require(got>=0 && got<vocab,"sample outside vocabulary");
            double psum=0, l1=0;for(int v=0;v<vocab;++v) {
                require(std::isfinite(actual[v]) && actual[v]>=0,"invalid probability");
                max_abs=std::max(max_abs,(float)std::abs(ref[v]-actual[v]));psum+=actual[v];l1+=std::abs(ref[v]-actual[v]);
            }
            max_l1=std::max(max_l1,l1);
            require(l1<0.025,"Markov probability L1 error exceeds Q8 activation tolerance");
            require(std::abs(psum-1.0)<2e-5,"probabilities not normalized");
            double cdf=0;for(int v=0;v<got;++v) { cdf+=actual[v]; }
            require(u[i]>=cdf-2e-6 && u[i]<=cdf+actual[got]+2e-6,"sample not in reference CDF interval");
            last=got;
        }
    }
    printf("Q8 activation quantization max_abs=%g max_l1=%g\n",max_abs,max_l1);
    require(max_abs<5e-4,"Markov probability mismatch");
    printf("chain backend=%s vocab=%d rank=%d positions=7 repeats=3 max_abs=%g PASS\n",ggml_backend_name(backend),vocab,rank,max_abs);
}
static void check_features(ggml_backend_t backend) {
    const int width=32,layers=5,capacity=29,rows=7;
    ggml_context_ptr ctx(ggml_init({64*ggml_tensor_overhead()+ggml_graph_overhead_custom(64,false),nullptr,true}));
    auto * graph=ggml_new_graph_custom(ctx.get(),64,false);
    std::vector<ggml_tensor *> source,input;
    auto * indices=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_I32,rows);
    ggml_tensor * combined=nullptr;
    for(int k=0;k<layers;++k) {
        source.push_back(ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,width,capacity));
        input.push_back(ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,width,capacity));
        auto * selected=ggml_get_rows(ctx.get(),input.back(),indices);
        combined=combined?ggml_concat(ctx.get(),combined,selected,0):selected;
    }
    ggml_build_forward_expand(graph,combined);supported(backend,graph);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),backend));require(bool(buffer),"feature allocation failed");
    std::vector<float> data(width*capacity),out(width*layers*rows);
    for(int k=0;k<layers;++k) {
        for(int i=0;i<width*capacity;++i) { data[i]=float(k*10000+i); }
        ggml_backend_tensor_set(source[k],data.data(),0,data.size()*sizeof(float));
    }
    const int32_t permutation[rows]={3,0,6,1,5,2,4};
    for(int offset: {0,5,22}) {
        int32_t selected[rows];for(int i=0;i<rows;++i) { selected[i]=offset+permutation[i]; }
        ggml_backend_tensor_set(indices,selected,0,sizeof(selected));
        for(int k=0;k<layers;++k) { llama_dspark_copy_feature_rows(backend,source[k],input[k],0); }
        require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"feature graph failed");
        ggml_backend_tensor_get(combined,out.data(),0,out.size()*sizeof(float));
        for(int row=0;row<rows;++row) for(int k=0;k<layers;++k) for(int j=0;j<width;++j) {
            require(out[(row*layers+k)*width+j]==float(k*10000+(offset+permutation[row])*width+j),"feature row or layer order mismatch");
        }
    }
    printf("features backend=%s layers=5 offsets=0,5,22 exact PASS\n",ggml_backend_name(backend));
}

static void check_sampling_edges(ggml_backend_t backend) {
    constexpr int vocab=131072;
    ggml_context_ptr ctx(ggml_init({64*ggml_tensor_overhead()+ggml_graph_overhead_custom(64,false),nullptr,true}));
    auto * probs=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,vocab,1);
    auto * uniform=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,1);
    auto * sampled=llama_dspark_sample_probs(ctx.get(),probs,uniform);
    auto * graph=ggml_new_graph_custom(ctx.get(),64,false);
    ggml_build_forward_expand(graph,sampled);supported(backend,graph);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),backend));require(bool(buffer),"edge allocation failed");
    std::vector<float> p(vocab,0.0f);p[3]=0.25f;p[8192]=0.5f;p[vocab-2]=0.25f;
    ggml_backend_tensor_set(probs,p.data(),0,p.size()*sizeof(float));
    const float values[]={0.0f,std::nextafter(0.25f,0.0f),0.25f,std::nextafter(0.75f,0.0f),0.75f,std::nextafter(1.0f,0.0f)};
    const int32_t expected[]={3,3,8192,8192,vocab-2,vocab-2};
    for(size_t i=0;i<sizeof(values)/sizeof(values[0]);++i) {
        ggml_backend_tensor_set(uniform,&values[i],0,sizeof(float));
        require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"edge graph failed");
        int32_t actual=-1;ggml_backend_tensor_get(sampled,&actual,0,sizeof(actual));
        require(actual==expected[i],"CDF boundary or zero-probability sample failed");
    }
    printf("sampling edges backend=%s sparse vocabulary and six boundaries PASS\n",ggml_backend_name(backend));
}

static void check_token_handoff(ggml_backend_t consumer) {
    auto * device = ggml_backend_get_device(consumer);
    ggml_backend_ptr producer(ggml_backend_dev_init(device, nullptr));
    require(bool(producer), "producer backend init failed");
    ggml_backend_event_ptr ready(ggml_backend_event_new(device));
    if (ggml_backend_is_cpu(producer.get())) { ggml_backend_cpu_set_n_threads(producer.get(), 4); }
    for (const int count : {1, 3, 7}) {
        ggml_context_ptr ctx(ggml_init({64*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(32, false), nullptr, true}));
        auto * values = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, count);
        auto * ids = ggml_cast(ctx.get(), values, GGML_TYPE_I32);
        auto * table = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_BF16, 8, 16);
        auto * tokens = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, count + 1);
        auto * rows = ggml_get_rows(ctx.get(), table, tokens);
        auto * draft_graph = ggml_new_graph_custom(ctx.get(), 32, false);
        auto * target_graph = ggml_new_graph_custom(ctx.get(), 32, false);
        ggml_build_forward_expand(draft_graph, ids);
        ggml_build_forward_expand(target_graph, rows);
        supported(producer.get(), draft_graph);
        supported(consumer, target_graph);
        // Both backends use the same device, as required by device DSpark.
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), consumer));
        require(bool(buffer), "handoff allocation failed");
        std::vector<float> embeddings(16*8);
        std::vector<ggml_bf16_t> packed(embeddings.size());
        for (size_t i = 0; i < embeddings.size(); ++i) {
            packed[i] = ggml_fp32_to_bf16(((float) i - 63.0f)*0.017f);
            embeddings[i] = ggml_bf16_to_fp32(packed[i]);
        }
        ggml_backend_tensor_set(table, packed.data(), 0, ggml_nbytes(table));
        auto * buft = ggml_backend_dev_host_buffer_type(device);
        if (!buft) { buft = ggml_backend_cpu_buffer_type(); }
        ggml_backend_buffer_ptr host(ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(rows)));
        require(bool(host), "handoff host allocation failed");
        auto * output = (float *) ggml_backend_buffer_get_base(host.get());
        for (int round = 0; round < 3; ++round) {
            const int32_t anchor = 15 - round;
            std::vector<float> input(count);
            for (int i = 0; i < count; ++i) { input[i] = (i*3 + round) % 15; }
            ggml_backend_tensor_set(values, input.data(), 0, ggml_nbytes(values));
            ggml_backend_tensor_set(tokens, &anchor, 0, sizeof(anchor));
            require(ggml_backend_graph_compute_async(producer.get(), draft_graph) == GGML_STATUS_SUCCESS, "producer graph failed");
            if (ready) {
                ggml_backend_event_record(ready.get(), producer.get());
                ggml_backend_event_wait(consumer, ready.get());
            } else {
                ggml_backend_synchronize(producer.get());
            }
            llama_dspark_copy_tokens(producer.get(), consumer, ids, tokens, 0, 1, count);
            require(ggml_backend_graph_compute_async(consumer, target_graph) == GGML_STATUS_SUCCESS, "consumer graph failed");
            ggml_backend_tensor_get_async(consumer, rows, output, 0, ggml_nbytes(rows));
            ggml_backend_synchronize(consumer);
            if (ready) { ggml_backend_event_synchronize(ready.get()); }
            for (int i = 0; i <= count; ++i) {
                const int32_t id = i == 0 ? anchor : (int32_t) input[i - 1];
                for (int j = 0; j < 8; ++j) {
                    require(output[i*8 + j] == embeddings[id*8 + j], "device token handoff mismatch");
                }
            }
        }
        bool rejected = false;
        try { llama_dspark_copy_tokens(producer.get(), consumer, ids, tokens, 0, 1, count + 1); }
        catch (const std::runtime_error &) { rejected = true; }
        require(rejected, "out-of-range token handoff accepted");
    }
    ggml_backend_synchronize(producer.get());
    printf("token handoff backend=%s type=bf16 counts=1,3,7 reuse=3 exact PASS\n", ggml_backend_name(consumer));
}

int main(int argc,char ** argv) {
    try {
        ggml_backend_load_all();
        auto * device=ggml_backend_dev_by_name(argc>1?argv[1]:"CPU");require(device,"device not found");
        ggml_backend_ptr backend(ggml_backend_dev_init(device,nullptr));require(bool(backend),"backend init failed");
        if(ggml_backend_is_cpu(backend.get())) { ggml_backend_cpu_set_n_threads(backend.get(),4); }
        check_features(backend.get());check_sampling_edges(backend.get());check_chain(backend.get(),64,32);check_chain(backend.get(),131072,256);
        check_token_handoff(backend.get());
    } catch (const std::exception & e) { fprintf(stderr,"FAIL: %s\n",e.what());return 1; }
    return 0;
}
