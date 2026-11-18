#include "ggml.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-vulkan/ggml-vulkan-vplane.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

using context_owner = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using buffer_owner = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
using backend_owner = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
static void require(bool ok, const char * why) { if (!ok) throw std::runtime_error(why); }
static std::vector<float> input(size_t n, uint32_t seed) {
    std::vector<float> a(n);
    for (size_t i = 0; i < n; ++i) {
        uint32_t x = seed + uint32_t(i) * 0x9e3779b9u;
        x = (x ^ (x >> 16)) * 0x7feb352du;
        x = (x ^ (x >> 15)) * 0x846ca68bu;
        x ^= x >> 16;
        a[i] = (int32_t(x & 65535u) - 32768) / 32768.0f;
    }
    return a;
}
static void compute(ggml_backend_t gpu, ggml_cgraph * g) {
    require(ggml_backend_graph_compute(gpu, g) == GGML_STATUS_SUCCESS, "GPU graph failed");
    ggml_backend_synchronize(gpu);
}
static std::vector<float> read(ggml_tensor * t) {
    std::vector<float> v(ggml_nelements(t)); ggml_backend_tensor_get(t,v.data(),0,v.size()*4); return v;
}
static std::vector<float> upload(ggml_tensor * t, const std::vector<float> & original) {
    std::vector<unsigned char> packed(ggml_nbytes(t));
    if (t->type == GGML_TYPE_F32) std::memcpy(packed.data(),original.data(),packed.size());
    else if (t->type == GGML_TYPE_F16) ggml_fp32_to_fp16_row(original.data(),reinterpret_cast<ggml_fp16_t *>(packed.data()),original.size());
    else if (t->type == GGML_TYPE_Q8_1) ggml_get_type_traits(t->type)->from_float_ref(original.data(),packed.data(),original.size());
    else require(ggml_quantize_chunk(t->type,original.data(),packed.data(),0,original.size()/256,256,nullptr)==packed.size(),"Quantized size mismatch");
    ggml_backend_tensor_set(t,packed.data(),0,packed.size());
    std::vector<unsigned char> returned(packed.size()); ggml_backend_tensor_get(t,returned.data(),0,returned.size());
    require(returned==packed,"Complete cache upload/readback differs");
    std::vector<float> decoded(original.size());
    if (t->type == GGML_TYPE_F32) decoded=original;
    else if (t->type == GGML_TYPE_Q8_1) {
        for (size_t block=0;block<decoded.size()/32;++block) {
            ggml_fp16_t scale;std::memcpy(&scale,returned.data()+block*36,2);
            float d=ggml_fp16_to_fp32(scale);
            for(size_t lane=0;lane<32;++lane) {
                int8_t code;std::memcpy(&code,returned.data()+block*36+4+lane,1);
                decoded[block*32+lane]=d*code;
            }
        }
    }
    else ggml_get_type_traits(t->type)->to_float(returned.data(),decoded.data(),decoded.size());
    return decoded;
}
static std::vector<float> reference(const std::vector<float>& q,const std::vector<float>& k,const std::vector<float>& v,int kv,int m) {
    std::vector<float> out(size_t(256)*8*m); std::vector<double> logits(kv), sums(256);
    const int history=kv>1024?kv-256:kv-m;
    const int window=kv==768?(m==3?513:512):0;
    for (int row=0;row<m;++row) for (int h=0;h<8;++h) {
        const int end=std::min(kv,history+row+1), first=window?std::max(0,end-window):0, kh=h/4;
        double maximum=-INFINITY;
        for (int col=first;col<end;++col) {
            double dot=0;
            for (int d=0;d<256;++d) dot+=double(q[d+256*(h+8*row)])*k[d+256*(kh+2*col)];
            logits[col]=dot/16.;maximum=std::max(maximum,logits[col]);
        }
        double denominator=0; std::fill(sums.begin(),sums.end(),0.);
        for (int col=first;col<end;++col) {
            const double weight=std::exp(logits[col]-maximum);denominator+=weight;
            for (int d=0;d<256;++d) sums[d]+=weight*v[d+256*(kh+2*col)];
        }
        for (int d=0;d<256;++d) out[d+256*(h+8*row)]=float(sums[d]/denominator);
    }
    return out;
}
struct errors { double nrmse,max_abs; };
static errors compare(const std::vector<float>& a,const std::vector<float>& b) {
    require(a.size()==b.size(),"Output length differs");double sum=0,denom=0,maximum=0;
    for(size_t i=0;i<a.size();++i) {
        require(std::isfinite(a[i])&&std::isfinite(b[i]),"Nonfinite complete output");
        double d=double(a[i])-b[i];sum+=d*d;denom+=double(b[i])*b[i];maximum=std::max(maximum,std::abs(d));
    }
    return {std::sqrt(sum/std::max(denom,1e-30)),maximum};
}
struct variant {
    context_owner ctx{nullptr,ggml_free}; buffer_owner buffer{nullptr,ggml_backend_buffer_free};
    ggml_cgraph * graph=nullptr;ggml_tensor * out=nullptr;
    std::vector<float> k,v;
    ggml_type kt,vt;size_t bytes=0;
};
static std::unique_ptr<variant> make_time_fa(ggml_backend_t gpu,int kv,int m,const std::vector<float>&q,const std::vector<float>&k,const std::vector<float>&v,int groups,ggml_type kt,int bits) {
    require((groups==4||groups==8)&&(bits==4||bits==8),"Invalid plane format");
    const ggml_type vt=bits==4?GGML_TYPE_Q4_0:GGML_TYPE_Q8_0;
    const int block_bytes=bits==4?18:34,chunk_bytes=256*block_bytes;
    auto a=std::make_unique<variant>();a->kt=kt;a->vt=vt;
    a->ctx.reset(ggml_init({2*1024*1024,nullptr,true}));require(bool(a->ctx),"Plane context failed");auto ctx=a->ctx.get();
    auto qr=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,256,8,m),kr=ggml_new_tensor_3d(ctx,kt,256,2,kv);
    auto vr=ggml_new_tensor_3d(ctx,GGML_TYPE_I8,chunk_bytes,kv/32,2);
    auto mask=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,kv,m);auto qp=ggml_permute(ctx,qr,0,2,1,3),kp=ggml_permute(ctx,kr,0,2,1,3);
    a->out=ggml_vk_flash_attn_vplane(ctx,qp,kp,vr,mask,bits,1.f/16.f);require(a->out!=nullptr,"Plane constructor rejected valid inputs");ggml_set_name(a->out,"vplane_attention");
    a->graph=ggml_new_graph_custom(ctx,128,false);ggml_build_forward_expand(a->graph,a->out);require(ggml_backend_supports_op(gpu,a->out),"Raw plane FA unsupported");
    a->buffer.reset(ggml_backend_alloc_ctx_tensors(ctx,gpu));require(bool(a->buffer),"Plane allocation failed");
    std::vector<float> raw(v.size());const int blocks=kv/32;
    for(int h=0;h<2;++h)for(int b=0;b<blocks;++b)for(int d=0;d<256;++d)for(int t=0;t<32;++t)raw[t+32*(d+256*(b+blocks*h))]=v[d+256*(h+2*(b*32+t))];
    upload(qr,q);a->k=upload(kr,k);
    std::vector<uint8_t> canonical(ggml_row_size(vt,raw.size()));auto traits=ggml_get_type_traits(vt);
    require(traits->from_float_ref&&traits->to_float,"Missing canonical quantizer");traits->from_float_ref(raw.data(),canonical.data(),raw.size());
    traits->to_float(canonical.data(),raw.data(),raw.size());
    std::vector<uint8_t> packed(ggml_nbytes(vr),0);
    for(int h=0;h<2;++h)for(int b=0;b<blocks;++b)for(int d=0;d<256;++d){
        auto src=canonical.data()+block_bytes*(d+256*(b+blocks*h));auto dst=packed.data()+chunk_bytes*(b+blocks*h);std::memcpy(dst+2*d,src,2);
        for(int t=0;t<32;++t){int g=t%groups,c=t/groups;
            if(bits==8){dst[512+(g*64+d/4)*(128/groups)+(c/4)*16+(d%4)*4+c%4]=src[2+t];}
            else {uint8_t code=(src[2+t%16]>>(4*(t/16)))&15;dst[512+(g*64+d/4)*(64/groups)+(d%4)*(16/groups)+c/2]|=uint8_t(code<<(4*(c%2)));}
        }
    }
    ggml_backend_tensor_set(vr,packed.data(),0,packed.size());std::vector<uint8_t> back(packed.size());ggml_backend_tensor_get(vr,back.data(),0,back.size());require(back==packed,"Plane full readback differs");
    a->v.resize(v.size());
    for(int h=0;h<2;++h)for(int b=0;b<blocks;++b)for(int d=0;d<256;++d){
        auto src=back.data()+chunk_bytes*(b+blocks*h);ggml_fp16_t scale;std::memcpy(&scale,src+2*d,2);
        for(int t=0;t<32;++t){int g=t%groups,c=t/groups,code;
            if(bits==8){int8_t value;std::memcpy(&value,src+512+(g*64+d/4)*(128/groups)+(c/4)*16+(d%4)*4+c%4,1);code=value;}
            else {code=int((src[512+(g*64+d/4)*(64/groups)+(d%4)*(16/groups)+c/2]>>(4*(c%2)))&15)-8;}
            float value=ggml_fp16_to_fp32(scale)*code;
            require(value==raw[t+32*(d+256*(b+blocks*h))],"Plane decode differs from canonical GGML reference");
            a->v[d+256*(h+2*(b*32+t))]=value;
        }
    }
    const int history=kv>1024?kv-256:kv-m,window=kv==768?(m==3?513:512):0;std::vector<ggml_fp16_t> masks(size_t(kv)*m);
    for(int row=0;row<m;++row)for(int col=0;col<kv;++col){int end=std::min(kv,history+row+1),first=window?std::max(0,end-window):0;masks[size_t(row)*kv+col]=ggml_fp32_to_fp16(col>=first&&col<end?0.f:-INFINITY);}
    ggml_backend_tensor_set(mask,masks.data(),0,masks.size()*2);a->bytes=ggml_nbytes(kr)+ggml_nbytes(vr);
    std::cerr<<"VPLANE_LAYOUT bits="<<bits<<" groups="<<groups<<" kv="<<kv<<" rows="<<m<<" bytes="<<a->bytes<<std::endl;return a;
}

static void contract_tests() {
    context_owner ctx(ggml_init({2*1024*1024, nullptr, true}), ggml_free);
    require(bool(ctx), "Context failed");
    for (int bits : {4, 8}) {
        auto type = bits == 4 ? GGML_TYPE_Q4_0 : GGML_TYPE_Q8_0;
        auto q = ggml_permute(ctx.get(), ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 256, 8, 1), 0, 2, 1, 3);
        auto k = ggml_permute(ctx.get(), ggml_new_tensor_3d(ctx.get(), type, 256, 2, 33024), 0, 2, 1, 3);
        auto v = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_I8, bits == 4 ? 4608 : 8704, 33024/32, 2);
        auto mask = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, 33024, 1);
        require(ggml_vk_flash_attn_vplane(ctx.get(), q, k, v, mask, bits, 1.0f/16) != nullptr, "Valid contract rejected");
        require(ggml_vk_flash_attn_vplane(ctx.get(), q, k, v, mask, 3, 1.0f/16) == nullptr, "Invalid bits accepted");
        require(ggml_vk_flash_attn_vplane(ctx.get(), q, k, v, mask, bits, INFINITY) == nullptr, "Infinite scale accepted");
        auto saved = *v;
        v->ne[1] -= 1;
        require(!ggml_vk_vplane_inputs(q, k, v, mask, bits), "Truncated payload accepted");
        *v = saved; v->nb[1] += 1;
        require(!ggml_vk_vplane_inputs(q, k, v, mask, bits), "Invalid payload stride accepted");
        *v = saved;
        auto saved_k = *k; k->ne[1] = INT64_MAX;
        require(!ggml_vk_vplane_inputs(q, k, v, mask, bits), "Oversize KV accepted");
        *k = saved_k;
        auto saved_q = *q; q->ne[1] = 0;
        require(!ggml_vk_vplane_inputs(q, k, v, mask, bits), "Zero query rows accepted");
        *q = saved_q;
        mask->ne[0] -= 1;
        require(!ggml_vk_vplane_inputs(q, k, v, mask, bits), "Short mask accepted");
    }
    for (uint32_t word = 0; word < 65536; ++word) {
        uint32_t spread = (word & 15) | ((word & 240) << 4) | ((word & 3840) << 8) | ((word & 61440) << 12);
        uint32_t actual = (spread + 0x78787878u) ^ 0x80808080u, expected = 0;
        for (int i = 0; i < 4; ++i) {
            expected |= uint32_t((int((word >> (4*i)) & 15) - 8) & 255) << (8*i);
        }
        require(actual == expected, "Nibble expansion mismatch");
    }
    std::cout << "Contract and 65536 nibble cases passed\n";
}

int main(int argc, char ** argv) {
    try {
        contract_tests();
        if (argc == 1 || (argc == 2 && std::string(argv[1]) == "--contract")) { return 0; }
        require(argc >= 2 && std::string(argv[1]) == "--gpu", "Usage: test-vulkan-vplane [--contract | --gpu [reference.bin]]");
        ggml_backend_load_all();
        auto device = ggml_backend_dev_by_name("Vulkan0");
        require(device != nullptr, "Vulkan0 missing");
        backend_owner gpu(ggml_backend_dev_init(device, nullptr), ggml_backend_free);
        require(bool(gpu), "Vulkan initialization failed");
        std::ifstream prior;
        if (argc == 3) { prior.open(argv[2], std::ios::binary); require(prior.good(), "Cannot open historical reference"); }
        const std::vector<std::array<int,3>> shapes = {{{768,1,102}},{{768,3,13}},{{33024,1,103}},{{33024,2,7}},{{33024,4,6}},{{33024,8,100}},{{131328,1,104}},{{131328,2,11}},{{131328,4,10}},{{131328,8,101}}};
        int count = 0;
        for (auto shape : shapes) {
            int kv = shape[0], m = shape[1];
            uint32_t seed = 0x127348u + shape[2]*109u;
            auto q = input(size_t(256)*8*m, seed), k = input(size_t(256)*2*kv, seed+1), v = input(size_t(256)*2*kv, seed+2);
            for (int bits : {8, 4}) {
                auto config = ggml_vk_vplane_tuning(kv, m, bits);
                auto a = make_time_fa(gpu.get(), kv, m, q, k, v, config.groups, bits == 4 ? GGML_TYPE_Q4_0 : GGML_TYPE_Q8_0, bits);
                if (!config.integer) {
                    for (auto & x : a->v) { x = ggml_fp16_to_fp32(ggml_fp32_to_fp16(x)); }
                }
                auto ref = reference(q, a->k, a->v, kv, m);
                compute(gpu.get(), a->graph); auto output = read(a->out);
                compute(gpu.get(), a->graph); require(output == read(a->out), "Repeated output changed");
                auto metric = compare(output, ref);
                require(metric.nrmse <= .03 && metric.max_abs <= .02, "FP64 numerical contract failed");
                bool exact = false;
                if (prior.is_open()) {
                    std::vector<float> expected(output.size());
                    prior.read(reinterpret_cast<char *>(expected.data()), expected.size()*sizeof(float));
                    require(bool(prior), "Truncated historical reference");
                    exact = output == expected;
                    auto delta = compare(output, expected);
                    require(delta.nrmse <= .01 && delta.max_abs <= .01, "Historical output changed beyond 1% contract");
                }
                std::cout << "VPLANE_RESULT kv=" << kv << " rows=" << m << " bits=" << bits
                          << " groups=" << config.groups << " integer=" << config.integer
                          << " nrmse=" << metric.nrmse << " max_abs=" << metric.max_abs << " historical_exact=" << exact << std::endl;
                // An invalid tag/layout must fail supports_op before a dispatch.
                auto saved = a->out->op_params[7]; a->out->op_params[7] = 1;
                require(!ggml_backend_supports_op(gpu.get(), a->out), "Unsupported grouping accepted");
                a->out->op_params[7] = saved;
                a->out->op_params[4] = 32;
                require(!ggml_backend_supports_op(gpu.get(), a->out), "Sparse packed cache accepted");
                ++count;
            }
        }
        if (prior.is_open()) { require(prior.peek() == std::char_traits<char>::eof(), "Trailing historical reference data"); }
        std::cout << "VPLANE_COMPLETE cases=" << count << std::endl;
        return 0;
    } catch (const std::exception & e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
