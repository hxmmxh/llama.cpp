# DSpark Vulkan 优化

`vulkan_opt` 基于 `cpu_opt` 的 `4d36b146147f30b3ce0e12d2b92547c079b63f75`，保留已有 CPU 优化，移植在 Intel Arc 140T 上验证过的 DSpark 设备内执行路径。代码来源为工作区实验 `20261108-192722-vulkan-dspark-device` 的 `provenance/device.patch`。没有加入未取得收益的矩阵乘 shader 候选。

Dense-q 补丁基于 `0cc8e02d81b9df6f19f55022afcf0c6523cefff9`，涉及 `common/sampling.{h,cpp}`、`common/speculative.{h,cpp}`、`tools/server/server-context.cpp` 和 `tests/test-speculative-rejection-sampling.cpp` 六个文件。2026-11-12 核对时，这六个文件已在当前工作区，逐文件 SHA-256 与 `20261109-142000-vulkan-denseq-unroll` 的实测冻结源码一致，原导入补丁的反向检查也通过。[源码核对与补丁](../../experiments/vulkan-denseq-port-20261112/source-verification.json) · [本地构建与回归记录](../../experiments/vulkan-denseq-port-20261112/validation.json)。

## 实现与适用范围

设置 `LLAMA_DSPARK_DEVICE=1` 后，Markov 条件链、draft 全词表 softmax/CDF 采样、目标特征的复制、行重排与拼接在 GPU 上执行。CPU 提供随机数和少量索引；原有接受/拒绝验证仍在 CPU 上执行，并读取 draft 概率向量。默认不设置此变量时保持原路径，包括 CPU W1 Q8 缓存和 CPU_REPACK 专用 shared-head padding。

另外设置 `LLAMA_DSPARK_DENSE_Q=1` 时，device DSpark 将全词表 draft 概率按 token ID 顺序复制到连续 F32 存储，CPU rejection verifier 直接读取，省去构造及扫描 `llama_token_data` 数组。接受/拒绝算法、grammar 处理及 target/residual RNG 调用顺序与原路径共用；未设置该变量时保留原表示。该开关在 drafter 初始化时读取，切换后须重启 server。启动后首次命中时打印 `DSpark dense q verifier active: rows=7 vocab=131072`，尾轮可以少于7行。

Dense-q 容器按 slot 持有，clear/truncate 保留已分配容量。7行、131072词表时有效载荷为3.5 MiB，随 slot 销毁释放；这不包括 context 概率缓冲区和 Vulkan staging。`include/llama.h` 的 C API 不变，common C++ 结构及新增重载需要调用方一起重编。

目前支持单序列、主模型和 draft 位于同一 GPU、完整词表、anchor-first DSpark。已测 Spark-X2.6-4B 和 1.7B 的 Q4_K_M 主模型与 Q8_0-backbone draft；1.7B draft 按既有 backbone 配方从原始 BF16 新量化，为有损混合精度产物。不涵盖并发多序列、异构 GPU、其他采样配方或上下文状态保存/恢复。

Flash Attention 是已有 Vulkan 后端能力，通过 `-fa on` 启用，并非本分支新写的 kernel。它融合注意力核心的 QK、softmax 和 PV；QKV 投影、输出投影、FFN 和 Markov 采样仍独立执行。草稿上限为 7 时，正常完整轮次的 draft 前向为 M7、主模型验证为 M8，末尾轮次可更短。

Arc 140T 驱动 `32.0.101.6313` 的默认 cooperative-matrix 路径曾出现相同 seed 输出不稳定。有效测试均设置 `GGML_VK_DISABLE_COOPMAT=1`；这仍使用 Vulkan GPU 矩阵乘。该结论限定于已测机器和驱动。

## 2026-11-11 全量数据集结果

Vulkan0、FA on、dense-q、draft7、F16 KV、12线程、batch512/ubatch128。两种模型分别运行六个过滤数据集的429题普通生成和概率 DSpark 对照，共1716个正式请求。Thinking low，target T=1/P=.95，draft T=1/P=1，每题最多512 token，按总 decode token / 总 decode 时间汇总。

| 模型 | 普通 decode token/s | DSpark decode token/s | 加速比 | 草稿接受率 | 每轮提交 L |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4B | 19.4061 | 26.8043 | 1.3812x | 41.60% | 3.90 |
| 1.7B | 40.4405 | 46.5184 | 1.1503x | 37.35% | 3.60 |

L包含纠正/bonus token。1.7B在Arena-Hard上为0.9607x负收益；本轮未评测准确率。完整逐数据集结果、上下文吞吐、采样口径及文件身份见[实测报告](../../experiments/windows-cpu/windows_core_ultra_7_255h/20261111-115611-vulkan-benchmark-4b-17b/REPORT.zh.md)。冻结程序另编入了 `GGML_VK_Q4_K_UNROLL=1/2` 实验候选，但本轮均未启用；当前分支只保留已采用的dense-q补丁，不包含这些 shader 候选。

## 早期速度结果

以下是 **旧冻结实验快照** 的 GSM8K 前 38 题，每题输出上限 512 token，Thinking low，target T=1/top_p=0.95，draft T=1/top_p=1。吞吐按总 decode token 除以总 decode 时间计算，不含加载、预热和 prefill。它们不是本分支的新完整测速。

| 配置 | Decode token/s | 严格 boxed 答案正确数 |
| --- | ---: | ---: |
| GPU DSpark，FA 关闭，草稿上限 7 | 28.8220 | 32/38 |
| GPU DSpark，FA 开启，草稿上限 7 | **32.8622** | 30/38 |
| GPU DSpark，FA 开启，草稿上限 5 | 31.3643 | 30/38 |
| GPU DSpark，FA 开启，草稿上限 6 | 32.0679 | 26/38 |

上述旧快照中最快的完整共同子集结果是 **32.8622 token/s**，比关闭 FA 的结果高约 14.0%；尚未达到 35。FA 和草稿长度会改变浮点舍入、采样与接受路径，生成内容和工作量也不同，因此不把这个比例当作同输出条件下的算子加速比。正确数只是 38 题、512-token 限额下的结果，不是完整 GSM8K 准确率，也不能据此宣称质量无损。

新分支保留了实验快照之外的 `cpu_opt` 改动，包括 target top-p 排序优化。已通过本地 CPU 编译及 5 项 CTest、Windows Vulkan 原生编译、GPU 设备算子与 CPU DSpark 单元检查。两道完整 GSM8K 回归题启用 CDF 检查后，输出 token、提议数和接受数与已测 FA 配置完全一致，server 正常退出；这不等于重新完成 38 题吞吐测量。验证状态与身份见 [验证记录](dspark-vulkan-validation.json)，完整分析见工作区 [Vulkan DSpark 耗时与优化报告](../../../reports/2026-11-08-Vulkan-DSpark耗时与优化/README.md)。

## Windows 编译

使用 Visual Studio 2022 的 x64 开发者 PowerShell，安装 CMake、Ninja、Vulkan SDK，并将 CMake、Ninja 放入 PATH。`VULKAN_SDK` 应指向已安装的 SDK；实验使用 SDK `1.4.363.1`。在仓库根目录运行：

```powershell
cmake -S . -B build-vulkan -G Ninja `
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl `
  "-DCMAKE_PREFIX_PATH=$env:VULKAN_SDK" `
  -DGGML_VULKAN=ON -DGGML_SYCL=OFF -DGGML_OPENVINO=OFF `
  -DGGML_NATIVE=OFF -DGGML_CPU_ALL_VARIANTS=OFF `
  -DGGML_AVX2=ON -DGGML_AVX512=OFF -DGGML_AVX_VNNI=OFF `
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF `
  -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF `
  -DLLAMA_BUILD_APP=OFF -DLLAMA_OPENSSL=OFF
if ($LASTEXITCODE -ne 0) { throw 'CMake configure failed' }
cmake --build build-vulkan --parallel 4 --target `
  llama-cli llama-server llama-bench test-dspark-device test-speculative-dspark `
  test-speculative-rejection-sampling test-llama-archs
if ($LASTEXITCODE -ne 0) { throw 'Build failed' }
```

## 运行最快的已测配置

先将下面两个模型路径改成实际路径。主模型与 draft 都选择 `Vulkan0`，F16 KV、context 8192、batch 512、ubatch 128、线程数 12、单序列。不要在测量中开启算子计时日志。

```powershell
$env:LLAMA_DSPARK_DEVICE = '1'
$env:LLAMA_DSPARK_DENSE_Q = '1'
$env:GGML_VK_DISABLE_COOPMAT = '1'
Remove-Item Env:LLAMA_DSPARK_DEVICE_VALIDATE -ErrorAction SilentlyContinue
Remove-Item Env:LLAMA_DSPARK_PROFILE -ErrorAction SilentlyContinue
Remove-Item Env:GGML_VK_PERF_LOGGER -ErrorAction SilentlyContinue
Remove-Item Env:GGML_VK_PERF_LOGGER_DETAIL -ErrorAction SilentlyContinue
Remove-Item Env:GGML_VK_Q4_K_UNROLL -ErrorAction SilentlyContinue

.\build-vulkan\bin\llama-server.exe `
  -m C:\models\Spark-X2.6-4B-Q4_K_M.gguf `
  -md C:\models\Spark-X2.6-4B-DSpark-Q8_0-backbone.gguf `
  --alias spark-test --device Vulkan0 --device-draft Vulkan0 -ngl 99 -ngld 99 `
  -t 12 -tb 12 -td 12 -tbd 12 `
  -ctk f16 -ctv f16 -ctkd f16 -ctvd f16 -fa on `
  -c 8192 -np 1 -b 512 -ub 128 --fit off `
  --cache-ram 0 --no-cache-idle-slots `
  --spec-type draft-dspark --spec-draft-sampling probabilistic `
  --spec-draft-n-max 7 --spec-draft-p-min 0 --no-spec-draft-backend-sampling `
  --temp 1 --top-p 0.95 --top-k 0 --min-p 0 `
  --repeat-penalty 1 --frequency-penalty 0 --presence-penalty 0 `
  --jinja --reasoning on --reasoning-effort low `
  --host 127.0.0.1 --port 8080
```

请求也应显式设置 `temperature=1`、`top_p=0.95`、`top_k=0`、`min_p=0`、中性 penalties、固定 seed、最大输出 512，关闭 prompt cache，并用 thinking enabled / reasoning_effort=low 渲染模板。该 full-vocabulary draft 路径使用请求温度，draft top_p 固定为 1；这里的配置因此是 draft T=1/P=1。复现某轮实验时使用该轮归档的输入及逐题 seed；全量429题和早期GSM38为不同口径。

启动日志应显示主模型 37/37、draft 6/6 层卸载到 GPU，以及 `DSpark Markov, categorical sampling and feature concatenation on Vulkan0`。仅看到 GPU 被枚举不等于所有这些操作已在 GPU 执行。

## 验证

```powershell
$env:GGML_VK_DISABLE_COOPMAT = '1'
.\build-vulkan\bin\test-dspark-device.exe Vulkan0
.\build-vulkan\bin\test-dspark-device.exe CPU
.\build-vulkan\bin\test-speculative-dspark.exe
ctest --test-dir build-vulkan -R '^test-generate-speculative-target$' --output-on-failure
if ($LASTEXITCODE -ne 0) { throw 'Test model generation failed' }
.\build-vulkan\bin\test-speculative-rejection-sampling.exe `
  -m .\build-vulkan\tests\test-speculative-models\target\llama-dense.gguf `
  -ngl 0 -t 1 --dense-q-only
if ($LASTEXITCODE -ne 0) { throw 'Dense-q replay failed' }
```

设备测试覆盖 5 层特征复制/重排、稀疏概率边界、64 与 131072 词表、7 步 Q8 Markov 条件链。模型检查可在启动 server 前额外设置 `LLAMA_DSPARK_DEVICE_VALIDATE=1`，检查概率归一化及采样 token 的 CDF 区间；完成后移除此变量再测速度。Windows 的 CMake 配置不提供 `test-sampling` 目标，该测试已在本地 CPU 构建中运行。实际加载库来自本次独立构建目录，运行库和两个模型的测后 SHA-256 均通过核验；详见 [验证记录](dspark-vulkan-validation.json)。

Dense-q replay 使用本地生成的小模型，比较2016组新旧表示的输出、候选概率及后续RNG轨迹，覆盖0/1/3/7行、无grammar及两种grammar顺序、全接受、首/中/尾拒绝、零概率和越界token。另检查131072词表存储的复用、截断、越界及尺寸溢出。它验证采样表示转换，不代替真实GPU模型正确性或吞吐测量。
