# llama.cpp Spark Intel GPU Edition

<p align="center">
  <img src="assets/spark-llama-intel-gpu-logo.png" alt="llama.cpp Spark Intel GPU Edition" width="640">
</p>

**Windows x64 Vulkan 便携发布包：面向 Intel 显卡特别优化的 Spark / DSpark 推理版本。**

适用于 Ultra7 的 Windows 11 x64 / Core Ultra 7 255H / Intel Arc 140T。源码来自 `llama_cpp_vulkan_opt`，具体提交和文件 SHA-256 见 `version.json`。

将整个 ZIP 解压到本地目录，准备好 GGUF 模型即可使用。无需 Python、CMake、Visual Studio、Vulkan SDK 或额外安装 VC++ 运行库。电脑需要已安装支持 Vulkan 的显卡驱动；发布包不包含显卡驱动，也不包含模型。请保留完整目录结构。

## 开始使用

在解压目录的终端执行，模型路径可在任意磁盘，带空格时保留双引号：

```bat
.\check-vulkan.cmd
.\run-cli.cmd "D:\models\your-model.gguf"
```

设备列表应出现 `Vulkan0: Intel(R) Arc(TM) 140T`。CLI 会进入交互对话；也可双击 `run-cli.cmd`，按提示输入模型路径。单次提问：

```bat
.\run-cli.cmd "D:\models\your-model.gguf" -Prompt "请介绍一下自己"
```

UTF-8 文本文件输入，或调整上下文、生成上限：

```bat
.\run-cli.cmd "D:\models\your-model.gguf" -PromptFile "D:\inputs\prompt.txt" -ContextSize 16384 -MaxTokens 1024
```

CLI 默认上下文 8192、12 个 CPU 线程、最大生成 512 token（含思考）、Vulkan0、最多卸载 99 层、Flash Attention 开启、F16 KV。`-MaxTokens -1` 允许生成到模型自然结束或上下文耗尽。默认使用 thinking low、target T=1 / top_p=0.95。脚本为 Arc 140T 设置 `GGML_VK_DISABLE_COOPMAT=1`。

## HTTP API

```bat
.\run-server.cmd "D:\models\Spark-X2.6-1.7B-Q4_K_M-Q4_1.gguf" -DraftModel "D:\models\Spark-X2.6-1.7B-DSpark-Q8_0-backbone.gguf" -Port 8080
```

`run-server.cmd` 使用 Spark 1.7B server 配置：默认模型路径为上面的两个 `D:\models` 文件，模型名 `spark-x2.6-1.7b`、128K 上下文及输出上限、12 线程、Vulkan0、D3 概率采样、Full Attention token32 K8V8、SWA/draft F16、seed 0、8192 MiB RAM 提示缓存及 idle slot 缓存、32 个 checkpoint、最小间隔 8192。脚本自动设置 `GGML_VK_DISABLE_COOPMAT=1` 和三个 `LLAMA_DSPARK_*` 环境变量。省略模型路径时采用上述默认值；模型位置不同则显式传 `-Model` / `-DraftModel`。

可通过 `-ModelAlias`、`-ContextSize`、`-MaxTokens`、`-Threads`、`-DraftMax`、`-KVCache`、`-CacheRamMiB`、`-NoCacheIdleSlots`、`-ContextCheckpoints`、`-CheckpointMinStep`、`-Seed`、`-Port` 调整。默认参数保存在同目录 `run.ps1` 的 server 分支；CLI 保留原来的 8192 上下文、512 输出和 F16 KV 默认值。

默认只监听本机 `127.0.0.1:8080`。局域网访问可使用 `-Host 0.0.0.0 -Port 8080`，也支持同义参数 `-ListenHost`；指定某个本机 IP 时仅监听该地址。参数接受 IPv4 或 IPv6 地址。按 Ctrl+C 结束。此包提供 HTTP API，不含网页聊天界面。

```bat
.\run-server.cmd -Model "D:\models\Spark-X2.6-1.7B-Q4_K_M-Q4_1.gguf" -DraftModel "D:\models\Spark-X2.6-1.7B-DSpark-Q8_0-backbone.gguf" -Host 0.0.0.0 -Port 8080
```

局域网客户端使用 `http://<服务机器的局域网 IP>:8080/v1`。监听所有 IPv4 地址时，脚本仍通过 `127.0.0.1` 完成预热；监听 `::` 时使用 `::1`，指定具体 IP 时使用该地址。在另一个 PowerShell 窗口发送请求：

```powershell
$body = @{
    model = 'spark-x2.6-1.7b'
    messages = @(@{ role = 'user'; content = '请介绍一下自己' })
    max_tokens = 512
    temperature = 1
    top_p = 0.95
} | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri 'http://127.0.0.1:8080/v1/chat/completions' -Method Post -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($body))
```

### 启动时预热固定前缀

发布包自带 `astudio_template\template.txt`。HTTP server 默认读取该文件预热，路径相对于 `run.ps1`，与当前工作目录无关。脚本读取 UTF-8，默认把 CRLF 规范为 LF，以匹配客户端渲染的模板；其余空白和特殊消息标记保留，不会再次套用聊天模板。需要保留文件换行时使用 `-PrefillNewlines preserve`。直接运行 `run-server.cmd` 即会启动并预热；模型不在默认位置时显式指定模型路径。自定义前缀时使用 `-PrefillFile`：

```bat
.\run-server.cmd "D:\models\Spark-target.gguf" -DraftModel "D:\models\Spark-DSpark-Q8_0-backbone.gguf" -KVCache q8_0 -ContextSize 131072 -MaxTokens 131072 -PrefillFile "D:\inputs\astudio_template\template.txt"
```

脚本启动服务，轮询 `/health`，就绪后先调用 `/tokenize`（`parse_special=true`、`add_special=false`），再向 `/completion` 发送 token ID 数组、`id_slot=0`、`n_predict=0`、`cache_prompt=true`。这样避免对已渲染的前缀额外添加 BOS 或聊天模板。此引擎的零预算响应可能包含一个未追加到 KV 的采样 token，脚本丢弃预热响应正文，仅校验预算和已计算的输入 token 数。终端显示 `Prefill complete: ... tokens in slot 0` 和 `HTTP API ready` 后再接入工作软件。默认等待启动及单次 prefill 的超时各为 1800 秒，可用 `-PrefillTimeoutSec` 调整。预热失败会报错并清理本脚本启动的服务；端口已被占用时直接报错，不会向已有服务发送前缀。

预热时使用 `-Host` / `-ListenHost` 配置的监听地址、`-Port` 指定的端口、单 slot 和 `--cache-prompt`；这些选项优先于 `-ExtraArgs`。RAM 缓存配置保持不变，默认仍是 `--cache-ram 8192 --cache-idle-slots --ctx-checkpoints 32 --checkpoint-min-step 8192`。不传 `-PrefillFile` 时加载包内默认模板；加 `-SkipPrefill` 才会跳过预热。两个参数不要同时使用。

后续工作软件仍须发送完整输入，模板渲染后的开头应与文件中的 token 前缀一致；仅发送“你好”不会自动补上这份前缀。若使用 `/v1/chat/completions`，按原来的结构提供 `messages` 和 `tools`，不要将整个渲染后的文件再塞进一条 system 消息。不要发送 `cache_prompt=false`。当前捕获的 P 约 25k tokens，server 默认 128K 上下文可以容纳；输入与输出合计仍受上下文容量限制。

验证缓存命中时检查响应的 `timings.cache_n` 和 `timings.prompt_n`，前者应接近固定前缀的 token 数；文件中的捕获统计仅作参考，以实际运行的 tokenizer 和响应为准。动态时间、路径、工具定义或换行差异会缩短命中前缀；SWA 模型回退时可能从 checkpoint 重算少量尾部。其他前缀覆盖 slot、清空缓存或上下文滑动也可能影响复用。这是进程内预热，每次重启都会重新计算一次，不保存磁盘 KV 文件。

标准 Vulkan 打包器默认包含上述启动脚本、`astudio_template\template.txt`、模板元数据和说明，后续发布包完整解压后即可使用。打包资源源文件保存在 `tools/distribution/vulkan/astudio_template`，不依赖旧发布目录；打包时会核对模板身份及最终 ZIP 内容。若手动更新旧解压目录，需要同时替换 `run.ps1`、`run-server.cmd` 并复制 `astudio_template` 目录；原始清单中的脚本哈希不再适用于手动更新，新发布包提供完整的新清单。

## Spark DSpark 和量化 KV

使用与主模型匹配的 DSpark draft，示例路径换成自己的文件：

```bat
.\run-cli.cmd "D:\models\Spark-target.gguf" -DraftModel "D:\models\Spark-DSpark-Q8_0-backbone.gguf" -DraftMax 3
.\run-server.cmd "D:\models\Spark-target.gguf" -DraftModel "D:\models\Spark-DSpark-Q8_0-backbone.gguf" -DraftMax 3
```

target 和 draft 均使用 Vulkan；draft 使用 F16 KV、T=1 / top_p=1、概率采样。CLI 普通运行只需要一个主模型；server 默认启用上述配套 draft，使用其他模型时需要显式调整配置。最大草稿长度支持 1 至 7。

Spark D256/GQA4 模型可显式选择 `-KVCache q8_0` 使用 token32 K8V8；滑动窗口层和 draft KV 仍为 F16。token32 KV 已由本版本默认启用，`q4_0` 也可选，但量化 KV 会引入误差，默认仍使用 F16。其他模型请先使用 F16；长上下文是否能加载取决于模型和可用内存。

```bat
.\run-server.cmd "D:\models\Spark-target.gguf" -DraftModel "D:\models\Spark-DSpark-Q8_0-backbone.gguf" -KVCache q8_0 -ContextSize 32768
```

## 原生程序与文件校验

`bin` 中提供 `llama-cli.exe`、`llama-server.exe`、`llama-bench.exe` 和 `vcomp140.dll`。可以直接调用原生程序并通过 `--help` 查看完整选项。原生调用需自行设置环境和参数，例如：

```bat
set GGML_VK_DISABLE_COOPMAT=1
.\bin\llama-bench.exe -m "D:\models\your-model.gguf" -dev Vulkan0 -ngl 99 -fa on -ctk f16 -ctv f16 -t 12 -p 512 -n 128 -r 3
```

PowerShell 高级参数示例：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -Command "& .\run.ps1 -Model 'D:\models\your-model.gguf' -Prompt '你好' -ExtraArgs @('--seed','42')"
```

在解压目录执行以下命令可核对 `version.json` 记录的所有文件：

```powershell
$v = Get-Content .\version.json -Raw | ConvertFrom-Json
foreach ($f in $v.files) {
    $p = Join-Path $PWD $f.path
    if ((Get-Item -LiteralPath $p).Length -ne $f.bytes -or (Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLowerInvariant() -cne $f.sha256) { throw "文件校验失败: $p" }
}
'所有文件校验通过'
```

若提示缺少 `vcomp140.dll`，请完整解压并检查 `bin` 中的 DLL；若提示缺少 `vulkan-1.dll` 或找不到设备，请安装或修复 Intel 显卡驱动。日志应显示 Vulkan0 和 GPU 卸载层数。模型加载内存不足时降低 `-ContextSize`，并关闭占用内存的应用。

本次发布验证范围见 `version.json` 和发布目录的 `RELEASE-REPORT.md`。本次未执行模型生成或性能基准测试。
