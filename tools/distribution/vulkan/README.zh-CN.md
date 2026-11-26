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

CLI 和 server 默认 `-b 512 -ub 512`。较大的 ubatch 可减少长提示的分批开销，也会增加计算缓冲和 SWA KV 缓存的预留内存；DSpark 的 draft 默认继承此上限。内存紧张时，可在 PowerShell 调用 `run.ps1` 时用 `-ExtraArgs @('-ub', '128')` 覆盖。更新启动脚本后，下次启动生效。 兼容 Arc 140T 的 token32 V4/V8 缓存默认启用 packed V prefill 和 mask tile 分类（9..1024 个 query，无 precision tail）；可设置 `LLAMA_VPLANE_PREFILL=0` 回退，或 `LLAMA_VPLANE_PREFILL_MASK_TILES=0` 仅关闭 prefill mask 分类。

## HTTP API

```bat
.\run-server.cmd "D:\models\Spark-X2.6-1.7B-Q4_K_M-Q4_1.gguf" -DraftModel "D:\models\Spark-X2.6-1.7B-DSpark-Q8_0-backbone.gguf" -Port 8080
```

`run-server.cmd` 默认以 Spark 1.7B、128K 上下文、12 线程、Vulkan0、D3 概率采样、Full Attention token32 K8V8 和 F16 SWA/draft KV 启动。模型别名为 `spark-x2.6-1.7b`，seed 为 0。默认配置启用六项 KV 驻留和三档预热。脚本设置 `GGML_VK_DISABLE_COOPMAT=1` 和三个 `LLAMA_DSPARK_*` 环境变量。省略模型路径时使用上面的两个 `D:\models` 文件。

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

发布包包含已渲染的主请求模板 `astudio_template\template.txt`、标题模板 `title.txt` 和切点元数据。`run-server.cmd` 默认先预热标题，再依次保存 **System 1**、**System 1 + System 2**、**System 1 + System 2 + User 1 的固定部分**。User 1 截到 `<environment_context>` 开头。第二、三档复用前一档，后续请求使用最长且滑窗覆盖有效的前缀。当前 1.7B 模型的参考长度分别是 **26,320、34,795、36,451** tokens，标题是 **2,388** tokens。

三档预热使用统一 KV，关闭 RAM 提示缓存、空闲槽缓存和 checkpoint。适用于包内默认模板、K8V8、普通解码或固定 D3。显式设置 RAM 缓存或 checkpoint、使用其他 KV 或草稿深度，或提供 `-ExtraArgs` 时，启动器使用单段预热。也可以用 `-SinglePrefill` 选择单段，或者用 `-PrefillFile` 指定自己的已渲染前缀。

```bat
.\run-server.cmd "D:\models\Spark-target.gguf" -DraftModel "D:\models\Spark-DSpark-Q8_0-backbone.gguf" -KVCache q8_0 -ContextSize 131072 -MaxTokens 131072 -PrefillFile "D:\inputs\astudio_template\template.txt"
```

启动器默认将 CRLF 规范为 LF。`-PrefillNewlines preserve` 保留原始换行。启动后先用 `/tokenize` 获取 token ID，校验三个主前缀逐 token 嵌套，再用 `/completion` 发送零输出预算的预热请求。终端逐档打印命中和补算量，显示 `HTTP API ready` 后可以接入客户端。启动及每次预热的默认超时为 1800 秒，可用 `-PrefillTimeoutSec` 调整。预热失败时启动器退出并回收本次启动的进程。

预热使用 `-Host` / `-ListenHost` 和 `-Port` 指定的地址、单 slot 和 `--cache-prompt`。单段模式沿用 RAM 缓存及 checkpoint 配置，默认 8192 MiB 和 32 个 checkpoint。`-SkipPrefill` 可直接启动 HTTP 服务。

客户端继续发送完整请求。使用 `/v1/chat/completions` 时，沿用原有的 `messages` 和 `tools`，让服务端按当前请求渲染。缓存复用要求 `cache_prompt=true`。启用的技能或固定工作区说明发生变化时，可命中较短的档位。

可用响应的 `timings.cache_n` 和 `timings.prompt_n` 检查命中及补算量。参考长度来自随包提供的模板和 Spark tokenizer。每次启动都会重新生成进程内 KV。

标准 Vulkan 打包器默认包含上述启动脚本、`astudio_template\template.txt`、模板元数据和说明，后续发布包完整解压后即可使用。打包资源源文件保存在 `tools/distribution/vulkan/astudio_template`，不依赖旧发布目录；打包时会核对模板身份及最终 ZIP 内容。若手动更新旧解压目录，需要同时替换 `run.ps1`、`run-server.cmd` 并复制 `astudio_template` 目录；原始清单中的脚本哈希不再适用于手动更新，新发布包提供完整的新清单。

## Spark DSpark 和量化 KV

使用与主模型匹配的 DSpark draft，示例路径换成自己的文件：

```bat
.\run-cli.cmd "D:\models\Spark-target.gguf" -DraftModel "D:\models\Spark-DSpark-Q8_0-backbone.gguf" -DraftMax 3
.\run-server.cmd "D:\models\Spark-target.gguf" -DraftModel "D:\models\Spark-DSpark-Q8_0-backbone.gguf" -DraftMax 3
```

target 和 draft 均使用 Vulkan；draft 使用 F16 KV、T=1 / top_p=1、概率采样。CLI 普通运行只需要一个主模型；server 默认启用上述配套 draft，使用其他模型时需要显式调整配置。最大草稿长度支持 1 至 7。

Spark D256/GQA4 的 HTTP 默认使用 token32 K8V8，滑动窗口层和 draft KV 使用 F16。CLI 默认使用 F16。可以显式选择其他 KV 类型，长上下文容量取决于模型和可用内存。

```bat
.\run-server.cmd "D:\models\Spark-target.gguf" -DraftModel "D:\models\Spark-DSpark-Q8_0-backbone.gguf" -KVCache q8_0 -ContextSize 32768 -SkipPrefill
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

发布验证范围见 `version.json` 和发布目录的 `RELEASE-REPORT.md`。
