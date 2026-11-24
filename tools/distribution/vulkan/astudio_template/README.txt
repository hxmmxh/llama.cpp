AStudio 固定前缀（发布包默认资源）

run-server.cmd 默认读取本目录的 template.txt 并预热，路径相对于脚本所在目录。
模板不含最后一条具体提问和模型生成前缀；保留前面的固定系统、工具及指令内容。
模板止于 </INSTRUCTIONS><environment_context>，只保留环境段的开头标记。
其后的 cwd、shell、current_date、timezone、filesystem/权限等环境内容全部排除。
客户端仍发送原来的完整请求，环境内容作为未预热的尾部继续处理。
文件保留 UTF-8 字节及 CRLF，不添加末尾换行。
运行时默认把 CRLF 规范为 LF；需要原换行时加 -PrefillNewlines preserve。
参考 Spark-X2.6-1.7B tokenizer：LF 为 25080 tokens，原 CRLF 为 25671 tokens；其他模型以实际统计为准。
脚本先经 /tokenize 获取 token ID，再用 n_predict=0 预热，不重复添加 BOS 或聊天模板。
引擎零预算响应中的单次采样结果会被丢弃，不追加到 KV。

参数：
-Host 0.0.0.0 -Port 8080：允许局域网访问（默认仅 127.0.0.1）。
-PrefillFile "D:\inputs\another-template.txt"：使用其他模板。
-SkipPrefill：跳过启动预热。
-PrefillTimeoutSec：启动及单次预热超时，默认 1800 秒。

看到 Prefill complete 和 HTTP API ready 后再接入工作软件。
后续请求仍需发送完整上下文，只有共同 token 前缀可以复用 KV。
模板中的技能路径及工具说明保留本次捕获值，不会自动更新；环境段中的日期和权限不存放于模板。

当前固定快照来自 2026-11-24 的实际 AStudio 请求，sites-skills 路径版本为 1.0.17。
已剔除环境段内容、最后的“你好”及其消息/回复生成标记。
已用服务的 apply-template 和 tokenize 接口验证：模板 LF 的 25080 个 token
与该次 25216-token 完整输入的开头逐 token 一致；剩余 136 tokens 包括环境内容、用户消息和生成标记。
这项检查不执行模型生成，也不改变运行中服务的 KV。
客户端的技能版本或其他固定字段更新后，需要重新捕获并验证；仅保留相同 token 数不能保证命中。

构建者更新模板后，须同步 template-info.json 的 bytes 和 sha256，打包器会校验。
此目录已受本仓库 Git 跟踪，是发布模板的唯一源文件；后续发布不依赖 dist/astudio_template。
