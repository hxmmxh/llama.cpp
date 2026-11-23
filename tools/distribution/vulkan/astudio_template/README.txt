AStudio 固定前缀（发布包默认资源）

run-server.cmd 默认读取本目录的 template.txt 并预热，路径相对于脚本所在目录。
模板不含具体用户消息和模型生成前缀。文件保留原始 UTF-8 字节及 CRLF。
运行时默认把 CRLF 规范为 LF；需要原换行时加 -PrefillNewlines preserve。
参考 Spark-X2.6-1.7B tokenizer：LF 为 25209 tokens，原 CRLF 为 25806 tokens；其他模型以实际统计为准。
脚本先经 /tokenize 获取 token ID，再用 n_predict=0 预热，不重复添加 BOS 或聊天模板。
引擎零预算响应中的单次采样结果会被丢弃，不追加到 KV。

参数：
-Host 0.0.0.0 -Port 8080：允许局域网访问（默认仅 127.0.0.1）。
-PrefillFile "D:\inputs\another-template.txt"：使用其他模板。
-SkipPrefill：跳过启动预热。
-PrefillTimeoutSec：启动及单次预热超时，默认 1800 秒。

看到 Prefill complete 和 HTTP API ready 后再接入工作软件。
后续请求仍需发送完整上下文，只有共同 token 前缀可以复用 KV。
模板中的日期、路径、权限、技能及工具说明保留本次捕获值，不会自动更新。

构建者更新模板后，须同步 template-info.json 的 bytes 和 sha256，打包器会校验。
此目录为打包资源源文件，后续发布不依赖 dist/astudio_template。
