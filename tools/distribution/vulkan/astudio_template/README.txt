AStudio 固定前缀

template.txt 和 title.txt 是 Vulkan 服务记录的实际渲染输入。文件使用 UTF-8 和 LF，
保留空白及角色标记。主模板和标题均停在环境段开头，环境内容和用户问题由客户端发送。

默认预热顺序为标题、System 1、System 1 + System 2、
System 1 + System 2 + User 1 的固定部分。参考 Spark-X2.6-1.7B，
标题为 2,388 tokens，三档主前缀分别为 26,320、34,795、36,451 tokens。
主请求的后两档复用前一档，正式请求使用最长且滑窗覆盖充分的驻留快照。

run-server.cmd 使用 /tokenize 获取原始 token ID，再向 /completion 发送零预算请求。
终端显示 HTTP API ready 后可连接客户端。-SinglePrefill 使用单段预热，
-PrefillFile 指定自己的已渲染前缀，-SkipPrefill 跳过预热。
完整参数和运行示例见 README.zh-CN.md。

客户端技能列表和工具配置发生变化时，固定前缀也会变化。
构建者更新模板后，须同步 template-info.json 中两个文件的哈希和字节数，
以及主模板的三个累计字节边界，并重新验证 token 前缀。
