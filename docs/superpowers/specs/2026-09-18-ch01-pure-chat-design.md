# MewHelp Ch01 设计:纯对话客服(SSE 流式 + 结构化提取)

## 目标

电商智能客服系统第一章:跑通纯对话。不做工具调用、不做 Agent 循环。

验收标准:
1. curl 调对话接口能看到流式回复(逐 token)。
2. 连续问两轮,第二轮能接住第一轮的上下文。
3. 发一段售后描述,能拿到结构化 JSON(订单号/诉求类型/期望方案)。

## 技术栈(定死,不可自行更换)

- Python 3.12 + FastAPI + LangChain
- 模型接入直连上游,应用侧统一说 OpenAI 协议;地址、模型名、密钥全在 `.env`,分 `CHAT_*` / `EMBED_*` / `RERANK_*` 三组(本章只用聊天这一组)
- 依赖管理:uv
- 验收模型:DeepSeek 官方,`base_url=https://api.deepseek.com/v1`,模型 `deepseek-chat`,token 只放 `.env` 不进 git

## 架构:单进程

一条命令(`make dev` 或 `scripts/dev.sh`)拉起 FastAPI 应用(:8000)。LangChain 的 `ChatOpenAI` 直接指向 `settings.chat_base_url`,换上游改 `.env` 里两行,不动代码。

换模型时注意 `CHAT_MODEL` 要写上游认的真实模型名。这一步没有别名可用,写错了不会在启动时报错,而是第一次调模型时回一句 `Model does not exist`。

## 目录结构

```
app/
  main.py              # FastAPI 入口 + 应用日志配置
  config.py            # pydantic-settings 读 .env
  api/chat.py          # POST /api/chat (SSE)
  api/extract.py       # POST /api/extract
  core/llm.py          # ChatOpenAI 工厂
  core/prompts.py      # 所有 PromptTemplate 集中在此
  core/memory.py       # 内存会话存储 + token 预算裁剪
  schemas/             # pydantic 请求/响应/结构化提取模型
scripts/dev.sh
tests/
dev-notes/ch01.md
.env / .env.example
```

## 接口契约

### POST /api/chat(SSE 流式对话)

请求:

```json
{"session_id": "abc", "message": "你们家猫粮发什么快递?"}
```

响应:`text/event-stream`,逐 token 推送:

```
data: {"delta": "我"}
data: {"delta": "们"}
...
data: [DONE]
```

- 历史按 `session_id` 存服务端内存 dict(本章不做持久化,重启丢历史,接受)。
- 每次调用前用 LangChain `trim_messages` 按 token 预算裁剪(默认 2000 token,可经 `.env` 配 `TOKEN_BUDGET`),策略:保 system prompt + 从最近往前保留完整轮次,保证从 HumanMessage 开始。
- 上游 LLM 出错:SSE 推 `event: error` + 错误信息后关流。**不能抛 HTTP 500**——响应头在第一帧发出时已定。

### POST /api/extract(结构化提取)

请求:

```json
{"text": "我上周买的猫爬架,订单号 MH20260701123,到货就散架了,我要退款"}
```

响应(`with_structured_output` 绑定 pydantic 模型 `AfterSalesTicket`):

```json
{"order_id": "MH20260701123", "request_type": "退款", "expected_solution": "商品到货损坏,要求退款"}
```

- `request_type` 枚举:退款 / 换货 / 维修 / 投诉 / 其他。
- 订单号缺失返回 `null`,不臆造。
- 上游失败返回 502 JSON。

## Prompt 管理

`core/prompts.py` 集中管理两个 `ChatPromptTemplate`:

1. **客服对话 prompt**:system 写清角色设定(「喵喵优选」电商客服「小喵」)与行为约束——不臆造订单/物流信息、本阶段无查询系统权限(涉及具体订单状态明说转人工、不承诺赔偿时效)、超范围问题引导回购物话题、语气规范。
2. **提取任务 prompt**:面向售后描述的字段提取指令,明说订单号原文没有则为 null、诉求类型五选一、方案忠于原文。

角色名统一写 `system`(LangChain 的统一叫法),多轮历史用 `MessagesPlaceholder("history")` 整段展开,保持消息结构。

## 结构化输出与 DeepSeek 兼容

`with_structured_output(AfterSalesTicket, method="function_calling")`,显式指定 method,并关闭思考模式(`extra_body={"thinking": {"type": "disabled"}}`)。

**实测发现(验收阶段,已回写)**:langchain-openai 的 `with_structured_output` 默认 method 实为 `json_schema`(发 `response_format`),DeepSeek 报 `This response_format type is unavailable now`;直接 `function_calling` 被 `deepseek-v4-pro` 的思考模式挡(`Thinking mode does not support this tool_choice`);`json_mode` 能返回 JSON 但漏 `expected_solution` 字段。最终 **`function_calling` + 关 thinking** 三字段齐全、枚举正确,5/5 标注样例通过。

`order_id` 加 `field_validator(mode="before")`,把 `""`/`"null"`/`"none"`/`"n/a"`/`"无"` 等占位文本归一为 `None`——这是课程 1.3 源码走读里点名的真实模型行为,两道防线(prompt 劝 + validator 兜)都要。

## 会话与 token 裁剪

- `SessionStore`:session_id -> 消息列表,进程内存 dict,`get` 不存在返回空列表、`append` 追加。
- `trim_history(messages, max_tokens)`:调 `trim_messages(strategy="last", token_counter=count_tokens_approximately, max_tokens=..., start_on="human", allow_partial=False)`。system 由 prompt 模板负责,不进历史。

## 测试与验证

按"可单测代码走 TDD、纯 Prompt 产出走标注样例验证"拆两类:

**TDD(pytest,先红后绿,不打真实模型):**
- `core/memory.py`:会话存储、token 预算裁剪逻辑
- SSE 组帧(用 FakeLLM / mock)
- schema 校验(AfterSalesTicket 枚举、可空字段、占位符归一)
- config 必填字段、llm 工厂指向配置

**标注样例验证(真实模型,Task 8):**
- 5 条标注售后描述跑 `/api/extract` 核对 order_id 与 request_type,覆盖 edge case:订单号缺失、诉求模糊、诉求类型"其他"。
- 两轮对话脚本验证第二轮接住第一轮上下文。

## 本章不做

- 工具调用、Agent 循环
- 会话持久化(重启丢历史,接受)
- 鉴权、限流、多副本
- 对话中自动触发提取(提取是独立接口)
- 聊天网页前端(本章验收全走 curl;前端另起一轮 Vibe Coding)

## 关键决策记录

| 决策 | 选择 | 备选与理由 |
|---|---|---|
| 上游接入形态 | 应用直连,配置分三组(本章用 CHAT) | 中间垫网关只为归一 rerank 格式,多一个进程、多一份配置、换模型改两处 |
| 验收模型 | DeepSeek 官方 deepseek-chat | 用户确认;应用侧 OpenAI 协议,换模型只改 .env |
| 会话管理 | 服务端 session_id 内存 dict | 客户端全量 history 对 curl 验收不友好 |
| 提取形态 | 独立 /extract 接口 | 融合进对话需意图判断,超出本章"纯对话"范围 |
| 结构化输出方法 | function_calling + 关 thinking(显式 method) | 实测 json_schema 默认不可用、function_calling 被思考模式挡、json_mode 漏字段;用户确认 function_calling+关 thinking |
| 推理模型适配 | extract 关 thinking,chat 保持默认 | deepseek-v4-pro 思考模式下不支持 tool_choice;对话流式不受影响 |
| 依赖管理 | uv + Python 3.12 | uv 已装,课程约定 |
| 前端 | 本章不做 | 验收标准全 curl;用户确认范围最小优先 |
