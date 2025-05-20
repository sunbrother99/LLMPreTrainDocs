# qwen3技术报告

Qwen3-235B-A22B 在代码、数学、通用能力等基准测试中，与 DeepSeek-R1、o1、o3-mini、Grok-3 和 Gemini-2.5-Pro 等顶级模型相比，表现出极具竞争力的结果。此外，小型 MoE 模型 Qwen3-30B-A3B 的激活参数数量是 QwQ-32B 的 10%，表现更胜一筹，甚至像 Qwen3-4B 这样的小模型也能匹敌 Qwen2.5-72B-Instruct 的性能。

此次开源包括

两款MoE模型：Qwen3-235B-A22B（2350多亿总参数、 220多亿激活参），以及Qwen3-30B-A3B（300亿总参数、30亿激活参数）；

六个Dense模型：Qwen3-32B、Qwen3-14B、Qwen3-8B、Qwen3-4B、Qwen3-1.7B和Qwen3-0.6B。

Dense模型的参数：

![image](https://github.com/user-attachments/assets/d68c3859-7d65-4c2b-a449-150887814524)

MoE模型的参数：

![image](https://github.com/user-attachments/assets/bf083afc-7dd9-483b-922b-ca4b9d48063c)


Qwen3 MoE模型沿用了Qwen2.5-MoE的细粒度专家分割实现 。Qwen3 MoE模型共有128个专家，每token激活8个专家 。与Qwen2.5-MoE不同的是，Qwen3-MoE设计不包含共享专家 ，通过全局负载均衡损失强制专家专业化，提升任务特定性能。这些架构和训练上的创新显著提升了模型在下游任务中的性能 。

## 分词方式

词典：Qwen3模型使用了Qwen的tokenizer，该tokenizer实现了字节级字节对编码（BBPE），词汇量大小为151,669 。

```text
[原始文本]  你好
   ↓
[UTF-8 字节序列化]  [228, 189, 160, 229, 165, 189] #16进制表示
   ↓
[初始 symbol 表达：每个字节为一个 symbol]  ['e4', 'bd', 'a0', 'e5', 'a5', 'bd']
   ↓
[根据 merges.txt 规则进行 symbol 合并]  ['e4bda0', 'e5a5bd']
   ↓
[得到最终 token 序列（subword）]  ['你','好']
   ↓
[查 vocab.json 得到 token_id 序列]  [19489, 22459]
```

---

## Step-by-Step：详细流程拆解

### Step 0：加载 vocab.json 和 merges.txt

#### vocab.json

```json
{
  "你": 19489,
  "好": 22459,
  "world": 3293,
  "!": 38,
  ...
}
```

这是 token → id 的字典（最终输出的是 id）

#### merges.txt（带频率排序的合并规则）

```txt
# version: 0.2
e4 bd
bd a0
e4bd a0
e5 a5
...
```

这表示 **symbol 合并规则**，即哪两个 symbol 如果相邻，则合并成一个更高频的 symbol。

---

### Step 1：输入字符串 UTF-8 编码（字节表示）

```python
text = "你好"
utf8_bytes = text.encode('utf-8')
print(list(utf8_bytes))
# 输出：[228, 189, 160, 229, 165, 189]
# 即：\xe4\xbd\xa0\xe5\xa5\xbd
```

### Step 2：初始 symbol 表达（Byte-Level Tokenization）

将 UTF-8 每个字节转为 symbol 字符串：

```python
['\xe4', '\xbd', '\xa0', '\xe5', '\xa5', '\xbd']
```

或者按十六进制简写为：

```plaintext
['e4', 'bd', 'a0', 'e5', 'a5', 'bd']
```

此时每个字节是一个“symbol”单位。

---

### Step 3：根据 merges.txt 执行 BPE 合并

按 `merges.txt` 中的顺序，重复合并符号对：

初始状态：

```plaintext
['e4', 'bd', 'a0', 'e5', 'a5', 'bd']
```

执行合并规则：

1. `e4 + bd → e4bd`
2. `e4bd + a0 → e4bda0`（即 “你” 的 UTF-8 全字节合并）
3. `e5 + a5 → e5a5`
4. `e5a5 + bd → e5a5bd`（即 “好” 的 UTF-8 全字节合并）

得到：

```plaintext
['e4bda0', 'e5a5bd']
```

这些合并好的 symbol 称为 **最终 subwords**。

---

### Step 4：从 vocab.json 映射为 token id

token → id：

```json
{
  "你"      → 19489,
  "好"      → 22459
}
```

在 vocab 中，'你' 的 UTF-8 合并字符串 `e4bda0` 被映射为 token `'你'`，并对应某个 ID。

最终输出 token-id 序列：

```python
[19489, 22459]
```

---

## utf-8编码后的symbol如何通过BPE的方式合并为token(subword)的？

非常好，这个问题是理解 BPE（包括 Qwen 的 BBPE）编码的关键。我们下面将通过**具体机制 + 举例 + 伪代码**来完整解释 **第三步 merges.txt 的合并过程**。

---

## 背景问题简化重述：

你已经得到了输入句子的 UTF-8 字节序列，例如：

```plaintext
['e4', 'bd', 'a0', 'e5', 'a5', 'bd']  # “你好”
```

现在我们要按 merges.txt 来合并这串 symbol，变成：

```plaintext
['e4bda0', 'e5a5bd']  # 最终 token
```

**问题是：怎么合并？是怎么用 merges.txt 合并这串 symbol 的？是不是逐行遍历？**

---

## 答案概要：

 > **合并过程是基于 merges.txt 定义的 symbol 对合并优先级，迭代执行最高优先级的合并对，直到无法再合并为止。**

---

## Step-by-Step 举例说明

### 准备材料：

#### 初始 symbol 序列（UTF-8 byte 序列）：

```python
symbols = ['e4', 'bd', 'a0', 'e5', 'a5', 'bd']
```

#### merges.txt（前几行）：

```txt
# version: 0.2
e4 bd       → 合并为 e4bd
bd a0       → 合并为 bda0
e4bd a0     → 合并为 e4bda0
e5 a5       → 合并为 e5a5
a5 bd       → 合并为 a5bd
e5a5 bd     → 合并为 e5a5bd
```

### 🎯 目标：对 symbol 列表进行迭代合并，直到没有可以合并的对为止。

---

## 合并算法原理（经典 BPE 合并）

每一轮操作如下：

1. 遍历当前 symbol 列表中的 **所有相邻二元对（bigrams）**；
2. 在 merges.txt 中找到这些对中优先级最高的（即最靠前的）；
3. 合并这个对；
4. 回到第 1 步，直到不能再合并。

---

## 举例过程：以“你好”为例

初始 symbols：

```plaintext
['e4', 'bd', 'a0', 'e5', 'a5', 'bd']
```

### 第一次遍历相邻对：

```plaintext
('e4', 'bd') ✅ 在 merges.txt，且优先级最高
('bd', 'a0') ✅ 存在，但后面一点
('a0', 'e5')
('e5', 'a5') ✅ 存在
('a5', 'bd') ✅ 存在
```

合并 `'e4' + 'bd'` → `'e4bd'`（优先级最高）

更新序列：

```plaintext
['e4bd', 'a0', 'e5', 'a5', 'bd']
```

---

### 第二轮遍历相邻对：

```plaintext
('e4bd', 'a0') ✅ 存在 → 合并
('a0', 'e5')
('e5', 'a5') ✅
('a5', 'bd') ✅
```

合并 `'e4bd' + 'a0'` → `'e4bda0'`

更新：

```plaintext
['e4bda0', 'e5', 'a5', 'bd']
```

---

### 第三轮遍历：

```plaintext
('e4bda0', 'e5') ❌ 不在 merges
('e5', 'a5') ✅
('a5', 'bd') ✅
```

合并 `'e5' + 'a5'` → `'e5a5'`

更新：

```plaintext
['e4bda0', 'e5a5', 'bd']
```

---

### 第四轮遍历：

```plaintext
('e4bda0', 'e5a5')
('e5a5', 'bd') ✅ → 合并 → 'e5a5bd'
```

最终合并：

```plaintext
['e4bda0', 'e5a5bd']  ✅ “你”和“好”
```

---

## ⚙️ 合并伪代码（简化版）

```python
def bpe_merge(symbols, merge_ranks):
    while True:
        # 找所有相邻对
        pairs = [(symbols[i], symbols[i+1]) for i in range(len(symbols) - 1)]
        # 找可合并的对及其优先级
        candidates = [(pair, merge_ranks.get(pair)) for pair in pairs if pair in merge_ranks]
        if not candidates:
            break
        # 找优先级最高（rank最小）的
        best_pair, _ = min(candidates, key=lambda x: x[1])
        # 合并该对
        new_symbols = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i+1]) == best_pair:
                new_symbols.append(symbols[i] + symbols[i+1])
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        symbols = new_symbols
    return symbols
```

其中 `merge_ranks` 是从 merges.txt 读取并排序生成的 pair → rank 字典：

```python
merge_ranks = {('e4', 'bd'): 0, ('bd', 'a0'): 1, ...}
```

---

## ✅ 总结一句话：

> 合并过程是一个 **按 merges.txt 顺序定义优先级的、迭代贪心合并相邻 symbol 对的过程**。不是暴力全遍历，而是智能挑出“当前能合并的最优对”。

---




# 训练方式：预训练+后训练

## 预训练（Pre-training）：数据构建与三阶段策略

在预训练方面，Qwen3 的数据集相比 Qwen2.5 有了显著扩展。Qwen2.5是在 18 万亿个 token 上进行预训练的，而 Qwen3 使用的数据量几乎是其两倍，达到了约 __36 万亿个 token__ ，涵盖了 119 种语言和方言。为了构建这个庞大的数据集，我们不仅从网络上收集数据，还从 PDF 文档中提取信息。我们使用 Qwen2.5-VL 从这些文档中提取文本，并用 Qwen2.5 改进提取内容的质量。为了增加数学和代码数据的数量，我们利用 Qwen2.5-Math 和 Qwen2.5-Coder 这两个数学和代码领域的专家模型合成数据，合成了包括教科书、问答对以及代码片段等多种形式的数据。

### Qwen3模型采用三阶段预训练过程：

__第一阶段：基本语言知识和通用能力训练。__ 所有Qwen3模型使用4,096 token的序列长度，在超过30万亿token的数据上进行训练 。此阶段旨在建立模型的语言能力和通用世界知识基础，训练数据覆盖119种语言和方言 。

__第二阶段：复杂推理能力训练。__ 此阶段的模型使用4,096 token的序列长度，在约5万亿高质量token上进行进一步预训练，预训练数据包括STEM、编码、推理、数学和合成数据（知识密集型数据）等 。在此阶段还加速了学习率衰减 。

__第三阶段：长上下文能力训练。__ 此阶段将模型4K的输入长度扩展到32K，高质量长上下文语料中，75%的文本长度在16,384到32,768 token之间，25%的文本长度在4,096到16,384 token之间 。报告提及沿用Qwen2.5的做法，使用ABF技术将RoPE的基础频率从10,000提高到1,000,000 。同时，引入YARN和Dual Chunk Attention (DCA)技术，在推理过程中实现序列长度容量的四倍增长 。

## 后训练（Post-training）：四阶段能力精炼，实现逐步推理和快速响应

<img width="827" alt="image" src="https://github.com/user-attachments/assets/28cc873d-0a43-428c-8d8e-657dd5c3a229" />

为了开发能够同时具备思考推理和快速响应能力的混合模型，我们实施了一个四阶段的训练流程。该流程包括：（1）长思维链冷启动，（2）长思维链强化学习，（3）思维模式融合，以及（4）通用强化学习。

__第一阶段：长思维链冷启动。__ 使用多样的长思维链数据对模型进行微调（SFT），涵盖了数学、代码、逻辑推理和 STEM 问题等多种任务和领域。这一过程旨在为模型配备基本的推理能力。

__第二阶段：长思维链强化学习。__ 这一阶段的重点是大规模强化学习，利用基于规则的奖励来增强模型的探索和钻研能力，进一步提升模型的推理能力麻烦让模型能够更加有效的寻找最佳答案。

__第三阶段：思维模式融合。__ 我们在一份包括长思维链数据和常用的指令微调数据的组合数据上对模型进行微调（SFT），将非思考模式整合到思考模型中。确保了推理和快速响应能力的无缝结合。

__第四阶段：通用强化学习。__ 我们在包括指令遵循、格式遵循和 Agent 能力等在内的 20 多个通用领域的任务上应用了强化学习，以进一步增强模型的通用能力并纠正不良行为。

## Qwen3如何实现思考/不思考的的控制

### 硬开关：enable_thinking=True 启用思考模式，enable_thinking=False 禁用思考模式。

qwen3通过tokenizer.apply_chat_template的enable_thinking参数来实现思考模式和非思考模式的切换，默认情况下，qwen3启用了思考模式。

```python
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True  # True is the default value for enable_thinking.
)
```

在enable_thinking=True（思考模式）下，模型会生成包裹在\<think\>...\</think\> 块中的思考内容，然后是最终响应。
思考模式下，输入给模型的模板为
```text
<|im_start|>system
你是一名乐于助人的助手。<|im_end|>
<|im_start|>user
给我讲讲大语言模型。<|im_end|>
<|im_start|>assistant
```

```text
对于思考模式，官方提示请使用Temperature=0.6、TopP=0.95、TopK=20和MinP=0（ 中的默认设置generation_config.json）。
请勿使用贪婪解码，因为它会导致性能下降和无休止的重复。https://huggingface.co/Qwen/Qwen3-32B
```

在enable_thinking=False（非思考模式）下，在模式下，模型不会生成任何思考内容。
非思考模式下，输入给模型的模板为

```text

<|im_start|>system
你是一名乐于助人的助手。<|im_end|>
<|im_start|>user
给我讲讲大语言模型。<|im_end|>
<|im_start|>assistant
<think>

</think>

```


```text
对于非思考模式，我们建议使用Temperature=0.7、TopP=0.8、TopK=20和MinP=0。
```

**注意：** 思考模式下，对话模板与普通模板相同，没有任何变化。非思考模式下，对话模板会在<|im_start|>assistant 的后面添加一个空的 \<think\>\</think\>。在实际使用中，用户输入只会影响到<|im_start|>user\n给我讲讲大语言模型。<|im_end|> 这一段，从<|im_start|>assistant开始的内容都是模型应该要生成的内容，整个Qwen3控制混合思考切换的流程为：

1、首先，Qwen3 会默认思考，也就是生成 \<think\> ... \</think\> 的内容。

2、如果我们不想让模型思考，我们只需要提前“注入”一段空白的思考内容，也就是 \<think\>「空白」\</think\>，让模型认为「思考」这个过程已经结束了，接下来都是普通回复。

3、这样就完成了混合思考的启停。

### 软开关 enable_thinking=True 启用思考模式时，通过用户输入，在思考模式和费思考模式之间切换。

实现方式：在用户prompt或system prompt中添加/think或/no_think,实现在不同轮输入之间切换模型的思维模式。在多回合对话中，模型将遵循最后一条的指令。

为了实现API层级的兼容，当enable_thinking=True时，无论用户使用/think还是/no_think，模型都会始终输出一个包裹在\<think\>...\</think\>中的块。当用户输入/think时，模型输出的\<think\>...\</think\>块中为正常的思考内容，当用户输入/no_think时，模型仍然输出包含\<think\>\</think\>的空白思考块。此方案为Qwen3的「空白思考注入」方案。

首先，我们先按照如今通用训练思考模型的方式，训练出一个会思考的模型。接下来，我们只需要在训练中设计这样一套数据：加了 /think 提示的，对应回复就是有思考内容的；加了 /no_think 提示的，对应回复就是思考内容为空白的。这样模型就能够学会响应软提示了。

正如Qwen3技术报告中的后训练的第三个阶段，就是用来训练模型思考模式的软控制的。

## 思考、非思考模式切换示例

```pyton
from transformers import AutoModelForCausalLM, AutoTokenizer

class QwenChatbot:
    def __init__(self, model_name="Qwen3-30B-A3B/Qwen3-30B-A3B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.history = []

    def generate_response(self, user_input):
        messages = self.history + [{"role": "user", "content": user_input}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt")
        response_ids = self.model.generate(**inputs, max_new_tokens=32768)[0][len(inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # Update history
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})

        return response

# Example Usage
if __name__ == "__main__":
    chatbot = QwenChatbot()

    # First input (without /think or /no_think tags, thinking mode is enabled by default)
    user_input_1 = "How many r's in strawberries?"
    print(f"User: {user_input_1}")
    response_1 = chatbot.generate_response(user_input_1)
    print(f"Bot: {response_1}")
    print("----------------------")

    # Second input with /no_think
    user_input_2 = "Then, how many r's in blueberries? /no_think"
    print(f"User: {user_input_2}")
    response_2 = chatbot.generate_response(user_input_2)
    print(f"Bot: {response_2}") 
    print("----------------------")

    # Third input with /think
    user_input_3 = "Really? /think"
    print(f"User: {user_input_3}")
    response_3 = chatbot.generate_response(user_input_3)
    print(f"Bot: {response_3}")
```






