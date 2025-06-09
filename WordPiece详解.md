## 什么是BPE（byte pair encoding）?

BPE最开始是一种文本的数据压缩算法，其核心是基于字符对频率统计的字符合并算法，分为两个阶段，训练阶段和推理应用阶段。

### 训练阶段：

经典 BPE 的基本流程简述

1. 输入语料（已分词）→ ["low", "lowest", "newer", "wider", ...]

2. 统计词频 → { "low": 5, "lowest": 2, ... }

3. 把每个词拆成字符序列 → ["l", "o", "w"] + 词尾标记（如 </w>）

4. 统计所有相邻 token（字符/子词）对的频率

5. 选择频率最高的一对，合并为新 token

6. 更新所有词中的 token

7. 重复步骤 4~6，直到达到设定的词表大小

### 推理使用阶段

BPE 的推理阶段是一个“基于 merge 合并规则的多轮迭代过程”，每轮按规则合并 token 对，直到无法再合并为止。

输入一段文本 → 文本标准化（大小写转化，编码规范化，去掉非打印字符等）[此步骤可选，试分词器而定] → 预切词（Pre-tokenization）→ 单词级别拆为字符序列 → 迭代式合并（Merge Loop）→ 输出 token（子词）序列 + 对应 ID

其中：

**单词级别 → 拆为字符序列**:1.把每个 token（word 或 byte）拆分为单字符/字节形式 （与训练阶段不同，推理阶段的预处理不需要加</w>后缀。）

**迭代式合并（Merge Loop）**:

根据训练阶段生成的 merges.txt 中的合并对（有序列表），依次尝试合并字符对，有的模型文件中没有显示的存在merges.txt文件，合并规则集成到API内部。如（bert）
,qwen系列则在模型文件中有merges.txt文件，二者都是按照合并规则来进行分词（token）的。

每一轮只能执行一个合并：把频率最高的 pair 替换成新 token

重复迭代，直到当前 token 序列中没有任何可合并的对

```python
def apply_bpe(text, merge_ranks, vocab):
    # 初始切分：按字符分
    tokens = list(text)
    print(f"\n初始 token 序列: {tokens}")

    while True:
        # 构造所有相邻 token pair
        pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens) - 1)]

        # 找出所有可以合并的 pair 且在 merge 表中出现的
        candidates = [(pair, merge_ranks[pair]) for pair in pairs if pair in merge_ranks]
        if not candidates:
            break

        # 选择优先级最高的 pair 进行合并
        best_pair = min(candidates, key=lambda x: x[1])[0]
        print(f"合并 pair: {best_pair}")

        # 执行合并
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                new_tokens.append(tokens[i] + tokens[i + 1])  # 合并为一个 token
                i += 2  # 跳过下一个
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
        print(f"当前 token 序列: {tokens}")

    # 可选：只保留 vocab 中合法 token（或 fallback 为 [UNK]）
    tokens = [tok if tok in vocab else '[UNK]' for tok in tokens]
    print(f"\n最终输出 token: {tokens}")
    return tokens
text = "unwanted"

# 模拟 merges.txt：pair → 合并顺序优先级（越小优先级越高）
merges = [
    ('u', 'n'),
    ('un', 'w'),
    ('w', 'a'),
    ('a', 'n'),
    ('t', 'e'),
    ('te', 'd'),
    ('n', 't'),
]

merge_ranks = {pair: i for i, pair in enumerate(merges)}

# 模拟 vocab（合并后合法 token）
vocab = {
    'u', 'n', 'w', 'a', 't', 'e', 'd',
    'un', 'wa', 'an', 'te', 'ted',
    'unw', 'want', 'ed', 'unwanted'
}

apply_bpe(text, merge_ranks, vocab)
```

| 步骤       | 内容                           |
| -------- | ---------------------------- |
| 合并策略     | 每轮找 merge rank 最小的 pair      |
| 终止条件     | 没有可合并的 token pair            |
| vocab 检查 | 只保留合法 token，其余变成 `[UNK]`（可选） |

### BPE推理阶段的关键特点

| 特点          | 说明                       |
| ----------- | ------------------------ |
| ✅ **非贪心**   | 不选最长匹配，而是按合并顺序执行         |
| ✅ **迭代式合并** | 每轮合并一个 pair，直到不能合并       |
| ✅ **顺序敏感**  | 合并次序固定，不能乱序执行            |
| ✅ **稳定输出**  | 相同输入总是分词一致，便于缓存          |
| ✅ **高效**    | 一般几十轮以内完成全部合并，数据结构支持快速合并 |



## 什么是wordpiece

WordPiece 最早由 Schuster 和 Nakajima 在 2012 年提出，用于日语和韩语的语音识别任务。2018 年，Google 将其应用于 BERT，成为该模型的分词核心。
WordPiece 的基本思想与 BPE 类似：从字符级别开始，逐步构建子词词汇表。但与 BPE 不同的是，WordPiece 在合并子词时不依赖频率，而是通过一个基于似然概率的评分函数来选择最优的子词单元。

简单来说，WordPiece 的目标是：**在给定语料库上，找到一组子词，使得这些子词组合成的语言模型具有最高的似然概率**。 这种方法引入了语言模型的视角，使得分词结果更贴近语言的实际分布，而不仅仅是机械地合并高频字符对。

wordpiece是一种分词方式，google提出的bert模型中使用这种分词方法。

**google最初版的BERT是基于wordpiece的分词，分词后的训练语料是子词形式的token序列，做mask时会把一个完整的词只mask掉其中的子词，剩余部分不mask,这就导致模型的学习任务较简单。后来的改进版做了全词mask(whole word mask,WWM),子词mask的同时，子词所在的整个word都会被msk,现在所说的bert都是指的全词mask版本的bert。**

谷歌中文版BERT在分词器方面也是使用的wordpiece,所有**中文版的wordpiece的训练是基于字的**，可以看到中文版bert的vocab中的中文都是单独的汉字，没有中文词语。并且中文版bert没有做全词mask,后来Google出的改进版bert,也没有出中文版本。**

哈工大训练的中文版bert，其词典与Google出的bert-base-chinese用的同一套词典，其wordpiece也是基于单个字的切分，但是哈**工大在训练中文bert时，用哈工大的分词工具对语料做了分词，以便训练阶段做全词mask**，所有哈工大的中文bert是bert-wwm-chinese. 

哈工大 BERT-wwm-ext-Chinese: https://github.com/ymcui/Chinese-BERT-wwm

---

**wordpiece分为两步：1、如何得到vocab词表？ 2、得到词表后如何应用wordpiece分词？**

### wordpiece如何得到vocab词表？（训练阶段）

得到词表vocab的过程就是wprdpiece的训练过程，分为以下几个步骤：

1、初始化

  1）准备一个大型语料库，对语料库中的文本按照空格进行切分，得到单词列表。
  
  2）将每个单词拆分为字符序列，并统计其频率。比如：
  
  ```text
  "low": l o w, 5次
  "lower": l o w e r, 3次
  "new": n e w, 4次
  ```
  
  3)初始化词汇表，包含所有单个字符，如 l,o,w,e,r,n，最开始的词汇表是字母，中文则是汉字。与BPE相同，词汇表中包含语料库中的所有唯一字符，但处理方式略有不同：对于每个单词，除了首个字符外，其他字符前都要加上##前缀。

2、构建初始语言模型

  1）使用当前词汇表对语料库进行分词（初始时词汇表为字符级别）。
  
  2）训练一个简单的语言模型（一般是unigram基于似然概率的语言模型），计算每个字符的似然概率，比如基于字符的概率分布（P(l),P(o),P(w),...）

3、候选子词评分

  1)遍历语料库，生成所有可能的相邻字符对（如lo,ow）
  
  2)对于每个候选子词（lo），计算合并后的似然增益：
    
    ```text
    未合并时的似然概率：P(w) = P(l)*P(o)*P(w)  (这个模型的假设就是假设每个字符独立出现)
    合并后的似然概率:P(w|lo) = P(lo)*P(w)
    增益公式 ：score(s) = log(P(merge)/P(unmerge)) = log(P(merge)) - log(P(unmerge))
    ```
  
  3)选择得分最高的子词合并。
  
4、迭代合并

  将得分最高的子词加入词汇表
  
  用新的词汇表更新语料库的分词表示，重新训练语言模型，重复步骤3，直到词汇表的大小达到预定大小（bert的词汇表是3万个子词）。

5、输出词汇表

  最终得到一个包含字符和子词的词汇表，比如
  
  ```text
  [l, o, w, e, r, n, lo, low, er, new, ...]
  ```

### 得到词表后如何应用wordpiece分词？（应用阶段）

1.单词拆分

先把句子按照空格切分（中文按照字切分，或者用结巴之类的分词工具先分词再切分），输入单词（lowest），拆分为字符序列（l o w e s t）

在 BERT 中，WordPiece 使用特殊标记（如 ##）表示词内子词，例如非首子词会带有 ## 前缀。

2.贪心最长匹配

对于上面拆分后的字符序列（l o w e s t），从左到右扫描，尝试匹配词汇表中最长的子词。
```python
def encode_word(word):
    tokens = []
    while len(word) > 0:
        i = len(word)
        # 从整个单词开始，查看是否在词典中出现
        while i > 0 and word[:i] not in vocab:
            # 不在词典中，末端指针往前移动一个位置
            i -= 1
        # 如果i从尾部移动头部的所有子词都不在词典中，则为未登陆词【UNK】
        if i == 0:
            return ["[UNK]"]
        # 如果word[:i]子词在词典中，切分成功
        tokens.append(word[:i])
        word = word[i:]
        if len(word) > 0:
            word = f"##{word}"
    return tokens
```

  最终结果：(low, e, s, t) 或 BERT 风格的 (low, ##e, ##s, ##t)。

**注意**：训练阶段和推理阶段所用的核心算法不同，训练阶段的核心是通过**训练unigram概率语言模型**获得使语料库的似然概率最大的子词，推理阶段的核心是**基于词表的贪心最大匹配算法**，将单词拆分成子词序列。


![image](https://github.com/user-attachments/assets/cdc39b8a-afa6-43d4-b5f8-244d678f29f5)


这种合并方式称为贪心合并，贪心合并是指：在分词推理阶段，从左到右扫描 token 序列，每次优先匹配词表中最长的子词（或合并对），立即使用，不回退、不尝试全局最优组合。

### WordPiece 的数学基础
  
WordPiece 是 Google 在 2016 年用于机器翻译模型的一种子词（subword）分词算法，后来被广泛用于 **BERT、ELECTRA、ALBERT 等模型**。它和 BPE 非常相似，但在合并策略上不同，**WordPiece 使用了概率建模的思想**，也就是说，它背后是有**数学基础的语言建模原理**。

---

## 一句话总结 WordPiece 的数学核心：

> **WordPiece 是通过最大化子词序列联合概率（或最小化困惑度）来选择子词的合并方式的子词编码方法。**

---

### WordPiece 的数学基础详解

### 🔸1. 目标公式（Maximum Likelihood）

WordPiece 的训练目标是：

> **最大化训练语料中所有句子的似然概率（likelihood）**

形式化地，对于一句话 \$w = w\_1, w\_2, ..., w\_T\$，我们希望找到最优的分词方式，使得：

$$
\text{maximize } \prod_{t=1}^{T} P(w_t \mid w_1, ..., w_{t-1})
$$

但在 WordPiece 中，我们不预测单词，而是预测子词（subword）序列，所以我们将句子分词成 \$s\_1, s\_2, ..., s\_n\$（每个 \$s\_i\$ 是一个子词），目标转化为：

$$
\text{maximize } \prod_{i=1}^{n} P(s_i \mid s_1, ..., s_{i-1})
$$

由于训练语料中 token 太多，这个联合概率不好直接计算，所以 WordPiece 采用了简化假设 + 动态规划。

---

### 🔸2. 简化：使用 Unigram 近似（n-gram）

WordPiece 实际用的是一种 **简化版的 Unigram 语言模型**：

* 假设每个子词的概率是独立的（这点和 SentencePiece 的 Unigram 类似）

即：

$$
\text{maximize } \prod_{i=1}^{n} P(s_i)
\Rightarrow \text{maximize } \sum_{i=1}^{n} \log P(s_i)
$$

通过最大化这个目标函数，WordPiece 在训练时评估合并后的子词是否能让语料的整体分词概率提升。

---

### 🔸3. 合并策略：最小化语言模型困惑度

WordPiece 的词表构建是一个**贪心式词汇生长过程**：

1. 从初始字符开始（单字/单字符）
2. 统计所有子词组合的词频
3. 合并能 **最大化总 log-likelihood 增加** 的子词组合（不是合并频率最高的）

即每次合并都是选择：

$$
\Delta \log P = \log P_{\text{new}}(S') - \log P_{\text{old}}(S)
$$

如果合并能提高总似然（或降低负对数似然），就进行合并，直到达到最大词表大小（如 30,000）。

---

### WordPiece 对比 BPE 的改进

WordPiece 解决了 BPE 的哪些问题？

语义相关性：

BPE 问题：BPE 只看频率，可能合并无意义的字符对（如 xy），忽略语言结构。
> WordPiece 改进：通过似然评分，优先选择对语言模型有意义的子词。例如，ing 比随机组合更可能被选中，因为它在英语中频繁作为后缀出现。
> 过度拆分或合并：

BPE 问题：频率驱动可能导致常见词被拆散（如 play 和 ing 分开），或罕见词被过度合并。
> WordPiece 改进：似然优化平衡了子词粒度，使得分词结果更符合语言分布。
语言模型支持：

BPE 问题：缺乏语言模型指导，分词结果可能不利于下游任务。
> WordPiece 改进：结合 unigram 语言模型，分词更贴近模型训练目标，尤其适合 BERT 的掩码语言模型（MLM）任务。

**在深度学习中的应用**

WordPiece 在 BERT 中得到了广泛验证：

BERT 的分词：

BERT 使用 30,000 大小的 WordPiece 词汇表，覆盖英文和其他语言。
输入文本被拆分为子词，例如 “playing” → [play, ##ing]，## 表示词内后续部分。
这种表示保留了词根和后缀信息，有助于模型理解形态变化。
多语言支持：

WordPiece 在多语言 BERT（如 mBERT）中训练统一的词汇表，支持跨语言迁移。
下游任务：

在分类、问答等任务中，WordPiece 的子词表示提高了模型对新词（OOV）的泛化能力。

### WordPiece 的优缺点
6.1 优点

语义优化：比 BPE 更贴近语言模型需求。

灵活性：通过似然调整子词粒度，适应不同语言。

高效性：训练和分词过程仍保持较低复杂度。

6.2 缺点

计算开销：相比 BPE，训练时需要额外计算似然，复杂度稍高。

依赖语料：似然评分依赖训练语料的质量和分布。

不可逆性：与 BPE 类似，分词后难以完美还原原始文本。

### 总结
WordPiece 是 BPE 的一个重要改进，通过引入最大化似然概率的评分机制，克服了 BPE 单纯依赖频率的局限性。它在 BERT 等模型中证明了其价值，尤其在需要语义相关性和语言模型支持的场景中表现出色。对于深度学习研究者来说，理解 WordPiece 的原理不仅有助于掌握分词技术，还能启发对预处理与模型设计的优化思考。未来，WordPiece 可能进一步与上下文信息结合，带来更智能的分词方案。




### 参考链接：
https://blog.csdn.net/shizheng_Li/article/details/146556265
https://blog.csdn.net/weixin_42426841/article/details/143170728?utm_source=chatgpt.com
