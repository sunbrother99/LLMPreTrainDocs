# 导入所需库
from collections import defaultdict, Counter
import math

# WordPiece 训练函数
def train_wordpiece(corpus, vocab_size=1000):
    """
    训练 WordPiece 模型，生成词汇表
    :param corpus: 输入语料库（单词列表）
    :param vocab_size: 目标词汇表大小
    :return: 词汇表（子词集合）
    """
    # 步骤 1：初始化单词为字符序列，统计频率
    word_freq = Counter(corpus)  # 统计每个单词的频率
    vocab = set()  # 初始化词汇表
    word_splits = {}  # 存储每个单词的当前分割状态
    
    total_words = sum(word_freq.values())  # 语料库总词数
    for word in word_freq:
        chars = list(word)  # 拆分为字符
        word_splits[word] = chars
        vocab.update(chars)  # 添加初始字符到词汇表
    
    # 初始似然估计：基于字符频率
    char_freq = defaultdict(int)
    for word, freq in word_freq.items():
        for char in word_splits[word]:
            char_freq[char] += freq
    
    # 计算初始概率
    token_probs = {token: freq / total_words for token, freq in char_freq.items()}
    
    # 步骤 2：迭代合并子词
    while len(vocab) < vocab_size:
        # 统计所有相邻字符对及其似然增益
        pair_scores = defaultdict(float)
        for word, freq in word_freq.items():
            chars = word_splits[word]
            for i in range(len(chars) - 1):
                pair = (chars[i], chars[i + 1])
                merged = ''.join(pair)
                # 计算合并前后的似然增益
                if chars[i] in token_probs and chars[i + 1] in token_probs:
                    unmerged_prob = token_probs[chars[i]] * token_probs[chars[i + 1]]
                    merged_freq = sum(f for w, f in word_freq.items() if merged in ''.join(word_splits[w]))
                    merged_prob = merged_freq / total_words
                    if unmerged_prob > 0:  # 避免除以零
                        score = math.log(merged_prob) - math.log(unmerged_prob)
                        pair_scores[pair] += score * freq
        
        if not pair_scores:  # 如果没有可合并的字符对，退出
            break
        
        # 选择得分最高的字符对
        best_pair = max(pair_scores, key=pair_scores.get)
        new_token = ''.join(best_pair)
        
        # 步骤 3：更新所有单词的分割状态
        for word in word_freq:
            chars = word_splits[word]
            i = 0
            new_chars = []
            while i < len(chars):
                if i < len(chars) - 1 and (chars[i], chars[i + 1]) == best_pair:
                    new_chars.append(new_token)
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1
            word_splits[word] = new_chars
        
        # 更新词汇表和概率
        vocab.add(new_token)
        merged_freq = sum(f for w, f in word_freq.items() if new_token in word_splits[w])
        token_probs[new_token] = merged_freq / total_words
    
    return vocab

# WordPiece 分词函数
def apply_wordpiece(word, vocab):
    """
    对单个单词应用 WordPiece 分词
    :param word: 输入单词
    :param vocab: 训练好的词汇表
    :return: 分词后的子词列表（带 ## 前缀）
    """
    if not word:
        return []
    
    # 步骤 1：初始化为字符序列
    chars = list(word)
    tokens = []
    
    # 步骤 2：贪心最长匹配
    i = 0
    while i < len(chars):
        # 尝试匹配最长的子词
        longest_match = None
        for j in range(len(chars), i, -1):
            candidate = ''.join(chars[i:j])
            if candidate in vocab:
                longest_match = candidate
                break
        
        if longest_match:
            # 如果是第一个子词，不加 ##，否则加 ## 前缀
            if i == 0:
                tokens.append(longest_match)
            else:
                tokens.append('##' + longest_match)
            i += len(longest_match)
        else:
            # 如果没有匹配，添加单个字符并移动
            if i == 0:
                tokens.append(chars[i])
            else:
                tokens.append('##' + chars[i])
            i += 1
    
    return tokens

# 测试代码
def main():
    # 示例语料库
    corpus = ["low", "low", "lower", "lowest", "new", "newer"]
    print("原始语料库:", corpus)
    
    # 训练 WordPiece 模型
    vocab_size = 10  # 设置较小的词汇表大小以便演示
    vocab = train_wordpiece(corpus, vocab_size)
    print("训练得到的词汇表:", sorted(vocab))
    
    # 应用 WordPiece 分词
    test_words = ["low", "lowest", "newest"]
    for word in test_words:
        tokens = apply_wordpiece(word, vocab)
        print(f"单词 '{word}' 分词结果: {tokens}")

if __name__ == "__main__":
    main()
