---
layout: article
title: "功能测试页面"
date: 2025-12-25
key: hello-world-2025
bilingual: true
lang: zh
hidden: true
# published: False
# tags: [Demo]
---

这是中文内容。

这是一个双语博客的测试。你应该能在标题下方看到语言切换按钮。

如果在主页列表看到这篇文章，应该能看到 EN/CN 的标志。

---

本文档用于展示博客主题支持的各种自定义内容块及其使用方法。每种样式都提供了源码示例和实际渲染效果。

## 1. 基础提示块 (Basic Blocks)

支持四种基础状态颜色：Standard, Success, Info, Warning, Error。

### HTML 语法

使用 `<div class="*-block" markdown="1">` (注意 `markdown="1"` 对于处理内部 MD 内容是必须的)。

**源码示例：**

````markdown
<!-- 常规/默认 -->
<div class="neutral-block" markdown="1">
<div class="block-title">Neutral Block</div>
这是一个默认样式的块。
</div>

<!-- 成功/绿色 -->
<div class="success-block" markdown="1">
<div class="block-title">Success Block</div>
操作成功提示。
</div>

<!-- 信息/蓝色 -->
<div class="info-block" markdown="1">
<div class="block-title">Info Block</div>
一般信息提示。
</div>

<!-- 警告/黄色 -->
<div class="warning-block" markdown="1">
<div class="block-title">Warning Block</div>
警告信息提示。
</div>

<!-- 错误/红色 -->
<div class="error-block" markdown="1">
<div class="block-title">Error Block</div>
错误或危险操作提示。
</div>
````

**渲染效果：**

<div class="neutral-block" markdown="1">
<div class="block-title">Neutral Block</div>
这是一个默认样式的块。
</div>

<div class="success-block" markdown="1">
<div class="block-title">Success Block</div>
操作成功提示。
</div>

<div class="info-block" markdown="1">
<div class="block-title">Info Block</div>
一般信息提示。
</div>

<div class="warning-block" markdown="1">
<div class="block-title">Warning Block</div>
警告信息提示。
</div>

<div class="error-block" markdown="1">
<div class="block-title">Error Block</div>
错误或危险操作提示。
</div>

### Liquid 标签语法

使用 `{% raw %}{% plain type title="..." %}{% endraw %}`。

**源码示例：**

````liquid
{% raw %}
{% plain success title="Liquid Success" %}
这是使用 Liquid 标签生成的块。
{% endplain %}

{% plain error title="Liquid Error" %}
这是使用 Liquid 标签生成的错误块。
{% endplain %}
{% endraw %}
````

**渲染效果：**

{% plain success title="Liquid Success" %}
这是使用 Liquid 标签生成的块。
{% endplain %}

{% plain error title="Liquid Error" %}
这是使用 Liquid 标签生成的错误块。
{% endplain %}

---

## 2. 学术与数学块 (Academic Blocks)

支持常见的学术环境定义：`proof`, `theorem`, `lemma`, `proposition`, `definition`, `example`, `remark`, `note`, `solution`。

### 基础用法 (默认换行标题)

**HTML 源码：**

````markdown
<div class="theorem" markdown="1">
这是一个定理。
</div>

<div class="proof" markdown="1">
这是一个证明。
</div>
````

**Liquid 源码：**

````liquid
{% raw %}
{% theorem %}
这是一个定理 (Liquid)。
{% endtheorem %}
{% endraw %}
````

**渲染效果：**

<div class="theorem" markdown="1">
这是一个定理。
</div>

<div class="proof" markdown="1">
这是一个证明。
</div>

### 行内标题样式 (Inline)

添加 `inline` 类或参数。

**源码：**

````markdown
<div class="proof inline" markdown="1">
标题与内容在同一行。
</div>

{% raw %}
{% note inline %}
注意：这是一个行内标题的 Note。
{% endnote %}
{% endraw %}
````

**渲染效果：**

<div class="proof inline" markdown="1">
标题与内容在同一行。
</div>

{% note inline %}
注意：这是一个行内标题的 Note。
{% endnote %}

### 自定义标题

使用 `data-title` 属性或 `title` 参数。

**源码：**

````markdown
<div class="lemma" data-title="Zorn's Lemma" markdown="1">
每个非空偏序集都有一个最大元...
</div>

{% raw %}
{% proposition title="My Proposition" %}
这是一个自定义标题的命题。
{% endproposition %}
{% endraw %}
````

**渲染效果：**

<div class="lemma" data-title="Zorn's Lemma" markdown="1">
每个非空偏序集都有一个最大元...
</div>

{% proposition title="My Proposition" %}
这是一个自定义标题的命题。
{% endproposition %}

---

## 3. 可折叠块 (Collapsible Blocks)

### HTML 语法 (`details` & `summary`)

**源码：**

````markdown
<details class="info" markdown="1">
<summary data-title="点击展开详情"></summary>
这里是隐藏的详细内容。
</details>
````

**渲染效果：**

<details class="info" markdown="1">
<summary data-title="点击展开详情"></summary>
这里是隐藏的详细内容。
</details>

### Liquid 语法 (`fold` 参数)

**源码：**

````liquid
{% raw %}
{% example fold title="查看代码示例" %}
```python
print("Hidden Code")
```
{% endexample %}
{% endraw %}
````

**渲染效果：**

{% example fold title="查看代码示例" %}
```python
print("Hidden Code")
```
{% endexample %}

---

## 4. 代码块增强 (Code Block Enhancements)

支持给代码块添加标题、默认折叠状态以及特殊样式。

### 带标题的代码块

在代码块下方添加 `{: title="..." }`。

**源码：**

````markdown
```python
def main():
    pass
```
{: title="main.py" }
````

**渲染效果：**

```python
def main():
    pass
```
{: title="main.py" }

### 默认折叠的代码块

使用 `fold="true"` (默认折叠) 或 `fold="open"` (默认展开)。

**源码：**

````markdown
```javascript
// 长代码折叠
const bigFile = "...";
```
{: title="config.js" fold="true" }
````

**渲染效果：**

```javascript
// 长代码折叠
const bigFile = "...";
```
{: title="config.js" fold="true" }

### 特殊语义颜色代码块

使用 `type="..."` 参数。支持 `example` (绿), `exploit` (蓝), `error` (红)。

**源码：**

````markdown
```bash
# 这是一个正确示例
npm install
```
{: type="example" }

```c
// 这是一个漏洞利用代码
char buf[10];
strcpy(buf, input);
```
{: type="exploit" title="vulnerable.c" }

```bash
# 这是一个错误操作
rm -rf /
```
{: type="error" }
````

**渲染效果：**

```bash
# 这是一个正确示例
npm install
```
{: type="example" }

```c
// 这是一个漏洞利用代码
char buf[10];
strcpy(buf, input);
```
{: type="exploit" title="vulnerable.c" }

```bash
# 这是一个错误操作
rm -rf /
```
{: type="error" }

### Liquid 代码块包装

**源码：**

````liquid
{% raw %}
{% code_block success fold title="Wrapped Code" %}
```python
print("Wrapped in Liquid")
```
{% endcode_block %}
{% endraw %}
````

**渲染效果：**

{% code_block success fold title="Wrapped Code" %}
```python
print("Wrapped in Liquid")
```
{% endcode_block %}
