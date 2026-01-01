---
layout: article
title: "功能测试页面"
date: 2025-12-25
key: hello-world-2025
bilingual: true
lang: zh
hidden: true
# published: False
tags: [Demo]
---

这是中文内容。

这是一个双语博客的测试。你应该能在标题下方看到语言切换按钮。

如果在主页列表看到这篇文章，应该能看到 EN/CN 的标志。

## 基础样式块测试

<div class="success-block" markdown="1">
<div class="block-title">跨行 Success 块测试</div>

这是一个可以包含多行内容的 Success 块。

- 支持列表
- 支持 **Markdown** 语法
- 支持代码块

```python
print("Hello World")
```
</div>

<div class="info-block" markdown="1">
<div class="block-title">Info 块测试</div>

这是一条信息提示。
</div>

<div class="warning-block" markdown="1">
<div class="block-title">Warning 块测试</div>

这是一条警告信息。
</div>

<div class="error-block" markdown="1">
<div class="block-title">Error 块测试</div>
这是一条错误信息。
</div>

## 数学/学术块测试 (HTML 语法)

<div class="proof" markdown="1">
这是一个 Proof 块（默认换行标题）。
</div>

<div class="proof inline" markdown="1">
这是一个 Proof 块（行内标题）。注意标题 "Proof" 会自动加粗并与此文本在同一行。
</div>

<div class="remark" data-title="Custom Remark" markdown="1">
这是一个 Remark 块（自定义标题）。显示为：注记 (Custom Remark)
</div>

<div class="note inline" markdown="1">
这是一个 Note 块（行内标题）。
</div>

<div class="theorem" markdown="1">
这是一个 Theorem 块（默认换行标题）。
</div>

<div class="proposition inline" markdown="1">
这是一个 Proposition 块（行内标题）。
</div>

<div class="lemma" markdown="1">
这是一个 Lemma 块。
</div>

## 可折叠块测试 (HTML 语法)

<details class="proof" markdown="1">
<summary></summary>
这是一个可折叠的 Proof 块（默认标题）。
</details>

<details class="note" markdown="1">
<summary data-title="关于配置"></summary>
这是一个可折叠的 Note 块（自定义标题）。显示为：注意 (关于配置)
</details>

<details class="theorem" markdown="1">
<summary></summary>
这是一个可折叠的 Theorem 块。
</details>

## Liquid 标签测试

{% theorem fold %}
这是一个使用 Liquid 标签生成的折叠 Theorem 块。
{% endtheorem %}

{% note fold title="Liquid Tag" %}
这是一个使用 Liquid 标签生成的折叠 Note 块（带标题）。
{% endnote %}

## 原生代码块增强测试

#### 1. 带标题的代码块

```python
def hello():
    print("Hello World")
```
{: title="main.py" }

#### 2. 可折叠代码块 (默认折叠)

```javascript
console.log("This is hidden by default");
```
{: fold="true" title="hidden.js" }

#### 3. 可折叠代码块 (默认展开)

```ruby
puts "This is open by default"
```
{: fold="open" title="open.rb" }

#### 4. 多种风格代码块

**Example Code (Success/Green)**

```python
# This is an example
def example():
    pass
```
{: type="example" }

**Exploit Code (Info/Blue)**

```c
// This is an exploit
int main() {
    return 0;
}
```
{: type="exploit" }

**Error Style Code**

```bash
rm -rf /
```
{: type="error" title="Dangerous Command" }

**Liquid Tag Example**

{% code_block example fold %}
```python
print("This is a folded example code block")
```
{% endcode_block %}



{% proof inline %}
这是一个使用 Liquid 标签生成的行内 Proof 块。
{% endproof %}

{% remark title="Liquid Remark" %}
这是一个使用 Liquid 标签生成的 Remark 块（带标题）。
{% endremark %}

<!-- <div class="info-block" markdown="1">
跨行 Info 块测试

这是一个可以包含多行内容的 Info 块。

- 支持列表
- 支持 **Markdown** 语法
- 支持代码块

```python
print("Hello World")
```
</div> -->
