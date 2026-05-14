---
license: gpl-3.0
language:
- en
pretty_name: LeetCode Solutions
tags:
- code
- leetcode
- programming
---

# LeetCode Solutions

Solutions to LeetCode problems in C++, Java, Python, SQL, and TypeScript.

## Source

Derived from [walkccc/LeetCode](https://github.com/walkccc/LeetCode) by [@walkccc](https://github.com/walkccc), licensed under MIT.

Dataset built using [tkeskin/llm-fine-tune](https://github.com/tkeskin/llm-fine-tune).

## Dataset structure

Each row is one LeetCode problem. Language columns are `null` when no solution exists for that language.

| Column       | Description                        |
|--------------|------------------------------------|
| `problem_id` | LeetCode problem number            |
| `title`      | Problem title                      |
| `cpp`        | C++ solution (~3,495 problems)     |
| `java`       | Java solution (~3,371 problems)    |
| `python`     | Python solution (~3,169 problems)  |
| `sql`        | SQL solution (~307 problems)       |
| `typescript` | TypeScript solution (~69 problems) |
