#!/usr/bin/env python3
import re
import os

def fix_file(filepath):
    """修复常见的代码问题"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修复省略号字符
    content = content.replace('...', '...')
    
    # 移除文件末尾的空白行
    content = content.rstrip() + '\n'
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

# 修复所有Python文件
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            try:
                fix_file(filepath)
                print(f"Fixed: {filepath}")
            except Exception as e:
                print(f"Error fixing {filepath}: {e}")
