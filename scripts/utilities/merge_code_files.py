#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

def merge_py_to_txt(root_dir='.', output_filename='merged.txt', exclude_dirs=None):
    """
    将指定目录下（递归）的所有 .py 文件内容合并到一个 txt 文件中，
    并排除掉用户指定的虚拟环境等目录。

    :param root_dir: 查找 .py 文件的根目录
    :param output_filename: 合并后输出的文件名
    :param exclude_dirs: 要排除的目录名列表
    """
    if exclude_dirs is None:
        exclude_dirs = ['venv', '__pycache__']

    py_files = []
    # 遍历目录树
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 如果当前目录在排除列表，就跳过整个目录
        if any(excl in dirpath.split(os.sep) for excl in exclude_dirs):
            continue
        for fname in filenames:
            if fname.endswith('.py'):
                py_files.append(os.path.join(dirpath, fname))

    py_files.sort()

    with open(output_filename, 'w', encoding='utf-8') as outfile:
        for py_file in py_files:
            outfile.write(f'# ===== File: {py_file} =====\n')
            try:
                with open(py_file, 'r', encoding='utf-8') as infile:
                    outfile.write(infile.read())
            except Exception as e:
                outfile.write(f'# ERROR reading {py_file}: {e}\n')
            outfile.write('\n\n')

    print(f"已合并 {len(py_files)} 个 .py 文件到 '{output_filename}'。")

if __name__ == '__main__':
    # 将 root_dir 改为你的项目根目录：
    merge_py_to_txt(
        root_dir='/Users/hydra/Downloads/MRI-AS-MRI_AS/FINAL_AS',
        output_filename='merged.txt',
        exclude_dirs=['venv', '__pycache__']
    )
