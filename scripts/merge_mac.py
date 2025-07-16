import os
import re

def merge_code_files(directory_path, output_file, file_extensions=None):
    """
    合并指定目录下所有代码文件到一个文件中
    
    参数:
    directory_path: 项目目录路径
    output_file: 输出文件路径
    file_extensions: 要合并的文件扩展名列表，如 ['.cs', '.js', '.py']，默认为常见代码文件类型
    """
    if file_extensions is None:
        # 默认的代码文件扩展名
        file_extensions = [
            '.cs', '.vb', '.js', '.ts', '.html', '.css', '.py', '.cpp', '.h', 
            '.c', '.java', '.sql', '.xml', '.json', '.config', '.csproj', '.sln'
        ]
    
    # 要忽略的目录
    ignore_dirs = ['bin', 'obj', '.vs', 'packages', 'node_modules', '.git']
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        print(f"开始处理目录: {directory_path}")
        file_count = 0
        
        for root, dirs, files in os.walk(directory_path):
            # 跳过忽略的目录
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                
                if file_ext in file_extensions:
                    rel_path = os.path.relpath(file_path, directory_path)
                    print(f"处理文件: {rel_path}")
                    
                    outfile.write(f"\n\n{'='*80}\n")
                    outfile.write(f"// FILE: {rel_path}\n")
                    outfile.write(f"{'='*80}\n\n")
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            content = infile.read()
                            outfile.write(content)
                        file_count += 1
                    except UnicodeDecodeError:
                        try:
                            # 尝试其他编码
                            with open(file_path, 'r', encoding='latin-1') as infile:
                                content = infile.read()
                                outfile.write(content)
                            file_count += 1
                        except Exception as e:
                            outfile.write(f"// 无法读取文件: {e}\n")
                            print(f"无法读取文件 {rel_path}: {e}")
                    except Exception as e:
                        outfile.write(f"// 处理文件时出错: {e}\n")
                        print(f"处理文件时出错 {rel_path}: {e}")
        
        print(f"处理完成! 共合并了 {file_count} 个文件。")

if __name__ == "__main__":
    # Mac格式的路径 - 修改为您的实际路径
    project_directory = "/Users/hydra/Downloads/MRI-AS-MRI_AS/FINAL_AS"
    output_filepath = "/Users/hydra/Downloads/merged_code.txt"  # 放在下载文件夹，容易找到
    
    # 可以自定义要合并的文件类型
    # custom_extensions = ['.py', '.cpp', '.h']  # 如果只想合并Python和C++文件
    
    print("开始合并代码文件...")
    merge_code_files(project_directory, output_filepath)
    print(f"文件已合并到: {output_filepath}")
