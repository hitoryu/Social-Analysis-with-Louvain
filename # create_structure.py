# create_structure.py
import os

def create_project_structure():
    """Tạo cấu trúc thư mục tự động"""
    directories = [
        'src',
        'data/raw', 'data/processed', 'data/external',
        'results/images', 'results/reports', 'results/models',
        'notebooks',
        'tests',
        'docs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Đã tạo: {directory}")
    
    # Tạo file __init__.py
    with open('src/__init__.py', 'w') as f:
        f.write('"""Package cho phân tích cộng đồng Facebook"""\n')
    
    with open('tests/__init__.py', 'w') as f:
        f.write('"""Tests cho đồ án"""')

if __name__ == "__main__":
    create_project_structure()