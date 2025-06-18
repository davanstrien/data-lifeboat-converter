#!/usr/bin/env python3
"""
Test static space deployment for Data Lifeboats

# /// script
# dependencies = [
#     "huggingface_hub>=0.19.0",
# ]
# ///
"""

import shutil
import tempfile
from pathlib import Path
from huggingface_hub import HfApi, create_repo


def test_redirect_approach(lifeboat_path: Path, repo_id: str, private: bool = True):
    """Test Approach 1: Minimal index.html redirect"""
    api = HfApi()
    
    print(f"Testing Approach 1: Redirect index.html")
    print(f"Data Lifeboat: {lifeboat_path}")
    print(f"Target repo: {repo_id}")
    
    # Create repository
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="static",
            private=private,
            exist_ok=True
        )
        print(f"✅ Created/verified Space repository: {repo_id}")
    except Exception as e:
        print(f"❌ Error creating repository: {e}")
        raise
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Copy entire Data Lifeboat
        print("📁 Copying Data Lifeboat files...")
        for item in lifeboat_path.iterdir():
            if item.is_file():
                shutil.copy2(item, temp_path / item.name)
            elif item.is_dir() and not item.name.startswith('.'):
                shutil.copytree(item, temp_path / item.name, 
                              ignore=shutil.ignore_patterns('.DS_Store', '__pycache__'))
        
        # Add redirect index.html
        print("🔄 Adding redirect index.html...")
        redirect_html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="0; url=README.html">
    <title>Redirecting to Data Lifeboat...</title>
</head>
<body>
    <p>Redirecting to Data Lifeboat documentation...</p>
    <p>If you are not redirected, <a href="README.html">click here</a>.</p>
</body>
</html>"""
        (temp_path / "index.html").write_text(redirect_html)
        
        # Create README.md
        readme_content = f"""---
title: Data Lifeboat Static Test - Redirect
emoji: 🚢
colorFrom: blue
colorTo: purple
sdk: static
pinned: false
---

# Data Lifeboat Static Space Test

Testing direct deployment with redirect index.html
"""
        (temp_path / "README.md").write_text(readme_content)
        
        # Upload
        print("🚀 Uploading to HuggingFace Spaces...")
        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(temp_path),
            repo_type="space",
            commit_message="Test static space with redirect"
        )
        
    print(f"✅ Static space created: https://huggingface.co/spaces/{repo_id}")


def test_app_file_approach(lifeboat_path: Path, repo_id: str, private: bool = True):
    """Test Approach 2: Using app_file directive"""
    api = HfApi()
    
    print(f"\nTesting Approach 2: app_file directive")
    print(f"Data Lifeboat: {lifeboat_path}")
    print(f"Target repo: {repo_id}")
    
    # Create repository
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="space", 
            space_sdk="static",
            private=private,
            exist_ok=True
        )
        print(f"✅ Created/verified Space repository: {repo_id}")
    except Exception as e:
        print(f"❌ Error creating repository: {e}")
        raise
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Copy entire Data Lifeboat
        print("📁 Copying Data Lifeboat files...")
        for item in lifeboat_path.iterdir():
            if item.is_file():
                shutil.copy2(item, temp_path / item.name)
            elif item.is_dir() and not item.name.startswith('.'):
                shutil.copytree(item, temp_path / item.name,
                              ignore=shutil.ignore_patterns('.DS_Store', '__pycache__'))
        
        # Create README.md with app_file
        readme_content = f"""---
title: Data Lifeboat Static Test - App File
emoji: 🚢
colorFrom: blue  
colorTo: purple
sdk: static
app_file: README.html
pinned: false
---

# Data Lifeboat Static Space Test

Testing direct deployment with app_file pointing to README.html
"""
        (temp_path / "README.md").write_text(readme_content)
        
        # Upload
        print("🚀 Uploading to HuggingFace Spaces...")
        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(temp_path),
            repo_type="space",
            commit_message="Test static space with app_file"
        )
        
    print(f"✅ Static space created: https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test static space deployment")
    parser.add_argument("lifeboat_path", help="Path to Data Lifeboat")
    parser.add_argument("--approach", choices=["redirect", "app_file", "both"], 
                       default="both", help="Which approach to test")
    parser.add_argument("--repo-prefix", default="davanstrien/static-test",
                       help="Repository prefix for test spaces")
    parser.add_argument("--private", action="store_true", help="Make spaces private")
    
    args = parser.parse_args()
    
    lifeboat_path = Path(args.lifeboat_path)
    
    if args.approach in ["redirect", "both"]:
        test_redirect_approach(
            lifeboat_path, 
            f"{args.repo_prefix}-redirect",
            args.private
        )
    
    if args.approach in ["app_file", "both"]:
        test_app_file_approach(
            lifeboat_path,
            f"{args.repo_prefix}-appfile", 
            args.private
        )