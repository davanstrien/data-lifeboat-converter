#!/usr/bin/env python3
"""
Convert Data Lifeboat to Hugging Face Dataset
"""

from pathlib import Path
from typing import Any, Dict, List

import polars as pl
from datasets import Dataset, Features, Image, Sequence, Value
from huggingface_hub import HfApi, create_repo

from lifeboat_to_hf import LifeboatLoader, LifeboatToPolars, Photo


class LifeboatToDynamicSpace:
    """Create Dynamic Docker Space that downloads Data Lifeboat at runtime"""

    def __init__(self, lifeboat_path: Path):
        self.lifeboat_path = Path(lifeboat_path)
        self.loader = LifeboatLoader(lifeboat_path)

    def create_dynamic_space(
        self, repo_id: str, raw_dataset_repo_id: str, private: bool = False
    ) -> str:
        """Create and upload Dynamic Docker Space that downloads raw Data Lifeboat"""
        import tempfile

        api = HfApi()
        lifeboat_meta = self.loader.load_lifeboat_metadata()

        print(f"Creating Dynamic Docker Space for Data Lifeboat: {lifeboat_meta.name}")
        print(f"Target repository: {repo_id}")
        print(f"Raw dataset source: {raw_dataset_repo_id}")

        # Create repository
        try:
            create_repo(
                repo_id=repo_id,
                repo_type="space",
                space_sdk="docker",
                private=private,
                exist_ok=True,
            )
            print(f"✅ Created/verified Space repository: {repo_id}")
        except Exception as e:
            print(f"❌ Error creating repository: {e}")
            raise

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create Dockerfile
            print("🐳 Creating dynamic Dockerfile...")
            dockerfile_content = self.create_dynamic_dockerfile(raw_dataset_repo_id)
            (temp_path / "Dockerfile").write_text(dockerfile_content)

            # Create download script
            print("📝 Creating download script...")
            script_content = self.create_download_script()
            (temp_path / "download_and_serve.py").write_text(script_content)

            # Create README.md
            print("📝 Creating Space README...")
            readme_content = self.create_space_readme(
                repo_id, raw_dataset_repo_id, lifeboat_meta
            )
            (temp_path / "README.md").write_text(readme_content)

            # Upload everything
            print("🚀 Uploading to HuggingFace Spaces...")
            api.upload_folder(
                repo_id=repo_id,
                folder_path=str(temp_path),
                repo_type="space",
                commit_message="Create Dynamic Data Lifeboat Space",
            )

        print("✅ Dynamic Space created successfully!")
        print(
            f"🌐 Access your Data Lifeboat at: https://huggingface.co/spaces/{repo_id}"
        )
        print("⚡ Data Lifeboat will be downloaded automatically when the Space starts")

        return repo_id

    def create_dynamic_dockerfile(self, raw_dataset_repo_id: str) -> str:
        """Generate Dockerfile for dynamic Data Lifeboat hosting"""
        dockerfile_template = Path("templates/Dockerfile.dynamic.template")
        template = dockerfile_template.read_text()
        return template.format(raw_dataset_repo_id=raw_dataset_repo_id)

    def create_download_script(self) -> str:
        """Generate download and serve script"""
        script_template = Path("templates/download_and_serve.py.template")
        if script_template.exists():
            return script_template.read_text()
        else:
            raise FileNotFoundError

    def create_space_readme(
        self, repo_id: str, raw_dataset_repo_id: str, lifeboat_meta
    ) -> str:
        """Generate README.md for Dynamic Space"""
        return f"""---
title: "{lifeboat_meta.name} - Dynamic Data Lifeboat"
emoji: 🚢
colorFrom: blue
colorTo: purple
sdk: docker
datasets:
- {raw_dataset_repo_id}
tags:
- flickr-commons
- data-lifeboat
pinned: false
---

# {lifeboat_meta.name} - Dynamic Data Lifeboat

This is a **Dynamic Data Lifeboat Space** that downloads and serves a Data Lifeboat collection at runtime.

## About This Collection

**Purpose:** {lifeboat_meta.purpose}

## How It Works

This Space dynamically downloads the raw Data Lifeboat from:
**[📦 {raw_dataset_repo_id}](https://huggingface.co/datasets/{raw_dataset_repo_id})**

When the Space starts, it:
1. 📥 Downloads the complete Data Lifeboat archive
2. 🚀 Extracts and serves it using Python HTTP server  
3. 🌐 Provides the same interactive experience as the original

## Advantages of Dynamic Hosting

- ✅ **No size limits** - Downloads happen at runtime within HuggingFace's infrastructure
- ✅ **Always fresh** - Pulls the latest version of the Data Lifeboat
- ✅ **Minimal upload** - Only configuration files need to be uploaded to the Space
- ✅ **Preserves integrity** - Raw Data Lifeboat served exactly as archived

## About Data Lifeboats

Data Lifeboats are **self-contained digital preservation archives** from the [Flickr Foundation](https://www.flickr.org/) that preserve complete cultural collections with:

- 📷 Original high-quality photos and thumbnails
- 📝 Complete metadata (titles, descriptions, tags, dates, locations)  
- 🌐 Interactive web viewer with no external dependencies
- 📊 Structured data ready for research and analysis

---

⚡ **Dynamic hosting** - This Data Lifeboat is downloaded fresh each time the Space starts.

*Technical implementation: Docker Space with runtime dataset download via `huggingface_hub`*
"""


class LifeboatToHuggingFace:
    """Convert Data Lifeboat to Hugging Face Dataset format"""

    def __init__(self, lifeboat_path: Path):
        self.lifeboat_path = Path(lifeboat_path)
        self.loader = LifeboatLoader(lifeboat_path)
        self.converter = LifeboatToPolars(self.loader)
        self.media_path = self.lifeboat_path / "media"

    def prepare_image_paths(self, photos_df: pl.DataFrame) -> pl.DataFrame:
        """Add absolute image paths to the dataframe"""
        # Convert relative paths to absolute paths
        photos_df = photos_df.with_columns(
            [
                (f"{pl.lit(str(self.lifeboat_path))}/" + pl.col("original_path")).alias(
                    "original_path_abs"
                ),
                (
                    f"{pl.lit(str(self.lifeboat_path))}/" + pl.col("thumbnail_path")
                ).alias("thumbnail_path_abs"),
            ]
        )
        return photos_df

    def create_features(self) -> Features:
        """Define the features/schema for the HuggingFace dataset"""
        return Features(
            {
                # Images first for better visual display
                "image": Image(),  # Original image
                "thumbnail": Image(),  # Thumbnail image
                # Core identifiers
                "photo_id": Value("string"),
                "secret": Value(
                    "string"
                ),  # Flickr URL construction parameter - enables direct image access for public domain photos
                "url": Value("string"),
                # Basic metadata
                "title": Value("string"),
                "description": Value("string"),
                # Attribution
                "uploader_id": Value("string"),
                "uploader_username": Value("string"),
                "license_id": Value("string"),
                "license_label": Value("string"),
                "license_url": Value("string"),
                # Temporal data
                "date_taken": Value("string"),
                "date_taken_granularity": Value("string"),
                "date_uploaded": Value("timestamp[s]"),
                # Location (optional)
                "latitude": Value("float64"),
                "longitude": Value("float64"),
                # Engagement metrics
                "count_faves": Value("int64"),
                "count_views": Value("int64"),
                "count_comments": Value("int64"),
                # Content metadata
                "safety_level": Value("string"),
                "visibility": Value("string"),
                "original_format": Value("string"),
                "media_type": Value("string"),
                # Counts
                "num_tags": Value("int64"),
                "num_albums": Value("int64"),
                "num_galleries": Value("int64"),
                "num_groups": Value("int64"),
                # Tags as a list of strings (simplified)
                "tags": Sequence(Value("string")),
                # Comments as list of dicts
                "comments": Sequence(
                    {
                        "comment_id": Value("string"),
                        "author_id": Value("string"),
                        "text": Value("string"),
                        "date": Value("timestamp[s]"),
                    }
                ),
            }
        )

    def prepare_dataset_records(
        self, photos: List[Photo], photos_enriched_df: pl.DataFrame
    ) -> List[Dict[str, Any]]:
        """Prepare records for HuggingFace Dataset"""
        # Create mapping of photo_id to enriched data
        enriched_data = {row["photo_id"]: row for row in photos_enriched_df.to_dicts()}

        records = []
        for photo in photos:
            enriched = enriched_data.get(photo.id, {})

            # Prepare tags as simple string list
            tags = [tag.normalizedValue for tag in photo.tags if tag.normalizedValue]

            # Prepare comments
            comments = [
                {
                    "comment_id": comment.id,
                    "author_id": comment.authorId,
                    "text": comment.text,
                    "date": comment.date,
                }
                for comment in photo.comments
                if comment.id  # Only include comments with valid IDs
            ]

            # Extract file paths using the new method
            original_path, thumbnail_path = photo.get_file_paths()

            # Build record (ordered to match Features)
            record = {
                # Images first
                "image": str(self.lifeboat_path / original_path)
                if original_path
                else None,
                "thumbnail": str(self.lifeboat_path / thumbnail_path)
                if thumbnail_path
                else None,
                # Core identifiers
                "photo_id": photo.id,
                "secret": photo.secret,
                "url": photo.url,
                # Basic metadata
                "title": photo.title,
                "description": photo.description,
                # Attribution
                "uploader_id": photo.uploaderId,
                "uploader_username": enriched.get("username"),
                "license_id": photo.licenseId,
                "license_label": enriched.get("label"),
                "license_url": enriched.get("url"),
                # Temporal
                "date_taken": photo.dateTaken.value if photo.dateTaken else None,
                "date_taken_granularity": photo.dateTaken.granularity
                if photo.dateTaken
                else None,
                "date_uploaded": photo.dateUploaded,
                # Location
                "latitude": photo.location.latitude if photo.location else None,
                "longitude": photo.location.longitude if photo.location else None,
                # Engagement
                "count_faves": photo.countFaves,
                "count_views": photo.countViews,
                "count_comments": photo.countComments,
                # Content metadata
                "safety_level": photo.safetyLevel,
                "visibility": photo.visibility,
                "original_format": photo.originalFormat,
                "media_type": photo.get_media_type(),
                # Counts
                "num_tags": len(photo.tags),
                "num_albums": len(photo.albumIds),
                "num_galleries": len(photo.galleryIds),
                "num_groups": len(photo.groupIds),
                # Nested data
                "tags": tags,
                "comments": comments,
            }

            records.append(record)

        return records

    def create_dataset(self) -> Dataset:
        """Create HuggingFace Dataset from Data Lifeboat"""
        print("Loading all photos...")
        photos = self.loader.load_all_photos()

        print("Creating Polars DataFrames...")
        datasets = self.converter.create_full_dataset()
        photos_enriched_df = datasets["photos_enriched"]

        print("Preparing dataset records...")
        records = self.prepare_dataset_records(photos, photos_enriched_df)

        print("Creating HuggingFace Dataset...")
        return Dataset.from_list(records, features=self.create_features())

    def calculate_dataset_stats(
        self, photos: List[Photo], datasets: Dict[str, pl.DataFrame]
    ) -> Dict[str, Any]:
        """Calculate statistics about the dataset"""
        from collections import Counter

        stats = {
            "total_photos": len(photos),
            "total_tags": datasets["tags"].height,
            "total_comments": datasets["comments"].height,
            "total_contributors": datasets["contributors"].height,
            "total_licenses": datasets["licenses"].height,
        }

        if upload_dates := [p.dateUploaded for p in photos if p.dateUploaded]:
            stats["earliest_upload"] = min(upload_dates).strftime("%Y-%m-%d")
            stats["latest_upload"] = max(upload_dates).strftime("%Y-%m-%d")

        # License distribution
        license_counts = Counter(p.licenseId for p in photos if p.licenseId)
        stats["license_distribution"] = dict(license_counts.most_common())

        # Format distribution
        format_counts = Counter(p.originalFormat for p in photos if p.originalFormat)
        stats["format_distribution"] = dict(format_counts.most_common())

        # Engagement stats
        faves = [p.countFaves for p in photos if p.countFaves]
        views = [p.countViews for p in photos if p.countViews]
        comments = [p.countComments for p in photos if p.countComments]

        if faves:
            stats["avg_faves"] = sum(faves) / len(faves)
            stats["max_faves"] = max(faves)
        if views:
            stats["avg_views"] = sum(views) / len(views)
            stats["max_views"] = max(views)
        if comments:
            stats["avg_comments"] = sum(comments) / len(comments)
            stats["max_comments"] = max(comments)

        # Top contributors by photo count
        contrib_df = datasets["contributors"]
        if contrib_df.height > 0:
            top_contributors = (
                contrib_df.filter(pl.col("photo_contributions") > 0)
                .sort("photo_contributions", descending=True)
                .head(5)
                .select(["username", "photo_contributions"])
                .to_dicts()
            )
            stats["top_contributors"] = top_contributors

        # Geographic coverage
        locations = [
            (p.location.latitude, p.location.longitude)
            for p in photos
            if p.location and p.location.latitude and p.location.longitude
        ]
        stats["photos_with_location"] = len(locations)
        stats["location_percentage"] = (
            (len(locations) / len(photos)) * 100 if photos else 0
        )

        return stats

    def create_dataset_card(
        self, photos: List[Photo], datasets: Dict[str, pl.DataFrame]
    ) -> str:
        """Generate a generic Data Lifeboat dataset card for HuggingFace Hub"""
        lifeboat_meta = self.loader.load_lifeboat_metadata()
        stats = self.calculate_dataset_stats(photos, datasets)

        # Determine size category
        size_cat = (
            "n<1K"
            if stats["total_photos"] < 1000
            else "1K<n<10K"
            if stats["total_photos"] < 10000
            else "10K<n<100K"
        )

        # Format license distribution
        license_list = "\n".join(
            [
                f"- {license}: {count} photos"
                for license, count in stats.get("license_distribution", {}).items()
            ]
        )

        # Format top contributors
        top_contrib_list = ""
        if stats.get("top_contributors"):
            top_contrib_list = "\n".join(
                [
                    f"- {contrib['username']}: {contrib['photo_contributions']} photos"
                    for contrib in stats["top_contributors"]
                ]
            )

        card = f"""---
license: other
license_name: various-open-licenses
license_link: https://www.flickr.com/commons/usage/
tags:
- flickr
- flickr-commons
- data-lifeboat
size_categories:
- {size_cat}
language:
- en
task_categories:
- image-classification
- image-to-text
- text-retrieval
- visual-question-answering
pretty_name: "{lifeboat_meta.name}"
---

# {lifeboat_meta.name}

## Dataset Description

This dataset is a **Flickr Data Lifeboat** converted to Hugging Face format. Data Lifeboats are digital preservation archives created by the [Flickr Foundation](https://www.flickr.org/programs/content-mobility/data-lifeboat/) to ensure long-term access to meaningful collections of Flickr photos and their rich community metadata.

### What is a Data Lifeboat?

Data Lifeboats are self-contained archives designed to preserve not just images, but the **social and cultural context** that makes them meaningful:
- Complete photo metadata and community interactions
- User-generated tags, descriptions, and comments  
- Attribution and licensing information
- No external dependencies for long-term preservation

**Collection Details:**
- **Original Collection ID**: `{lifeboat_meta.id}`
- **Created**: {lifeboat_meta.dateCreated}
- **Curator**: {lifeboat_meta.creator.name}

### Purpose

{lifeboat_meta.purpose}

## Dataset Statistics

### Collection Overview
- **Total Photos**: {stats["total_photos"]:,}
- **Total Tags**: {stats["total_tags"]:,}
- **Total Comments**: {stats["total_comments"]:,}
- **Contributors**: {stats["total_contributors"]:,}
- **Date Range**: {stats.get("earliest_upload", "Unknown")} to {stats.get("latest_upload", "Unknown")}
- **Photos with Location**: {stats["photos_with_location"]} ({stats["location_percentage"]:.1f}%)

### License Distribution
{license_list}

### Format Distribution
{chr(10).join([f"- {fmt}: {count} photos" for fmt, count in stats.get("format_distribution", {}).items()])}

### Engagement Metrics
- **Average Views**: {stats.get("avg_views", 0):,.0f} (max: {stats.get("max_views", 0):,})
- **Average Favorites**: {stats.get("avg_faves", 0):,.0f} (max: {stats.get("max_faves", 0):,})
- **Average Comments**: {stats.get("avg_comments", 0):.1f} (max: {stats.get("max_comments", 0)})

### Top Contributing Institutions/Users
{top_contrib_list}

## Dataset Structure

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `image` | Image | Original resolution photo |
| `thumbnail` | Image | Thumbnail version |
| `photo_id` | string | Unique Flickr identifier |
| `secret` | string | Flickr URL construction parameter (enables direct access to public domain photos) |
| `url` | string | Original Flickr URL |
| `title` | string | Photo title |
| `description` | string | Photo description |
| `uploader_username` | string | Username of uploader |
| `license_label` | string | Human-readable license |
| `date_taken` | string | When photo was captured |
| `date_uploaded` | timestamp | When uploaded to Flickr |
| `latitude/longitude` | float | GPS coordinates (if available) |
| `count_views/faves/comments` | int | Community engagement metrics |
| `tags` | list[string] | User-generated tags |
| `comments` | list[dict] | Full comment threads with metadata |

**Note on `secret` field**: This is a Flickr-specific parameter required for constructing direct image URLs. While normally sensitive, it's appropriate to include here because all photos are public domain and this enables the self-contained nature of Data Lifeboat archives.

## Usage

### Basic Loading

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("your-username/dataset-name")

# Access images and metadata
for example in dataset["train"]:
    image = example["image"]
    title = example["title"]
    tags = example["tags"]
    comments = example["comments"]
```

### Research Applications

This dataset is suitable for:
- **Computer Vision**: Image classification, scene understanding, visual content analysis
- **Vision-Language**: Image-to-text generation, visual question answering
- **Digital Humanities**: Cultural heritage analysis, community interaction studies
- **Information Retrieval**: Text-based image search, tag prediction, metadata analysis
- **Social Media Research**: Community engagement patterns, collaborative annotation

### Working with Comments and Tags

```python
# Analyze community engagement
for example in dataset["train"]:
    photo_id = example["photo_id"]
    num_comments = len(example["comments"])
    tags = example["tags"]
    
    print(f"Photo {{photo_id}}: {{num_comments}} comments, {{len(tags)}} tags")
```

## Source Data

### Data Collection

This collection originates from Flickr's Commons program, which partners with cultural heritage institutions to share photographs with no known copyright restrictions. Photos are selected based on cultural significance, community engagement, and preservation value.

### Community Annotations

All metadata reflects authentic community interaction:
- **Tags**: User-generated keywords and descriptions
- **Comments**: Community discussions and reactions
- **Engagement metrics**: Real view counts, favorites, and interactions

## Licensing and Ethics

### License Information

Photos in this collection have open licenses enabling reuse:
- Most common: "No known copyright restrictions"
- Also includes: Creative Commons licenses (CC0, CC BY, etc.)
- See `license_label` and `license_url` fields for specific licensing per image

### Ethical Considerations

- **Historical Context**: Some photos may depict people, places, or events from historical periods
- **Community Content**: Comments and tags reflect community perspectives at time of creation
- **Attribution**: Original photographers and institutions are preserved in metadata
- **Public Domain**: All content was publicly accessible on Flickr under open licenses

## Citation

```bibtex
@misc{{{lifeboat_meta.id.lower().replace("_", "_")}}},
  title = {{{lifeboat_meta.name}}},
  author = {{{lifeboat_meta.creator.name}}},
  year = {{2025}},
  publisher = {{Hugging Face}},
  url = {{https://huggingface.co/datasets/[dataset-url]}},
  note = {{Flickr Data Lifeboat digital preservation format}}
}}
```

## Additional Information

### Future Considerations

{lifeboat_meta.futureConsiderations}

### About Data Lifeboats

Learn more about the Data Lifeboat initiative:
- [Data Lifeboat Overview](https://www.flickr.org/programs/content-mobility/data-lifeboat/)
- [Flickr Foundation](https://www.flickr.org/foundation/)
- [Commons Program](https://www.flickr.com/commons)

### Dataset Maintenance

This dataset was created using the Flickr Foundation's Data Lifeboat format and converted to Hugging Face using automated tools that preserve all original metadata and community context.
"""
        return card

    def save_dataset_locally(
        self,
        dataset: Dataset,
        output_dir: Path,
        photos: List[Photo],
        datasets: Dict[str, pl.DataFrame],
    ):
        """Save dataset locally for testing"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Save dataset
        dataset.save_to_disk(str(output_dir / "dataset"))

        # Save dataset card
        card = self.create_dataset_card(photos, datasets)
        (output_dir / "README.md").write_text(card)

        print(f"Dataset saved to {output_dir}")

    def push_to_hub(
        self,
        dataset: Dataset,
        repo_id: str,
        photos: List[Photo],
        datasets: Dict[str, pl.DataFrame],
        private: bool = True,
    ):
        """Push dataset to Hugging Face Hub"""
        # Create dataset card
        dataset_card = self.create_dataset_card(photos, datasets)

        # Push to hub
        dataset.push_to_hub(
            repo_id,
            private=private,
            commit_message="Upload Flickr Commons Data Lifeboat dataset",
        )

        # Update README
        api = HfApi()
        api.upload_file(
            path_or_fileobj=dataset_card.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Update dataset card",
        )

        print(f"Dataset uploaded to https://huggingface.co/datasets/{repo_id}")

    def create_raw_dataset_card(self, lifeboat_meta=None) -> str:
        """Generate a dataset card for the raw Data Lifeboat format"""
        if lifeboat_meta is None:
            lifeboat_meta = self.loader.load_lifeboat_metadata()

        # Calculate basic stats for the raw format
        media_path = self.lifeboat_path / "media"
        total_files = 0
        total_size_mb = 0

        if media_path.exists():
            for file_path in media_path.rglob("*"):
                if file_path.is_file() and not file_path.name.startswith("."):
                    total_files += 1
                    total_size_mb += file_path.stat().st_size / (1024 * 1024)

        # Determine size category based on total size
        if total_size_mb < 1000:  # < 1GB
            size_cat = "n<1K"
        elif total_size_mb < 10000:  # < 10GB
            size_cat = "1K<n<10K"
        else:
            size_cat = "10K<n<100K"

        return f"""---
license: other
license_name: various-open-licenses
license_link: https://www.flickr.com/commons/usage/
tags:
- flickr-commons
- data-lifeboat
size_categories:
- {size_cat}
language:
- en
pretty_name: "{lifeboat_meta.name} (Raw Data Lifeboat)"
---

# {lifeboat_meta.name} - Raw Data Lifeboat

## Overview

This is the **raw, unprocessed version** of the {lifeboat_meta.name} Data Lifeboat - a self-contained digital preservation archive created by the [Flickr Foundation](https://www.flickr.org/programs/content-mobility/data-lifeboat/).

**🔗 ML-Ready Version**: For machine learning applications, see the processed dataset: [`{lifeboat_meta.name.lower().replace(" ", "-")}`](https://huggingface.co/datasets/USERNAME/PLACEHOLDER-PROCESSED)

## What is a Data Lifeboat?

Data Lifeboats are **self-contained archives** designed to preserve not just images, but the complete social and cultural context that makes them meaningful. Unlike traditional datasets, they include:

- 📁 **Complete file structure** with original organization
- 🌐 **Built-in web viewer** for browsing without external tools
- 📊 **Rich metadata** preserved in JavaScript format
- 🔗 **No external dependencies** - everything needed is included
- 🏛️ **Community context** - tags, comments, and social interactions

## Collection Details

- **Collection ID**: `{lifeboat_meta.id}`
- **Created**: {lifeboat_meta.dateCreated}
- **Curator**: {lifeboat_meta.creator.name}
- **Total Files**: ~{total_files:,}
- **Archive Size**: ~{total_size_mb:,.0f} MB

### Purpose

{lifeboat_meta.purpose}

## Archive Structure

```
{lifeboat_meta.id}/
├── viewer/                    # Built-in web viewer application
│   ├── index.html            # Main viewer interface  
│   ├── browse.html           # Browse photos interface
│   ├── photo.html           # Individual photo viewer
│   └── static/              # CSS, JavaScript, and assets
├── metadata/                 # All metadata in JavaScript format
│   ├── lifeboat.js          # Collection metadata
│   ├── photoIndex.js        # Index of all photos
│   ├── tagIndex.js          # Tag index and frequencies
│   ├── licenseIndex.js      # License information
│   ├── contributorIndex.js  # User/institution data
│   ├── albumIndex.js        # Album information
│   ├── galleryIndex.js      # Gallery data
│   ├── groupIndex.js        # Group information
│   └── photos/              # Individual photo metadata
│       ├── [photo_id].js    # One file per photo
│       └── ...
├── media/                   # All image files
│   ├── originals/           # Full resolution images
│   └── thumbnails/          # Thumbnail versions
└── README.html             # Collection documentation
```

## How to Use This Archive

### Option 1: Web Viewer (Recommended)
1. Download the entire archive
2. Open `viewer/index.html` in any web browser
3. Browse photos, view metadata, and explore the collection

### Option 2: Direct File Access
- **Images**: `media/originals/` and `media/thumbnails/`
- **Metadata**: `metadata/` directory contains all structured data
- **Documentation**: `README.html` for collection details

### Option 3: Programmatic Access
The metadata is stored in JavaScript format but can be easily parsed:

```python
import json
import re
from pathlib import Path

def parse_js_metadata(js_file_path):
    \"\"\"Parse JavaScript metadata files\"\"\"
    content = Path(js_file_path).read_text()
    # Extract JSON from: var variableName = {...};
    json_match = re.search(r'var\s+\w+\s*=\s*({{.*}});?', content, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(1))
    return None

# Load collection metadata
collection_info = parse_js_metadata('metadata/lifeboat.js')
photo_index = parse_js_metadata('metadata/photoIndex.js')
```

## Data Format and Standards

### Metadata Format
- **JavaScript Objects**: Structured data in `.js` files
- **UTF-8 Encoding**: All text files use UTF-8
- **Consistent Naming**: File names follow Flickr conventions
- **Cross-References**: IDs link photos, tags, comments, and users

### Image Format
- **Originals**: JPEG/PNG in original resolution and format
- **Thumbnails**: JPEG thumbnails for web viewing
- **Naming**: `[photo_id]_[secret]_[size].[format]`

### Licensing
All photos in this collection have open licenses:
- **Primary**: "No known copyright restrictions" (Commons)
- **Others**: Various Creative Commons licenses
- **See**: `metadata/licenseIndex.js` for complete license information

## Preservation Principles

This Data Lifeboat follows digital preservation best practices:

✅ **Self-Contained**: No external dependencies or API calls  
✅ **Standards-Based**: Uses HTML, CSS, JavaScript - universally supported  
✅ **Human-Readable**: Can be understood without specialized software  
✅ **Machine-Readable**: Structured data for computational analysis  
✅ **Documented**: Comprehensive metadata and documentation included  
✅ **Portable**: Works on any system with a web browser  

## Research Applications

This raw format is ideal for:

- **Digital Preservation Research**: Studying self-contained archive formats
- **Metadata Analysis**: Examining community-generated tags and comments
- **Cultural Heritage**: Preserving social context of cultural artifacts  
- **Web Archaeology**: Understanding historical web interfaces and formats
- **Custom Processing**: Building your own analysis tools

## Related Resources

### Processed Version
For machine learning and computational analysis, use the processed version:
- **Dataset**: [`{lifeboat_meta.name.lower().replace(" ", "-")}`](https://huggingface.co/datasets/USERNAME/PLACEHOLDER-PROCESSED)
- **Features**: Images as HuggingFace Image features, structured metadata
- **Ready-to-use**: Compatible with `datasets` library and ML frameworks

### Learn More
- [Data Lifeboat Initiative](https://www.flickr.org/programs/content-mobility/data-lifeboat/)
- [Flickr Foundation](https://www.flickr.org/foundation/)  
- [Flickr Commons Program](https://www.flickr.com/commons)

## Technical Notes

### Browser Compatibility
The viewer works in all modern browsers:
- Chrome/Chromium 70+
- Firefox 65+
- Safari 12+
- Edge 79+

### File Handling
- Some browsers may restrict local file access for security
- For full functionality, serve files through a local web server:
  ```bash
  # Python 3
  python -m http.server 8000
  
  # Node.js  
  npx serve .
  ```

## Citation

```bibtex
@misc{{{lifeboat_meta.id.lower()}_raw}},
  title = {{{lifeboat_meta.name} - Raw Data Lifeboat}},
  author = {{{lifeboat_meta.creator.name}}},
  year = {{2025}},
  publisher = {{Hugging Face}},
  url = {{https://huggingface.co/datasets/[your-username]/[dataset-name]-raw}},
  note = {{Self-contained digital preservation archive}}
}}
```

## Future Considerations

{lifeboat_meta.futureConsiderations}

---

**Preservation Notice**: This archive is designed to remain accessible indefinitely. The self-contained format ensures that future researchers can access and understand this collection even if external services or APIs change.
"""

    def upload_raw_lifeboat(self, repo_id: str, private: bool = True) -> str:
        """Upload the raw Data Lifeboat using upload_large_folder"""
        import shutil
        import tempfile

        api = HfApi()

        # Load metadata before any temporary operations
        lifeboat_meta = self.loader.load_lifeboat_metadata()

        # Define patterns to ignore during upload
        ignore_patterns = [
            ".DS_Store",
            "**/.DS_Store",
            "**/.git/**",
            "**/cache/**",
            "**/__pycache__/**",
            "**/*.pyc",
            "**/Thumbs.db",
        ]

        print(f"Preparing raw Data Lifeboat for upload to {repo_id}...")

        # Create a temporary directory to organize the upload
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            data_dir = temp_path / "data"
            data_dir.mkdir()

            # Copy the lifeboat directory into the data folder
            lifeboat_name = self.lifeboat_path.name
            target_dir = data_dir / lifeboat_name

            print(f"Copying {self.lifeboat_path} to temporary data directory...")
            shutil.copytree(
                str(self.lifeboat_path),
                str(target_dir),
                ignore=shutil.ignore_patterns(
                    ".DS_Store", ".git", "__pycache__", "*.pyc", "Thumbs.db"
                ),
            )

            print(f"Uploading raw Data Lifeboat to {repo_id}...")

            # Upload the data directory (which contains the lifeboat)
            api.upload_large_folder(
                repo_id=repo_id,
                folder_path=str(
                    temp_path
                ),  # Upload the temp directory containing "data/"
                repo_type="dataset",
                ignore_patterns=ignore_patterns,
                private=private,
                print_report=True,
            )

        # Create and upload the dataset card for raw format
        raw_card = self.create_raw_dataset_card()

        # Update the card with the correct paths and repo links
        processed_repo = repo_id.replace("-raw", "")
        raw_card = raw_card.replace("USERNAME/PLACEHOLDER-PROCESSED", processed_repo)

        # Update the archive structure in the card to show the data/ prefix
        lifeboat_name = self.lifeboat_path.name
        raw_card = raw_card.replace(
            f"```\n{lifeboat_meta.id}/",
            f"```\ndata/{lifeboat_name}/",
        )

        # Update usage instructions to reference the data/ path
        raw_card = raw_card.replace(
            "1. Download the entire archive\n2. Open `viewer/index.html` in any web browser",
            f"1. Download the entire archive\n2. Open `data/{lifeboat_name}/viewer/index.html` in any web browser",
        )

        raw_card = raw_card.replace(
            "- **Images**: `media/originals/` and `media/thumbnails/`\n- **Metadata**: `metadata/` directory contains all structured data\n- **Documentation**: `README.html` for collection details",
            f"- **Images**: `data/{lifeboat_name}/media/originals/` and `data/{lifeboat_name}/media/thumbnails/`\n- **Metadata**: `data/{lifeboat_name}/metadata/` directory contains all structured data\n- **Documentation**: `data/{lifeboat_name}/README.html` for collection details",
        )

        raw_card = raw_card.replace(
            "# Load collection metadata\ncollection_info = parse_js_metadata('metadata/lifeboat.js')\nphoto_index = parse_js_metadata('metadata/photoIndex.js')",
            f"# Load collection metadata\ncollection_info = parse_js_metadata('data/{lifeboat_name}/metadata/lifeboat.js')\nphoto_index = parse_js_metadata('data/{lifeboat_name}/metadata/photoIndex.js')",
        )

        api.upload_file(
            path_or_fileobj=raw_card.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Add dataset card for raw Data Lifeboat",
        )

        print(
            f"Raw Data Lifeboat uploaded to https://huggingface.co/datasets/{repo_id}"
        )
        print(f"Access the viewer at: data/{lifeboat_name}/viewer/index.html")
        return repo_id

    def upload_dual_datasets(
        self, base_repo_id: str, private: bool = True
    ) -> tuple[str, str]:
        """Upload both raw and processed versions of the Data Lifeboat"""
        raw_repo_id = f"{base_repo_id}-raw"
        processed_repo_id = base_repo_id

        print("=== Uploading Dual Data Lifeboat Datasets ===")

        # 1. Upload raw Data Lifeboat first
        print("\n1/2: Uploading Raw Data Lifeboat...")
        self.upload_raw_lifeboat(raw_repo_id, private=private)

        # 2. Get data for processed version
        print("\n2/2: Preparing Processed Dataset...")
        photos = self.loader.load_all_photos()
        datasets = self.converter.create_full_dataset()

        # Create processed dataset
        print("Creating HuggingFace Dataset...")
        processed_dataset = self.create_dataset()

        # 3. Upload processed version with cross-references
        print("Uploading Processed Dataset...")
        self.push_to_hub(
            processed_dataset, processed_repo_id, photos, datasets, private=private
        )

        # 4. Update processed dataset card to include raw version link
        processed_card = self.create_dataset_card(photos, datasets)

        # Add cross-reference section to processed card
        cross_ref_section = f"""
## Related Datasets

### Raw Data Lifeboat Version
This dataset also has a **raw, self-contained archive version** that preserves the original Data Lifeboat format:

**🗂️ Raw Archive**: [`{base_repo_id}-raw`](https://huggingface.co/datasets/{raw_repo_id})

The raw version includes:
- Built-in web viewer (`viewer/index.html`)
- Complete original file structure  
- Self-contained archive (no external dependencies)
- Ideal for digital preservation and archival research

Choose the raw version if you need the complete preservation format or want to use the built-in viewer.
"""

        # Insert cross-reference after "Dataset Description" section
        processed_card = processed_card.replace(
            "## Dataset Statistics", cross_ref_section + "\n## Dataset Statistics"
        )

        # Update processed dataset card
        api = HfApi()
        api.upload_file(
            path_or_fileobj=processed_card.encode(),
            path_in_repo="README.md",
            repo_id=processed_repo_id,
            repo_type="dataset",
            commit_message="Update dataset card with raw version cross-reference",
        )

        print("\n=== Upload Complete ===")
        print(
            f"📊 Processed Dataset: https://huggingface.co/datasets/{processed_repo_id}"
        )
        print(f"🗂️ Raw Data Lifeboat: https://huggingface.co/datasets/{raw_repo_id}")

        return processed_repo_id, raw_repo_id


def main():
    """Main CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert Data Lifeboat to HuggingFace Dataset"
    )
    parser.add_argument("lifeboat_path", help="Path to Data Lifeboat directory")
    parser.add_argument("--save-local", help="Save dataset locally to this directory")
    parser.add_argument(
        "--push-to-hub",
        help="Push to HuggingFace Hub with this repo ID (e.g., 'username/dataset-name')",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the dataset private on HuggingFace Hub",
    )
    parser.add_argument(
        "--create-space",
        help="Create a Space for interactive Data Lifeboat viewing (e.g., 'username/space-name')",
    )
    parser.add_argument(
        "--raw-dataset-repo-id",
        help="Raw dataset repository ID (auto-detected from --push-to-hub if not specified)",
    )

    # New upload format options
    format_group = parser.add_mutually_exclusive_group()
    format_group.add_argument(
        "--raw-only",
        action="store_true",
        help="Upload only the raw Data Lifeboat format",
    )
    format_group.add_argument(
        "--processed-only",
        action="store_true",
        help="Upload only the processed HuggingFace dataset",
    )
    format_group.add_argument(
        "--both",
        action="store_true",
        default=True,
        help="Upload both raw and processed versions (default)",
    )

    args = parser.parse_args()

    # Convert Data Lifeboat
    converter = LifeboatToHuggingFace(args.lifeboat_path)

    # Handle Space creation
    if args.create_space:
        # Smart default: if uploading to hub and no explicit raw repo ID, auto-detect it
        raw_dataset_repo_id = args.raw_dataset_repo_id
        if not raw_dataset_repo_id and args.push_to_hub:
            raw_dataset_repo_id = f"{args.push_to_hub}-raw"
            print(f"🤖 Auto-detected raw dataset repository: {raw_dataset_repo_id}")

        if not raw_dataset_repo_id:
            print(
                "❌ Error: --raw-dataset-repo-id is required when using --create-space without --push-to-hub"
            )
            print("Examples:")
            print(
                "  # Auto-detect from upload: --create-space username/space-name --push-to-hub username/dataset-name"
            )
            print(
                "  # Explicit raw repo: --create-space username/space-name --raw-dataset-repo-id username/dataset-raw"
            )
            exit(1)

        space_creator = LifeboatToDynamicSpace(args.lifeboat_path)
        space_creator.create_dynamic_space(
            repo_id=args.create_space,
            raw_dataset_repo_id=raw_dataset_repo_id,
            private=args.private,
        )
        # Exit after creating space if no other operations requested
        if not (args.push_to_hub or args.save_local):
            exit(0)

    # Handle different upload modes
    if args.push_to_hub:
        if args.raw_only:
            # Upload only raw version
            raw_repo_id = f"{args.push_to_hub}-raw"
            converter.upload_raw_lifeboat(raw_repo_id, private=args.private)

        elif args.processed_only:
            # Upload only processed version (original behavior)
            print("Loading all photos...")
            photos = converter.loader.load_all_photos()

            print("Creating Polars DataFrames...")
            datasets = converter.converter.create_full_dataset()

            print("Creating HuggingFace Dataset...")
            dataset = converter.create_dataset()

            print("\nDataset created successfully!")
            print(f"Number of examples: {len(dataset)}")
            print(f"Features: {list(dataset.features.keys())}")

            converter.push_to_hub(
                dataset, args.push_to_hub, photos, datasets, private=args.private
            )

        else:  # --both (default)
            # Upload both versions with cross-references
            converter.upload_dual_datasets(args.push_to_hub, private=args.private)

    elif args.save_local:
        # For local save, only create processed version
        print("Loading all photos...")
        photos = converter.loader.load_all_photos()

        print("Creating Polars DataFrames...")
        datasets = converter.converter.create_full_dataset()

        print("Creating HuggingFace Dataset...")
        dataset = converter.create_dataset()

        print("\nDataset created successfully!")
        print(f"Number of examples: {len(dataset)}")
        print(f"Features: {list(dataset.features.keys())}")

        converter.save_dataset_locally(dataset, args.save_local, photos, datasets)

    else:
        # No upload specified, just create and show info
        print("Loading all photos...")
        photos = converter.loader.load_all_photos()

        print("Creating Polars DataFrames...")
        datasets = converter.converter.create_full_dataset()

        print("Creating HuggingFace Dataset...")
        dataset = converter.create_dataset()

        print("\nDataset created successfully!")
        print(f"Number of examples: {len(dataset)}")
        print(f"Features: {list(dataset.features.keys())}")
        print("\nUse --push-to-hub or --save-local to save the dataset.")


if __name__ == "__main__":
    main()
