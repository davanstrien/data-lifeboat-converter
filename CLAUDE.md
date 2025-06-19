# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Data Lifeboat - a digital preservation tool from the Flickr Foundation designed to archive meaningful collections of Flickr photos. It's a self-contained HTML/JavaScript viewer with no external dependencies, ensuring long-term accessibility of archived photo collections.

## Architecture

The project consists of two main components:

1. **Data Layer** (`Commons_1K_2025/`):
   - `media/`: Contains original photos and thumbnails
   - `metadata/`: JavaScript files containing structured data about the collection
   - Key metadata files include `lifeboat.js` (collection info), `photoIndex.js`, `tagIndex.js`, `licenseIndex.js`, etc.

2. **Viewer Application** (`viewer/`):
   - Static HTML pages that load and display the collection
   - JavaScript files in `viewer/static/` handle the interactive functionality
   - No build process required - runs directly in the browser

## Development Guidelines

Since this is a static site with no build process:
- Edit JavaScript and HTML files directly
- Test by opening HTML files in a web browser
- Ensure all paths are relative to maintain portability
- The viewer expects data files to be in specific locations relative to itself

## Key Technical Details

- All metadata is stored as JavaScript objects assigned to global variables
- The viewer dynamically loads these JavaScript files to access the data
- No external dependencies or API calls - everything is self-contained
- Designed for long-term preservation, prioritizing simplicity and durability over modern web frameworks

## Data Lifeboat to HuggingFace Conversion

This repository includes tools to convert Data Lifeboats to HuggingFace datasets in two complementary formats.

### Current Implementation Status

#### ✅ **Completed Features**
- **Pydantic data validation** with nullable fields for robust parsing of JavaScript metadata
- **Polars DataFrame conversion** for efficient data manipulation and statistics
- **Processed dataset creation** with HuggingFace Image features and rich metadata
- **Raw dataset upload** using `upload_large_folder()` with proper directory structure
- **Generic dataset cards** with auto-generated statistics and cross-references
- **CLI interface** supporting multiple upload modes

#### ⚠️ **Known Issues**
- **Dual upload bug**: Metadata loading fails after temporary directory cleanup in `upload_raw_lifeboat()`
  - Workaround: Use `--raw-only` or `--processed-only` for now
  - Fix in progress: Pass metadata as parameter to avoid re-loading

#### ✅ **Successfully Tested**
- Individual raw uploads with proper `data/Commons_1K_2025/` structure
- Individual processed uploads with 1,000 photos and full metadata
- Cross-linking between dataset versions in README cards
- **Static Space deployment** with app_file approach (ENOLA_GAY collection)
- **Docker Space deployment** with HTTP server (ENOLA_GAY collection)
- **Backward compatibility** across all Data Lifeboat format versions

### Dual Upload Strategy

The conversion creates **two complementary datasets**:

#### 1. **Raw Data Lifeboat** (`repo-name-raw`)
- **Structure**: `data/Commons_1K_2025/` containing complete original archive
- **Access**: Built-in viewer at `data/Commons_1K_2025/viewer/index.html`
- **Use cases**: Digital preservation, archival research, web archaeology
- **Format**: Self-contained HTML/JS with no external dependencies

#### 2. **Processed Dataset** (`repo-name`)
- **Structure**: HuggingFace Dataset with Image features and structured metadata
- **Access**: Standard `datasets` library compatibility
- **Use cases**: Machine learning, computer vision, NLP research
- **Format**: Optimized parquet files with automatic image loading

### Usage Commands

```bash
# Upload both versions (default - currently has bug)
uv run lifeboat_to_hf_dataset.py Commons_1K_2025 --push-to-hub username/dataset-name

# Upload only raw Data Lifeboat (✅ working)
uv run lifeboat_to_hf_dataset.py Commons_1K_2025 --push-to-hub username/dataset-name --raw-only

# Upload only processed dataset (✅ working)
uv run lifeboat_to_hf_dataset.py Commons_1K_2025 --push-to-hub username/dataset-name --processed-only

# Save locally for testing (✅ working)
uv run lifeboat_to_hf_dataset.py Commons_1K_2025 --save-local ./output

# Add --private flag for private repositories
uv run lifeboat_to_hf_dataset.py Commons_1K_2025 --push-to-hub username/dataset-name --raw-only --private
```

### Technical Architecture

#### **Data Processing Pipeline**
1. **JavaScript parsing** extracts JSON from `.js` metadata files
2. **Pydantic validation** ensures data integrity with nullable field handling
3. **Polars transformation** creates structured DataFrames with joins and statistics
4. **HuggingFace conversion** generates Dataset with Image features and rich metadata

#### **Upload Strategies**
- **Raw uploads**: `upload_large_folder()` with file filtering and `data/` organization
- **Processed uploads**: `datasets.push_to_hub()` with automatic image handling
- **Cross-linking**: README cards include references between versions

#### **File Organization**
```
Raw Dataset Repository:
├── README.md                    # Dataset card with preservation focus
└── data/
    └── Commons_1K_2025/         # Original Data Lifeboat structure
        ├── viewer/index.html    # Built-in web interface
        ├── metadata/            # JavaScript data files
        └── media/               # Images (originals + thumbnails)

Processed Dataset Repository:
├── README.md                    # Dataset card with ML focus
├── data/                        # HuggingFace dataset shards
└── dataset_info.json           # Dataset configuration
```

### HuggingFace Space Hosting

The converter creates **Dynamic Docker Spaces** that download Data Lifeboats at runtime, enabling hosting of any size collection without size limitations:

#### **✅ Dynamic Spaces (Runtime Download)**
**Scalable hosting** that downloads Data Lifeboats from raw dataset repositories:

```bash
# Upload dataset + create space (auto-detects raw repo)
uv run lifeboat_to_hf_dataset.py Commons_1K_2025 --push-to-hub username/dataset-name --create-space username/space-name

# Override raw dataset source if needed
uv run lifeboat_to_hf_dataset.py Commons_1K_2025 --push-to-hub username/dataset-name --create-space username/space-name --raw-dataset-repo-id username/different-raw

# Create space from existing raw dataset
uv run lifeboat_to_hf_dataset.py Commons_1K_2025 --create-space username/space-name --raw-dataset-repo-id username/existing-raw
```

**How it works:**
- Docker container downloads the raw Data Lifeboat from HuggingFace Hub at startup
- Serves the complete archive using Python HTTP server
- Handles any Data Lifeboat size (no 50GB limitation)
- Automatic redirect from root to `README.html` entry point
- Links to source raw dataset repository in Space metadata

**Key benefits:**
- **One command workflow** - Upload dataset + create space in single command
- **Intelligent defaults** - Auto-detects raw dataset repository from upload
- **No size limits** - Can host Data Lifeboats of any size
- **Automatic provisioning** - Downloads content dynamically when Space starts
- **Archival integrity** - Serves Data Lifeboat exactly as created
- **Flexible overrides** - Can specify custom raw dataset sources
- **Cross-referencing** - Links between Space and dataset repositories
- **Free hosting** - Leverages HuggingFace's infrastructure

**Technical approach:**
1. User visits Space URL
2. Docker container starts and downloads raw Data Lifeboat
3. HTTP server serves Data Lifeboat with redirect to README.html
4. All relative paths work correctly: `viewer/`, `metadata/`, `media/`

**Space features:**
- **Interactive viewers** - Full Data Lifeboat functionality preserved
- **Self-contained** - No external dependencies once downloaded
- **Cross-platform** - Works on any modern browser
- **Version control** - Git-based repository management
- **Metadata linking** - Space README references source dataset

### Next Steps

#### **High Priority**
1. **Fix dual upload bug** - Modify `upload_raw_lifeboat()` to pass metadata as parameter
2. **Test complete dual upload workflow** with proper cross-references
3. **Add error handling** for network interruptions and partial uploads

#### **Future Enhancements**
- **Batch processing** for multiple Data Lifeboats
- **Resume capability** for interrupted uploads
- **Validation checks** to ensure Data Lifeboat completeness before upload
- **Custom dataset splits** (train/validation/test) for ML applications
- **Geographic filtering** for location-based subsets
- **Combined workflow** - Create Space + Dataset in one command

### Data Quality and Validation

#### **Robust Parsing**
- **Nullable fields**: All Pydantic models handle missing/null data gracefully
- **Type safety**: Strict validation with informative error messages
- **Format detection**: Automatic handling of JPG/PNG images and various metadata formats

#### **Statistics Generation**
- **Collection overview**: Photo counts, date ranges, geographic coverage
- **Engagement metrics**: Views, favorites, comments with averages and maximums  
- **License distribution**: Breakdown of open license types
- **Contributor analysis**: Top contributing institutions and users

### Performance Characteristics

#### **Processing Speed**
- **1,000 photos**: ~2-3 minutes for complete processing
- **Large collections**: Polars enables efficient handling of 10K+ photos
- **Memory usage**: Optimized for collections up to 100K photos

#### **Upload Efficiency**
- **Raw uploads**: `upload_large_folder()` with resumable capability
- **Processed uploads**: Automatic sharding for optimal performance
- **File filtering**: Excludes `.DS_Store`, cache files, and development artifacts