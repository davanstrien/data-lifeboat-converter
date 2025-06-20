# Data Lifeboat to HuggingFace Converter

Convert [Data Lifeboats](https://www.flickr.org/new-research-report-on-data-lifeboat/) from the [Flickr Foundation](https://www.flickr.org/) into HuggingFace datasets and interactive Spaces for preservation, research, and machine learning.

## 🚀 Quick Start

### Installation Options

We suggest using [`uv`](https://docs.astral.sh/uv/) for installing this tool. 

#### Option 1: Install Tool Globally (Recommended)

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install the tool globally from GitHub
uv tool install git+https://github.com/davanstrien/data-lifeboat-converter.git

# Now use it anywhere
lifeboat-to-hf path/to/Data_Lifeboat --push-to-hub username/dataset-name
```

#### Option 2: Run Directly from GitHub

```bash
# Run without installing (downloads and runs each time)
uvx --from git+https://github.com/davanstrien/data-lifeboat-converter.git lifeboat-to-hf path/to/Data_Lifeboat --push-to-hub username/dataset-name
```

#### Option 3: Clone and Run Locally

```bash
# Clone repository and run with uv
git clone https://github.com/davanstrien/data-lifeboat-converter.git
cd data-lifeboat-converter
uv run lifeboat_to_hf_dataset.py path/to/Data_Lifeboat --push-to-hub username/dataset-name
```

### Basic Usage

```bash
# Upload dataset to HuggingFace Hub (creates both raw and processed versions)
lifeboat-to-hf path/to/Data_Lifeboat --push-to-hub username/dataset-name

# Upload dataset + create interactive Space in one command
lifeboat-to-hf path/to/Data_Lifeboat --push-to-hub username/dataset-name --create-space username/space-name

# Save dataset locally for testing
lifeboat-to-hf path/to/Data_Lifeboat --save-local ./output
```

## 📦 What This Tool Creates

### 1. **Raw Data Lifeboat** (`repo-name-raw`)
- **Purpose:** Digital preservation in original format
- **Contents:** Complete Data Lifeboat archive with built-in viewer
- **Access:** `data/LIFEBOAT_NAME/viewer/index.html`
- **Use cases:** Archival research, digital humanities, web archaeology

### 2. **Processed Dataset** (`repo-name`)
- **Purpose:** Machine learning and analysis
- **Contents:** HuggingFace Dataset with Image features and structured metadata
- **Access:** Standard `datasets` library compatibility
- **Use cases:** Computer vision, NLP research, data analysis

### 3. **Interactive Space** (optional)
- **Purpose:** Web-based Data Lifeboat viewer
- **Technology:** Dynamic Docker Space with runtime download
- **Features:** Full Data Lifeboat functionality, any size support
- **Use cases:** Public access, demos, collaborative research

## 📋 Examples

### Upload Datasets

```bash
# Upload both raw and processed versions (default)
lifeboat-to-hf Commons_1K_2025 --push-to-hub myusername/flickr-commons-1k

# Upload only raw Data Lifeboat
lifeboat-to-hf Commons_1K_2025 --push-to-hub myusername/flickr-commons-1k --raw-only

# Upload only processed dataset
lifeboat-to-hf Commons_1K_2025 --push-to-hub myusername/flickr-commons-1k --processed-only

# Make repositories private
lifeboat-to-hf Commons_1K_2025 --push-to-hub myusername/flickr-commons-1k --private
```

### Create Interactive Spaces

```bash
# Create Space (auto-detects raw dataset from upload)
lifeboat-to-hf Commons_1K_2025 --push-to-hub myusername/dataset --create-space myusername/viewer

# Create Space with custom raw dataset source
lifeboat-to-hf Commons_1K_2025 --create-space myusername/viewer --raw-dataset-repo-id myusername/existing-raw

# Create private Space
lifeboat-to-hf Commons_1K_2025 --push-to-hub myusername/dataset --create-space myusername/viewer --private
```

### Local Development

```bash
# Save locally for testing
lifeboat-to-hf Commons_1K_2025 --save-local ./my-dataset

# Process different format options locally
lifeboat-to-hf Commons_1K_2025 --save-local ./output --processed-only
```

## 🔧 Command Options

| Option | Description | Example |
|--------|-------------|---------|
| `lifeboat_path` | Path to Data Lifeboat directory (required) | `Commons_1K_2025` |
| `--push-to-hub` | Upload to HuggingFace Hub | `username/dataset-name` |
| `--create-space` | Create interactive Space | `username/space-name` |
| `--raw-dataset-repo-id` | Raw dataset source (auto-detected if not specified) | `username/raw-repo` |
| `--save-local` | Save dataset locally instead of uploading | `./output-directory` |
| `--private` | Make repositories private | (flag) |
| `--raw-only` | Upload only raw Data Lifeboat | (flag) |
| `--processed-only` | Upload only processed dataset | (flag) |

## 🌐 Interactive Spaces

Dynamic Spaces download Data Lifeboats at runtime, enabling hosting of any size collection:

### How They Work
1. User visits Space URL
2. Docker container downloads raw Data Lifeboat from HuggingFace Hub
3. HTTP server serves the complete archive
4. All Data Lifeboat features work: viewer, search, metadata browsing

### Benefits
- **Big size limits** - Can host multi-gigabyte collections
- **Automatic provisioning** - Content downloaded when Space starts
- **Archival integrity** - Serves Data Lifeboat exactly as created
- **Free hosting** - Leverages HuggingFace infrastructure

## 📊 Dataset Features

### Processed Dataset Structure
```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("username/dataset-name")

# Access features
print(dataset.features)
# ['image', 'thumbnail', 'photo_id', 'title', 'description', 
#  'uploader_username', 'license_label', 'date_taken', 'tags', ...]

# Work with images and metadata
for example in dataset['train']:
    image = example['image']  # PIL Image
    title = example['title']
    tags = example['tags']    # List of tag strings
    # ... analysis code
```

### Rich Metadata Included
- **Images:** Original photos + thumbnails
- **Descriptive:** Titles, descriptions, tags
- **Technical:** Upload dates, formats, dimensions
- **Social:** View counts, favorites, comments
- **Geographic:** Latitude/longitude (when available)
- **Legal:** License information and URLs

## 🔐 Authentication

For private repositories or uploading, authenticate with HuggingFace:

```bash
# Login to HuggingFace Hub
huggingface-cli login

# Or set token as environment variable
export HUGGINGFACE_HUB_TOKEN="your_token_here"
```

## 🐛 Troubleshooting

### Common Issues

**"Repository not found" errors:**
- Ensure you're authenticated for private repos
- Check repository names are spelled correctly
- Verify the raw dataset exists when creating Spaces

**Memory issues with large collections:**
- Use `--raw-only` for very large Data Lifeboats
- Process smaller subsets locally first with `--save-local`

**Space startup failures:**
- Check the raw dataset repository exists and is accessible
- Verify the Space logs in the HuggingFace Spaces interface

### Getting Help

1. Check the Space logs on HuggingFace for runtime errors
2. Verify your Data Lifeboat has the expected structure
3. Test locally with `--save-local` before uploading

## 📚 About Data Lifeboats

Data Lifeboats are self-contained digital preservation archives created by the [Flickr Foundation](https://www.flickr.org/). They preserve not just images, but complete social and cultural context:

- ✅ Original high-quality photos
- ✅ Complete metadata (titles, descriptions, tags, dates, locations)
- ✅ Interactive web viewer (no external dependencies)
- ✅ Structured data for research and analysis

Learn more about Data Lifeboats and their role in digital preservation at [flickr.org](https://www.flickr.org/).

---

*This tool helps bridge digital preservation (Data Lifeboats) with modern ML/research infrastructure (HuggingFace), ensuring valuable cultural collections remain accessible for future generations.*
