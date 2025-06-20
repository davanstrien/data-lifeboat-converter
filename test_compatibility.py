from pathlib import Path

from lifeboat_to_hf import LifeboatLoader

# Test loading all three Data Lifeboats
lifeboats = ["Commons_1K_2025", "Commons-Smiles", "ENOLA_GAY"]

for lb in lifeboats:
    print(f"\n=== Testing {lb} ===")
    try:
        loader = LifeboatLoader(Path(lb))

        # Test lifeboat metadata loading
        metadata = loader.load_lifeboat_metadata()
        print(f"✅ Metadata loaded: {metadata.name}")

        # Test photo index loading
        photo_index = loader.load_photo_index()
        print(f"✅ Photo index loaded: {len(photo_index)} photos")

        # Test loading first photo
        first_photo_id = next(iter(photo_index))
        photo = loader.load_photo(first_photo_id)
        print(f"✅ Sample photo loaded: {photo.title or photo.id}")
        print(f"   Media type: {photo.get_media_type()}")
        print(f"   File paths: {photo.get_file_paths()}")

    except Exception as e:
        print(f"❌ Error: {e}")
