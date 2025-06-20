#!/usr/bin/env python3
"""
Convert Flickr Data Lifeboat to Hugging Face Dataset using Pydantic and Polars
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import polars as pl
from pydantic import BaseModel, Field, field_validator

# Pydantic Models for Data Validation


class DateTaken(BaseModel):
    """Date taken with granularity information"""

    value: str
    granularity: Literal["second", "minute", "hour", "day", "month", "year", "circa"]


class Tag(BaseModel):
    """Tag with author and normalization info"""

    authorId: Optional[str] = Field(None, alias="authorId")
    rawValue: Optional[str] = Field(None, alias="rawValue")
    normalizedValue: Optional[str] = Field(None, alias="normalizedValue")
    isMachineTag: bool = Field(False, alias="isMachineTag")

    class Config:
        populate_by_name = True


class Comment(BaseModel):
    """Comment on a photo"""

    id: Optional[str] = None
    authorId: Optional[str] = Field(None, alias="authorId")
    text: Optional[str] = None
    date: Optional[datetime] = None


class ThumbnailInfo(BaseModel):
    """Thumbnail metadata"""

    path: str
    width: int
    height: int


class DetailedFileInfo(BaseModel):
    """Detailed file info for older Data Lifeboat format"""

    path: str
    label: str
    content_type: str
    width: int
    height: int
    media: str
    size: int
    checksum: str


class SimpleFiles(BaseModel):
    """Simple files format for newer Data Lifeboats"""

    original: str
    thumbnail: str


class ComplexFiles(BaseModel):
    """Complex files format for older Data Lifeboats"""

    original: DetailedFileInfo
    thumbnail: DetailedFileInfo


# Union type to handle both file formats
FilesType = Union[SimpleFiles, ComplexFiles]


class Location(BaseModel):
    """Geographic location (if available)"""

    latitude: Optional[float] = None
    longitude: Optional[float] = None
    accuracy: Optional[int] = None
    context: Optional[str] = None

    # Add place information if available
    placeId: Optional[str] = Field(None, alias="placeId")
    woeid: Optional[str] = None

    class Config:
        populate_by_name = True


class Photo(BaseModel):
    """Complete photo metadata"""

    id: str  # Keep required - essential for identification
    secret: Optional[str] = None
    url: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    uploaderId: Optional[str] = Field(None, alias="uploaderId")
    licenseId: Optional[str] = Field(None, alias="licenseId")
    dateTaken: Optional[DateTaken] = Field(None, alias="dateTaken")
    dateUploaded: Optional[datetime] = Field(None, alias="dateUploaded")
    location: Optional[Location] = None
    tags: List[Tag] = []
    countFaves: int = Field(0, alias="countFaves")
    countViews: int = Field(0, alias="countViews")
    countComments: int = Field(0, alias="countComments")
    albumIds: List[str] = Field(default_factory=list, alias="albumIds")
    galleryIds: List[str] = Field(default_factory=list, alias="galleryIds")
    groupIds: List[str] = Field(default_factory=list, alias="groupIds")
    safetyLevel: Optional[str] = Field(None, alias="safetyLevel")
    visibility: Optional[str] = "public"
    originalFormat: Optional[str] = Field(None, alias="originalFormat")
    mediaType: Optional[str] = Field(None, alias="mediaType")
    media: Optional[str] = None  # For older lifeboats
    files: Optional[FilesType] = None
    comments: List[Comment] = []

    @field_validator("files", mode="before")
    @classmethod
    def validate_files(cls, v):
        """Handle both simple and complex file formats"""
        if v is None:
            return None

        # Check if it's the simple format (strings)
        if isinstance(v.get("original"), str) and isinstance(v.get("thumbnail"), str):
            return SimpleFiles(**v)

        # Check if it's the complex format (objects)
        elif isinstance(v.get("original"), dict) and isinstance(
            v.get("thumbnail"), dict
        ):
            return ComplexFiles(**v)

        return v

    def get_media_type(self) -> str:
        """Get media type from either field"""
        return self.mediaType or self.media or "photo"

    def get_file_paths(self) -> tuple[Optional[str], Optional[str]]:
        """Extract file paths from either format"""
        if not self.files:
            return None, None

        if isinstance(self.files, SimpleFiles):
            return self.files.original, self.files.thumbnail
        elif isinstance(self.files, ComplexFiles):
            return self.files.original.path, self.files.thumbnail.path
        else:
            return None, None

    class Config:
        populate_by_name = True


class PhotoIndexEntry(BaseModel):
    """Entry in the photo index"""

    title: Optional[str] = None
    uploaderId: Optional[str] = Field(None, alias="uploaderId")
    licenseId: Optional[str] = Field(None, alias="licenseId")
    dateTaken: Optional[DateTaken] = Field(None, alias="dateTaken")
    dateUploaded: Optional[datetime] = Field(None, alias="dateUploaded")
    tags: List[str] = []
    countComments: int = Field(0, alias="countComments")
    countFaves: int = Field(0, alias="countFaves")
    countViews: int = Field(0, alias="countViews")
    visibility: Optional[str] = "public"
    thumbnail: Optional[ThumbnailInfo] = None
    metadataPath: Optional[str] = Field(None, alias="metadataPath")

    class Config:
        populate_by_name = True


class License(BaseModel):
    """License information"""

    label: Optional[str] = None
    url: Optional[str] = None


class ContributorInfo(BaseModel):
    """Contribution counts"""

    photo: int = 0
    tag: int = 0
    comment: int = 0


class Contributor(BaseModel):
    """Contributor/user information"""

    username: Optional[str] = None
    realname: Optional[str] = None
    pathAlias: Optional[str] = Field(None, alias="pathAlias")
    contributions: Optional[ContributorInfo] = None

    class Config:
        populate_by_name = True


class Album(BaseModel):
    """Album metadata"""

    title: Optional[str] = None
    ownerId: Optional[str] = Field(None, alias="ownerId")
    countPhotos: int = Field(0, alias="countPhotos")
    countVideos: int = Field(0, alias="countVideos")
    countViews: int = Field(0, alias="countViews")
    countComments: int = Field(0, alias="countComments")
    photoIds: List[str] = Field(default_factory=list, alias="photoIds")

    class Config:
        populate_by_name = True


class Gallery(BaseModel):
    """Gallery metadata"""

    title: Optional[str] = None
    ownerId: Optional[str] = Field(None, alias="ownerId")
    countPhotos: int = Field(0, alias="countPhotos")
    countVideos: int = Field(0, alias="countVideos")
    countViews: int = Field(0, alias="countViews")
    countComments: int = Field(0, alias="countComments")
    description: Optional[str] = None
    photoIds: List[str] = Field(default_factory=list, alias="photoIds")

    class Config:
        populate_by_name = True


class Group(BaseModel):
    """Group metadata"""

    name: Optional[str] = None
    nsid: Optional[str] = None
    eighteenPlus: bool = Field(False, alias="eighteenPlus")
    iconserver: Optional[str] = None
    iconfarm: int = 0
    members: int = 0
    poolCount: int = Field(0, alias="poolCount")
    topicCount: int = Field(0, alias="topicCount")
    privacy: int = 0
    lang: Optional[str] = None
    ispoolmoderated: bool = False
    photoIds: List[str] = Field(default_factory=list, alias="photoIds")

    class Config:
        populate_by_name = True


class Creator(BaseModel):
    """Creator information"""

    id: str
    name: str


class LifeboatMetadata(BaseModel):
    """Main collection metadata"""

    id: str
    name: str
    creator: Creator
    dateCreated: datetime = Field(alias="dateCreated")
    purpose: str
    futureConsiderations: str = Field(alias="futureConsiderations")

    class Config:
        populate_by_name = True


# JavaScript Parser


class JavaScriptParser:
    """Parse JavaScript files containing data objects"""

    @staticmethod
    def parse_js_file(file_path: Path) -> dict:
        """
        Extract JSON data from JavaScript file
        Handles format: var variableName = {...};
        """
        content = file_path.read_text(encoding="utf-8")

        # Remove JavaScript variable declaration
        # Pattern matches: var name = {data}; or const name = {data};
        pattern = r"(?:var|const|let)\s+\w+\s*=\s*({[\s\S]*});?\s*$"
        match = re.search(pattern, content)

        if match:
            json_str = match.group(1)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # Try to fix common issues
                # Remove trailing commas
                json_str = re.sub(r",\s*}", "}", json_str)
                json_str = re.sub(r",\s*]", "]", json_str)
                return json.loads(json_str)
        else:
            raise ValueError(f"Could not extract JSON data from {file_path}")

    @staticmethod
    def parse_js_metadata(js_file_path: str) -> dict:
        """
        Parse JavaScript metadata files (user-friendly wrapper)
        Takes string path and returns parsed JSON data
        """
        from pathlib import Path
        
        file_path = Path(js_file_path)
        # Use the existing robust parse_js_file method
        return JavaScriptParser.parse_js_file(file_path)


# Data Lifeboat Loader


class LifeboatLoader:
    """Load and validate Data Lifeboat contents"""

    def __init__(self, lifeboat_path: Path):
        self.root_path = Path(lifeboat_path)
        self.metadata_path = self.root_path / "metadata"
        self.media_path = self.root_path / "media"
        self.parser = JavaScriptParser()

    def load_lifeboat_metadata(self) -> LifeboatMetadata:
        """Load main collection metadata, with defaults for older formats"""
        lifeboat_file = self.metadata_path / "lifeboat.js"

        if lifeboat_file.exists():
            data = self.parser.parse_js_file(lifeboat_file)
            return LifeboatMetadata(**data)
        else:
            # Return default metadata for older lifeboats without lifeboat.js
            print(
                f"Warning: No lifeboat.js found, using defaults for {self.root_path.name}"
            )
            return LifeboatMetadata(
                id=self.root_path.name.upper(),
                name=f"Flickr Data Lifeboat: {self.root_path.name}",
                creator=Creator(id="unknown", name="Unknown Creator"),
                dateCreated=datetime.now(),
                purpose="Legacy Data Lifeboat collection",
                futureConsiderations="This is an older format Data Lifeboat without lifeboat.js metadata.",
            )

    def load_photo_index(self) -> Dict[str, PhotoIndexEntry]:
        """Load photo index"""
        data = self.parser.parse_js_file(self.metadata_path / "photoIndex.js")
        return {
            photo_id: PhotoIndexEntry(**photo_data)
            for photo_id, photo_data in data.items()
        }

    def load_licenses(self) -> Dict[str, License]:
        """Load license index"""
        data = self.parser.parse_js_file(self.metadata_path / "licenseIndex.js")
        return {
            license_id: License(**license_data)
            for license_id, license_data in data.items()
        }

    def load_contributors(self) -> Dict[str, Contributor]:
        """Load contributor index"""
        data = self.parser.parse_js_file(self.metadata_path / "contributorIndex.js")
        return {
            user_id: Contributor(**user_data) for user_id, user_data in data.items()
        }

    def load_albums(self) -> Dict[str, Album]:
        """Load album index"""
        data = self.parser.parse_js_file(self.metadata_path / "albumIndex.js")
        return {album_id: Album(**album_data) for album_id, album_data in data.items()}

    def load_galleries(self) -> Dict[str, Gallery]:
        """Load gallery index"""
        data = self.parser.parse_js_file(self.metadata_path / "galleryIndex.js")
        return {
            gallery_id: Gallery(**gallery_data)
            for gallery_id, gallery_data in data.items()
        }

    def load_groups(self) -> Dict[str, Group]:
        """Load group index"""
        data = self.parser.parse_js_file(self.metadata_path / "groupIndex.js")
        return {group_id: Group(**group_data) for group_id, group_data in data.items()}

    def load_tags(self) -> Dict[str, int]:
        """Load tag frequencies"""
        return self.parser.parse_js_file(self.metadata_path / "tagIndex.js")

    def load_photo(self, photo_id: str) -> Photo:
        """Load individual photo metadata"""
        photo_path = self.metadata_path / "photos" / f"{photo_id}.js"
        data = self.parser.parse_js_file(photo_path)
        return Photo(**data)

    def load_all_photos(self) -> List[Photo]:
        """Load all photo metadata"""
        photo_index = self.load_photo_index()
        photos = []

        for photo_id in photo_index:
            try:
                photo = self.load_photo(photo_id)
                photos.append(photo)
            except Exception as e:
                print(f"Error loading photo {photo_id}: {e}")

        return photos


# Polars DataFrame Conversion


class LifeboatToPolars:
    """Convert Lifeboat data to Polars DataFrames"""

    def __init__(self, loader: LifeboatLoader):
        self.loader = loader

    def create_photos_df(self, photos: List[Photo]) -> pl.DataFrame:
        """Create main photos DataFrame"""
        records = []

        for photo in photos:
            record = {
                "photo_id": photo.id,
                "secret": photo.secret,
                "url": photo.url,
                "title": photo.title,
                "description": photo.description,
                "uploader_id": photo.uploaderId,
                "license_id": photo.licenseId,
                "date_taken": photo.dateTaken.value if photo.dateTaken else None,
                "date_taken_granularity": photo.dateTaken.granularity
                if photo.dateTaken
                else None,
                "date_uploaded": photo.dateUploaded,
                "latitude": photo.location.latitude if photo.location else None,
                "longitude": photo.location.longitude if photo.location else None,
                "count_faves": photo.countFaves,
                "count_views": photo.countViews,
                "count_comments": photo.countComments,
                "safety_level": photo.safetyLevel,
                "visibility": photo.visibility,
                "original_format": photo.originalFormat,
                "media_type": photo.get_media_type(),
                "original_path": photo.get_file_paths()[0],
                "thumbnail_path": photo.get_file_paths()[1],
                "num_tags": len(photo.tags),
                "num_albums": len(photo.albumIds),
                "num_galleries": len(photo.galleryIds),
                "num_groups": len(photo.groupIds),
            }
            records.append(record)

        return pl.DataFrame(records)

    def create_tags_df(self, photos: List[Photo]) -> pl.DataFrame:
        """Create tags DataFrame with photo relationships"""
        records = []

        for photo in photos:
            for tag in photo.tags:
                record = {
                    "photo_id": photo.id,
                    "author_id": tag.authorId,
                    "raw_value": tag.rawValue,
                    "normalized_value": tag.normalizedValue,
                    "is_machine_tag": tag.isMachineTag,
                }
                records.append(record)

        return pl.DataFrame(records)

    def create_comments_df(self, photos: List[Photo]) -> pl.DataFrame:
        """Create comments DataFrame"""
        records = []

        for photo in photos:
            for comment in photo.comments:
                record = {
                    "photo_id": photo.id,
                    "comment_id": comment.id,
                    "author_id": comment.authorId,
                    "text": comment.text,
                    "date": comment.date,
                }
                records.append(record)

        return pl.DataFrame(records)

    def create_licenses_df(self) -> pl.DataFrame:
        """Create licenses DataFrame"""
        licenses = self.loader.load_licenses()
        records = [
            {"license_id": license_id, "label": license.label, "url": license.url}
            for license_id, license in licenses.items()
        ]
        return pl.DataFrame(records)

    def create_contributors_df(self) -> pl.DataFrame:
        """Create contributors DataFrame"""
        contributors = self.loader.load_contributors()
        records = [
            {
                "contributor_id": user_id,
                "username": contributor.username,
                "realname": contributor.realname,
                "path_alias": contributor.pathAlias,
                "photo_contributions": contributor.contributions.photo
                if contributor.contributions
                else 0,
                "tag_contributions": contributor.contributions.tag
                if contributor.contributions
                else 0,
                "comment_contributions": contributor.contributions.comment
                if contributor.contributions
                else 0,
            }
            for user_id, contributor in contributors.items()
        ]
        return pl.DataFrame(records)

    def create_full_dataset(self) -> Dict[str, pl.DataFrame]:
        """Create all DataFrames for the dataset"""
        print("Loading photos...")
        photos = self.loader.load_all_photos()

        print("Creating DataFrames...")
        datasets = {
            "photos": self.create_photos_df(photos),
            "tags": self.create_tags_df(photos),
            "comments": self.create_comments_df(photos),
            "licenses": self.create_licenses_df(),
            "contributors": self.create_contributors_df(),
        }

        # Add enriched view with license and contributor info
        datasets["photos_enriched"] = (
            datasets["photos"]
            .join(
                datasets["licenses"],
                left_on="license_id",
                right_on="license_id",
                how="left",
            )
            .join(
                datasets["contributors"].select(["contributor_id", "username"]),
                left_on="uploader_id",
                right_on="contributor_id",
                how="left",
            )
        )

        return datasets


# Usage example
if __name__ == "__main__":
    # Load and convert a Data Lifeboat
    lifeboat_path = Path("Commons_1K_2025")

    # Initialize loader
    loader = LifeboatLoader(lifeboat_path)

    # Load metadata
    print("Loading lifeboat metadata...")
    lifeboat_meta = loader.load_lifeboat_metadata()
    print(f"Collection: {lifeboat_meta.name}")
    print(f"Created: {lifeboat_meta.dateCreated}")

    # Convert to Polars
    converter = LifeboatToPolars(loader)
    datasets = converter.create_full_dataset()

    # Display summary
    for name, df in datasets.items():
        print(f"\n{name} DataFrame:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {df.columns}")
        if df.height > 0:
            print(f"  Sample:\n{df.head(3)}")
