"""
Woodpile Detection Module
Integrates YOLO detection with coordinate mapping for satellite images.
"""

import re
import cv2
import math
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# Import our model manager (optional - detector can work without it for testing)
YOLOModelManager = None
try:
    from .yolo_model import YOLOModelManager
except ImportError:
    try:
        from yolo_model import YOLOModelManager
    except ImportError:
        pass  # Will be None if ultralytics not installed


@dataclass
class DetectionResult:
    """Single detection result with GPS coordinates."""
    # Pixel coordinates
    pixel_bbox: List[float]  # [x1, y1, x2, y2]
    center_pixel: Tuple[float, float]  # (x, y)
    
    # GPS coordinates
    latitude: float
    longitude: float
    altitude: Optional[float]
    
    # Detection info
    confidence: float
    class_id: int
    class_name: str
    
    # Metadata
    image_width: int
    image_height: int
    slice_origin: Optional[Tuple[int, int]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pixel_bbox": self.pixel_bbox,
            "center_pixel": self.center_pixel,
            "latitude": round(self.latitude, 6),
            "longitude": round(self.longitude, 6),
            "altitude": self.altitude,
            "confidence": round(self.confidence, 4),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "google_maps_url": self.get_google_maps_url(),
            "image_size": f"{self.image_width}x{self.image_height}"
        }
    
    def to_list(self) -> List:
        """
        Convert to simple list format.
        
        Returns: [x1, y1, x2, y2, lat, lon, confidence, maps_url]
        """
        return [
            self.pixel_bbox[0],
            self.pixel_bbox[1],
            self.pixel_bbox[2],
            self.pixel_bbox[3],
            round(self.latitude, 6),
            round(self.longitude, 6),
            round(self.confidence, 4),
            self.get_google_maps_url()
        ]
    
    def get_google_maps_url(self) -> str:
        """Generate Google Maps URL."""
        return f"https://www.google.com/maps?q={self.latitude},{self.longitude}"


@dataclass
class ImageInfo:
    """Information extracted from image filename."""
    filepath: Path
    center_lat: float
    center_lon: float
    altitude: Optional[float]  # meters
    
    @classmethod
    def from_filename(cls, filepath: str) -> Optional['ImageInfo']:
        """
        Parse filename for coordinates.
        
        Expected format: name_45.5231_-122.6765_1000m.jpg
        or: name_45.5231_-122.6765.jpg
        """
        path = Path(filepath)
        stem = path.stem
        
        # Pattern: anything followed by _lat_lon or _lat_lon_altm
        # Example: woodpile_45.5231_-122.6765_1000m
        pattern = r'.*?_(-?\d+\.?\d*)_(-?\d+\.?\d*)(?:_(\d+)m?)?$'
        match = re.search(pattern, stem)
        
        if match:
            lat = float(match.group(1))
            lon = float(match.group(2))
            alt = float(match.group(3)) if match.group(3) else None
            
            # Validate coordinates
            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                print(f"Warning: Invalid coordinates in filename: {stem}")
                return None
            
            return cls(
                filepath=path,
                center_lat=lat,
                center_lon=lon,
                altitude=alt
            )
        
        print(f"Warning: Could not parse coordinates from filename: {stem}")
        print(f"Expected format: name_45.5231_-122.6765_1000m.jpg")
        return None


class CoordinateMapper:
    """
    Maps pixel coordinates to GPS coordinates.
    
    Based on image center point and altitude (eye altitude from Google Earth).
    
    Calibration: At 1000m eye altitude, typical Google Earth view covers
    approximately 1500m horizontally (for a 1920px wide screenshot).
    This gives ~0.78 meters per pixel at 1000m altitude.
    """
    
    # Earth's radius in meters
    EARTH_RADIUS = 6371000
    
    # Calibration: meters per pixel at 1000m altitude
    # For 1920px width covering 1500m ground distance: 1500/1920 = 0.78125
    MPP_AT_1000M = 0.78125  # meters per pixel at 1000m altitude
    
    def __init__(self, image_width: int, image_height: int,
                 center_lat: float, center_lon: float,
                 altitude: Optional[float] = None):
        """
        Initialize coordinate mapper.
        
        Args:
            image_width: Image width in pixels
            image_height: Image height in pixels
            center_lat: Center latitude (decimal degrees)
            center_lon: Center longitude (decimal degrees)
            altitude: Eye altitude in meters (for scale calculation)
        """
        self.image_width = image_width
        self.image_height = image_height
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.altitude = altitude or 1000  # Default to 1000m
        
        # Calculate meters per pixel
        self.meters_per_pixel = self._calculate_meters_per_pixel()
        
        # Pre-calculate ground coverage for verification
        self.ground_width_m = self.image_width * self.meters_per_pixel
        self.ground_height_m = self.image_height * self.meters_per_pixel
    
    def _calculate_meters_per_pixel(self) -> float:
        """
        Calculate meters per pixel based on altitude.
        
        Scale is linear with altitude: at 2000m, coverage doubles.
        """
        scale_factor = self.altitude / 1000.0
        return self.MPP_AT_1000M * scale_factor
    
    def get_ground_coverage(self) -> Tuple[float, float]:
        """
        Get the ground coverage in meters (width, height).
        
        Returns:
            (ground_width_m, ground_height_m)
        """
        return (self.ground_width_m, self.ground_height_m)
    
    def pixel_to_gps(self, pixel_x: float, pixel_y: float) -> Tuple[float, float]:
        """
        Convert pixel coordinates to GPS coordinates.
        
        Args:
            pixel_x: X coordinate (0 to image_width)
            pixel_y: Y coordinate (0 to image_height, 0 at top)
            
        Returns:
            (latitude, longitude) in decimal degrees
        """
        # Calculate offset from center (in pixels)
        # Y increases downward in images, but latitude increases northward
        offset_x = pixel_x - self.image_width / 2
        offset_y = -(pixel_y - self.image_height / 2)  # Invert Y
        
        # Convert to meters
        offset_x_m = offset_x * self.meters_per_pixel
        offset_y_m = offset_y * self.meters_per_pixel
        
        # Convert meters to degrees
        # Latitude: 1 degree = ~111km
        lat_offset_deg = offset_y_m / 111000
        
        # Longitude: 1 degree varies with latitude
        lon_offset_deg = offset_x_m / (111000 * math.cos(math.radians(self.center_lat)))
        
        # Calculate final coordinates
        lat = self.center_lat + lat_offset_deg
        lon = self.center_lon + lon_offset_deg
        
        return (lat, lon)
    
    def bbox_center_to_gps(self, bbox: List[float]) -> Tuple[float, float]:
        """
        Convert bounding box center to GPS coordinates.
        
        Args:
            bbox: [x1, y1, x2, y2] in pixel coordinates
            
        Returns:
            (latitude, longitude)
        """
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        return self.pixel_to_gps(center_x, center_y)


class WoodpileDetector:
    """
    Main detector class for woodpile detection in satellite images.
    
    Integrates YOLO model, SAHI slicing, and coordinate mapping.
    """
    
    # Supported slice sizes
    SLICE_SIZES = [160, 240, 320, 480, 640]
    
    def __init__(self, model_manager=None):
        """
        Initialize detector.
        
        Args:
            model_manager: YOLOModelManager instance (creates new if None)
        """
        if YOLOModelManager is None:
            raise RuntimeError("ultralytics not installed. Install with: pip install ultralytics")
        
        self.model_manager = model_manager or YOLOModelManager()
        self.stats = {
            "images_processed": 0,
            "total_detections": 0,
            "errors": []
        }
    
    def is_ready(self) -> bool:
        """Check if detector is ready (model loaded)."""
        return self.model_manager.is_loaded()
    
    def detect_single(self,
                     image_path: str,
                     slice_size: int = 640,
                     overlap_ratio: float = 0.2,
                     conf_threshold: float = 0.25,
                     iou_threshold: float = 0.45) -> List[DetectionResult]:
        """
        Detect woodpiles in a single image.
        
        Args:
            image_path: Path to image file
            slice_size: Size of detection slices (160/240/320/480/640)
            overlap_ratio: Overlap between slices (0-1)
            conf_threshold: Detection confidence threshold
            iou_threshold: IoU threshold for NMS
            
        Returns:
            List of DetectionResult objects
        """
        if not self.is_ready():
            raise RuntimeError("No model loaded. Please import a model first.")
        
        # Validate slice size
        if slice_size not in self.SLICE_SIZES:
            raise ValueError(f"Slice size must be one of {self.SLICE_SIZES}")
        
        # Parse image info from filename
        image_info = ImageInfo.from_filename(image_path)
        if image_info is None:
            self.stats["errors"].append(f"Failed to parse coordinates: {image_path}")
            return []
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            self.stats["errors"].append(f"Failed to load image: {image_path}")
            return []
        
        h, w = image.shape[:2]
        
        # Initialize coordinate mapper
        mapper = CoordinateMapper(
            image_width=w,
            image_height=h,
            center_lat=image_info.center_lat,
            center_lon=image_info.center_lon,
            altitude=image_info.altitude
        )
        
        # Run detection with slicing
        detections = self.model_manager.predict_sliced(
            image=image,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_ratio=overlap_ratio,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )
        
        # Convert to DetectionResult objects
        results = []
        for det in detections:
            bbox = det["bbox"]
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            # Convert to GPS
            lat, lon = mapper.pixel_to_gps(center_x, center_y)
            
            result = DetectionResult(
                pixel_bbox=bbox,
                center_pixel=(center_x, center_y),
                latitude=lat,
                longitude=lon,
                altitude=image_info.altitude,
                confidence=det["confidence"],
                class_id=det["class_id"],
                class_name=det["class_name"],
                image_width=w,
                image_height=h,
                slice_origin=det.get("slice_origin")
            )
            results.append(result)
        
        self.stats["images_processed"] += 1
        self.stats["total_detections"] += len(results)
        
        return results
    
    def detect_batch(self,
                    image_paths: List[str],
                    slice_size: int = 640,
                    overlap_ratio: float = 0.2,
                    conf_threshold: float = 0.25,
                    iou_threshold: float = 0.45,
                    progress_callback=None) -> Dict[str, List[DetectionResult]]:
        """
        Detect woodpiles in multiple images.
        
        Args:
            image_paths: List of image file paths
            slice_size: Size of detection slices
            overlap_ratio: Overlap between slices
            conf_threshold: Detection confidence threshold
            iou_threshold: IoU threshold for NMS
            progress_callback: Optional callback(current, total, current_file)
            
        Returns:
            Dictionary mapping image paths to detection results
        """
        results = {}
        total = len(image_paths)
        
        for i, path in enumerate(image_paths):
            if progress_callback:
                progress_callback(i + 1, total, Path(path).name)
            
            try:
                detections = self.detect_single(
                    image_path=path,
                    slice_size=slice_size,
                    overlap_ratio=overlap_ratio,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold
                )
                results[path] = detections
            except Exception as e:
                self.stats["errors"].append(f"{path}: {str(e)}")
                results[path] = []
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detection statistics."""
        return {
            **self.stats,
            "model_loaded": self.is_ready(),
            "model_name": self.model_manager.get_loaded_model_name()
        }
    
    def reset_stats(self):
        """Reset detection statistics."""
        self.stats = {
            "images_processed": 0,
            "total_detections": 0,
            "errors": []
        }


# ==================== TEST CODE ====================

def test_without_dependencies():
    """Test parsing and coordinate mapping without YOLO dependencies."""
    print("=" * 60)
    print("Woodpile Detector Test (Standalone)")
    print("=" * 60)
    
    # Test 1: ImageInfo parsing
    print("\n1. Testing ImageInfo parsing...")
    test_filenames = [
        "woodpile_45.5231_-122.6765_1000m.jpg",
        "test_40.7128_-74.0060.png",
        "area1_51.5074_-0.1278_500m.jpeg",
        "site_43.6425_-70.2646_1000m.jpg",
        "invalid_filename.jpg",
    ]
    
    for filename in test_filenames:
        info = ImageInfo.from_filename(filename)
        if info:
            print(f"   ✓ {filename}")
            print(f"     Center: ({info.center_lat}, {info.center_lon})")
            print(f"     Altitude: {info.altitude}m")
        else:
            print(f"   ✗ {filename} (parse failed)")
    
    # Test 2: Coordinate mapping
    print("\n2. Testing CoordinateMapper...")
    mapper = CoordinateMapper(
        image_width=1920,
        image_height=1080,
        center_lat=45.5231,
        center_lon=-122.6765,
        altitude=1000
    )
    print(f"   Image: {mapper.image_width}x{mapper.image_height}")
    print(f"   Center: ({mapper.center_lat}, {mapper.center_lon})")
    print(f"   Altitude: {mapper.altitude}m")
    print(f"   Meters/pixel: {mapper.meters_per_pixel:.4f}")
    
    # Verify ground coverage
    coverage = mapper.get_ground_coverage()
    print(f"   Ground coverage: {coverage[0]:.0f}m x {coverage[1]:.0f}m")
    print(f"   (Expected at 1000m: ~1500m x ~844m)")
    
    # Test pixel to GPS conversion
    test_pixels = [
        (960, 540),    # Center
        (0, 0),        # Top-left
        (1920, 1080),  # Bottom-right
    ]
    
    print("   Pixel to GPS conversions:")
    for px, py in test_pixels:
        lat, lon = mapper.pixel_to_gps(px, py)
        print(f"   ({px:4d}, {py:4d}) → ({lat:.6f}, {lon:.6f})")
    
    # Test 3: Output formats
    print("\n3. Testing output formats...")
    sample_result = DetectionResult(
        pixel_bbox=[100.5, 200.3, 150.7, 250.9],
        center_pixel=(125.6, 225.6),
        latitude=45.523456,
        longitude=-122.676543,
        altitude=1000.0,
        confidence=0.8765,
        class_id=0,
        class_name="woodpile",
        image_width=1920,
        image_height=1080
    )
    print(f"   to_dict(): {sample_result.to_dict()}")
    print(f"   to_list(): {sample_result.to_list()}")
    
    print("\n" + "=" * 60)
    print("Standalone tests completed")
    print("=" * 60)


def test_with_model():
    """Test with YOLO model (requires ultralytics)."""
    if YOLOModelManager is None:
        raise ImportError("ultralytics not installed")
    
    print("\n" + "=" * 60)
    print("Woodpile Detector Test (With Model)")
    print("=" * 60)
    
    # Test 4: Detector initialization
    print("\n4. Testing Detector initialization...")
    detector = WoodpileDetector()
    print(f"   Ready: {detector.is_ready()}")
    print(f"   Available models: {detector.model_manager.list_available_models()}")
    print(f"   Supported slice sizes: {detector.SLICE_SIZES}")
    
    # Test 5: Batch detection (requires actual model and images)
    print("\n5. Testing batch detection...")
    print("   Expected filename format: name_43.6425_-70.2646_1000m.jpg")
    
    if detector.is_ready():
        test_images = [
            "test_45.5231_-122.6765_1000m.jpg",
        ]
        
        for img_path in test_images:
            if Path(img_path).exists():
                print(f"   Processing: {img_path}")
                results = detector.detect_single(img_path, slice_size=640)
                print(f"   Found {len(results)} detections")
                for r in results[:3]:
                    print(f"     - {r.class_name} @ ({r.latitude:.6f}, {r.longitude:.6f})")
                    print(f"       List format: {r.to_list()}")
            else:
                print(f"   Skipped (not found): {img_path}")
    else:
        print("   Skipped: No model loaded")
        print("   To test with real model:")
        print("   1. Import a YOLO model: manager.import_model('model.pt')")
        print("   2. Run detection: detector.detect_single('image.jpg')")
    
    print("\n" + "=" * 60)
    print("Model tests completed")
    print("=" * 60)


if __name__ == "__main__":
    # Always run standalone tests
    test_without_dependencies()
    
    # Try to run model tests if dependencies available
    try:
        test_with_model()
    except Exception as e:
        print(f"\n   Model tests skipped: {e}")
        print("   (Install ultralytics to run full tests)")
