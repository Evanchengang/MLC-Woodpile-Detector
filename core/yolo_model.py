"""
YOLOv8 Model Loader Module
Handles model loading, caching, and model history management.
"""

import os
import json
import torch
from pathlib import Path
from typing import Union, List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np
from ultralytics import YOLO


@dataclass
class ModelInfo:
    """Information about a registered model."""
    name: str
    path: str
    task: str = ""
    num_classes: int = 0
    class_names: Dict[int, str] = None
    date_added: str = ""
    
    def __post_init__(self):
        if self.class_names is None:
            self.class_names = {}
        if not self.date_added:
            from datetime import datetime
            self.date_added = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelInfo':
        return cls(**data)


class ModelConfig:
    """Manages model configuration and history."""
    
    CONFIG_FILENAME = "model_config.json"
    
    def __init__(self, config_dir: Optional[str] = None):
        if config_dir is None:
            # Store in user's home directory
            config_dir = Path.home() / ".woodpile_detector"
        
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.config_dir / self.CONFIG_FILENAME
        
        self.models: Dict[str, ModelInfo] = {}
        self.active_model_name: Optional[str] = None
        
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                
                self.models = {
                    name: ModelInfo.from_dict(info)
                    for name, info in data.get("models", {}).items()
                }
                self.active_model_name = data.get("active_model_name")
            except Exception as e:
                print(f"Error loading config: {e}")
                self.models = {}
                self.active_model_name = None
    
    def save_config(self):
        """Save configuration to file."""
        data = {
            "models": {
                name: info.to_dict()
                for name, info in self.models.items()
            },
            "active_model_name": self.active_model_name
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add_model(self, name: str, model_info: ModelInfo) -> bool:
        """Add a model to the registry."""
        self.models[name] = model_info
        if self.active_model_name is None:
            self.active_model_name = name
        self.save_config()
        return True
    
    def remove_model(self, name: str) -> bool:
        """Remove a model from the registry."""
        if name not in self.models:
            return False
        
        del self.models[name]
        
        if self.active_model_name == name:
            self.active_model_name = next(iter(self.models.keys()), None)
        
        self.save_config()
        return True
    
    def get_model(self, name: str) -> Optional[ModelInfo]:
        """Get model info by name."""
        return self.models.get(name)
    
    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self.models.keys())
    
    def set_active(self, name: str) -> bool:
        """Set the active model."""
        if name not in self.models:
            return False
        self.active_model_name = name
        self.save_config()
        return True
    
    def get_active_model(self) -> Optional[ModelInfo]:
        """Get the currently active model info."""
        if self.active_model_name is None:
            return None
        return self.models.get(self.active_model_name)


class YOLOModelManager:
    """Manages YOLOv8 model loading, inference, and model registry."""
    
    def __init__(self, config_dir: Optional[str] = None):
        self.config = ModelConfig(config_dir)
        self.model: Optional[YOLO] = None
        self.current_model_info: Optional[ModelInfo] = None
        self.device = self._get_device()
        
        # Auto-load active model if exists
        self._auto_load_active()
    
    def _get_device(self) -> str:
        """Determine the best available device."""
        if False:  # Force CPU
            return "mps"  # Apple Silicon
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def _auto_load_active(self):
        """Automatically load the active model from config."""
        active = self.config.get_active_model()
        
        # 打包环境：如果配置路径不存在，尝试用内置的 best.pt
        if active:
            import sys
            if getattr(sys, 'frozen', False) and not Path(active.path).exists():
                bundle_dir = Path(sys._MEIPASS) if hasattr(sys, '_MEIPASS') else Path.cwd()
                builtin_model = bundle_dir / 'best.pt'
                if builtin_model.exists():
                    active.path = str(builtin_model)
        
        if active:
            self._load_model_internal(active.path, active.name)
    def _load_model_internal(self, model_path: str, name: str) -> bool:
        """Internal method to load model without config updates."""
        try:
            path = Path(model_path)
            if not path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            if path.suffix != ".pt":
                raise ValueError("Model file must be .pt format")
            
            # Load the model
            self.model = YOLO(str(path))
            
            # Warm up
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model.predict(dummy, verbose=False)
            
            # Update current model info
            self.current_model_info = self.config.get_model(name)
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            self.current_model_info = None
            return False
    
    def import_model(self, model_path: Union[str, Path], name: Optional[str] = None) -> bool:
        """
        Import a new YOLOv8 model and add to registry.
        
        Args:
            model_path: Path to the .pt model file
            name: Optional custom name for the model
            
        Returns:
            True if imported and loaded successfully
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            print(f"Error: Model file not found: {model_path}")
            return False
        
        # Generate name if not provided
        if name is None:
            name = model_path.stem
        
        # Check for duplicate names
        base_name = name
        counter = 1
        while name in self.config.models:
            name = f"{base_name}_{counter}"
            counter += 1
        
        # First load to get model info
        temp_model = YOLO(str(model_path))
        
        # Extract model info
        model_info = ModelInfo(
            name=name,
            path=str(model_path.resolve()),
            task=getattr(temp_model, 'task', 'detect'),
            num_classes=len(temp_model.names) if temp_model.names else 0,
            class_names=dict(temp_model.names) if temp_model.names else {}
        )
        
        # Add to config
        self.config.add_model(name, model_info)
        
        # Load as current model
        success = self._load_model_internal(str(model_path), name)
        
        if success:
            # Set as active
            self.config.set_active(name)
        
        return success
    
    def switch_model(self, name: str) -> bool:
        """
        Switch to a different registered model.
        
        Args:
            name: Name of the model to switch to
            
        Returns:
            True if switched successfully
        """
        model_info = self.config.get_model(name)
        if model_info is None:
            print(f"Error: Model '{name}' not found in registry")
            return False
        
        success = self._load_model_internal(model_info.path, name)
        
        if success:
            self.config.set_active(name)
            print(f"Switched to model: {name}")
        
        return success
    
    def remove_model(self, name: str) -> bool:
        """
        Remove a model from the registry.
        Does not delete the actual .pt file.
        """
        if name == self.current_model_info.name if self.current_model_info else None:
            print("Cannot remove currently loaded model")
            return False
        
        return self.config.remove_model(name)
    
    def is_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self.model is not None
    
    def get_loaded_model_name(self) -> Optional[str]:
        """Get the name of the currently loaded model."""
        if self.current_model_info:
            return self.current_model_info.name
        return None
    
    def list_available_models(self) -> List[str]:
        """List all registered model names."""
        return self.config.list_models()
    
    def get_model_details(self, name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get detailed info about a model.
        
        Args:
            name: Model name, or None for currently loaded model
        """
        if name is None:
            if self.current_model_info is None:
                return None
            return self.current_model_info.to_dict()
        
        model_info = self.config.get_model(name)
        if model_info:
            return model_info.to_dict()
        return None
    
    def predict(self, 
                image: np.ndarray,
                conf_threshold: float = 0.25,
                iou_threshold: float = 0.45,
                verbose: bool = False) -> List[Dict[str, Any]]:
        """
        Run detection on an image.
        
        Args:
            image: numpy array (H, W, C) in BGR format
            conf_threshold: confidence threshold
            iou_threshold: IoU threshold for NMS
            verbose: print detailed info
            
        Returns:
            List of detection dicts with keys: bbox, confidence, class_id, class_name
        """
        if not self.is_loaded():
            raise RuntimeError("No model loaded. Call import_model() or switch_model() first.")
        
        results = self.model.predict(
            image,
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device,
            verbose=verbose
        )
        
        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                
                for box, conf, cls_id in zip(boxes, confs, classes):
                    detections.append({
                        "bbox": box.tolist(),  # [x1, y1, x2, y2]
                        "confidence": float(conf),
                        "class_id": int(cls_id),
                        "class_name": self.model.names.get(int(cls_id), "unknown")
                    })
        
        return detections
    
    def predict_sliced(self,
                       image: np.ndarray,
                       slice_height: int = 640,
                       slice_width: int = 640,
                       overlap_ratio: float = 0.2,
                       conf_threshold: float = 0.25,
                       iou_threshold: float = 0.45) -> List[Dict[str, Any]]:
        """
        Run detection with SAHI-style slicing for large images.
        
        Args:
            image: numpy array (H, W, C)
            slice_height: height of each slice
            slice_width: width of each slice
            overlap_ratio: overlap between slices (0-1)
            conf_threshold: confidence threshold
            iou_threshold: IoU threshold
            
        Returns:
            List of detection dicts with absolute coordinates
        """
        if not self.is_loaded():
            raise RuntimeError("No model loaded. Call import_model() or switch_model() first.")
        
        h, w = image.shape[:2]
        stride_y = int(slice_height * (1 - overlap_ratio))
        stride_x = int(slice_width * (1 - overlap_ratio))
        
        all_detections = []
        
        # Generate slice coordinates
        y_positions = list(range(0, h - slice_height + 1, stride_y))
        x_positions = list(range(0, w - slice_width + 1, stride_x))
        
        # Ensure we cover the edges
        if y_positions and y_positions[-1] + slice_height < h:
            y_positions.append(h - slice_height)
        if x_positions and x_positions[-1] + slice_width < w:
            x_positions.append(w - slice_width)
        
        for y in y_positions:
            for x in x_positions:
                # Extract slice
                slice_img = image[y:y + slice_height, x:x + slice_width]
                
                # Run detection on slice
                slice_detections = self.predict(
                    slice_img,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold
                )
                
                # Adjust coordinates to absolute image coordinates
                for det in slice_detections:
                    bbox = det["bbox"]
                    det["bbox"] = [
                        bbox[0] + x,  # x1
                        bbox[1] + y,  # y1
                        bbox[2] + x,  # x2
                        bbox[3] + y   # y2
                    ]
                    det["slice_origin"] = (x, y)
                    all_detections.append(det)
        
        # Apply NMS to remove duplicates across slices
        all_detections = self._apply_nms(all_detections, iou_threshold)
        
        return all_detections
    
    def _apply_nms(self, 
                   detections: List[Dict[str, Any]], 
                   iou_threshold: float) -> List[Dict[str, Any]]:
        """Apply Non-Maximum Suppression to remove overlapping detections."""
        if not detections:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
        
        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)
            
            # Remove overlapping detections
            detections = [
                d for d in detections
                if self._iou(best["bbox"], d["bbox"]) < iou_threshold
            ]
        
        return keep
    
    def _iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        
        # Union
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0


# ==================== TEST CODE ====================

if __name__ == "__main__":
    print("=" * 60)
    print("YOLO Model Manager Test")
    print("=" * 60)
    
    # Initialize manager
    print("\n1. Initializing Model Manager...")
    manager = YOLOModelManager()
    print(f"   Device: {manager.device}")
    print(f"   Config directory: {manager.config.config_dir}")
    
    # List registered models
    print("\n2. Registered Models:")
    models = manager.list_available_models()
    if models:
        for name in models:
            marker = " (active)" if name == manager.config.active_model_name else ""
            print(f"   - {name}{marker}")
    else:
        print("   (No models registered)")
    
    # Check if model is loaded
    print(f"\n3. Model Loaded: {manager.is_loaded()}")
    if manager.is_loaded():
        print(f"   Current model: {manager.get_loaded_model_name()}")
    
    # Test import (uncomment to test with actual model file)
    """
    print("\n4. Testing model import...")
    # Replace with actual path to your .pt file
    test_model_path = "path/to/your/model.pt"
    
    if os.path.exists(test_model_path):
        success = manager.import_model(test_model_path, name="my_woodpile_model")
        print(f"   Import result: {'SUCCESS' if success else 'FAILED'}")
        
        if success:
            print(f"\n5. Model Details:")
            details = manager.get_model_details()
            for key, value in details.items():
                print(f"   {key}: {value}")
    else:
        print(f"   Skipped: Test model not found at {test_model_path}")
    """
    
    print("\n" + "=" * 60)
    print("Test completed. To test with actual model:")
    print("1. Update test_model_path variable")
    print("2. Uncomment the test block above")
    print("=" * 60)
