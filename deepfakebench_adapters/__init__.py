"""
DeepfakeBench Adapters for Anti-Deepfake-Box.

To integrate with DeepfakeBench:
1. Copy all *_detector.py files to <deepfakebench>/training/detectors/
2. Copy *.yaml config files to <deepfakebench>/training/config/detector/
3. Add to <deepfakebench>/training/detectors/__init__.py:
       from .adb_visual_detector   import ADBVisualDetector
       from .adb_rppg_detector     import ADBRPPGDetector
       from .adb_sync_detector     import ADBSyncDetector
       from .adb_ensemble_detector import ADBEnsembleDetector
"""
