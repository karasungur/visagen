"""English translations (default)."""

TRANSLATIONS = {
    # Common
    "common": {
        "start": "Start",
        "stop": "Stop",
        "save": "Save",
        "load": "Load",
        "cancel": "Cancel",
        "apply": "Apply",
        "refresh": "Refresh",
        "export": "Export",
        "browse": "Browse",
    },
    # App
    "app": {
        "title": "Visagen - Face Swapping Framework",
        "subtitle": "Modern Face Swapping with PyTorch Lightning",
        "footer": "Visagen v2.0.0-alpha | Powered by PyTorch Lightning",
    },
    # Workflow
    "workflow": {
        "title": "Workflow Steps",
        "steps": {
            "extract": "Extract",
            "sort": "Sort",
            "train": "Train",
            "merge": "Merge",
            "export": "Export",
        },
        "descriptions": {
            "extract": "Extract faces from videos or images",
            "sort": "Filter and organize extracted faces",
            "train": "Train the face swap model",
            "merge": "Apply trained model to videos",
            "export": "Export processed videos",
        },
    },
    # Errors
    "errors": {
        "path_required": "Path is required",
        "path_not_found": "Path not found",
        "not_a_directory": "Path is not a directory",
        "invalid_file_type": "Invalid file type. Expected: {types}",
        "no_model_loaded": "No model loaded. Please load a checkpoint first.",
        "process_failed": "Process failed with exit code {code}",
        "missing_images": "Required images are missing",
        # New error messages
        "no_output_dir": "No output directory specified",
        "source_image_required": "Please provide a source image",
    },
    # Status messages
    "status": {
        "ready": "Ready",
        "loading": "Loading...",
        "processing": "Processing...",
        "completed": "Completed",
        "failed": "Failed",
        "model_loaded": "Model loaded: {name}",
        "model_unloaded": "Model unloaded",
        "no_model": "No model loaded",
        "files_found": "Found {count} files",
        "stopped": "Process stopped by user",
        # New status messages for hardcoded strings
        "no_training": "No training in progress",
        "no_merge": "No merge in progress",
        "no_sorting": "No sorting in progress",
        "no_extraction": "No extraction in progress",
        "no_export": "No export in progress",
        "preview_available": "Preview available",
        "extraction_completed": "Extraction completed successfully!",
        "extraction_stopped": "Extraction stopped",
    },
    # Training tab
    "training": {
        "title": "Model Training",
        "src_dir": {
            "label": "Source Directory",
            "placeholder": "./workspace/data_src/aligned",
            "info": "Directory containing source face images",
        },
        "dst_dir": {
            "label": "Destination Directory",
            "placeholder": "./workspace/data_dst/aligned",
            "info": "Directory containing destination face images",
        },
        "output_dir": {
            "label": "Output Directory",
            "info": "Directory for checkpoints and logs",
        },
        "batch_size": {
            "label": "Batch Size",
        },
        "max_epochs": {
            "label": "Max Epochs",
        },
        "learning_rate": {
            "label": "Learning Rate",
        },
        "start": "Start Training",
        "stop": "Stop Training",
        "log": {
            "label": "Training Log",
        },
        "preview": {
            "title": "Training Preview",
            "status": {
                "label": "Preview Status",
            },
            "image": {
                "label": "Preview Grid",
            },
        },
        "dssim_weight": {
            "label": "DSSIM Weight",
        },
        "l1_weight": {
            "label": "L1 Weight",
        },
        "lpips_weight": {
            "label": "LPIPS Weight",
            "info": "Requires lpips package",
        },
        "gan_power": {
            "label": "GAN Power",
            "info": "0 = disabled, > 0 = adversarial training",
        },
        "precision": {
            "label": "Precision",
            "choices": {
                "32": "FP32 (Standard)",
                "16-mixed": "FP16 Mixed (Faster)",
                "bf16-mixed": "BF16 Mixed (Newer GPUs)",
            },
        },
        "model_type": {
            "label": "Model Type",
            "info": "standard=ConvNeXt, diffusion=SD VAE hybrid, eg3d=3D-aware",
            "choices": {
                "standard": "Standard (ConvNeXt)",
                "diffusion": "Diffusion (SD VAE)",
                "eg3d": "EG3D (3D Aware)",
            },
        },
        "texture_weight": {
            "label": "Texture Weight",
            "info": "Texture consistency loss (for diffusion model)",
        },
        "use_pretrained_vae": {
            "label": "Use Pretrained VAE",
            "info": "Use SD VAE (requires diffusers package)",
        },
        "uniform_yaw": {
            "label": "Uniform Yaw",
            "info": "Balance training samples across different face angles",
        },
        "masked_training": {
            "label": "Masked Training",
            "info": "Focus training on face area (blur background)",
        },
        "resume_ckpt": {
            "label": "Resume Checkpoint",
            "placeholder": "./workspace/model/checkpoints/last.ckpt",
        },
        "refresh_preview": "🔄 Refresh Preview",
        "preset": {
            "label": "Training Preset",
            "load": "Load",
            "save": "Save As...",
            "name_input": "Preset Name",
            "confirm_save": "Save Preset",
            "saved": "Preset saved: {name}",
            "deleted": "Preset deleted: {name}",
            "load_error": "Failed to load preset",
        },
    },
    # Inference tab
    "inference": {
        "title": "Face Swap Inference",
        "checkpoint": {
            "label": "Model Checkpoint",
            "placeholder": "./workspace/model/checkpoints/last.ckpt",
        },
        "load_model": "Load Model",
        "unload_model": "Unload Model",
        "model_status": {
            "label": "Model Status",
        },
        "source_image": {
            "label": "Source Face",
        },
        "target_image": {
            "label": "Target Face",
        },
        "output_image": {
            "label": "Result",
        },
        "swap": "Swap Face",
    },
    # Extract tab
    "extract": {
        "title": "Face Extraction",
        "description": "Extract faces from images or videos for training.",
        "input_path": {
            "label": "Input (image, video, or directory)",
            "placeholder": "./input_video.mp4",
        },
        "output_dir": {
            "label": "Output Directory",
        },
        "face_type": {
            "label": "Face Type",
            "choices": {
                "whole_face": "Whole Face",
                "full": "Full",
                "mid_full": "Mid Full",
                "half": "Half",
                "head": "Head",
            },
        },
        "output_size": {
            "label": "Output Size",
        },
        "min_confidence": {
            "label": "Min Confidence",
        },
        "start": "Extract Faces",
        "log": {
            "label": "Extraction Log",
        },
    },
    # Settings tab
    "settings": {
        "title": "Settings",
        "device_section": "Device Configuration",
        "performance_section": "Performance",
        "language_section": "Language",
        "device": {
            "label": "Compute Device",
            "info": "Select the device for model inference and training",
            "choices": {
                "auto": "Auto (Detect Best)",
                "cuda": "CUDA (GPU)",
                "cpu": "CPU",
            },
        },
        "batch_size": {
            "label": "Default Batch Size",
            "info": "Number of samples processed simultaneously",
        },
        "num_workers": {
            "label": "Number of Workers",
            "info": "Worker threads for data loading (0 = main thread only)",
        },
        "locale": {
            "label": "Language",
            "info": "Application display language",
            "choices": {
                "en": "English",
                "tr": "Turkish",
            },
        },
        "status": {
            "label": "Status",
            "saved": "Settings saved successfully",
            "saved_reload": "Settings saved. Restart app to apply language change.",
        },
    },
    # Merge tab
    "merge": {
        "title": "Video Face Swap",
        "description": "Process videos using trained models with customizable blending and color transfer.",
        "input_video": {
            "label": "Input Video",
            "placeholder": "./input.mp4",
            "info": "Path to source video file",
        },
        "output_video": {
            "label": "Output Video",
            "placeholder": "./output.mp4",
            "info": "Path for processed video output",
        },
        "checkpoint": {
            "label": "Model Checkpoint",
            "placeholder": "./workspace/model/checkpoints/last.ckpt",
            "info": "Path to trained model checkpoint",
        },
        "color_transfer": {
            "label": "Color Transfer Mode",
            "info": "RCT=Reinhard, LCT=Linear, SOT=Sliced OT",
            "choices": {
                "rct": "RCT (Reinhard)",
                "lct": "LCT (Linear)",
                "sot": "SOT (Sliced OT)",
                "none": "None",
            },
        },
        "blend_mode": {
            "label": "Blend Mode",
            "info": "Laplacian=pyramid, Poisson=seamless, Feather=alpha",
            "choices": {
                "laplacian": "Laplacian (Pyramid)",
                "poisson": "Poisson (Seamless)",
                "feather": "Feather (Alpha)",
            },
        },
        "restoration": {
            "title": "Face Restoration",
            "enable": {
                "label": "Enable GFPGAN",
                "info": "Enhance face quality with GFPGAN",
            },
            "strength": {
                "label": "Restoration Strength",
            },
            "version": {
                "label": "GFPGAN Version",
            },
        },
        "encoding": {
            "title": "Video Encoding",
            "codec": {
                "label": "Encoder",
                "info": "'auto' selects NVENC if available",
                "choices": {
                    "auto": "Auto (Best Available)",
                    "libx264": "libx264 (CPU H.264)",
                    "libx265": "libx265 (CPU H.265)",
                    "h264_nvenc": "NVENC H.264 (GPU)",
                    "hevc_nvenc": "NVENC H.265 (GPU)",
                },
            },
            "crf": {
                "label": "Quality (CRF)",
                "info": "Lower = better quality, higher file size",
            },
        },
        "start": "Start Merge",
        "stop": "Stop Merge",
        "log": {
            "label": "Merge Log",
        },
    },
    # Interactive Merge tab
    "interactive_merge": {
        "title": "Interactive Merge",
        "description": "Real-time preview with adjustable parameters. Load a trained model and frame sequence to begin.",
        # Session setup
        "session": {
            "title": "Session Setup",
        },
        "checkpoint": {
            "label": "Model Checkpoint",
            "placeholder": "./workspace/model/checkpoints/last.ckpt",
            "info": "Path to trained model checkpoint",
        },
        "frames_dir": {
            "label": "Frames Directory",
            "placeholder": "./frames",
            "info": "Directory containing input frame images",
        },
        "output_dir": {
            "label": "Output Directory",
            "placeholder": "./output",
            "info": "Directory for exported frames",
        },
        "load_session": "Load Session",
        "session_status": {
            "label": "Session Status",
            "not_loaded": "No session loaded",
            "loaded": "Loaded: {count} frames from {path}",
        },
        # Settings
        "settings": {
            "title": "Merge Settings",
        },
        "mode": {
            "label": "Merge Mode",
            "info": "How to blend swapped face",
            "choices": {
                "original": "Original",
                "overlay": "Overlay",
                "hist-match": "Histogram Match",
                "seamless": "Seamless",
                "seamless-hist-match": "Seamless + Hist Match",
                "raw-rgb": "Raw RGB",
                "raw-predict": "Raw Predict",
            },
        },
        "mask_mode": {
            "label": "Mask Mode",
            "info": "Face region segmentation method",
            "choices": {
                "full": "Full",
                "convex_hull": "Convex Hull",
                "segmented": "Segmented",
            },
        },
        "color_transfer": {
            "label": "Color Transfer",
            "info": "Color matching algorithm",
            "choices": {
                "none": "None",
                "rct": "RCT (Reinhard)",
                "lct": "LCT (Linear)",
                "mkl": "MKL (Monge-Kantorovitch)",
                "idt": "IDT (Iterative)",
                "sot": "SOT (Sliced OT)",
            },
        },
        # Mask processing
        "mask_processing": {
            "title": "Mask Processing",
        },
        "erode_mask": {
            "label": "Erode Mask",
            "info": "Negative = dilate, Positive = erode",
        },
        "blur_mask": {
            "label": "Blur Mask",
            "info": "Mask edge softness",
        },
        "face_scale": {
            "label": "Face Scale",
            "info": "Adjust swapped face size",
        },
        # Sharpening
        "sharpening": {
            "title": "Sharpening",
        },
        "sharpen_mode": {
            "label": "Sharpen Mode",
            "choices": {
                "none": "None",
                "box": "Box",
                "gaussian": "Gaussian",
            },
        },
        "sharpen_amount": {
            "label": "Sharpen Amount",
            "info": "Negative = blur, Positive = sharpen",
        },
        # Advanced
        "advanced": {
            "title": "Advanced",
        },
        "hist_threshold": {
            "label": "Histogram Match Threshold",
        },
        "restore_face": {
            "label": "GFPGAN Restoration",
            "info": "Enhance face quality",
        },
        "restore_strength": {
            "label": "Restoration Strength",
        },
        # Apply
        "apply_settings": "Apply Settings",
        "config_status": {
            "label": "Current Config",
            "showing_original": "Showing original (settings not applied)",
        },
        # Preview
        "preview": {
            "title": "Preview",
            "image_label": "Preview",
        },
        "show_original": {
            "label": "Show Original",
        },
        "frame_info": {
            "label": "Frame Info",
            "no_frame": "No frame loaded",
            "format": "Frame {current}/{total}",
            "detail": "Frame {current}/{total}: {filename}",
        },
        # Navigation
        "nav": {
            "prev": "◀ Previous",
            "next": "Next ▶",
            "frame": "Frame",
        },
        # Export
        "export": {
            "title": "Export",
            "current": "Export Current Frame",
            "all": "Export All Frames",
            "save_session": "Save Session",
            "status_label": "Export Status",
            "current_success": "Exported to: {path}",
            "all_success": "Exported {count} frames",
            "session_saved": "Session saved to: {path}",
            "failed": "Export failed",
            "save_failed": "Save failed",
        },
        # Errors
        "errors": {
            "no_session": "No session loaded",
            "load_failed": "Error loading session",
            "update_failed": "Error updating config",
            "navigate_failed": "Error navigating",
        },
    },
    # Sort tab
    "sort": {
        "title": "Dataset Sorting",
        "description": "Sort and filter face images by various criteria.",
        "input_dir": {
            "label": "Input Directory",
            "placeholder": "./workspace/data_src/aligned",
            "info": "Directory containing aligned face images",
        },
        "output_dir": {
            "label": "Output Directory (optional)",
            "placeholder": "Leave empty to sort in place",
            "info": "Optional output directory for sorted images",
        },
        "method": {
            "label": "Sort Method",
            "info": "Select sorting/filtering algorithm",
            "choices": {
                "blur": "Blur (Sharpness)",
                "motion-blur": "Motion Blur",
                "face-yaw": "Face Yaw (Left-Right)",
                "face-pitch": "Face Pitch (Up-Down)",
                "face-source-rect-size": "Face Size",
                "hist": "Histogram Similarity",
                "hist-dissim": "Histogram Dissimilarity",
                "brightness": "Brightness",
                "hue": "Hue",
                "black": "Black Pixels",
                "origname": "Original Name",
                "oneface": "Single Face Only",
                "final": "Final (Best Selection)",
                "final-fast": "Final Fast",
            },
        },
        "target_count": {
            "label": "Target Count",
            "info": "Used only for 'final' and 'final-fast' methods",
        },
        "dry_run": {
            "label": "Dry Run (Preview)",
            "info": "Show what would happen without making changes",
        },
        "start": "Start Sorting",
        "stop": "Stop Sorting",
        "log": {
            "label": "Sorting Log",
        },
    },
    # Export tab
    "export": {
        "title": "Model Export",
        "description": "Export trained models to ONNX or TensorRT for optimized inference.",
        "input_path": {
            "label": "Input Path",
            "placeholder": "./workspace/model/checkpoints/last.ckpt",
            "info": "Checkpoint (.ckpt) for ONNX, or ONNX (.onnx) for TensorRT",
        },
        "output_path": {
            "label": "Output Path",
            "placeholder": "./model.onnx",
            "info": "Output file path (.onnx or .engine)",
        },
        "format": {
            "label": "Export Format",
            "info": "ONNX for cross-platform, TensorRT for NVIDIA GPUs",
            "choices": {
                "onnx": "ONNX",
                "tensorrt": "TensorRT",
            },
        },
        "precision": {
            "label": "Precision",
            "info": "FP16 recommended for balance of speed and quality",
            "choices": {
                "fp32": "FP32 (Full Precision)",
                "fp16": "FP16 (Half Precision)",
                "int8": "INT8 (Quantized)",
            },
        },
        "validate": {
            "label": "Validate Export",
            "info": "Compare exported model against PyTorch original",
        },
        "start": "Export Model",
        "stop": "Stop Export",
        "log": {
            "label": "Export Log",
        },
    },
    # Video Tools tab
    "video_tools": {
        "title": "Video Tools",
        "description": "Tools for video-to-frame and frame-to-video conversion.",
        "extract": {
            "title": "Extract Frames from Video",
            "input": {
                "label": "Input Video",
                "placeholder": "./input.mp4",
                "info": "Path to video file",
            },
            "output": {
                "label": "Output Directory",
                "placeholder": "./frames",
                "info": "Directory to save extracted frames",
            },
            "fps": {
                "label": "FPS (0 = original)",
                "info": "Target frame rate (0 to keep original)",
            },
            "format": {
                "label": "Output Format",
            },
            "start": "Extract Frames",
            "log": {
                "label": "Log",
            },
        },
        "create": {
            "title": "Create Video from Frames",
            "input": {
                "label": "Input Directory",
                "placeholder": "./frames",
                "info": "Directory containing image sequence",
            },
            "output": {
                "label": "Output Video",
                "placeholder": "./output.mp4",
                "info": "Output video path",
            },
            "fps": {
                "label": "FPS",
            },
            "codec": {
                "label": "Codec",
            },
            "bitrate": {
                "label": "Bitrate",
                "info": "Video bitrate (e.g., 16M, 25M)",
            },
            "start": "Create Video",
            "log": {
                "label": "Log",
            },
        },
        "cut": {
            "title": "Cut Video Segment",
            "input": {
                "label": "Input Video",
                "placeholder": "./input.mp4",
            },
            "output": {
                "label": "Output Video",
                "placeholder": "./cut_output.mp4",
            },
            "start_time": {
                "label": "Start Time",
                "info": "Format: HH:MM:SS or seconds",
            },
            "end_time": {
                "label": "End Time",
                "info": "Format: HH:MM:SS or seconds",
            },
            "start": "Cut Video",
            "log": {
                "label": "Log",
            },
        },
        "denoise": {
            "title": "Temporal Denoising",
            "description": "Apply temporal denoising to reduce flickering in frame sequences.",
            "input": {
                "label": "Input Directory",
                "placeholder": "./frames",
                "info": "Directory containing image sequence",
            },
            "output": {
                "label": "Output Directory (optional)",
                "placeholder": "Leave empty for in-place",
            },
            "factor": {
                "label": "Denoise Factor",
                "info": "Temporal window size (must be odd)",
            },
            "start": "Apply Denoising",
            "log": {
                "label": "Log",
            },
        },
    },
    # Faceset Tools tab
    "faceset_tools": {
        "title": "Faceset Tools",
        "description": "Tools for enhancing and resizing face datasets.",
        "enhance": {
            "title": "Face Enhancement (GFPGAN)",
            "description": "Enhance face quality using GFPGAN restoration.",
            "input": {
                "label": "Input Directory",
                "placeholder": "./workspace/data_src/aligned",
                "info": "Directory containing face images",
            },
            "output": {
                "label": "Output Directory (optional)",
                "placeholder": "Leave empty for auto-naming",
                "info": "Output directory (default: input_enhanced)",
            },
            "strength": {
                "label": "Enhancement Strength",
                "info": "0 = original, 1 = fully enhanced",
            },
            "model": {
                "label": "GFPGAN Version",
            },
            "start": "Enhance Faceset",
            "log": {
                "label": "Log",
            },
        },
        "resize": {
            "title": "Faceset Resizing",
            "description": "Resize face images with DFL metadata preservation.",
            "input": {
                "label": "Input Directory",
                "placeholder": "./workspace/data_src/aligned",
                "info": "Directory containing face images",
            },
            "output": {
                "label": "Output Directory (optional)",
                "placeholder": "Leave empty for auto-naming",
                "info": "Output directory (default: input_SIZE)",
            },
            "size": {
                "label": "Target Size",
                "info": "Output image size (width = height)",
            },
            "face_type": {
                "label": "Face Type",
                "info": "Target face type (keep = preserve original)",
            },
            "interp": {
                "label": "Interpolation",
            },
            "start": "Resize Faceset",
            "log": {
                "label": "Log",
            },
        },
    },
    # Batch processing tab
    "batch": {
        "title": "Batch Processing",
        "description": "Process multiple videos in a queue with the same settings.",
        "files": {
            "label": "Select Videos",
        },
        "output_dir": {
            "label": "Output Directory",
            "placeholder": "./batch_output",
            "info": "Directory for processed video outputs",
        },
        "checkpoint": {
            "label": "Model Checkpoint",
            "placeholder": "./workspace/model/checkpoints/last.ckpt",
            "info": "Path to trained model checkpoint",
        },
        "operation": {
            "label": "Operation",
        },
        "add_to_queue": "Add to Queue",
        "queue": {
            "title": "Processing Queue",
            "file": "File",
            "status": "Status",
            "progress": "Progress",
        },
        "progress": {
            "label": "Overall Progress",
        },
        "start_all": "Start All",
        "stop_all": "Stop All",
        "clear_completed": "Clear Completed",
        "status": {
            "label": "Status",
            "no_files": "No files selected",
            "added": "Added {count} files to queue",
            "started": "Started processing {count} items",
            "stopped": "Processing stopped",
            "cleared": "Cleared {count} completed items",
            "already_running": "Batch processing is already running",
            "no_pending": "No pending items in queue",
        },
    },
    # Model comparison tab
    "compare": {
        "title": "Model Comparison",
        "description": "Compare outputs from two different model checkpoints side by side.",
        "checkpoint_a": {
            "label": "Model A Checkpoint",
            "placeholder": "./model_a.ckpt",
        },
        "checkpoint_b": {
            "label": "Model B Checkpoint",
            "placeholder": "./model_b.ckpt",
        },
        "load_model_a": "Load Model A",
        "load_model_b": "Load Model B",
        "status_a": {"label": "Model A Status"},
        "status_b": {"label": "Model B Status"},
        "test_image": {"label": "Test Image"},
        "compare": "Compare Models",
        "results": {"title": "Comparison Results"},
        "metrics": {"label": "Quality Metrics (SSIM/PSNR)"},
        "unload_all": "Unload All Models",
        "errors": {
            "both_models_required": "Please load both models before comparing",
        },
    },
    # Postprocess tab
    "postprocess": {
        "title": "Postprocess",
        "color_transfer": {
            "title": "Color Transfer Demo",
            "apply": "Apply Color Transfer",
        },
        "blending": {
            "title": "Blending Demo",
            "apply": "Blend Images",
        },
        "restoration": {
            "title": "Face Restoration Demo",
            "description": "Enhance face quality using GFPGAN or GPEN.",
            "apply": "Restore Face",
        },
        "neural": {
            "title": "Neural Color Transfer",
            "description": "VGG-based semantic color matching for more realistic results.",
            "apply": "Apply Neural Color",
        },
        "ct_source": {"label": "Source (color reference)"},
        "ct_target": {"label": "Target (to modify)"},
        "ct_result": {"label": "Result"},
        "ct_mode": {
            "label": "Color Transfer Mode",
            "info": "RCT=Reinhard, LCT=Linear, SOT=Sliced OT, MKL=Monge-Kantorovitch, IDT=Iterative",
        },
        "bl_fg": {"label": "Foreground"},
        "bl_bg": {"label": "Background"},
        "bl_mask": {"label": "Mask"},
        "bl_result": {"label": "Result"},
        "bl_mode": {
            "label": "Blend Mode",
            "info": "Laplacian=multi-band pyramid, Poisson=seamless clone, Feather=alpha blend",
        },
        "restore_input": {"label": "Input Face"},
        "restore_result": {"label": "Restored Face"},
        "restore_mode": {
            "label": "Restoration Mode",
            "info": "GFPGAN: Best quality, GPEN: Better structure preservation",
        },
        "restore_strength": {
            "label": "Restoration Strength",
            "info": "0 = original, 1 = fully restored",
        },
        "restore_version": {
            "label": "GFPGAN Version",
            "info": "Only used when mode is GFPGAN",
        },
        "gpen_size": {
            "label": "GPEN Model Size",
            "info": "Only used when mode is GPEN. Larger = better quality, slower",
        },
        "nct_source": {"label": "Style Reference (color source)"},
        "nct_target": {"label": "Target Image (to modify)"},
        "nct_result": {"label": "Result"},
        "nct_mode": {
            "label": "Transfer Mode",
            "info": "histogram=LAB space, statistics=mean/std, gram=style (requires torchvision)",
        },
        "nct_strength": {"label": "Transfer Strength"},
        "nct_preserve": {"label": "Preserve Luminance"},
    },
    # Wizard tab
    "wizard": {
        "title": "Quick Start Wizard",
        "description": "Step-by-step guide to create your first face swap video. Perfect for beginners!",
        "next_step": "Next Step →",
        "back": "← Back",
        "steps": {
            "upload": "Upload",
            "extract": "Extract",
            "train": "Train",
            "apply": "Apply",
        },
        "step1": {
            "title": "Step 1: Upload Videos",
            "description": "Upload the source video (face to copy) and destination video (face to replace).",
            "src_video": "Source Video (face to copy)",
            "dst_video": "Destination Video (face to replace)",
            "status": "Status",
            "error_missing_videos": "Please upload both source and destination videos.",
            "success": "Videos uploaded successfully! Proceed to extraction.",
        },
        "step2": {
            "title": "Step 2: Extract Faces",
            "description": "Extract and align faces from both videos. This may take a few minutes.",
            "face_type": "Face Type",
            "output_size": "Output Size",
            "log": "Extraction Log",
            "extract": "Start Extraction",
        },
        "step3": {
            "title": "Step 3: Train Model",
            "description": "Train the face swap model. This may take 30-60 minutes depending on settings.",
            "epochs": "Training Epochs",
            "batch_size": "Batch Size",
            "preset": "Training Preset",
            "log": "Training Log",
            "preview": "Training Preview",
            "train": "Start Training",
            "stop": "Stop Training",
            "stopped": "Training stopped by user.",
        },
        "step4": {
            "title": "Step 4: Apply to Video",
            "description": "Apply the trained model to create the final face-swapped video.",
            "color_transfer": "Color Transfer",
            "blend_mode": "Blend Mode",
            "log": "Merge Log",
            "apply": "Create Video",
            "status": "Final Status",
            "result": "Result Video",
            "error_no_model": "No trained model found. Please complete training first.",
            "success": "Video created successfully: {path}",
        },
    },
}
