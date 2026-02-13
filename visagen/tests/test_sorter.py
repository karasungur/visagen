"""
Tests for Visagen sorting module.

Tests cover:
- Base classes and dataclasses
- Blur sorting methods
- Pose sorting methods
- Histogram sorting methods
- Color sorting methods
- Metadata sorting methods
- Composite FinalSorter
- CLI argument parsing
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_image():
    """Create a sample BGR image."""
    return np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)


@pytest.fixture
def sharp_image():
    """Create a sharp image with high frequency content."""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    # Add sharp edges
    cv2.rectangle(img, (50, 50), (200, 200), (255, 255, 255), 2)
    cv2.circle(img, (128, 128), 50, (128, 128, 128), 2)
    # Add some noise for texture
    noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    return img


@pytest.fixture
def blurry_image(sharp_image):
    """Create a blurry version of sharp image."""
    return cv2.GaussianBlur(sharp_image, (21, 21), 10)


@pytest.fixture
def sample_landmarks():
    """Create sample 68-point landmarks."""
    # Simple face-like arrangement
    landmarks = np.zeros((68, 2), dtype=np.float32)

    # Jaw (0-16)
    for i in range(17):
        angle = np.pi * (1 - i / 16)
        landmarks[i] = [128 + 80 * np.cos(angle), 128 + 80 * np.sin(angle)]

    # Eyebrows (17-26)
    for i in range(5):
        landmarks[17 + i] = [60 + i * 20, 80]
        landmarks[22 + i] = [140 + i * 20, 80]

    # Nose (27-35)
    for i in range(9):
        landmarks[27 + i] = [128, 100 + i * 8]

    # Eyes (36-47)
    for i in range(6):
        landmarks[36 + i] = [80 + i * 8, 100]
        landmarks[42 + i] = [160 + i * 8, 100]

    # Mouth (48-67)
    for i in range(20):
        angle = 2 * np.pi * i / 20
        landmarks[48 + i] = [128 + 30 * np.cos(angle), 180 + 15 * np.sin(angle)]

    return landmarks


@pytest.fixture
def mock_metadata(sample_landmarks):
    """Create mock FaceMetadata."""
    from visagen.vision.dflimg import FaceMetadata

    return FaceMetadata(
        landmarks=sample_landmarks,
        source_landmarks=sample_landmarks,
        source_rect=(50, 50, 200, 200),
        source_filename="test_001.jpg",
        face_type="whole_face",
        image_to_face_mat=np.eye(2, 3, dtype=np.float32),
    )


@pytest.fixture
def sample_images_dir(temp_dir, sample_image):
    """Create directory with sample images."""
    for i in range(5):
        path = temp_dir / f"image_{i:04d}.jpg"
        cv2.imwrite(str(path), sample_image)
    return temp_dir


# =============================================================================
# Base Classes Tests
# =============================================================================


class TestSortResult:
    """Tests for SortResult dataclass."""

    def test_creation(self, temp_dir):
        """Test SortResult creation."""
        from visagen.sorting.base import SortResult

        path = temp_dir / "test.jpg"
        result = SortResult(filepath=path, score=0.5)

        assert result.filepath == path
        assert result.score == 0.5
        assert result.metadata is None

    def test_with_metadata(self, temp_dir):
        """Test SortResult with metadata."""
        from visagen.sorting.base import SortResult

        path = temp_dir / "test.jpg"
        result = SortResult(filepath=path, score=0.5, metadata={"key": "value"})

        assert result.metadata == {"key": "value"}

    def test_comparison(self, temp_dir):
        """Test SortResult comparison."""
        from visagen.sorting.base import SortResult

        r1 = SortResult(temp_dir / "a.jpg", 0.3)
        r2 = SortResult(temp_dir / "b.jpg", 0.7)

        assert r1 < r2


class TestSortOutput:
    """Tests for SortOutput dataclass."""

    def test_creation(self, temp_dir):
        """Test SortOutput creation."""
        from visagen.sorting.base import SortOutput, SortResult

        sorted_images = [SortResult(temp_dir / "a.jpg", 0.8)]
        trash_images = [SortResult(temp_dir / "b.jpg", 0.1)]

        output = SortOutput(
            sorted_images=sorted_images,
            trash_images=trash_images,
            method="blur",
            elapsed_seconds=1.5,
        )

        assert len(output.sorted_images) == 1
        assert len(output.trash_images) == 1
        assert output.method == "blur"
        assert output.elapsed_seconds == 1.5


# =============================================================================
# Blur Sorter Tests
# =============================================================================


class TestBlurSorter:
    """Tests for BlurSorter."""

    def test_sharp_higher_score(self, sharp_image, blurry_image):
        """Sharp image should have higher score than blurry."""
        from visagen.sorting.blur import BlurSorter

        sorter = BlurSorter()

        sharp_score = sorter.compute_score(sharp_image)
        blurry_score = sorter.compute_score(blurry_image)

        assert sharp_score > blurry_score

    def test_with_metadata(self, sharp_image, mock_metadata):
        """Test with face metadata."""
        from visagen.sorting.blur import BlurSorter

        sorter = BlurSorter()
        score = sorter.compute_score(sharp_image, mock_metadata)

        assert score > 0

    def test_name_and_description(self):
        """Test sorter attributes."""
        from visagen.sorting.blur import BlurSorter

        sorter = BlurSorter()

        assert sorter.name == "blur"
        assert "sharpness" in sorter.description.lower()
        assert sorter.requires_dfl_metadata is True


class TestMotionBlurSorter:
    """Tests for MotionBlurSorter."""

    def test_compute_score(self, sample_image):
        """Test score computation."""
        from visagen.sorting.blur import MotionBlurSorter

        sorter = MotionBlurSorter()
        score = sorter.compute_score(sample_image)

        assert isinstance(score, float)
        assert score >= 0

    def test_name(self):
        """Test sorter name."""
        from visagen.sorting.blur import MotionBlurSorter

        sorter = MotionBlurSorter()
        assert sorter.name == "motion-blur"


class TestBlurFastSorter:
    """Tests for BlurFastSorter."""

    def test_name(self):
        """Test sorter name."""
        from visagen.sorting.blur import BlurFastSorter

        sorter = BlurFastSorter()
        assert sorter.name == "blur-fast"


class TestFaceHullMask:
    """Tests for face hull mask utility."""

    def test_mask_shape(self, sample_image, sample_landmarks):
        """Test mask has correct shape."""
        from visagen.sorting.blur import get_face_hull_mask

        mask = get_face_hull_mask(sample_image.shape, sample_landmarks)

        assert mask.shape[:2] == sample_image.shape[:2]
        assert mask.shape[2] == 1

    def test_mask_values(self, sample_image, sample_landmarks):
        """Test mask contains valid values."""
        from visagen.sorting.blur import get_face_hull_mask

        mask = get_face_hull_mask(sample_image.shape, sample_landmarks)

        assert mask.min() >= 0.0
        assert mask.max() <= 1.0


# =============================================================================
# Pose Sorter Tests
# =============================================================================


class TestYawSorter:
    """Tests for YawSorter."""

    def test_compute_score(self, sample_image, mock_metadata):
        """Test yaw score computation."""
        from visagen.sorting.pose import YawSorter

        sorter = YawSorter()
        score = sorter.compute_score(sample_image, mock_metadata)

        assert isinstance(score, float)

    def test_no_metadata(self, sample_image):
        """Test with no metadata returns 0."""
        from visagen.sorting.pose import YawSorter

        sorter = YawSorter()
        score = sorter.compute_score(sample_image, None)

        assert score == 0.0

    def test_name(self):
        """Test sorter name."""
        from visagen.sorting.pose import YawSorter

        sorter = YawSorter()
        assert sorter.name == "face-yaw"


class TestPitchSorter:
    """Tests for PitchSorter."""

    def test_compute_score(self, sample_image, mock_metadata):
        """Test pitch score computation."""
        from visagen.sorting.pose import PitchSorter

        sorter = PitchSorter()
        score = sorter.compute_score(sample_image, mock_metadata)

        assert isinstance(score, float)

    def test_name(self):
        """Test sorter name."""
        from visagen.sorting.pose import PitchSorter

        sorter = PitchSorter()
        assert sorter.name == "face-pitch"


# =============================================================================
# Color Sorter Tests
# =============================================================================


class TestBrightnessSorter:
    """Tests for BrightnessSorter."""

    def test_bright_higher_score(self):
        """Bright image should have higher score."""
        from visagen.sorting.color import BrightnessSorter

        bright = np.ones((100, 100, 3), dtype=np.uint8) * 200
        dark = np.ones((100, 100, 3), dtype=np.uint8) * 50

        sorter = BrightnessSorter()

        bright_score = sorter.compute_score(bright)
        dark_score = sorter.compute_score(dark)

        assert bright_score > dark_score

    def test_no_metadata_required(self):
        """Test metadata not required."""
        from visagen.sorting.color import BrightnessSorter

        sorter = BrightnessSorter()
        assert sorter.requires_dfl_metadata is False


class TestBlackPixelSorter:
    """Tests for BlackPixelSorter."""

    def test_black_image_higher_score(self):
        """Image with more black pixels should have higher score."""
        from visagen.sorting.color import BlackPixelSorter

        mostly_black = np.zeros((100, 100, 3), dtype=np.uint8)
        mostly_white = np.ones((100, 100, 3), dtype=np.uint8) * 255

        sorter = BlackPixelSorter()

        black_score = sorter.compute_score(mostly_black)
        white_score = sorter.compute_score(mostly_white)

        assert black_score > white_score

    def test_ascending_order(self):
        """Test that sorting is ascending (fewer black = first)."""
        from visagen.sorting.color import BlackPixelSorter

        sorter = BlackPixelSorter()
        assert sorter.reverse_sort is False


# =============================================================================
# Histogram Sorter Tests
# =============================================================================


class TestHistogramSimilaritySorter:
    """Tests for HistogramSimilaritySorter."""

    def test_sort_groups_similar(self, sample_images_dir):
        """Test that similar images are grouped together."""
        from visagen.sorting.histogram import HistogramSimilaritySorter

        image_paths = list(sample_images_dir.glob("*.jpg"))
        sorter = HistogramSimilaritySorter()

        result = sorter.sort(image_paths)

        assert len(result.sorted_images) == len(image_paths)
        assert result.method == "hist"

    def test_default_disables_exact_path(self):
        """Default configuration should avoid exact O(n^2) mode."""
        from visagen.sorting.histogram import HistogramSimilaritySorter

        sorter = HistogramSimilaritySorter()
        assert sorter.exact_limit == 0


class TestHistogramDissimilaritySorter:
    """Tests for HistogramDissimilaritySorter."""

    def test_name(self):
        """Test sorter name."""
        from visagen.sorting.histogram import HistogramDissimilaritySorter

        sorter = HistogramDissimilaritySorter()
        assert sorter.name == "hist-dissim"


class TestSSIMSorters:
    """Tests for SSIM-based sorters."""

    def test_ssim_dissimilarity_outlier_first(self, temp_dir):
        """Outlier image should be ranked first by SSIM dissimilarity."""
        from visagen.sorting.similarity import SSIMDissimilaritySorter

        img_base = np.full((128, 128, 3), 128, dtype=np.uint8)
        img_near = np.full((128, 128, 3), 132, dtype=np.uint8)
        img_outlier = np.full((128, 128, 3), 0, dtype=np.uint8)

        p1 = temp_dir / "a.jpg"
        p2 = temp_dir / "b.jpg"
        p3 = temp_dir / "c.jpg"
        cv2.imwrite(str(p1), img_base)
        cv2.imwrite(str(p2), img_near)
        cv2.imwrite(str(p3), img_outlier)

        sorter = SSIMDissimilaritySorter(exact_limit=10, target_size=64)
        result = sorter.sort([p1, p2, p3])

        assert result.method == "ssim-dissim"
        assert result.sorted_images[0].filepath == p3

    def test_ssim_similarity_groups_similar_first(self, temp_dir):
        """Nearest neighbor should be selected before distant outlier."""
        from visagen.sorting.similarity import SSIMSimilaritySorter

        img_base = np.full((128, 128, 3), 128, dtype=np.uint8)
        img_near = np.full((128, 128, 3), 132, dtype=np.uint8)
        img_outlier = np.full((128, 128, 3), 0, dtype=np.uint8)

        p1 = temp_dir / "a.jpg"
        p2 = temp_dir / "b.jpg"
        p3 = temp_dir / "c.jpg"
        cv2.imwrite(str(p1), img_base)
        cv2.imwrite(str(p2), img_near)
        cv2.imwrite(str(p3), img_outlier)

        sorter = SSIMSimilaritySorter(exact_limit=10, target_size=64)
        result = sorter.sort([p1, p2, p3])

        assert result.method == "ssim"
        assert result.sorted_images[-1].filepath == p3

    def test_default_disables_exact_path(self):
        """Default configuration should avoid exact O(n^2) mode."""
        from visagen.sorting.similarity import SSIMSimilaritySorter

        sorter = SSIMSimilaritySorter()
        assert sorter.exact_limit == 0


# =============================================================================
# Metadata Sorter Tests
# =============================================================================


class TestOneFaceSorter:
    """Tests for OneFaceSorter."""

    def test_filters_multiface(self, temp_dir):
        """Test that multi-face frames are filtered."""
        from visagen.sorting.metadata import OneFaceSorter

        # Create files simulating multi-face detection
        (temp_dir / "00001_0.jpg").touch()
        (temp_dir / "00001_1.jpg").touch()  # Second face from same frame
        (temp_dir / "00002_0.jpg").touch()  # Single face frame

        image_paths = list(temp_dir.glob("*.jpg"))
        sorter = OneFaceSorter()

        result = sorter.sort(image_paths)

        # 00001 frame has 2 faces - both should be trashed
        # 00002 frame has 1 face - should be kept
        assert len(result.sorted_images) == 1
        assert len(result.trash_images) == 2


class TestSourceRectSorter:
    """Tests for SourceRectSorter."""

    def test_larger_rect_higher_score(self, sample_image, mock_metadata):
        """Larger source rect should have higher score."""
        from visagen.sorting.metadata import SourceRectSorter
        from visagen.vision.dflimg import FaceMetadata

        small_meta = FaceMetadata(
            landmarks=mock_metadata.landmarks,
            source_landmarks=mock_metadata.source_landmarks,
            source_rect=(100, 100, 150, 150),  # 50x50
            source_filename="small.jpg",
            face_type="whole_face",
            image_to_face_mat=np.eye(2, 3, dtype=np.float32),
        )

        large_meta = FaceMetadata(
            landmarks=mock_metadata.landmarks,
            source_landmarks=mock_metadata.source_landmarks,
            source_rect=(0, 0, 200, 200),  # 200x200
            source_filename="large.jpg",
            face_type="whole_face",
            image_to_face_mat=np.eye(2, 3, dtype=np.float32),
        )

        sorter = SourceRectSorter()

        small_score = sorter.compute_score(sample_image, small_meta)
        large_score = sorter.compute_score(sample_image, large_meta)

        assert large_score > small_score


# =============================================================================
# Composite Sorter Tests
# =============================================================================


class TestFinalSorter:
    """Tests for FinalSorter."""

    def test_init_defaults(self):
        """Test default initialization."""
        from visagen.sorting.composite import FinalSorter

        sorter = FinalSorter()

        assert sorter.target_count == 2000
        assert sorter.faster is False
        assert sorter.yaw_bins == 128

    def test_init_custom(self):
        """Test custom initialization."""
        from visagen.sorting.composite import FinalSorter

        sorter = FinalSorter(target_count=500, faster=True, yaw_bins=64)

        assert sorter.target_count == 500
        assert sorter.faster is True
        assert sorter.yaw_bins == 64

    def test_name(self):
        """Test sorter name."""
        from visagen.sorting.composite import FinalSorter

        sorter = FinalSorter()
        assert sorter.name == "final"

    def test_empty_input(self):
        """Test with empty input."""
        from visagen.sorting.composite import FinalSorter

        sorter = FinalSorter()
        result = sorter.sort([])

        assert len(result.sorted_images) == 0
        assert len(result.trash_images) == 0


class TestFinalFastSorter:
    """Tests for FinalFastSorter."""

    def test_faster_flag(self):
        """Test faster flag is set."""
        from visagen.sorting.composite import FinalFastSorter

        sorter = FinalFastSorter()
        assert sorter.faster is True

    def test_name(self):
        """Test sorter name."""
        from visagen.sorting.composite import FinalFastSorter

        sorter = FinalFastSorter()
        assert sorter.name == "final-fast"

    def test_uses_source_rect_area_for_fast_filtering(self, temp_dir):
        """Final fast should keep larger source rects when trimming."""
        from visagen.sorting.composite import FinalFastSorter
        from visagen.sorting.processor import ProcessedImage

        image_paths = [temp_dir / f"img_{i:03d}.jpg" for i in range(11)]

        processed = []
        for i, path in enumerate(image_paths):
            processed.append(
                ProcessedImage(
                    filepath=path,
                    image=np.zeros((8, 8, 3), dtype=np.uint8),
                    sharpness=0.0,
                    yaw=0.0,
                    pitch=0.0,
                    histogram=np.zeros(256, dtype=np.float32),
                    source_rect_area=float(i),
                    error=None,
                )
            )

        class DummyProcessor:
            def load_and_process_all(self, *args, **kwargs):
                return processed

        sorter = FinalFastSorter(target_count=1, yaw_bins=1)
        result = sorter.sort(image_paths, processor=DummyProcessor())

        trashed = {item.filepath for item in result.trash_images}
        assert image_paths[0] in trashed  # Smallest area should be dropped first.
        assert image_paths[-1] not in trashed  # Largest area should be retained.


# =============================================================================
# Processor Tests
# =============================================================================


class TestParallelSortProcessor:
    """Tests for ParallelSortProcessor."""

    def test_init_defaults(self):
        """Test default initialization."""
        from visagen.sorting.processor import ParallelSortProcessor

        processor = ParallelSortProcessor()

        assert processor.max_workers > 0
        assert processor.use_threads is True

    def test_init_custom(self):
        """Test custom initialization."""
        from visagen.sorting.processor import ParallelSortProcessor

        processor = ParallelSortProcessor(max_workers=2, use_threads=False)

        assert processor.max_workers == 2
        assert processor.use_threads is False


# =============================================================================
# CLI Tests
# =============================================================================


class TestSorterCLI:
    """Tests for sorter CLI."""

    def test_parse_args_minimal(self, sample_images_dir, monkeypatch):
        """Test minimal argument parsing."""
        from visagen.tools.sorter import parse_args

        monkeypatch.setattr(
            "sys.argv",
            ["visagen-sort", str(sample_images_dir)],
        )

        args = parse_args()

        assert args.input == sample_images_dir
        assert args.method == "blur"
        assert args.target == 2000
        assert args.exact_limit is None
        assert args.undo_last_trash is False

    def test_parse_args_full(self, sample_images_dir, temp_dir, monkeypatch):
        """Test full argument parsing."""
        from visagen.tools.sorter import parse_args

        output_dir = temp_dir / "output"

        monkeypatch.setattr(
            "sys.argv",
            [
                "visagen-sort",
                str(sample_images_dir),
                "-m",
                "final",
                "-t",
                "500",
                "-o",
                str(output_dir),
                "-j",
                "4",
                "--dry-run",
                "-v",
            ],
        )

        args = parse_args()

        assert args.method == "final"
        assert args.target == 500
        assert args.output == output_dir
        assert args.jobs == 4
        assert args.dry_run is True
        assert args.verbose is True
        assert args.exact_limit is None

    def test_parse_args_with_exact_limit(self, sample_images_dir, monkeypatch):
        """Test explicit exact-limit parsing."""
        from visagen.tools.sorter import parse_args

        monkeypatch.setattr(
            "sys.argv",
            ["visagen-sort", str(sample_images_dir), "--exact-limit", "128"],
        )

        args = parse_args()
        assert args.exact_limit == 128

    def test_parse_args_undo(self, sample_images_dir, monkeypatch):
        """Undo mode should parse cleanly."""
        from visagen.tools.sorter import parse_args

        monkeypatch.setattr(
            "sys.argv",
            ["visagen-sort", str(sample_images_dir), "--undo-last-trash"],
        )

        args = parse_args()
        assert args.undo_last_trash is True

    def test_get_sort_methods(self):
        """Test available sort methods."""
        from visagen.tools.sorter import get_sort_methods

        methods = get_sort_methods()

        assert "blur" in methods
        assert "final" in methods
        assert "face-yaw" in methods
        assert "hist" in methods
        assert "blur-fast" in methods
        assert "absdiff-dissim" in methods
        assert "id-sim" in methods
        assert "id-dissim" in methods
        assert "ssim" in methods
        assert "ssim-dissim" in methods

    @pytest.mark.parametrize(
        ("profile", "expected_use_threads"),
        [
            ("cpu_bound", False),
            ("io_bound", True),
            ("gpu_bound", True),
        ],
    )
    def test_main_auto_exec_mode_uses_profile(
        self,
        sample_images_dir,
        monkeypatch,
        profile,
        expected_use_threads,
    ):
        """Auto mode should pick thread/process based on sorter profile."""
        from visagen.sorting.base import SortOutput, SortResult
        from visagen.tools import sorter as sorter_module

        captured: dict[str, object] = {}

        class DummyProcessor:
            def __init__(self, max_workers=None, use_threads=True):
                captured["max_workers"] = max_workers
                captured["use_threads"] = use_threads

        class DummySorter:
            description = "dummy"
            execution_profile = profile

            def sort(self, image_paths, processor):
                assert isinstance(processor, DummyProcessor)
                return SortOutput(
                    sorted_images=[SortResult(path, 0.0) for path in image_paths],
                    trash_images=[],
                    method="dummy",
                    elapsed_seconds=0.0,
                )

        monkeypatch.setattr(
            sorter_module, "get_sort_methods", lambda: {"blur": DummySorter}
        )
        monkeypatch.setattr(
            "visagen.sorting.processor.ParallelSortProcessor",
            DummyProcessor,
        )
        monkeypatch.setattr(
            sorter_module, "apply_sort_result", lambda *args, **kwargs: None
        )

        rc = sorter_module.main(
            [str(sample_images_dir), "--method", "blur", "--jobs", "3"],
        )

        assert rc == 0
        assert captured["max_workers"] == 3
        assert captured["use_threads"] is expected_use_threads

    def test_get_image_paths(self, sample_images_dir):
        """Test image path discovery."""
        from visagen.tools.sorter import get_image_paths

        paths = get_image_paths(sample_images_dir)

        assert len(paths) == 5
        assert all(p.suffix == ".jpg" for p in paths)

    def test_apply_sort_result_custom_trash_collision(self, tmp_path):
        """Custom trash dir should resolve destination filename collisions."""
        from visagen.sorting.base import SortOutput, SortResult
        from visagen.tools.sorter import apply_sort_result

        dataset = tmp_path / "aligned"
        trash_dir = tmp_path / "trash"
        dataset.mkdir(parents=True, exist_ok=True)
        trash_dir.mkdir(parents=True, exist_ok=True)

        src = dataset / "dup.jpg"
        src.write_bytes(b"source")
        (trash_dir / "dup.jpg").write_bytes(b"existing")

        output = SortOutput(
            sorted_images=[],
            trash_images=[SortResult(src, 0.0, {"reason": "test"})],
            method="hist",
            elapsed_seconds=0.0,
        )

        apply_sort_result(
            output,
            input_dir=dataset,
            output_dir=None,
            trash_dir=trash_dir,
            no_rename=False,
            dry_run=False,
            verbose=False,
        )

        assert not src.exists()
        assert (trash_dir / "dup.jpg").exists()
        assert (trash_dir / "dup_restored_1.jpg").exists()

    def test_main_undo_last_trash(self, tmp_path):
        """Sorter CLI should support undoing the last managed trash batch."""
        from visagen.tools import sorter as sorter_module
        from visagen.tools.dataset_trash import move_to_trash

        dataset = tmp_path / "aligned"
        dataset.mkdir(parents=True, exist_ok=True)
        target = dataset / "restore.jpg"
        target.write_bytes(b"x")

        batch = move_to_trash([target], dataset_root=dataset, reason="test-sort")
        assert batch.count_moved == 1
        assert not target.exists()

        rc = sorter_module.main([str(dataset), "--undo-last-trash"])
        assert rc == 0
        assert target.exists()


# =============================================================================
# Integration Tests
# =============================================================================


class TestSortingIntegration:
    """Integration tests for sorting pipeline."""

    def test_blur_sort_pipeline(self, sample_images_dir):
        """Test full blur sorting pipeline."""
        from visagen.sorting import BlurSorter, ParallelSortProcessor

        image_paths = list(sample_images_dir.glob("*.jpg"))
        sorter = BlurSorter()
        processor = ParallelSortProcessor()

        result = sorter.sort(image_paths, processor)

        assert len(result.sorted_images) + len(result.trash_images) == len(image_paths)
        assert result.method == "blur"
        assert result.elapsed_seconds >= 0

    def test_brightness_sort_pipeline(self, sample_images_dir):
        """Test brightness sorting pipeline."""
        from visagen.sorting import BrightnessSorter

        image_paths = list(sample_images_dir.glob("*.jpg"))
        sorter = BrightnessSorter()

        result = sorter.sort(image_paths)

        assert len(result.sorted_images) == len(image_paths)
        # Verify descending order (reverse_sort=True is default)
        scores = [r.score for r in result.sorted_images]
        assert scores == sorted(scores, reverse=True)
