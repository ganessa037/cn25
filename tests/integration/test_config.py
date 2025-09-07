#!/usr/bin/env python3
"""
Integration Test Configuration for Document Parser

Comprehensive configuration management for integration tests including
test settings, mock data, external service configurations, and test utilities.
"""

import os
import json
import time
import random
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class IntegrationTestConfig:
    """Configuration for integration tests."""
    
    # Test environment settings
    test_environment: str = "integration"
    debug_mode: bool = False
    verbose_logging: bool = True
    
    # API settings
    api_base_url: str = "http://localhost:8000"
    api_timeout: int = 30
    api_retry_count: int = 3
    api_retry_delay: float = 1.0
    
    # Performance thresholds
    max_response_time: float = 10.0  # seconds
    max_memory_usage: int = 1024 * 1024 * 1024  # 1GB
    max_cpu_usage: float = 80.0  # percentage
    
    # Load testing settings
    load_test_duration: int = 300  # 5 minutes
    concurrent_users: int = 10
    requests_per_second: int = 5
    ramp_up_time: int = 60  # seconds
    
    # File settings
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    supported_formats: List[str] = field(default_factory=lambda: ['jpg', 'jpeg', 'png', 'pdf', 'tiff'])
    
    # OCR settings
    ocr_engines: List[str] = field(default_factory=lambda: ['tesseract', 'easyocr'])
    ocr_languages: List[str] = field(default_factory=lambda: ['eng', 'msa'])
    
    # Document types
    document_types: List[str] = field(default_factory=lambda: ['mykad', 'spk'])
    
    # Test data settings
    sample_data_dir: str = "test_data"
    mock_data_count: int = 100
    
    # Database settings (if applicable)
    test_database_url: str = "sqlite:///test_document_parser.db"
    
    # External service settings
    mock_external_services: bool = True
    external_service_timeout: int = 10
    
    # Cleanup settings
    cleanup_after_tests: bool = True
    preserve_test_artifacts: bool = False
    
    # Security settings
    enable_rate_limiting: bool = True
    max_requests_per_minute: int = 100
    
    # Monitoring settings
    enable_metrics_collection: bool = True
    metrics_interval: int = 5  # seconds
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Ensure test data directory exists
        self.test_data_path = Path(self.sample_data_dir)
        self.test_data_path.mkdir(parents=True, exist_ok=True)
        
        # Set environment variables
        os.environ['TEST_ENVIRONMENT'] = self.test_environment
        os.environ['API_BASE_URL'] = self.api_base_url
        
        # Configure logging level
        if self.debug_mode:
            os.environ['LOG_LEVEL'] = 'DEBUG'
        elif self.verbose_logging:
            os.environ['LOG_LEVEL'] = 'INFO'
        else:
            os.environ['LOG_LEVEL'] = 'WARNING'


class MockDataGenerator:
    """Generate mock data for integration testing."""
    
    def __init__(self, config: IntegrationTestConfig):
        self.config = config
        self.random = random.Random(42)  # Fixed seed for reproducibility
        
        # Malaysian names and data
        self.malay_names = [
            "Ahmad bin Abdullah", "Siti Nurhaliza binti Hassan", "Muhammad Faiz bin Omar",
            "Nurul Ain binti Ibrahim", "Mohd Rizal bin Zakaria", "Fatimah binti Yusof",
            "Ali bin Rahman", "Khadijah binti Ahmad", "Hassan bin Mohamed", "Zainab binti Ali"
        ]
        
        self.chinese_names = [
            "Tan Wei Ming", "Lim Mei Ling", "Wong Kar Wai", "Lee Siew Lan",
            "Ng Boon Huat", "Ong Li Hua", "Teo Ah Kow", "Goh Swee Lian"
        ]
        
        self.indian_names = [
            "Raj Kumar a/l Suresh", "Priya a/p Raman", "Sanjay a/l Krishnan",
            "Deepa a/p Murugan", "Arjun a/l Selvam", "Kavitha a/p Raj"
        ]
        
        # Malaysian states
        self.states = [
            "Johor", "Kedah", "Kelantan", "Melaka", "Negeri Sembilan",
            "Pahang", "Perak", "Perlis", "Pulau Pinang", "Sabah",
            "Sarawak", "Selangor", "Terengganu", "Kuala Lumpur", "Labuan", "Putrajaya"
        ]
        
        # SPK certificate types
        self.spk_types = [
            "Sijil Pelajaran Malaysia", "Sijil Tinggi Persekolahan Malaysia",
            "Diploma", "Ijazah Sarjana Muda", "Ijazah Sarjana", "Ijazah Doktor Falsafah"
        ]
        
        # Institutions
        self.institutions = [
            "Universiti Malaya", "Universiti Kebangsaan Malaysia", "Universiti Putra Malaysia",
            "Universiti Sains Malaysia", "Universiti Teknologi Malaysia", "Universiti Islam Antarabangsa Malaysia",
            "Universiti Utara Malaysia", "Universiti Malaysia Sarawak", "Universiti Malaysia Sabah"
        ]
    
    def generate_ic_number(self) -> str:
        """Generate a valid Malaysian IC number."""
        # Birth year (last 2 digits)
        year = self.random.randint(50, 99) if self.random.choice([True, False]) else self.random.randint(0, 23)
        
        # Birth month
        month = self.random.randint(1, 12)
        
        # Birth day
        day = self.random.randint(1, 28)  # Safe day range
        
        # Birth place code (state)
        place_codes = {
            "Johor": ["01", "21", "22", "24"],
            "Kedah": ["02", "25", "26", "27"],
            "Kelantan": ["03", "28", "29"],
            "Melaka": ["04", "30"],
            "Negeri Sembilan": ["05", "31", "59"],
            "Pahang": ["06", "32", "33"],
            "Perak": ["07", "34", "35"],
            "Perlis": ["08", "36"],
            "Pulau Pinang": ["09", "37", "38", "39"],
            "Sabah": ["12", "47", "48", "49"],
            "Sarawak": ["13", "50", "51", "52", "53"],
            "Selangor": ["10", "40", "41", "42", "43", "44"],
            "Terengganu": ["11", "45", "46"],
            "Kuala Lumpur": ["14", "54", "55", "56", "57"],
            "Labuan": ["15", "58"],
            "Putrajaya": ["16"]
        }
        
        state = self.random.choice(list(place_codes.keys()))
        place_code = self.random.choice(place_codes[state])
        
        # Random digits
        random_digits = f"{self.random.randint(0, 9999):04d}"
        
        return f"{year:02d}{month:02d}{day:02d}-{place_code}-{random_digits}"
    
    def generate_mykad_data(self) -> Dict[str, Any]:
        """Generate mock MyKad data."""
        # Select random ethnicity and corresponding name
        ethnicity = self.random.choice(["Malay", "Chinese", "Indian"])
        
        if ethnicity == "Malay":
            name = self.random.choice(self.malay_names)
            religion = "Islam"
        elif ethnicity == "Chinese":
            name = self.random.choice(self.chinese_names)
            religion = self.random.choice(["Buddha", "Kristian", "Taoisme"])
        else:  # Indian
            name = self.random.choice(self.indian_names)
            religion = self.random.choice(["Hindu", "Kristian", "Islam"])
        
        ic_number = self.generate_ic_number()
        
        # Extract birth date from IC
        year_part = int(ic_number[:2])
        birth_year = 1900 + year_part if year_part > 50 else 2000 + year_part
        birth_month = int(ic_number[2:4])
        birth_day = int(ic_number[4:6])
        
        # Determine gender from IC (last digit)
        last_digit = int(ic_number[-1])
        gender = "Lelaki" if last_digit % 2 == 1 else "Perempuan"
        
        # Generate address
        address_lines = [
            f"{self.random.randint(1, 999)} Jalan {self.random.choice(['Merdeka', 'Bukit Bintang', 'Ampang', 'Cheras', 'Damansara'])}",
            f"Taman {self.random.choice(['Melawati', 'Desa', 'Sri', 'Bandar'])}",
            f"{self.random.randint(40000, 99999)} {self.random.choice(self.states)}"
        ]
        
        return {
            "ic_number": ic_number,
            "name": name,
            "gender": gender,
            "birth_date": f"{birth_day:02d}/{birth_month:02d}/{birth_year}",
            "birth_place": self.random.choice(self.states),
            "nationality": "Warganegara",
            "religion": religion,
            "ethnicity": ethnicity,
            "address": address_lines,
            "issue_date": self.random_date(2010, 2023),
            "expiry_date": self.random_date(2024, 2030)
        }
    
    def generate_spk_data(self) -> Dict[str, Any]:
        """Generate mock SPK certificate data."""
        name = self.random.choice(self.malay_names + self.chinese_names + self.indian_names)
        ic_number = self.generate_ic_number()
        
        # Certificate details
        cert_type = self.random.choice(self.spk_types)
        institution = self.random.choice(self.institutions)
        
        # Generate subjects and grades (for SPM/STPM)
        subjects = []
        if "Sijil Pelajaran Malaysia" in cert_type:
            spm_subjects = [
                "Bahasa Melayu", "Bahasa Inggeris", "Matematik", "Sejarah",
                "Pendidikan Moral", "Sains", "Geografi", "Ekonomi", "Kimia", "Fizik"
            ]
            for subject in self.random.sample(spm_subjects, self.random.randint(8, 10)):
                grade = self.random.choice(["A+", "A", "A-", "B+", "B", "C+", "C"])
                subjects.append({"subject": subject, "grade": grade})
        
        # Generate graduation details
        graduation_year = self.random.randint(2015, 2023)
        
        return {
            "certificate_number": f"SPK{self.random.randint(100000, 999999)}",
            "name": name,
            "ic_number": ic_number,
            "certificate_type": cert_type,
            "institution": institution,
            "graduation_date": f"{self.random.randint(1, 12):02d}/{graduation_year}",
            "subjects": subjects,
            "cgpa": round(self.random.uniform(2.5, 4.0), 2) if "Ijazah" in cert_type else None,
            "classification": self.random.choice(["Kepujian Kelas Pertama", "Kepujian Kelas Kedua (Tinggi)", "Kepujian Kelas Kedua (Rendah)", "Kelas Ketiga"]) if "Ijazah" in cert_type else None,
            "issue_date": self.random_date(graduation_year, graduation_year + 1),
            "verification_code": f"VER{self.random.randint(100000, 999999)}"
        }
    
    def random_date(self, start_year: int, end_year: int) -> str:
        """Generate a random date between start_year and end_year."""
        start_date = datetime(start_year, 1, 1)
        end_date = datetime(end_year, 12, 31)
        
        time_between = end_date - start_date
        days_between = time_between.days
        random_days = self.random.randrange(days_between)
        
        random_date = start_date + timedelta(days=random_days)
        return random_date.strftime("%d/%m/%Y")
    
    def create_mock_image(self, document_type: str, data: Dict[str, Any], 
                         quality: str = "high") -> Image.Image:
        """Create a mock document image."""
        # Image dimensions based on document type
        if document_type == "mykad":
            width, height = 856, 540  # Standard MyKad dimensions
            bg_color = (0, 100, 200)  # Blue background
        else:  # SPK
            width, height = 595, 842  # A4 dimensions
            bg_color = (255, 255, 255)  # White background
        
        # Create image
        img = Image.new('RGB', (width, height), bg_color)
        draw = ImageDraw.Draw(img)
        
        # Try to load a font, fallback to default if not available
        try:
            font_large = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
            font_medium = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 18)
            font_small = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 14)
        except:
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        if document_type == "mykad":
            self._draw_mykad(draw, data, font_large, font_medium, font_small)
        else:
            self._draw_spk(draw, data, font_large, font_medium, font_small)
        
        # Apply quality degradation
        if quality == "low":
            # Add noise and blur
            img = img.resize((width//2, height//2), Image.LANCZOS)
            img = img.resize((width, height), Image.LANCZOS)
            
            # Add noise
            noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
            img_array = np.array(img)
            img_array = np.clip(img_array.astype(np.int16) + noise - 25, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_array)
        
        elif quality == "medium":
            # Slight compression artifacts
            img = img.resize((int(width*0.8), int(height*0.8)), Image.LANCZOS)
            img = img.resize((width, height), Image.LANCZOS)
        
        return img
    
    def _draw_mykad(self, draw, data, font_large, font_medium, font_small):
        """Draw MyKad layout."""
        text_color = (255, 255, 255)  # White text
        
        # Title
        draw.text((50, 30), "MALAYSIA", fill=text_color, font=font_large)
        draw.text((50, 60), "MYKAD", fill=text_color, font=font_large)
        
        # IC Number (prominent)
        draw.text((50, 120), data['ic_number'], fill=text_color, font=font_large)
        
        # Personal details
        y_pos = 180
        fields = [
            ("Nama/Name", data['name']),
            ("Jantina/Sex", data['gender']),
            ("Tarikh Lahir/Date of Birth", data['birth_date']),
            ("Tempat Lahir/Place of Birth", data['birth_place']),
            ("Kewarganegaraan/Nationality", data['nationality']),
            ("Agama/Religion", data['religion'])
        ]
        
        for label, value in fields:
            draw.text((50, y_pos), f"{label}:", fill=text_color, font=font_small)
            draw.text((50, y_pos + 20), str(value), fill=text_color, font=font_medium)
            y_pos += 50
        
        # Address
        draw.text((50, y_pos), "Alamat/Address:", fill=text_color, font=font_small)
        y_pos += 20
        for line in data['address']:
            draw.text((50, y_pos), line, fill=text_color, font=font_small)
            y_pos += 20
    
    def _draw_spk(self, draw, data, font_large, font_medium, font_small):
        """Draw SPK certificate layout."""
        text_color = (0, 0, 0)  # Black text
        
        # Header
        draw.text((200, 50), "KEMENTERIAN PENDIDIKAN MALAYSIA", fill=text_color, font=font_medium)
        draw.text((250, 80), data['certificate_type'], fill=text_color, font=font_large)
        
        # Certificate number
        draw.text((50, 150), f"No. Sijil: {data['certificate_number']}", fill=text_color, font=font_small)
        
        # Personal details
        y_pos = 200
        fields = [
            ("Nama", data['name']),
            ("No. K/P", data['ic_number']),
            ("Institusi", data['institution']),
            ("Tarikh Tamat Pengajian", data['graduation_date'])
        ]
        
        for label, value in fields:
            draw.text((50, y_pos), f"{label}: {value}", fill=text_color, font=font_medium)
            y_pos += 40
        
        # Subjects (if applicable)
        if data.get('subjects'):
            y_pos += 20
            draw.text((50, y_pos), "Keputusan Peperiksaan:", fill=text_color, font=font_medium)
            y_pos += 30
            
            for subject_data in data['subjects'][:5]:  # Show first 5 subjects
                draw.text((70, y_pos), f"{subject_data['subject']}: {subject_data['grade']}", 
                         fill=text_color, font=font_small)
                y_pos += 25
        
        # CGPA (if applicable)
        if data.get('cgpa'):
            y_pos += 20
            draw.text((50, y_pos), f"PNGK: {data['cgpa']}", fill=text_color, font=font_medium)
            y_pos += 30
            draw.text((50, y_pos), f"Kelas: {data['classification']}", fill=text_color, font=font_medium)
        
        # Footer
        draw.text((50, 750), f"Tarikh Dikeluarkan: {data['issue_date']}", fill=text_color, font=font_small)
        draw.text((50, 770), f"Kod Pengesahan: {data['verification_code']}", fill=text_color, font=font_small)
    
    def save_mock_image(self, img: Image.Image, filename: str, format: str = "PNG") -> str:
        """Save mock image to file."""
        filepath = self.config.test_data_path / filename
        img.save(filepath, format=format)
        return str(filepath)
    
    def generate_test_dataset(self, count: int = None) -> Dict[str, List[str]]:
        """Generate a complete test dataset."""
        count = count or self.config.mock_data_count
        
        dataset = {
            "mykad_high": [],
            "mykad_medium": [],
            "mykad_low": [],
            "spk_high": [],
            "spk_medium": [],
            "spk_low": []
        }
        
        for i in range(count):
            # Generate MyKad samples
            mykad_data = self.generate_mykad_data()
            for quality in ["high", "medium", "low"]:
                img = self.create_mock_image("mykad", mykad_data, quality)
                filename = f"mykad_{quality}_{i:04d}.png"
                filepath = self.save_mock_image(img, filename)
                dataset[f"mykad_{quality}"].append(filepath)
            
            # Generate SPK samples
            spk_data = self.generate_spk_data()
            for quality in ["high", "medium", "low"]:
                img = self.create_mock_image("spk", spk_data, quality)
                filename = f"spk_{quality}_{i:04d}.png"
                filepath = self.save_mock_image(img, filename)
                dataset[f"spk_{quality}"].append(filepath)
        
        # Save dataset metadata
        metadata_file = self.config.test_data_path / "dataset_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                "generated_at": datetime.now().isoformat(),
                "count_per_category": count,
                "total_images": count * 6,  # 2 doc types * 3 qualities
                "categories": list(dataset.keys()),
                "dataset": {k: len(v) for k, v in dataset.items()}
            }, f, indent=2)
        
        return dataset


class MockExternalServices:
    """Mock external services for integration testing."""
    
    def __init__(self, config: IntegrationTestConfig):
        self.config = config
        self.response_delays = {
            "fast": 0.1,
            "normal": 0.5,
            "slow": 2.0,
            "timeout": 15.0
        }
    
    def mock_ocr_service_response(self, delay_type: str = "normal") -> Dict[str, Any]:
        """Mock OCR service response."""
        time.sleep(self.response_delays.get(delay_type, 0.5))
        
        return {
            "status": "success",
            "confidence": random.uniform(0.8, 0.99),
            "text": "Sample extracted text from document",
            "bounding_boxes": [
                {"text": "Sample", "bbox": [10, 20, 80, 40], "confidence": 0.95},
                {"text": "extracted", "bbox": [90, 20, 180, 40], "confidence": 0.92},
                {"text": "text", "bbox": [190, 20, 230, 40], "confidence": 0.88}
            ],
            "processing_time": random.uniform(0.5, 2.0)
        }
    
    def mock_validation_service_response(self, delay_type: str = "normal") -> Dict[str, Any]:
        """Mock validation service response."""
        time.sleep(self.response_delays.get(delay_type, 0.5))
        
        return {
            "status": "success",
            "is_valid": random.choice([True, True, True, False]),  # 75% valid
            "validation_score": random.uniform(0.7, 1.0),
            "errors": [] if random.random() > 0.25 else ["Invalid format detected"],
            "warnings": [] if random.random() > 0.5 else ["Low confidence in some fields"]
        }
    
    def mock_database_response(self, delay_type: str = "fast") -> Dict[str, Any]:
        """Mock database response."""
        time.sleep(self.response_delays.get(delay_type, 0.1))
        
        return {
            "status": "success",
            "record_id": f"REC{random.randint(100000, 999999)}",
            "timestamp": datetime.now().isoformat(),
            "operation": "insert"
        }


class TestUtilities:
    """Utility functions for integration testing."""
    
    @staticmethod
    def create_temp_file(content: bytes, suffix: str = ".tmp") -> str:
        """Create a temporary file with given content."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(content)
            return f.name
    
    @staticmethod
    def cleanup_temp_files(file_paths: List[str]):
        """Clean up temporary files."""
        for file_path in file_paths:
            try:
                os.unlink(file_path)
            except (OSError, FileNotFoundError):
                pass
    
    @staticmethod
    def wait_for_service(url: str, timeout: int = 30) -> bool:
        """Wait for a service to become available."""
        import requests
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    return True
            except requests.RequestException:
                pass
            time.sleep(1)
        
        return False
    
    @staticmethod
    def measure_performance(func, *args, **kwargs) -> Dict[str, Any]:
        """Measure function performance."""
        import psutil
        import tracemalloc
        
        # Start memory tracking
        tracemalloc.start()
        process = psutil.Process()
        
        # Measure CPU and memory before
        cpu_before = process.cpu_percent()
        memory_before = process.memory_info().rss
        
        # Execute function
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        end_time = time.time()
        
        # Measure CPU and memory after
        cpu_after = process.cpu_percent()
        memory_after = process.memory_info().rss
        
        # Get memory peak
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return {
            "success": success,
            "result": result,
            "error": error,
            "execution_time": end_time - start_time,
            "cpu_usage": max(cpu_before, cpu_after),
            "memory_usage": memory_after - memory_before,
            "memory_peak": peak,
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def generate_load_test_data(user_count: int, duration: int) -> List[Dict[str, Any]]:
        """Generate load test scenarios."""
        scenarios = []
        
        for user_id in range(user_count):
            scenario = {
                "user_id": user_id,
                "start_time": random.uniform(0, duration * 0.1),  # Staggered start
                "actions": []
            }
            
            # Generate random actions for this user
            action_count = random.randint(5, 20)
            for action_id in range(action_count):
                action = {
                    "action_id": action_id,
                    "type": random.choice(["classify", "extract", "validate", "process"]),
                    "delay": random.uniform(1, 10),  # Delay between actions
                    "expected_duration": random.uniform(0.5, 5.0)
                }
                scenario["actions"].append(action)
            
            scenarios.append(scenario)
        
        return scenarios


# Global test configuration instance
TEST_CONFIG = IntegrationTestConfig()

# Global mock data generator
MOCK_DATA_GENERATOR = MockDataGenerator(TEST_CONFIG)

# Global mock services
MOCK_SERVICES = MockExternalServices(TEST_CONFIG)

# Test constants
TEST_CONSTANTS = {
    "API_ENDPOINTS": {
        "health": "/health",
        "classify": "/api/v1/classify",
        "extract": "/api/v1/extract",
        "validate": "/api/v1/validate",
        "process": "/api/v1/process",
        "batch": "/api/v1/batch",
        "status": "/api/v1/status",
        "metrics": "/api/v1/metrics"
    },
    "HTTP_STATUS_CODES": {
        "OK": 200,
        "CREATED": 201,
        "BAD_REQUEST": 400,
        "UNAUTHORIZED": 401,
        "FORBIDDEN": 403,
        "NOT_FOUND": 404,
        "METHOD_NOT_ALLOWED": 405,
        "CONFLICT": 409,
        "UNPROCESSABLE_ENTITY": 422,
        "TOO_MANY_REQUESTS": 429,
        "INTERNAL_SERVER_ERROR": 500,
        "SERVICE_UNAVAILABLE": 503
    },
    "PERFORMANCE_THRESHOLDS": {
        "response_time": {
            "fast": 1.0,
            "acceptable": 5.0,
            "slow": 10.0
        },
        "memory_usage": {
            "low": 100 * 1024 * 1024,  # 100MB
            "medium": 500 * 1024 * 1024,  # 500MB
            "high": 1024 * 1024 * 1024  # 1GB
        },
        "cpu_usage": {
            "low": 25.0,
            "medium": 50.0,
            "high": 80.0
        }
    }
}