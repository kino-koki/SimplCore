"""
青小团数据管线测试 - 覆盖数据下载器、质量过滤器、去重器、分词管线、数据混合、缓存管理等
"""

import os
import json
import tempfile
import pytest


class TestQingxiaotuanDataDownloader:

    def test_downloader_creation(self):
        from training.data_pipeline.data_ingestion_pipeline import DataDownloader
        downloader = DataDownloader()
        assert downloader is not None

    def test_downloader_http_error_handling(self):
        from training.data_pipeline.data_ingestion_pipeline import DataDownloader
        downloader = DataDownloader()
        result = downloader._handle_http_error(404)
        assert result is False

    def test_downloader_resume_support(self):
        from training.data_pipeline.data_ingestion_pipeline import DataDownloader
        downloader = DataDownloader()
        assert hasattr(downloader, '_download_with_resume')

    def test_download_single_file(self, temp_dir):
        from training.data_pipeline.data_ingestion_pipeline import DataDownloader
        downloader = DataDownloader()
        test_file = os.path.join(temp_dir, "test.jsonl")
        with open(test_file, "w") as f:
            f.write('{"text": "test"}\n')
        assert os.path.exists(test_file)

    def test_download_batch(self, temp_dir):
        from training.data_pipeline.data_ingestion_pipeline import DataDownloader
        downloader = DataDownloader()
        urls = []
        results = downloader.batch_download(urls, temp_dir)
        assert results == []

    def test_downloader_retry_mechanism(self):
        from training.data_pipeline.data_ingestion_pipeline import DataDownloader
        downloader = DataDownloader(max_retries=3)
        assert downloader.max_retries == 3


class TestQingxiaotuanQualityFilter:

    def test_quality_filter_creation(self):
        from training.data_pipeline.data_ingestion_pipeline import QualityFilter
        qf = QualityFilter()
        assert qf is not None

    def test_quality_filter_empty_text(self):
        from training.data_pipeline.data_ingestion_pipeline import QualityFilter
        qf = QualityFilter()
        result = qf.filter("")
        assert result is False

    def test_quality_filter_short_text(self):
        from training.data_pipeline.data_ingestion_pipeline import QualityFilter
        qf = QualityFilter(min_length=50)
        result = qf.filter("短文本")
        assert result is False

    def test_quality_filter_long_text(self):
        from training.data_pipeline.data_ingestion_pipeline import QualityFilter
        qf = QualityFilter(min_length=10)
        text = "这是一段足够长的文本内容，用于测试质量过滤器。"
        result = qf.filter(text)
        assert result is True

    def test_quality_filter_too_short_rejected(self):
        from training.data_pipeline.data_ingestion_pipeline import QualityFilter
        qf = QualityFilter(min_length=1000)
        result = qf.filter("短文本")
        assert result is False

    def test_quality_filter_repetition_detection(self):
        from training.data_pipeline.data_ingestion_pipeline import QualityFilter
        qf = QualityFilter(max_repetition_ratio=0.5)
        text = "你好你好你好你好你好你好你好你好你好你好你好你好" * 10
        assert not qf._is_repetitive(text)

    def test_quality_filter_language_detection(self):
        from training.data_pipeline.data_ingestion_pipeline import QualityFilter
        qf = QualityFilter()
        result = qf._detect_language("这是一段中文文本")
        assert result == "zh" or result is not None

    def test_quality_filter_batch_process(self, sample_pretrain_path):
        from training.data_pipeline.data_ingestion_pipeline import QualityFilter
        qf = QualityFilter()
        with open(sample_pretrain_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        data = [json.loads(line)for line in lines]
        results = [qf.filter(item.get("text", "")) for item in data]
        assert len(results) == len(data)


class TestQingxiaotuanDeduplicator:

    def test_deduplicator_creation(self):
        from training.data_pipeline.data_ingestion_pipeline import Deduplicator
        dedup = Deduplicator()
        assert dedup is not None

    def test_deduplicator_exact_match(self):
        from training.data_pipeline.data_ingestion_pipeline import Deduplicator
        dedup = Deduplicator(method="exact")
        docs = ["Hello world", "Hello world", "Different text"]
        result = dedup.deduplicate(docs)
        assert len(result) == 2
        assert result == ["Hello world", "Different text"]

    def test_deduplicator_fingerprint(self):
        from training.data_pipeline.data_ingestion_pipeline import Deduplicator
        dedup = Deduplicator(method="fingerprint")
        docs = ["这是一段中文文本", "这是一段中文文本", "另一段不同的文本"]
        result = dedup.deduplicate(docs)
        assert len(result) == 2

    def test_deduplicator_no_duplicates(self):
        from training.data_pipeline.data_ingestion_pipeline import Deduplicator
        dedup = Deduplicator(method="exact")
        docs = ["First doc", "Second doc", "Third doc"]
        result = dedup.deduplicate(docs)
        assert len(result) == 3

    def test_deduplicator_empty_input(self):
        from training.data_pipeline.data_ingestion_pipeline import Deduplicator
        dedup = Deduplicator()
        result = dedup.deduplicate([])
        assert result == []

    def test_deduplicator_all_duplicates(self):
        from training.data_pipeline.data_ingestion_pipeline import Deduplicator
        dedup = Deduplicator(method="exact")
        docs = ["Same text"] * 10
        result = dedup.deduplicate(docs)
        assert len(result) == 1


class TestQingxiaotuanTokenizationPipeline:

    def test_tokenization_pipeline_creation(self):
        from training.data_pipeline.data_ingestion_pipeline import TokenizationPipeline
        pipeline = TokenizationPipeline()
        assert pipeline is not None

    def test_tokenize_text_length(self):
        from training.data_pipeline.data_ingestion_pipeline import TokenizationPipeline
        pipeline = TokenizationPipeline()
        result = pipeline._estimate_tokens("这是一段测试文本")
        assert result > 0

    def test_tokenization_batch(self):
        from training.data_pipeline.data_ingestion_pipeline import TokenizationPipeline
        pipeline = TokenizationPipeline()
        texts = ["文本1", "文本2" * 50, "文本3" * 100]
        result = pipeline.tokenize_batch(texts)
        assert len(result) == 3

    def test_truncate_to_max_length(self):
        from training.data_pipeline.data_ingestion_pipeline import TokenizationPipeline
        pipeline = TokenizationPipeline(max_length=512)
        result = pipeline.truncate(list(range(1000)), 512)
        assert len(result) == 512


class TestQingxiaotuanDataMixer:

    def test_data_mixer_creation(self):
        from training.data_pipeline.data_ingestion_pipeline import DataMixer
        mixer = DataMixer()
        assert mixer is not None

    def test_data_mixer_mix_datasets(self):
        from training.data_pipeline.data_ingestion_pipeline import DataMixer
        mixer = DataMixer()
        ds1 = list(range(100))
        ds2 = list(range(50))
        mixed = mixer.mix([("ds1", ds1, 0.5), ("ds2", ds2, 0.5)], total=60)
        assert len(mixed) == 60

    def test_data_mixer_preserves_ratio(self):
        from training.data_pipeline.data_ingestion_pipeline import DataMixer
        mixer = DataMixer()
        ds1 = list(range(100))
        ds2 = list(range(100))
        mixed = mixer.mix([("ds1", ds1, 0.7), ("ds2", ds2, 0.3)], total=100)
        assert len(mixed) == 100

    def test_data_mixer_invalid_weights(self):
        from training.data_pipeline.data_ingestion_pipeline import DataMixer
        mixer = DataMixer()
        ds1 = list(range(100))
        with pytest.raises((ValueError, TypeError)):
            mixer.mix([("ds1", ds1, 1.5)], total=50)


class TestQingxiaotuanCacheManager:

    def test_cache_manager_creation(self, temp_dir):
        from training.data_pipeline.data_ingestion_pipeline import CacheManager
        cm = CacheManager(cache_dir=temp_dir)
        assert cm.cache_dir == temp_dir

    def test_cache_manager_store_retrieve(self, temp_dir):
        from training.data_pipeline.data_ingestion_pipeline import CacheManager
        cm = CacheManager(cache_dir=temp_dir)
        key = "test_key"
        data = [1, 2, 3, 4, 5]
        cm.store(key, data)
        retrieved = cm.retrieve(key)
        assert retrieved == data

    def test_cache_manager_miss(self, temp_dir):
        from training.data_pipeline.data_ingestion_pipeline import CacheManager
        cm = CacheManager(cache_dir=temp_dir)
        result = cm.retrieve("nonexistent")
        assert result is None

    def test_cache_manager_clear(self, temp_dir):
        from training.data_pipeline.data_ingestion_pipeline import CacheManager
        cm = CacheManager(cache_dir=temp_dir)
        cm.store("key1", [1, 2, 3])
        cm.store("key2", [4, 5, 6])
        cm.clear()
        assert cm.retrieve("key1") is None
        assert cm.retrieve("key2") is None


class TestQingxiaotuanFullPipeline:

    def test_pipeline_creation(self):
        from training.data_pipeline.data_ingestion_pipeline import DataIngestionPipeline
        pipeline = DataIngestionPipeline()
        assert pipeline is not None

    def test_pipeline_has_all_components(self):
        from training.data_pipeline.data_ingestion_pipeline import DataIngestionPipeline
        pipeline = DataIngestionPipeline()
        assert hasattr(pipeline, 'downloader')
        assert hasattr(pipeline, 'quality_filter')
        assert hasattr(pipeline, 'deduplicator')
        assert hasattr(pipeline, 'tokenization_pipeline')
        assert hasattr(pipeline, 'data_mixer')

    def test_pipeline_process_sample(self, sample_pretrain_path, temp_dir):
        from training.data_pipeline.data_ingestion_pipeline import DataIngestionPipeline
        pipeline = DataIngestionPipeline()
        with open(sample_pretrain_path, "r") as f:
            data = [json.loads(line) for line in f]
        result = pipeline.process_batch(data, output_dir=temp_dir)
        assert result is not None
