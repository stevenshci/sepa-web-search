import os
import asyncio
import json
import pickle
import hashlib
import time
import threading
import numpy as np
import re
from datetime import datetime
from filelock import FileLock, Timeout


class RAGCache:

    def __init__(self, base_dir=None, ttl_days=30, memory_size=50, similarity_threshold=0.85, embedding_model=None):
        """

        Args:
            base_dir: Base directory for cache storage
            ttl_days: Time-to-live in days for cache entries
            memory_size: Maximum number of entries in memory cache
            similarity_threshold: Minimum similarity score for semantic matches (0.0-1.0)
            embedding_model: Embedding model to use for semantic search
        """
        self.base_dir = base_dir or os.environ.get('CACHE_DIR', '/tmp/rag_cache')
        self.cache_dir = os.path.join(self.base_dir, 'exact')
        self.meta_dir = os.path.join(self.base_dir, 'metadata')
        self.embedding_dir = os.path.join(self.base_dir, 'embeddings')

        for directory in [self.cache_dir, self.meta_dir, self.embedding_dir]:
            os.makedirs(directory, exist_ok=True)

        self.ttl_seconds = ttl_days * 86400
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model  

        self.memory_cache = {}
        self.memory_cache_keys = []
        self.memory_cache_size = memory_size
        self.cache_lock = threading.RLock()

    def set_embedding_model(self, model):
        """Set the embedding model for semantic search"""
        self.embedding_model = model

    def _normalize_query(self, query):
        """Normalize query text for consistency"""
        if query is None:
            return ""
        # Normalize whitespace, remove punctuation, convert to lowercase
        query = query.lower().strip()
        query = re.sub(r'[^\w\s]', '', query)  # Remove punctuation
        query = re.sub(r'\s+', ' ', query)     # Normalize whitespace
        return query

    def _generate_key(self, query, trusted=True, include_video=False):
        """Generate cache key from query parameters"""
        normalized_query = self._normalize_query(query)
        key_parts = [normalized_query, str(trusted), str(include_video)]
        key_string = "_".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_metadata_path(self, cache_key):
        """Get path to metadata file"""
        return os.path.join(self.meta_dir, f"{cache_key}.json")

    def _get_cache_path(self, cache_key):
        """Get path to cache file"""
        return os.path.join(self.cache_dir, f"{cache_key}.pickle")

    def _get_embedding_path(self, cache_key):
        """Get path to embedding file"""
        return os.path.join(self.embedding_dir, f"{cache_key}.npy")

    async def save_embedding(self, cache_key, query):
        """Save query embedding for semantic search"""
        if self.embedding_model is None:
            return False

        try:
            normalized_query = self._normalize_query(query)

            # Offload embedding generation to thread pool
            def generate_embedding():
                return self.embedding_model.embed_query(normalized_query)

            embedding = await asyncio.to_thread(generate_embedding)

            embedding_path = self._get_embedding_path(cache_key)
            lock_path = embedding_path + ".lock"
            try:
                with FileLock(lock_path, timeout=1):
                    np.save(embedding_path, embedding)
                return True
            except Timeout:
                return False
            except Exception:
                return False
        except Exception:
            return False

    def _is_valid_entry(self, entry):
        """Check if a cache entry is still valid (not expired)"""
        if not entry or 'timestamp' not in entry:
            return False
        return (time.time() - entry['timestamp']) < self.ttl_seconds

    async def get(self, query, trusted=True, include_video=False):
        """Get search results from cache"""
        normalized_query = self._normalize_query(query)

        cache_key = self._generate_key(normalized_query, trusted, include_video)

        with self.cache_lock:
            if cache_key in self.memory_cache:
                entry = self.memory_cache[cache_key]
                if self._is_valid_entry(entry):
                    self.memory_cache_keys.remove(cache_key)
                    self.memory_cache_keys.append(cache_key)
                    return entry['data']

        cache_path = self._get_cache_path(cache_key)
        if os.path.exists(cache_path):
            try:
                lock_path = cache_path + ".lock"
                try:
                    with FileLock(lock_path, timeout=1):
                        with open(cache_path, 'rb') as f:
                            entry = pickle.load(f)

                        if self._is_valid_entry(entry):
                            # Update memory cache
                            self._update_memory_cache(cache_key, entry['data'])
                            return entry['data']
                except Timeout:
                    pass
                except Exception:
                    pass
            except Exception:
                pass

        if self.embedding_model is not None:
            semantic_key = await self._find_semantic_match(normalized_query)
            if semantic_key:
                cache_path = self._get_cache_path(semantic_key)
                if os.path.exists(cache_path):
                    try:
                        lock_path = cache_path + ".lock"
                        try:
                            with FileLock(lock_path, timeout=1):
                                with open(cache_path, 'rb') as f:
                                    entry = pickle.load(f)

                                if self._is_valid_entry(entry):
                                    self._update_memory_cache(cache_key, entry['data'])
                                    return entry['data']
                        except Timeout:
                            pass
                        except Exception:
                            pass
                    except Exception:
                        pass

        # No cache hit
        return None

    async def put(self, query, data, trusted=True, include_video=False):
        if data is None or len(data) == 0:
            return False

        normalized_query = self._normalize_query(query)
        cache_key = self._generate_key(normalized_query, trusted, include_video)

        # Create cache entry
        entry = {
            'data': data,
            'timestamp': time.time()
        }

        # Update memory cache
        self._update_memory_cache(cache_key, data)

        cache_path = self._get_cache_path(cache_key)
        meta_path = self._get_metadata_path(cache_key)

        try:
            lock_path = cache_path + ".lock"
            try:
                with FileLock(lock_path, timeout=1):
                    with open(cache_path, 'wb') as f:
                        pickle.dump(entry, f)
            except Timeout:
                return False
            except Exception:
                return False

            metadata = {
                'query': normalized_query,
                'trusted': trusted,
                'include_video': include_video,
                'timestamp': entry['timestamp'],
                'result_count': len(data)
            }

            meta_lock_path = meta_path + ".lock"
            try:
                with FileLock(meta_lock_path, timeout=1):
                    with open(meta_path, 'w') as f:
                        json.dump(metadata, f)
            except Timeout:
                pass
            except Exception:
                pass

            await self.save_embedding(cache_key, normalized_query)

            return True
        except Exception:
            return False

    def _update_memory_cache(self, key, data):
        """Update in-memory LRU cache"""
        with self.cache_lock:
            self.memory_cache[key] = {
                'data': data,
                'timestamp': time.time()
            }

            if key in self.memory_cache_keys:
                self.memory_cache_keys.remove(key)
            self.memory_cache_keys.append(key)

            while len(self.memory_cache_keys) > self.memory_cache_size:
                oldest_key = self.memory_cache_keys.pop(0)
                if oldest_key in self.memory_cache:
                    del self.memory_cache[oldest_key]

    async def _find_semantic_match(self, query):
        """Find semantically similar cached query"""
        if self.embedding_model is None:
            return None

        try:
            def generate_query_embedding():
                return self.embedding_model.embed_query(query)

            query_embedding = await asyncio.to_thread(generate_query_embedding)

            best_match = None
            best_score = self.similarity_threshold  # Minimum threshold

            for filename in os.listdir(self.embedding_dir):
                if not filename.endswith('.npy'):
                    continue

                cache_key = filename.split('.')[0]

                cache_path = self._get_cache_path(cache_key)
                if not os.path.exists(cache_path):
                    continue

                # Load embedding
                embedding_path = self._get_embedding_path(cache_key)
                try:
                    lock_path = embedding_path + ".lock"
                    try:
                        with FileLock(lock_path, timeout=0.5):  # Short timeout for reading
                            cached_embedding = np.load(embedding_path)
                    except Timeout:
                        continue
                    except Exception:
                        continue

                    # Calculate cosine similarity
                    similarity = self._cosine_similarity(query_embedding, cached_embedding)

                    if similarity > best_score:
                        # Check if the cache entry is valid before considering it
                        try:
                            lock_path = cache_path + ".lock"
                            try:
                                with FileLock(lock_path, timeout=0.5):
                                    with open(cache_path, 'rb') as f:
                                        entry = pickle.load(f)

                                    if self._is_valid_entry(entry):
                                        best_score = similarity
                                        best_match = cache_key
                            except Timeout:
                                continue
                            except Exception:
                                continue
                        except Exception:
                            continue
                except Exception:
                    continue

            return best_match

        except Exception:
            return None

    def _cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    async def cleanup_expired(self):
        """Clean up expired cache entries"""
        cleanup_before = time.time() - self.ttl_seconds
        removed = 0

        with self.cache_lock:
            keys_to_remove = []
            for key, entry in self.memory_cache.items():
                if entry['timestamp'] < cleanup_before:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del self.memory_cache[key]
                if key in self.memory_cache_keys:
                    self.memory_cache_keys.remove(key)
                removed += 1

        for directory in [self.cache_dir, self.meta_dir, self.embedding_dir]:
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)

                if not os.path.isfile(file_path):
                    continue

                if os.path.getmtime(file_path) < cleanup_before:
                    try:
                        lock_path = file_path + ".lock"
                        try:
                            with FileLock(lock_path, timeout=1):
                                if os.path.exists(file_path):  # Double-check it still exists
                                    os.remove(file_path)
                                    removed += 1
                        except Timeout:
                            pass
                        except Exception:
                            pass
                    except Exception:
                        pass
                    continue

                if directory == self.cache_dir and filename.endswith('.pickle'):
                    try:
                        lock_path = file_path + ".lock"
                        entry = None
                        try:
                            with FileLock(lock_path, timeout=1):
                                with open(file_path, 'rb') as f:
                                    entry = pickle.load(f)
                        except Timeout:
                            continue
                        except Exception:
                            continue

                        if entry and entry.get('timestamp', 0) < cleanup_before:
                            cache_key = filename.split('.')[0]

                            for ext, dir_path in [
                                ('.pickle', self.cache_dir),
                                ('.json', self.meta_dir),
                                ('.npy', self.embedding_dir)
                            ]:
                                related_path = os.path.join(dir_path, f"{cache_key}{ext}")
                                if os.path.exists(related_path):
                                    try:
                                        related_lock_path = related_path + ".lock"
                                        try:
                                            with FileLock(related_lock_path, timeout=1):
                                                if os.path.exists(related_path): 
                                                    os.remove(related_path)
                                                    removed += 1
                                        except Timeout:
                                            pass
                                        except Exception:
                                            pass
                                    except Exception:
                                        pass
                    except Exception:
                        pass

        return removed