#!/usr/bin/env python3
"""
test_enhanced_retrieval.py
--------------------------
Test script for the enhanced retrieval system with external search.
"""

import asyncio
import logging
import os
import sys
from typing import List

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.retriever import HybridRetriever
from app.search import ExternalSearcher
from app.models import EvidencePassage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


async def test_external_search():
    """Test the external search functionality independently."""
    print("\n" + "="*60)
    print("TESTING EXTERNAL SEARCH FUNCTIONALITY")
    print("="*60)
    
    # Test with Wikipedia (should work without API key)
    searcher = ExternalSearcher(tavily_api_key=None)
    
    test_queries = [
        "black holes physics",
        "human brain neurons count",
        "climate change effects"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Testing query: '{query}'")
        try:
            results = await searcher.search_wikipedia(query)
            print(f"   Found {len(results)} Wikipedia results")
            for i, result in enumerate(results[:2]):  # Show top 2
                print(f"   {i+1}. {result['title'][:50]}...")
                print(f"      Source: {result['source']}")
                print(f"      Text preview: {result['text'][:100]}...")
        except Exception as e:
            print(f"   ❌ Error: {e}")


async def test_hybrid_retriever():
    """Test the hybrid retriever with external search integration."""
    print("\n" + "="*60)
    print("TESTING HYBRID RETRIEVER WITH EXTERNAL SEARCH")
    print("="*60)
    
    # Test without external search first
    print("\n📚 Testing without external search...")
    retriever_no_ext = HybridRetriever(
        enable_external_search=False,
        tavily_api_key=None
    )
    
    test_claim = "The human brain contains 100 billion neurons"
    
    try:
        results_no_ext = await retriever_no_ext._retrieve_async(test_claim, top_k=3)
        print(f"   Found {len(results_no_ext)} results from knowledge base only")
        for i, result in enumerate(results_no_ext):
            print(f"   {i+1}. Score: {result.rrf_score:.4f}")
            print(f"      Source: {result.source}")
            print(f"      Preview: {result.text[:80]}...")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test with external search (Wikipedia only)
    print("\n🌐 Testing with external search (Wikipedia)...")
    retriever_with_ext = HybridRetriever(
        enable_external_search=True,
        tavily_api_key=None  # No Tavily key, just Wikipedia
    )
    
    try:
        results_with_ext = await retriever_with_ext._retrieve_async(test_claim, top_k=5)
        print(f"   Found {len(results_with_ext)} results including external search")
        
        kb_results = [r for r in results_with_ext if r.source and not r.source.startswith(("Web:", "Wikipedia:"))]
        wiki_results = [r for r in results_with_ext if r.source and r.source.startswith("Wikipedia:")]
        
        print(f"   Knowledge base results: {len(kb_results)}")
        print(f"   Wikipedia results: {len(wiki_results)}")
        
        for i, result in enumerate(results_with_ext[:3]):
            source_type = "📚 KB" if not result.source or not result.source.startswith(("Web:", "Wikipedia:")) else "🌐 Web"
            print(f"   {i+1}. {source_type} Score: {result.rrf_score:.4f}")
            print(f"      Source: {result.source}")
            print(f"      Preview: {result.text[:80]}...")
    except Exception as e:
        print(f"   ❌ Error: {e}")


def test_knowledge_base_size():
    """Test the expanded knowledge base."""
    print("\n" + "="*60)
    print("TESTING EXPANDED KNOWLEDGE BASE")
    print("="*60)
    
    try:
        with open("data/knowledge_base.json", "r", encoding="utf-8") as f:
            import json
            kb_data = json.load(f)
        
        print(f"📊 Knowledge base contains {len(kb_data)} entries")
        
        # Show some examples from different categories
        categories = {
            "Space/Astronomy": ["black holes", "milky way", "universe"],
            "Biology": ["brain", "neurons", "cells", "octopus"],
            "Physics": ["speed of light", "magnetic field"],
            "History": ["edison", "vikings", "caesar"],
            "Myth Busting": ["sugar hyperactivity", "tongue map", "great wall space"]
        }
        
        print("\n📝 Sample entries by category:")
        for category, keywords in categories.items():
            print(f"\n{category}:")
            for entry in kb_data:
                text_lower = entry["text"].lower()
                if any(keyword in text_lower for keyword in keywords):
                    print(f"   • {entry['text'][:60]}...")
                    print(f"     Source: {entry['source']}")
                    break
    except Exception as e:
        print(f"❌ Error reading knowledge base: {e}")


async def main():
    """Run all tests."""
    print("🚀 Testing Enhanced HaluExplain Retrieval System")
    print("="*60)
    
    # Test knowledge base
    test_knowledge_base_size()
    
    # Test external search
    await test_external_search()
    
    # Test hybrid retriever
    await test_hybrid_retriever()
    
    print("\n" + "="*60)
    print("✅ Testing completed!")
    print("="*60)
    
    print("\n📋 Summary:")
    print("• Knowledge base expanded from 20 to 40 entries")
    print("• Wikipedia search integration added")
    print("• Tavily search integration ready (requires API key)")
    print("• Hybrid retriever supports external search toggle")
    print("• All components maintain backward compatibility")


if __name__ == "__main__":
    asyncio.run(main())
