"""
AI Services Module - Centralized AI function management

This module contains all AI-related functions for the agricultural advisory chatbot:
- RAG (file search store) management
- Knowledge base initialization and file uploads
- AI decision logic for infographic generation
- Infographic generation (SVG and image formats)

All functions interact with the Gemini client and are designed to be imported
and used by the Flask app.
"""

import os
import time
import json
import re
import glob
import io
import logging
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime
from io import BytesIO
import concurrent.futures

from google import genai
from google.genai import types
from PIL import Image

# ============================================================================
# MODULE-LEVEL CONFIGURATION
# ============================================================================

logger = logging.getLogger(__name__)

# These will be set by the Flask app on startup
UPLOAD_FOLDER = 'uploads'
FLASK_APP = None  # Will be set to Flask app instance
CLIENT = None  # Will be set to genai.Client instance

# File search store persistence
_STORE_INFO_PATH = '.file_search_store.json'
_UPLOAD_CACHE_NAME = 'upload_cache.json'

# Language names for infographic generation
LANGUAGE_NAMES = {
    'english': 'English',
    'hindi': 'Hindi (हिंदी)',
    'hinglish': 'Hindi-English mix (Hinglish)',
    'marathi': 'Marathi (मराठी)',
    'tamil': 'Tamil (தமிழ்)',
    'telugu': 'Telugu (తెలుగు)',
    'kannada': 'Kannada (ಕನ್ನಡ)',
    'gujarati': 'Gujarati (ગુજરાતી)',
    'punjabi': 'Punjabi (ਪੰਜਾਬੀ)',
    'bengali': 'Bengali (বাংলা)',
    'malayalam': 'Malayalam (മലയാളം)',
    'odia': 'Odia (ଓଡ଼ିଆ)',
    'assamese': 'Assamese (অসমীয়া)',
    'urdu': 'Urdu (اردو)'
}


def classify_query_type(question: str) -> Dict[str, Any]:
    """
    Classify whether a query benefits from visual or text response.
    
    Uses a fast LLM to analyze the question and determine the best response format.
    
    Args:
        question: The user's question
    
    Returns:
        Dict with keys:
        - format (str): 'visual' or 'text'
        - confidence (float): 0.0 to 1.0
        - reason (str): Brief explanation
    """
    if CLIENT is None:
        logger.warning("⚠️ AI client not available for query classification")
        return {'format': 'text', 'confidence': 0.5, 'reason': 'AI unavailable, defaulting to text'}
    
    # Quick keyword-based classification for obvious cases
    question_lower = question.lower().strip()
    
    # Pleasantries and greetings - always text
    pleasantry_patterns = [
        'hello', 'hi', 'namaste', 'namaskar', 'good morning', 'good evening',
        'thank you', 'thanks', 'dhanyawad', 'shukriya', 'bye', 'goodbye',
        'how are you', 'what is your name', 'who are you', 'who made you'
    ]
    if any(pattern in question_lower for pattern in pleasantry_patterns):
        logger.info("📝 Query classified as TEXT (pleasantry detected)")
        return {'format': 'text', 'confidence': 0.95, 'reason': 'Greeting or pleasantry detected'}
    
    # Explicit visual requests - always visual
    visual_explicit = [
        'show me', 'diagram', 'chart', 'infographic', 'picture', 'image', 'visual',
        'steps', 'step by step', 'how to', 'how do i', 'schedule', 'calendar',
        'process', 'procedure', 'symptoms', 'identify', 'comparison', 'compare'
    ]
    if any(pattern in question_lower for pattern in visual_explicit):
        logger.info("🎨 Query classified as VISUAL (explicit request)")
        return {'format': 'visual', 'confidence': 0.95, 'reason': 'Visual content keywords detected'}
    
    # Use LLM for nuanced classification
    prompt = """You are a query classifier for a farmer-focused agricultural chatbot.
Classify whether this query would benefit MORE from a VISUAL response (infographic, diagram, chart) 
or a TEXT response (plain explanation).

VISUAL queries include:
- How-to processes (planting steps, treatment procedures)
- Schedules and timelines (fertilizer schedule, irrigation calendar)
- Disease/pest symptoms and identification
- Comparisons (varieties, methods, products)
- Statistics and data

TEXT queries include:
- Greetings and pleasantries
- Simple factual questions
- Clarification requests
- Opinion or advice seeking
- Conversational questions

Query: "{question}"

Respond ONLY with JSON:
{{"format": "visual" or "text", "confidence": 0.0-1.0, "reason": "brief explanation"}}"""

    try:
        resp = CLIENT.models.generate_content(
            model='gemini-3-pro-preview',
            contents=prompt.format(question=question)
        )
        
        raw = resp.text or ''
        parsed = _parse_json_from_text(raw)
        
        if parsed and 'format' in parsed:
            result = {
                'format': parsed.get('format', 'text').lower(),
                'confidence': float(parsed.get('confidence', 0.5)),
                'reason': parsed.get('reason', 'AI classification')
            }
            logger.info(f"🔍 Query classified as {result['format'].upper()} (confidence: {result['confidence']:.2f})")
            return result
            
    except Exception as e:
        logger.error(f'❌ Query classification failed: {e}')
    
    # Default to text when uncertain
    logger.info("📝 Query classification defaulting to TEXT")
    return {'format': 'text', 'confidence': 0.5, 'reason': 'Classification failed, defaulting to text'}


# ============================================================================
# RAG & FILE MANAGEMENT FUNCTIONS
# ============================================================================

def set_client_and_app(client: genai.Client, app=None, upload_folder: str = 'uploads'):
    """Initialize module-level references to the Gemini client and Flask app.
    
    Must be called from main app.py after client is created.
    """
    global CLIENT, FLASK_APP, UPLOAD_FOLDER
    CLIENT = client
    FLASK_APP = app
    UPLOAD_FOLDER = upload_folder
    logger.info("✅ AI services module initialized with Gemini client and Flask app")


def ensure_file_search_store():
    """
    Ensure a Gemini file-search store exists. Reuses persisted store on disk
    to avoid creating new stores on every startup.
    
    Returns: Store object with .name attribute
    Raises: RuntimeError if client not initialized
    """
    if CLIENT is None:
        raise RuntimeError('Gemini client not initialized. Call set_client_and_app() first.')
    
    # Try to read persisted store name from disk
    try:
        if os.path.exists(_STORE_INFO_PATH):
            with open(_STORE_INFO_PATH, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
                name = data.get('name')
                if name:
                    # Minimal holder for store.name
                    class _Store:
                        def __init__(self, store_name):
                            self.name = store_name
                    
                    store = _Store(name)
                    logger.info(f'✅ Reusing persisted file search store: {name}')
                    return store
    except Exception as e:
        logger.warning(f'⚠️ Failed to read persisted store info: {e}')
    
    # Create a new store and persist its name
    logger.info("🔄 Creating new file search store...")
    store = CLIENT.file_search_stores.create()
    
    try:
        with open(_STORE_INFO_PATH, 'w', encoding='utf-8') as fh:
            json.dump({'name': store.name, 'created_at': int(time.time())}, fh)
        logger.info(f'✅ Created and persisted new file search store: {store.name}')
    except Exception as e:
        logger.warning(f'⚠️ Failed to persist store info (will recreate on restart): {e}')
    
    return store


def upload_file_to_store(path: str) -> bool:
    """
    Upload a file to the Gemini file-search store with hash-based deduplication.
    
    Args:
        path: File path to upload
    
    Returns:
        True if upload successful or already uploaded, False otherwise
    """
    try:
        store = ensure_file_search_store()
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        cache_path = os.path.join(UPLOAD_FOLDER, _UPLOAD_CACHE_NAME)
        
        # Compute SHA256 hash of file content
        def _compute_hash(file_path: str) -> str:
            h = hashlib.sha256()
            with open(file_path, 'rb') as fh:
                for chunk in iter(lambda: fh.read(8192), b''):
                    h.update(chunk)
            return h.hexdigest()
        
        file_hash = _compute_hash(path)
        
        # Load upload cache
        cache = {}
        try:
            if os.path.exists(cache_path):
                with open(cache_path, 'r', encoding='utf-8') as ch:
                    cache = json.load(ch)
        except Exception:
            cache = {}
        
        # Check if already uploaded to this store
        if file_hash in cache and cache[file_hash].get('uploaded') and cache[file_hash].get('store_name') == store.name:
            logger.info(f"✅ Skipping upload for {path}; already in store {store.name}")
            return True
        
        # Upload the file
        logger.info(f"📤 Uploading {os.path.basename(path)} to file search store...")
        CLIENT.file_search_stores.upload_to_file_search_store(
            file_search_store_name=store.name,
            file=path
        )
        
        # The above call is blocking and will raise on error. Polling is not required.
        
        # Update cache
        try:
            cache[file_hash] = {
                'filename': os.path.basename(path),
                'uploaded': True,
                'timestamp': int(time.time()),
                'store_name': store.name
            }
            with open(cache_path, 'w', encoding='utf-8') as ch:
                json.dump(cache, ch)
        except Exception:
            logger.warning('⚠️ Failed to update upload cache')
        
        logger.info(f"✅ Successfully uploaded: {os.path.basename(path)}")
        return True
        
    except Exception as e:
        logger.error(f'❌ Upload to store failed for {path}: {e}')
        return False


def initialize_knowledge_base():
    """
    Automatically upload all knowledge base files to the file search store on startup.
    Called on first HTTP request to avoid blocking app initialization.
    """
    if CLIENT is None:
        logger.warning("⚠️ Gemini client not initialized; skipping knowledge base upload")
        return
    
    kb_dir = 'knowledge_base'
    if not os.path.isdir(kb_dir):
        logger.warning(f"⚠️ Knowledge base directory '{kb_dir}' not found")
        return
    
    logger.info("📚 Starting knowledge base initialization...")

    # Supported file extensions for RAG
    supported_extensions = ('.pdf', '.txt', '.json', '.doc', '.docx')

    # Collect all candidate files first
    file_paths: List[str] = []
    for root, dirs, files in os.walk(kb_dir):
        for filename in files:
            if filename.lower().endswith(supported_extensions):
                file_paths.append(os.path.join(root, filename))

    if not file_paths:
        logger.info("📚 No knowledge base files found to upload.")
        return

    logger.info(f"📚 Found {len(file_paths)} file(s) to upload. Using parallel uploader...")

    uploaded_count = 0

    # Use a ThreadPoolExecutor to upload files concurrently. Upload function is IO-bound.
    max_workers = min(8, (os.cpu_count() or 2) * 2)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as exe:
        future_to_path = {exe.submit(upload_file_to_store, p): p for p in file_paths}

        for fut in concurrent.futures.as_completed(future_to_path):
            path = future_to_path[fut]
            try:
                success = fut.result()
                if success:
                    uploaded_count += 1
                    logger.info(f"✅ Uploaded: {path}")
                else:
                    logger.warning(f"⚠️ Failed to upload: {path}")
            except Exception as e:
                logger.error(f"❌ Error uploading {path}: {e}")

    logger.info(f"📚 Knowledge base initialization complete! Uploaded {uploaded_count}/{len(file_paths)} files.")


def load_reference_images(category: str, max_images: int = 2) -> List[Dict[str, Any]]:
    """
    Load reference plant images for a given category (e.g., 'sugarcane', 'weeds').
    
    Args:
        category: Category folder name (e.g., 'sugarcane', 'weeds')
        max_images: Maximum number of images to load (default 2)
    
    Returns:
        List of dicts with 'data' (bytes) and 'filename' keys
    """
    folder = os.path.join('knowledge_base', 'plant_images', category)
    if not os.path.isdir(folder):
        logger.debug(f"Reference image folder not found: {folder}")
        return []
    
    # Find all image files
    paths: List[str] = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        paths.extend(glob.glob(os.path.join(folder, ext)))
    
    # Limit to max_images
    paths = paths[:max_images]
    
    # Load image data
    images = []
    for p in paths:
        try:
            with open(p, 'rb') as f:
                images.append({'data': f.read(), 'filename': os.path.basename(p)})
                logger.debug(f"Loaded reference image: {os.path.basename(p)}")
        except Exception as e:
            logger.warning(f'⚠️ Failed to read reference image {p}: {e}')
    
    return images


# ============================================================================
# INFOGRAPHIC DECISION & GENERATION FUNCTIONS
# ============================================================================

def _parse_json_from_text(raw: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from text response, handling both fenced and raw JSON.
    
    Tries three approaches:
    1. Extract fenced ```json...``` block
    2. Extract bare {...} block
    3. Parse entire raw text as JSON
    
    Args:
        raw: Raw text from AI model
    
    Returns:
        Parsed JSON dict or None if parsing fails
    """
    if not raw:
        return None
    
    # Try fenced JSON first
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
    txt = m.group(1) if m else raw.strip()

    # Helper: try multiple parsing strategies
    def try_load(s: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(s)
        except Exception:
            pass
        # Try to fix common issues: single quotes -> double quotes
        try:
            s2 = s.replace("\n", " ")
            s2 = re.sub(r"\bNone\b", 'null', s2)
            s2 = re.sub(r"\bTrue\b", 'true', s2)
            s2 = re.sub(r"\bFalse\b", 'false', s2)
            # naive single->double quote replacement
            if "'" in s2 and '"' not in s2:
                s3 = s2.replace("'", '"')
                try:
                    return json.loads(s3)
                except Exception:
                    pass
        except Exception:
            pass
        # Try ast.literal_eval as a last resort (can parse Python dicts)
        try:
            import ast
            val = ast.literal_eval(s)
            if isinstance(val, dict):
                return val
        except Exception:
            pass
        return None

    # First attempt
    parsed = try_load(txt)
    if parsed is not None:
        return parsed

    # If that failed, try to extract the largest {...} substring (balanced braces)
    start = raw.find('{')
    end = raw.rfind('}')
    if start != -1 and end != -1 and end > start:
        candidate = raw[start:end+1]
        parsed = try_load(candidate)
        if parsed is not None:
            return parsed

    logger.debug(f"Failed to parse JSON from model output (preview): {raw[:200]}...")
    return None


def parse_json_from_text(raw: str) -> Optional[Dict[str, Any]]:
    """Public wrapper that parses model text into JSON dict or returns None."""
    return _parse_json_from_text(raw)


def decide_make_infographic(content: str, original_question: str = '') -> Dict[str, Any]:
    """
    Intelligently decide whether to generate an infographic.
    
    Args:
        content: Response content to evaluate
        original_question: Original user question (checked for triggers)
    
    Returns:
        Dict with keys:
        - make (bool): Whether to generate infographic
        - reason (str): Explanation of decision
        - style (str): Suggested style ('simple', 'chart', 'timeline')
    """
    logger.info("🕵️ Starting infographic decision logic...")
    logger.info(f"   Question preview: '{original_question[:100]}...'")
    
    # Primary trigger: if the user explicitly asks for an image
    if "generate image" in original_question.lower():
        logger.info("✅ TRIGGER FOUND: User explicitly requested an image.")
        return {
            'make': True,
            'reason': 'User requested an infographic.',
            'style': 'detailed'
        }

    # If no client, skip AI decision
    if CLIENT is None:
        logger.warning("⚠️ AI client not available for decision")
        return {'make': False, 'reason': 'AI unavailable', 'style': 'simple'}
    
    # Ask Gemini for decision
    prompt = (
        "You are a concise assistant that decides whether a small infographic (SVG) "
        "would help make this information easier to understand for a smallholder farmer. "
        "Return ONLY JSON: {\n"
        "  \"make_infographic\": true|false,\n"
        "  \"reason\": \"short rationale\",\n"
        "  \"style\": \"simple|chart|timeline\"\n}\n"
        "Content:\n" + content
    )
    
    try:
        resp = CLIENT.models.generate_content(model='gemini-3-pro-preview', contents=prompt)
        parsed = _parse_json_from_text(resp.text or '')
        
        if parsed:
            final_decision = {
                'make': bool(parsed.get('make_infographic')),
                'reason': parsed.get('reason', ''),
                'style': parsed.get('style', 'simple')
            }
            
            if final_decision['make']:
                logger.info(f"✅ AI decided to generate: {final_decision['reason']}")
            else:
                logger.info(f"❌ AI decided not to generate: {final_decision['reason']}")
            
            return final_decision
    except Exception as e:
        logger.error(f'❌ AI decision call failed: {e}')
    
    logger.info("➡️ Decision: Do NOT generate (default)")
    return {'make': False, 'reason': 'default', 'style': 'simple'}


def generate_svg_infographic(content: str, style: str = 'simple') -> Optional[str]:
    """
    Generate a compact SVG infographic summarizing content.
    
    Args:
        content: Content to visualize
        style: Style hint ('simple', 'chart', 'timeline')
    
    Returns:
        Raw SVG string or None on failure
    """
    if CLIENT is None:
        logger.warning("⚠️ AI client not available for SVG generation")
        return None
    
    prompt = (
        f"Produce a single standalone SVG (no HTML) under 800px width that visually "
        f"summarizes the following content for a sugarcane farmer. Use simple shapes, "
        f"large readable labels, and an uncluttered layout. Output ONLY the raw SVG.\n"
        f"Style hint: {style}\nContent:\n" + content
    )
    
    try:
        logger.info("🎨 Generating SVG infographic...")
        resp = CLIENT.models.generate_content(model='gemini-3-pro-preview', contents=prompt)
        raw = resp.text or ''
        
        # Try to extract fenced SVG first
        m = re.search(r'```(?:svg)?\s*(<svg[\s\S]*?</svg>)\s*```', raw, re.DOTALL | re.IGNORECASE)
        if m:
            logger.info("✅ SVG extracted from fenced block")
            return m.group(1)
        
        # Try to find bare <svg>...</svg>
        m2 = re.search(r'(<svg[\s\S]*?</svg>)', raw, re.DOTALL | re.IGNORECASE)
        if m2:
            logger.info("✅ SVG extracted from raw text")
            return m2.group(1)
        
        # As fallback, return if it looks like SVG
        if raw.strip().startswith('<svg'):
            logger.info("✅ SVG found at start of response")
            return raw.strip()
        
        logger.warning("⚠️ No SVG found in response")
        
    except Exception as e:
        logger.error(f'❌ SVG generation failed: {e}')
    
    return None


def generate_infographic_image(content: str, topic: str, language: str = 'english') -> Optional[str]:
    """
    Generate an infographic using Gemini 3 Pro Image with Google Search grounding.
    
    This is the primary image generation method, using state-of-the-art Gemini 3 Pro Image
    with Google Search for real-time agricultural data and 4K resolution for clarity.
    
    Args:
        content: Content to visualize (context for the infographic)
        topic: Main topic for the infographic (used in prompt)
        language: Language for text labels in the infographic (default: 'english')
    
    Returns:
        Relative file path to saved PNG ('generated_infographics/infographic_YYYYMMDD_HHMMSS.png')
        or None if generation fails
    """
    if CLIENT is None:
        logger.error("❌ AI client not available for image generation")
        return None
    
    # Get the full language name for the prompt
    lang_name = LANGUAGE_NAMES.get(language.lower(), 'English')
    
    # Enhanced prompt with language-specific instructions
    prompt = (
        f"Generate a professional agricultural infographic on: {topic}\n"
        f"CRITICAL LANGUAGE REQUIREMENT: All text labels, headings, titles, and content in the infographic MUST be in {lang_name}.\n"
        f"Design requirements:\n"
        f"- Clean, modern flat design with agricultural theme\n"
        f"- Green and yellow color scheme (farmer-friendly)\n"
        f"- High clarity for mobile viewing\n"
        f"- Include icons, text labels in {lang_name}, and key statistics\n"
        f"- Aspect ratio: 16:9\n"
        f"- Include practical tips relevant to sugarcane farmers in India\n"
        f"- Use simple, easy-to-understand vocabulary suitable for farmers\n"
        f"- All numbers and measurements should use local conventions\n"
        f"Context information:\n{content[:500] if content else 'General agricultural topic'}"
    )
    
    try:
        # Create output directory for generated infographics
        output_dir = os.path.join(UPLOAD_FOLDER, 'generated_infographics')
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"📸 Calling Gemini 3 Pro Image to generate infographic for: {topic}")
        logger.info(f"   ✓ Model: gemini-3-pro-image-preview")
        logger.info(f"   ✓ Language: {lang_name}")
        logger.info(f"   ✓ Resolution: 4K")
        logger.info(f"   ✓ Aspect Ratio: 16:9")
        logger.info(f"   ✓ Tools: Google Search grounding")
        
        # Call Gemini 3 Pro Image with Google Search grounding
        response = CLIENT.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=prompt,
            config=types.GenerateContentConfig(
                # Use Google Search for real-time agricultural data
                tools=[{"google_search": {}}],
                # Configure image output quality and size
                image_config=types.ImageConfig(
                    aspect_ratio="16:9",
                    image_size="4K"  # High resolution for clarity
                )
            )
        )
        
        # Extract image parts from response
        image_parts = [part for part in response.parts if part.inline_data]
        
        if not image_parts:
            logger.error("❌ API call succeeded but returned no images")
            return None
        
        # Save the first generated image
        image_data = image_parts[0].inline_data
        image_bytes = image_data.data
        
        # Create unique filename with timestamp and language
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"infographic_{language}_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        
        # Save image using Pillow
        with Image.open(BytesIO(image_bytes)) as img:
            img.save(filepath, "PNG")
        
        logger.info(f"✅ Infographic saved to: {filepath}")
        logger.info(f"🎨 Generated using Gemini 3 Pro Image (4K resolution, {lang_name})")
        
        # Return relative path for URL
        return f"generated_infographics/{filename}"
        
    except Exception as e:
        logger.error(f'❌ Image generation failed: {type(e).__name__}: {e}')
        import traceback
        logger.error(traceback.format_exc())
    
    return None
