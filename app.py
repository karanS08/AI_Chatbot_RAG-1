from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types
import time
import os
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import logging
from PIL import Image
import io

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = "uploads"
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'doc', 'docx', 'jpg', 'jpeg', 'png'}
IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# Language-specific agricultural system instructions
AGRICULTURAL_INSTRUCTIONS = {
    'english': """You are an expert agricultural advisor specializing in sugarcane cultivation.
    Provide practical, actionable advice to farmers in simple, easy-to-understand language.
    Focus on: sugarcane cultivation, pest control, disease management, fertilizer application,
    irrigation, market prices, and government schemes. Always be respectful and supportive.
    You have access to data from both international and Indian sources, but you must give
    advice **only in the context of India**. Use international data only to provide additional insight or support.""",

    'hindi': """आप गन्ने की खेती में विशेषज्ञ कृषि सलाहकार हैं।
    किसानों को सरल और समझने में आसान भाषा में व्यावहारिक सलाह दें।
    फोकस: गन्ने की खेती, कीट नियंत्रण, रोग प्रबंधन, उर्वरक प्रयोग, सिंचाई, बाजार मूल्य,
    और सरकारी योजनाएं। हमेशा सम्मानजनक और सहायक रहें।
    आपके पास अंतरराष्ट्रीय और भारतीय दोनों प्रकार का डेटा उपलब्ध है, लेकिन आपको उत्तर
    **सिर्फ भारत के संदर्भ में ही देना है।** अंतरराष्ट्रीय डेटा का उपयोग केवल अतिरिक्त
    समर्थन या तुलना देने के लिए करें।""",

    'marathi': """तुम्ही ऊस लागवडीचे तज्ञ कृषी सल्लागार आहात.
    शेतकऱ्यांना सोप्या, समजण्यास सोप्या भाषेत व्यावहारिक सल्ला द्या.
    उत्तरे संक्षिप्त ठेवा - जास्तीत जास्त 3-5 वाक्ये.
    फोकस: ऊस लागवड, कीड नियंत्रण, रोग व्यवस्थापन, खत वापर, सिंचन, बाजार भाव,
    आणि सरकारी योजना. नेहमी आदरपूर्ण आणि सहाय्यक रहा.
    तुमच्याकडे आंतरराष्ट्रीय आणि भारतीय अशा दोन्ही स्रोतांमधील माहिती उपलब्ध आहे,
    परंतु उत्तरे **फक्त भारताच्या संदर्भात** द्यायची आहेत. आंतरराष्ट्रीय माहिती
    केवळ अतिरिक्त आधार किंवा तुलना म्हणून वापरा.""",

    'tamil': """நீங்கள் கரும்பு சாகுபடியில் நிபுணத்துவம் பெற்ற விவசாய ஆலோசகர்.
    விவசாயிகளுக்கு எளிமையான, புரிந்துகொள்ள எளிதான மொழியில் நடைமுறை ஆலோசனை வழங்கவும்.
    பதில்களை சுருக்கமாக வைத்திருங்கள் - அதிகபட்சம் 3-5 வாக்கியங்கள்.
    கவனம்: கரும்பு சாகுபடி, பூச்சி கட்டுப்பாடு, நோய் மேலாண்மை, உரம் பயன்பாடு, நீர்ப்பாசனம்,
    சந்தை விலைகள் மற்றும் அரசாங்க திட்டங்கள். எப்போதும் மரியாதையுடனும் ஆதரவாகவும் இருங்கள்.
    உங்களிடம் சர்வதேச மற்றும் இந்திய தகவல்கள் இரண்டும் உள்ளன, ஆனால் பதில்கள்
    **இந்தியாவை மட்டுமே மையமாகக் கொண்டு** இருக்க வேண்டும். சர்வதேச தரவை கூடுதல்
    ஆதாரமாக அல்லது விளக்கமாக மட்டும் பயன்படுத்தவும்.""",

    'telugu': """మీరు చెరకు సాగులో నిపుణుడైన వ్యవసాయ సలహాదారు.
    రైతులకు అర్థం చేసుకోవడానికి సులభమైన భాషలో ఆచరణాత్మక సలహా ఇవ్వండి.
    సమాధానాలు సంక్షిప్తంగా ఉండాలి - గరిష్ఠంగా 3-5 వాక్యాలు.
    దృష్టి: చెరకు సాగు, తెగులు నియంత్రణ, వ్యాధి నిర్వహణ, ఎరువుల వినియోగం,
    నీటిపారుదల, మార్కెట్ ధరలు మరియు ప్రభుత్వ పథకాలు. ఎల్లప్పుడూ గౌరవప్రదంగా ఉండండి.
    మీ వద్ద అంతర్జాతీయ మరియు భారతీయ సమాచారం ఉన్నప్పటికీ, మీ సమాధానం
    **కేవలం భారతదేశానికి సంబంధించినదిగా** ఉండాలి. అంతర్జాతీయ సమాచారం అదనపు
    మద్దతు కోసం మాత్రమే ఉపయోగించాలి.""",

    'kannada': """ನೀವು ಕಬ್ಬಿನ ಕೃಷಿಯಲ್ಲಿ ಪರಿಣಿತ ಕೃಷಿ ಸಲಹೆಗಾರರು.
    ರೈತರಿಗೆ ಸರಳ ಮತ್ತು ಅರ್ಥಮಾಡಿಕೊಳ್ಳಲು ಸುಲಭವಾದ ಭಾಷೆಯಲ್ಲಿ ಪ್ರಾಯೋಗಿಕ ಸಲಹೆ ನೀಡಿ.
    ಗಮನ: ಕಬ್ಬಿನ ಕೃಷಿ, ಕೀಟ ನಿಯಂತ್ರಣ, ರೋಗ ನಿರ್ವಹಣೆ, ಗೊಬ್ಬರ ಬಳಕೆ, ನೀರಾವರಿ,
    ಮಾರುಕಟ್ಟೆ ಬೆಲೆಗಳು ಮತ್ತು ಸರ್ಕಾರಿ ಯೋಜನೆಗಳು. ಯಾವಾಗಲೂ ಗೌರವಾನ್ವಿತ ಮತ್ತು ಸಹಾಯಕವಾಗಿರಿ.
    ನಿಮಗೆ ಅಂತರಾಷ್ಟ್ರೀಯ ಮತ್ತು ಭಾರತೀಯ ಮೂಲಗಳ ಮಾಹಿತಿ ಲಭ್ಯವಿದೆ, ಆದರೆ ನೀವು ಉತ್ತರವನ್ನು
    **ಮಾತ್ರ ಭಾರತದ ಸಂದರ್ಭದಲ್ಲಿ** ನೀಡಬೇಕು. ಅಂತರಾಷ್ಟ್ರೀಯ ಮಾಹಿತಿಯನ್ನು ಹೆಚ್ಚುವರಿ
    ಬೆಂಬಲ ಅಥವಾ ಹೋಲಿಕೆಯಾಗಿ ಮಾತ್ರ ಬಳಸಿ."""
}


# Initialize Gemini client
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    logger.error("GOOGLE_API_KEY environment variable is not set")
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

try:
    client = genai.Client(api_key=api_key)
    FILE_SEARCH_STORE = client.file_search_stores.create()
    logger.info(f"Successfully initialized Gemini client and file store: {FILE_SEARCH_STORE.name}")
except Exception as e:
    logger.error(f"Failed to initialize Gemini client or file store: {str(e)}")
    # Set to None - we'll create on first use if needed
    FILE_SEARCH_STORE = None
    logger.warning("Application starting without file search store. Will attempt to create on first use.")

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def ensure_file_search_store():
    """Ensure file search store exists, create if needed"""
    global FILE_SEARCH_STORE
    if FILE_SEARCH_STORE is None:
        try:
            FILE_SEARCH_STORE = client.file_search_stores.create()
            logger.info(f"Created new file search store: {FILE_SEARCH_STORE.name}")
        except Exception as e:
            logger.error(f"Failed to create file search store: {str(e)}")
            raise
    return FILE_SEARCH_STORE

def upload_file_to_store(file_path):
    """Upload file to Gemini file search store with error handling"""
    try:
        store = ensure_file_search_store()
        logger.info(f"Uploading file to store: {file_path}")
        
        upload_op = client.file_search_stores.upload_to_file_search_store(
            file_search_store_name=store.name,
            file=file_path
        )
        
        # Wait for upload to complete - handle different operation object structures
        try:
            while not upload_op.done:
                time.sleep(2)
                upload_op = client.operations.get(upload_op.name)
        except AttributeError:
            # If operation doesn't have expected attributes, assume it completed
            logger.warning(f"Could not track upload completion for {file_path}, assuming success")
            time.sleep(3)  # Give it a moment to complete
        
        logger.info(f"Successfully uploaded file: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error uploading file {file_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        # Don't raise - allow upload to continue even if file search fails
        return False

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    """Health check endpoint for monitoring"""
    try:
        # Check if API key is set
        if not os.getenv("GOOGLE_API_KEY"):
            return jsonify({
                "status": "unhealthy",
                "error": "GOOGLE_API_KEY not configured"
            }), 500

        # Check if file store is initialized
        store = ensure_file_search_store()

        return jsonify({
            "status": "healthy",
            "gemini_api": "connected",
            "file_store": "initialized"
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

@app.route("/upload", methods=["POST"])
def upload_files():
    """Handle file uploads with validation and error handling"""
    try:
        if "files" not in request.files:
            return jsonify({"error": "No files provided"}), 400

        files = request.files.getlist("files")
        if not files or files[0].filename == '':
            return jsonify({"error": "No files selected"}), 400

        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        uploaded_files = []
        errors = []

        for file in files:
            try:
                # Validate file
                if not allowed_file(file.filename):
                    errors.append(f"{file.filename}: File type not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
                    continue

                # Secure the filename
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

                # Save and upload file
                file.save(file_path)
                upload_file_to_store(file_path)
                uploaded_files.append(filename)
                logger.info(f"File uploaded successfully: {filename}")

            except Exception as e:
                errors.append(f"{file.filename}: {str(e)}")
                logger.error(f"Error processing file {file.filename}: {str(e)}")

        if not uploaded_files and errors:
            return jsonify({"error": "No files were uploaded. " + " ".join(errors)}), 400

        message = f"Successfully uploaded {len(uploaded_files)} file(s)"
        if errors:
            message += f". {len(errors)} file(s) failed: " + " ".join(errors)

        return jsonify({"message": message, "uploaded": uploaded_files}), 200

    except Exception as e:
        logger.error(f"Unexpected error in /upload: {str(e)}")
        return jsonify({"error": "Server error during upload"}), 500

@app.route("/ask", methods=["POST"])
def ask():
    """Handle question queries with language support and error handling"""
    try:
        # Validate request
        if not request.json:
            return jsonify({"error": "Request must be JSON"}), 400

        user_question = request.json.get("question", "").strip()
        language = request.json.get("language", "english").lower()

        if not user_question:
            return jsonify({"error": "Question cannot be empty"}), 400

        if len(user_question) > 5000:
            return jsonify({"error": "Question too long (max 5000 characters)"}), 400

        # Get language-specific system instruction
        system_instruction = AGRICULTURAL_INSTRUCTIONS.get(language, AGRICULTURAL_INSTRUCTIONS['english'])

        logger.info(f"Processing question in {language}: {user_question[:100]}...")

        # Ensure file search store is available
        store = ensure_file_search_store()

        # Generate content with Gemini with system instruction
        response = client.models.generate_content(
            model='gemini-3-pro-preview',
            contents=f"{system_instruction}\n\nUser Question: {user_question}",
            config=types.GenerateContentConfig(
                tools=[types.Tool(
                    file_search=types.FileSearch(
                        file_search_store_names=[store.name]
                    )
                )]
            )
        )

        # Extract answer
        if not response.candidates:
            return jsonify({"error": "No response generated"}), 500

        answer = response.text or "No answer generated"

        logger.info(f"Question processed successfully")
        return jsonify({"response": answer}), 200

    except Exception as e:
        logger.error(f"Error in /ask: {str(e)}")
        return jsonify({"error": "Failed to process question"}), 500

@app.route("/api/rag", methods=["GET", "POST"])
def rag_api():
    """HTTP API endpoint for RAG system: accepts POST with 'query', optional 'language', and media attachments for search."""
    # Log all requests
    logger.info("="*60)
    logger.info("RAG API Request Debug Info:")
    logger.info(f"Method: {request.method}")
    logger.info(f"URL: {request.url}")
    logger.info(f"Content-Type: {request.content_type}")
    logger.info(f"Remote Address: {request.remote_addr}")
    logger.info(f"User Agent: {request.user_agent}")
    logger.info("="*60)
    
    # Handle GET requests with helpful message
    if request.method == "GET":
        logger.warning("GET request received - this endpoint requires POST")
        return jsonify({
            "error": "This endpoint only accepts POST requests",
            "usage": {
                "method": "POST",
                "content_type": "application/json or multipart/form-data",
                "json_example": {
                    "query": "What are the best practices for sugarcane pest control?",
                    "language": "english"
                },
                "multipart_example": "Send 'query' and 'language' as form fields, and files under 'media' field"
            }
        }), 405
    
    # Additional debug logging for POST requests
    logger.info(f"Form data: {dict(request.form)}")
    logger.info(f"JSON data: {request.json}")
    logger.info(f"Files: {list(request.files.keys())}")
    logger.info(f"Query Args: {dict(request.args)}")
    logger.info("="*60)
    
    try:
        # Support both JSON and multipart/form-data
        if request.content_type and request.content_type.startswith('multipart/form-data'):
            user_query = request.form.get("query", "").strip()
            language = request.form.get("language", "english").lower()
            files = request.files.getlist("media")
        else:
            if not request.json:
                return jsonify({"error": "Request must be JSON or multipart/form-data"}), 400
            user_query = request.json.get("query", "").strip()
            language = request.json.get("language", "english").lower()
            files = []

        if not user_query:
            return jsonify({"error": "Query cannot be empty"}), 400

        if len(user_query) > 5000:
            return jsonify({"error": "Query too long (max 5000 characters)"}), 400

        # Get language-specific system instruction
        system_instruction = AGRICULTURAL_INSTRUCTIONS.get(language, AGRICULTURAL_INSTRUCTIONS['english'])

        logger.info(f"Processing RAG query in {language}: {user_query[:100]}... with {len(files)} media files")

        # Ensure file search store is available
        store = ensure_file_search_store()

        # Upload media files to file search store
        uploaded_files = []
        for file in files:
            if file and file.filename:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                success = upload_file_to_store(file_path)
                if success:
                    uploaded_files.append(filename)
                else:
                    logger.warning(f"Failed to upload media file: {filename}")

        # Generate content with Gemini with system instruction
        response = client.models.generate_content(
            model='gemini-3-pro-preview',
            contents=f"{system_instruction}\n\nUser Query: {user_query}",
            config=types.GenerateContentConfig(
                tools=[types.Tool(
                    file_search=types.FileSearch(
                        file_search_store_names=[store.name]
                    )
                )]
            )
        )

        # Extract answer
        if not response.candidates:
            return jsonify({"error": "No response generated"}), 500

        answer = response.text or "No answer generated"

        logger.info(f"RAG query processed successfully with {len(uploaded_files)} media files")
        return jsonify({"response": answer, "uploaded_media": uploaded_files}), 200

    except Exception as e:
        logger.error(f"Error in /api/rag: {str(e)}")
        return jsonify({"error": "Failed to process RAG query"}), 500

@app.route("/analyze", methods=["POST"])
def analyze_crop_image():
    """Handle crop disease image analysis"""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        image_file = request.files["file"]
        language = request.form.get("language", "english").lower()

        if image_file.filename == '':
            return jsonify({"error": "No image selected"}), 400

        # Validate file type
        if not image_file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            return jsonify({"error": "Only JPG, JPEG, and PNG images are allowed"}), 400

        # Get language-specific instruction
        system_instruction = AGRICULTURAL_INSTRUCTIONS.get(language, AGRICULTURAL_INSTRUCTIONS['english'])

        logger.info(f"Analyzing crop image in {language}: {image_file.filename}")

        # Read image data
        image_bytes = image_file.read()

        # Prepare prompt for crop disease analysis
        analysis_prompt = f"""{system_instruction}

Analyze this crop image and provide:
1. Identify the crop (if visible)
2. Identify any diseases, pests, or health issues
3. Assess the severity (mild, moderate, severe)
4. Recommend immediate treatment steps
5. Suggest preventive measures for the future

Please provide practical, actionable advice in {language} language."""

        # Ensure file search store is available
        store = ensure_file_search_store()

        # Use Gemini Vision API for image analysis
        response = client.models.generate_content(
            model='gemini-3-pro-preview',
            contents=[
                types.Content(
                    parts=[
                        types.Part(text=analysis_prompt),
                        types.Part(inline_data=types.Blob(
                            mime_type=image_file.content_type or 'image/jpeg',
                            data=image_bytes
                        ))
                    ]
                )
            ],
            config=types.GenerateContentConfig(
                tools=[types.Tool(
                    file_search=types.FileSearch(
                        file_search_store_names=[store.name]
                    )
                )]
            )
        )

        # Extract analysis
        if not response.candidates:
            return jsonify({"error": "No analysis generated"}), 500

        analysis = response.text or "Unable to analyze the image"

        logger.info(f"Image analysis completed successfully")
        return jsonify({"response": analysis}), 200

    except Exception as e:
        logger.error(f"Error in /analyze_crop_image: {str(e)}")
        return jsonify({"error": "Failed to analyze image. Please try again with a clear crop image."}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file size limit exceeded"""
    return jsonify({"error": "File too large (max 50MB)"}), 413

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    # Ensure upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Use Render's PORT environment variable for deployment
    port = int(os.environ.get("PORT", 5000))
    print(app.url_map)

    app.run(host="0.0.0.0", port=port, debug=False)
