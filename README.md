# 🌾 Farmer-Focused Agricultural Advisory System

A specialized AI-powered chatbot designed for Indian farmers, particularly those working with sugarcane cultivation. Provides expert agricultural advice in 6 Indian languages with voice input/output and crop disease identification capabilities.

## ✨ Features

### 🗣️ Multi-Language Support
- **6 Indian Languages**: English, Hindi, Marathi, Tamil, Telugu, Kannada
- Language-specific agricultural system instructions
- Native language responses for better farmer accessibility

### 🎤 Voice Input & Output
- **Voice Questions**: Speak your questions using microphone
- **Text-to-Speech**: Listen to answers in your language
- Perfect for low-literacy farmers
- Works on modern mobile browsers

### 📸 Crop Disease Identification
- Upload crop images (JPG, JPEG, PNG)
- AI-powered disease and pest detection
- Treatment recommendations from knowledge base
- Visual analysis with actionable advice

### 📄 Document-Based Knowledge
- Upload agricultural documents (PDF, TXT, DOC, DOCX)
- AI searches through documents to answer questions
- Source citations for transparency
- Specialized for sugarcane cultivation

### 📱 Farmer-Friendly UI
- Large fonts (18px minimum) for easy reading
- High contrast colors for outdoor visibility
- Large touch targets (56px) for easier tapping
- Mobile-first responsive design
- Bilingual labels (English + Hindi)

## 🚀 Live Demo

**Deployed on Render.com:** [https://ai-chatbot-sugarcane.onrender.com](https://ai-chatbot-sugarcane.onrender.com)

> Note: First load may take 30-60 seconds due to free tier cold start.

## 📋 Prerequisites

- Python 3.11 or higher
- Google Gemini API key ([Get it here](https://ai.google.dev/))
- Git (for deployment)

## 🛠️ Local Installation

### 1. Clone the repository

```bash
git clone https://github.com/shashanktamaskar/AI_Chatbot_Sugarcane.git
cd AI_Chatbot_Sugarcane
```

### 2. Create a virtual environment

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the root directory:

```bash
# .env
GOOGLE_API_KEY=your_actual_api_key_here
FLASK_ENV=development
FLASK_DEBUG=False
```

**Important:** Never commit the `.env` file to version control.

### 5. Run the application

```bash
# Development mode
python app.py

# Production mode (local)
gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 120 app:app
```

The application will be available at `http://localhost:5000`

## ☁️ Deployment to Render.com

### Step 1: Prepare Your Repository

All deployment files are already configured:
- ✅ `Procfile` - Tells Render how to run the app
- ✅ `runtime.txt` - Specifies Python version (3.11.0)
- ✅ `render.yaml` - Infrastructure as code configuration
- ✅ `requirements.txt` - Python dependencies

### Step 2: Create Render Account

1. Go to [https://render.com](https://render.com)
2. Sign up with GitHub (recommended)
3. Verify your email
4. Authorize Render to access your GitHub repositories

### Step 3: Deploy the Application

#### Option A: Using render.yaml (Recommended)

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click "New +" → "Blueprint"
3. Select this repository
4. Render will detect `render.yaml` and configure automatically
5. Add environment variable:
   - Key: `GOOGLE_API_KEY`
   - Value: Your actual Google Gemini API key
6. Click "Apply"
7. Wait 5-10 minutes for deployment

#### Option B: Manual Web Service Creation

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Name**: `ai-chatbot-sugarcane`
   - **Region**: Oregon (or closest to you)
   - **Branch**: `main`
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 app:app`
   - **Instance Type**: Free
5. Add Environment Variables:
   - `GOOGLE_API_KEY`: Your API key
   - `FLASK_ENV`: production
   - `FLASK_DEBUG`: False
   - `PYTHON_VERSION`: 3.11.0
6. Click "Create Web Service"
7. Wait for deployment (5-10 minutes)

### Step 4: Verify Deployment

Once deployment is complete:
1. Check the deployment logs for errors
2. Open the provided URL (e.g., `https://ai-chatbot-sugarcane.onrender.com`)
3. Test all features:
   - ✅ Language selector works
   - ✅ Document upload functions
   - ✅ Ask questions in different languages
   - ✅ Image upload for disease identification
   - ✅ Voice input (on supported browsers)
   - ✅ Text-to-speech output

### Step 5: Monitoring (Optional)

Set up uptime monitoring:
1. Create account at [UptimeRobot](https://uptimerobot.com) (free)
2. Add your Render URL as a monitor
3. Configure email alerts for downtime

## 📚 Knowledge Base Setup

The application uses a structured knowledge base for agricultural information.

### Folder Structure

```
knowledge_base/
├── sugarcane/              # Sugarcane cultivation guides
├── pest_control/           # Pest management information
├── diseases/               # Disease identification & treatment
├── market_info/            # Market prices & trends
└── government_schemes/     # Government programs & subsidies
```

### Adding Agricultural Content

1. Prepare documents (PDF, TXT, DOC, DOCX)
2. Place in appropriate `knowledge_base/` subfolder
3. Use clear, descriptive filenames
4. Include language suffix (e.g., `_hindi.pdf`)
5. Upload via web interface or copy to server

For detailed instructions, see [AGRICULTURAL_SETUP.md](AGRICULTURAL_SETUP.md)

## 📱 Usage Guide

### For Farmers

#### Select Language
1. Open the application
2. Click the language dropdown
3. Choose your preferred language (Hindi, Marathi, Tamil, etc.)

#### Upload Documents
1. Click "Choose Files" under "Upload Documents"
2. Select agricultural PDFs or documents
3. Click "Upload Files"
4. Wait for success message

#### Identify Crop Diseases
1. Take a clear photo of affected crop part
2. Click "Choose File" under "Crop Disease Identification"
3. Select the photo
4. Preview appears
5. Click "Analyze Image"
6. Review disease identification and treatment advice

#### Ask Questions

**Text Input:**
- Type question in the text box
- Press Enter or click Send (📤)

**Voice Input:**
- Click microphone button (🎤)
- Speak your question clearly
- Text appears automatically
- Click Send

**Listen to Answers:**
- After receiving answer
- Click "🔊 Listen to Answer"
- Audio plays in selected language
- Click "Stop" to pause

### Example Questions

- "गन्ने की बुवाई का सबसे अच्छा समय क्या है?" (Best time to plant sugarcane?)
- "ऊस लागवडीसाठी किती पाणी लागते?" (Water requirements for sugarcane?)
- "What subsidies are available for sugarcane farmers?"
- "கரும்பு நோய் கட்டுப்பாடு எப்படி?" (How to control sugarcane diseases?)

## 🔧 API Endpoints

### POST /upload
Upload agricultural documents

**Request:**
```
Content-Type: multipart/form-data
files: [file1, file2, ...]
```

**Response:**
```json
{
  "message": "Successfully uploaded 2 file(s)",
  "uploaded": ["sugarcane_guide.pdf", "pest_control.pdf"]
}
```

### POST /ask
Ask agricultural questions with language support

**Request:**
```json
{
  "question": "गन्ने में सफेद सुंडी का इलाज?",
  "language": "hindi"
}
```

**Response:**
```json
{
  "answer": "सफेद सुंडी के नियंत्रण के लिए...",
  "sources": ["pest_control_hindi.pdf"]
}
```

### POST /analyze_crop_image
Analyze crop images for disease identification

**Request:**
```
Content-Type: multipart/form-data
image: [crop_photo.jpg]
language: hindi
```

**Response:**
```json
{
  "analysis": "यह गन्ने में लाल सड़न रोग के लक्षण हैं..."
}
```

## 🔒 Security Features

- API keys stored in environment variables
- XSS protection through input sanitization
- File type validation (documents and images)
- File size limits (50MB documents, 10MB images)
- Input length validation
- CORS protection
- Secure error messages
- No sensitive data exposure

## 📊 File Specifications

**Documents:**
- Formats: PDF, TXT, DOC, DOCX
- Max size: 50MB per file
- Recommended: Searchable PDFs

**Images (Disease Identification):**
- Formats: JPG, JPEG, PNG
- Max size: 10MB
- Recommended: Clear, well-lit photos

**Questions:**
- Max length: 5000 characters
- All languages supported

## 🌐 Browser Compatibility

**Best Experience:**
- ✅ Chrome/Edge (Desktop & Mobile)
- ✅ Safari (iOS/macOS)
- ✅ Firefox (Desktop)

**Limited Features:**
- ⚠️ Older browsers may not support voice input
- ⚠️ Text-to-speech quality varies by browser

## 🐛 Troubleshooting

### Deployment Issues

**"Application Error" on Render**
- Check deployment logs in Render dashboard
- Verify `GOOGLE_API_KEY` is set correctly
- Ensure all dependencies are in `requirements.txt`

**Cold Start Delay (30-60s)**
- Normal for Render free tier
- Upgrade to paid plan for instant wake-up
- Use UptimeRobot to ping every 5 minutes (keeps warm)

### Application Issues

**Voice input not working**
- Use Chrome/Safari browser
- Grant microphone permissions
- Check internet connection

**Document upload fails**
- Verify file size (<50MB)
- Check supported format
- Review server logs

**Poor image analysis**
- Upload clear, well-lit images
- Focus on affected areas
- Try different angles

**Answer not relevant**
- Upload more specific documents
- Rephrase question
- Try different language

## 🗂️ Project Structure

```
.
├── app.py                      # Flask application with all routes
├── templates/
│   └── index.html              # Farmer-friendly web interface
├── uploads/                    # User-uploaded files (gitignored)
│   └── .gitkeep
├── knowledge_base/             # Agricultural knowledge base
│   ├── sugarcane/
│   ├── pest_control/
│   ├── diseases/
│   ├── market_info/
│   └── government_schemes/
├── .env                        # Environment variables (gitignored)
├── .env.example                # Environment template
├── requirements.txt            # Python dependencies
├── Procfile                    # Render deployment configuration
├── runtime.txt                 # Python version for Render
├── render.yaml                 # Infrastructure as Code
├── .gitignore                  # Git ignore rules
├── README.md                   # This file
├── AGRICULTURAL_SETUP.md       # Knowledge base setup guide
└── SETUP_INSTRUCTIONS.md       # Original setup guide
```

## 🚧 Development

### Running Tests
```bash
# Run with debug mode
FLASK_ENV=development FLASK_DEBUG=True python app.py

# Test with gunicorn locally
gunicorn --bind 0.0.0.0:5000 app:app
```

### Adding New Languages

1. Edit `app.py` - Add language to `AGRICULTURAL_INSTRUCTIONS` dict
2. Edit `templates/index.html` - Add option to language selector
3. Add language code to `languageCodes` in JavaScript
4. Test with native speakers

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Make your changes
4. Test thoroughly (all languages, all features)
5. Commit (`git commit -m "Add new feature"`)
6. Push (`git push origin feature/new-feature`)
7. Create Pull Request

## 📝 Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `GOOGLE_API_KEY` | Google Gemini API key | ✅ Yes | None |
| `FLASK_ENV` | Environment (development/production) | No | development |
| `FLASK_DEBUG` | Enable debug mode | No | False |
| `PORT` | Server port (auto-set by Render) | No | 5000 |
| `PYTHON_VERSION` | Python version for deployment | No | 3.11.0 |

## 💰 Cost Breakdown

| Service | Plan | Cost | Notes |
|---------|------|------|-------|
| Render.com | Free Tier | $0/month | 750 hrs/month, cold starts |
| Render.com | Starter | $7/month | No cold starts, better performance |
| Google Gemini API | Free Tier | $0 | Limited requests |
| Google Gemini API | Pay-as-you-go | ~$5-10/month | For production use |
| **Total (Free)** | | **$0/month** | Perfect for testing/MVP |
| **Total (Production)** | | **$12-17/month** | Recommended for real users |

## 🎯 Success Metrics

- ✅ Multi-language support (6 languages)
- ✅ Voice input/output functionality
- ✅ Crop disease image analysis
- ✅ Mobile-responsive design
- ✅ Deployed and accessible 24/7
- ✅ Fast response times (<10 seconds)
- ✅ Farmer-friendly UI (large fonts, clear buttons)

## 🔮 Future Enhancements

- [ ] Weather-based farming alerts
- [ ] Offline mode (PWA) for poor connectivity
- [ ] SMS-based Q&A for feature phones
- [ ] Market price API integration
- [ ] Community forum for farmers
- [ ] Video tutorials in regional languages
- [ ] Fertilizer dosage calculator
- [ ] Crop yield predictor
- [ ] Integration with government portals

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Credits

- **Target Users:** Indian sugarcane farmers
- **Powered by:** Google Gemini AI
- **Languages:** Hindi, Marathi, Tamil, Telugu, Kannada, English
- **Focus:** Practical agricultural advice
- **Developer:** Shashank Tamaskar

## 📞 Support

For issues and questions:
1. Check [AGRICULTURAL_SETUP.md](AGRICULTURAL_SETUP.md) for knowledge base help
2. Review troubleshooting section above
3. Check deployment logs on Render
4. Report issues on GitHub

---

**⚠️ Important Security Notes:**
- Never commit `.env` file to version control
- Keep your Google API key secure
- Use environment variables for all sensitive data
- Enable HTTPS in production (Render provides this automatically)

---

**Version:** 2.0 - Farmer-Focused Edition
**Last Updated:** 2025
**Status:** Production Ready ✅

---

Made with ❤️ for Indian Farmers | किसानों के लिए | शेतकऱ्यांसाठी
