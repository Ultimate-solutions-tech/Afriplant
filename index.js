require('dotenv').config();
const express = require('express');
const path = require('path');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const vision = require('@google-cloud/vision');

const app = express();
const port = process.env.PORT || 3000;
const apiKey = process.env.GEMINI_API_KEY;
const genAI = new GoogleGenerativeAI(apiKey);

// Initialize Google Vision client
const visionClient = new vision.ImageAnnotatorClient({
  keyFilename: process.env.GOOGLE_VISION_API_KEY
});

const model = genAI.getGenerativeModel({
  model: 'gemini-1.5-flash',
});

const generationConfig = {
  temperature: 1,
  topP: 0.95,
  topK: 40,
  maxOutputTokens: 8192,
  responseMimeType: 'text/plain',
};

// Serve static files
app.use(express.static(path.join(__dirname, 'public')));
app.use(express.json());

// Function to analyze image using Google Vision API
async function analyzeImageWithVisionAPI(base64Image) {
    try {
        const image = { content: base64Image };
        const [result] = await visionClient.labelDetection({ image });
        const labels = result.labelAnnotations;

        const description = labels
            .filter(label => label.description)
            .map(label => label.description)
            .slice(0, 3)
            .join(', ');

        return `Identified plant characteristics: ${description}`;
    } catch (error) {
        console.error("Error in Vision API:", error);
        return "Unable to identify plant from the image.";
    }
}

// Function to format chatbot response
function formatResponse(content) {
    return `
        <div class="chat-response">
            <h3 style="color: #4CAF50; margin-bottom: 8px;">Guide</h3>
            <div class="step">
                ${content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>').replace(/\*(.*?)\*/g, '<em>$1</em>')}
            </div>
        </div>
    `;
}

// Route to handle chat
app.post('/chat', async (req, res) => {
    const userMessage = req.body.message;

    try {
        const chatSession = model.startChat({
            generationConfig,
            history: [
                {
                    role: 'user',
                    parts: [
                        {
                            text: `"You are a highly specialized AI botanist, trained to provide expert guidance on plant identification, care, and diagnosis." Answer only questions related to plant and nothing more.
                            Make the response be a block response, user friendly and easy readability. Highlight and bold every subheading. Remove every asterics symbol`,
                        },
                    ],
                },
                {
                    role: 'user',
                    parts: [{ text: userMessage }],
                },
            ],
        });

        const result = await chatSession.sendMessage(userMessage);

        // Format the response using the formatResponse function
        const formattedResponse = formatResponse(result.response.text());

        res.json({ response: formattedResponse });
    } catch (error) {
        console.error('Error:', error);
        res.status(500).json({ error: 'Something went wrong. Please try again.' });
    }
});

// Route to handle image upload and analysis
app.post('/chat-image', async (req, res) => {
    const base64Image = req.body.image;
    const mimeType = 'data:image/jpeg;base64,';

    if (!base64Image.startsWith(mimeType)) {
        res.status(400).json({ error: 'Invalid image format. Please upload a valid image.' });
        return;
    }

    try {
        const plantDescription = await analyzeImageWithVisionAPI(base64Image);

        const chatSession = model.startChat({
            generationConfig,
            history: [
                {
                    role: 'user',
                    parts: [
                        {
                            text: `Provide care and maintenance tips for this plant based on the following description: ${plantDescription}`,
                        },
                    ],
                },
            ],
        });

        const result = await chatSession.sendMessage();

        // Format the response using the formatResponse function
        const formattedResponse = formatResponse(result.response.text());

        res.json({ response: formattedResponse });
    } catch (error) {
        console.error('Error:', error);
        res.status(500).json({ error: 'Something went wrong. Please try again.' });
    }
});

// Serve HTML file
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});
