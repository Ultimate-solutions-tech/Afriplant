require("dotenv").config();
const express = require("express");
const bodyParser = require("body-parser");
const cors = require("cors");
const path = require("path");
const { GoogleGenerativeAI } = require("@google/generative-ai");
const vision = require("@google-cloud/vision");

const app = express();
const port = process.env.PORT || 5000;

// Middleware
app.use(bodyParser.json({ limit: "10mb" })); // Allow larger payloads for image processing
app.use(cors());
app.use(express.static(path.join(__dirname, "public")));

// Google APIs Setup
const apiKey = process.env.GEMINI_API_KEY;
const genAI = new GoogleGenerativeAI(apiKey);
const visionClient = new vision.ImageAnnotatorClient({
    keyFilename: process.env.GOOGLE_VISION_API_KEY,
});

// Generative AI Configuration
const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
const generationConfig = {
    temperature: 1,
    topP: 0.95,
    topK: 40,
    maxOutputTokens: 8192,
    responseMimeType: "text/plain",
};

// Helper Functions
async function analyzeImageWithVisionAPI(base64Image) {
    try {
        const image = { content: base64Image };
        const [result] = await visionClient.labelDetection({ image });
        const labels = result.labelAnnotations;
        const description = labels
            .filter((label) => label.description)
            .map((label) => label.description)
            .slice(0, 3)
            .join(", ");
        return `Identified plant characteristics: ${description}`;
    } catch (error) {
        console.error("Error in Vision API:", error);
        return "Unable to identify plant from the image.";
    }
}

function formatResponse(content) {
    return `
        <div class="chat-response">
            <h3 style="color: #4CAF50; margin-bottom: 8px;">Guide</h3>
            <div class="step">
                ${content.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>").replace(/\*(.*?)\*/g, "<em>$1</em>")}
            </div>
        </div>
    `;
}

// Routes
app.post("/chat", async (req, res) => {
    const { message } = req.body;

    try {
        const chatSession = model.startChat({
            generationConfig,
            history: [
                {
                    role: "user",
                    parts: [
                        {
                            text: `You are a highly specialized AI botanist. Answer only questions related to plants. Make the response clear, user-friendly, and well-structured. Highlight subheadings in bold.`,
                        },
                    ],
                },
                {
                    role: "user",
                    parts: [{ text: message }],
                },
            ],
        });

        const result = await chatSession.sendMessage(message);
        const formattedResponse = formatResponse(result.response.text());
        res.json({ response: formattedResponse });
    } catch (error) {
        console.error("Error in /chat endpoint:", error);
        res.status(500).json({ response: "Something went wrong. Please try again." });
    }
});

app.post("/chat-image", async (req, res) => {
    const { image } = req.body;

    if (!image || !image.startsWith("data:image/")) {
        return res.status(400).json({ response: "Invalid image format. Please upload a valid image file." });
    }

    try {
        const base64Image = image.split(",")[1];
        const plantDescription = await analyzeImageWithVisionAPI(base64Image);

        const chatSession = model.startChat({
            generationConfig,
            history: [
                {
                    role: "user",
                    parts: [
                        {
                            text: `Provide care and maintenance tips for this plant based on the following description: ${plantDescription}`,
                        },
                    ],
                },
            ],
        });

        const result = await chatSession.sendMessage();
        const formattedResponse = formatResponse(result.response.text());
        res.json({ response: formattedResponse });
    } catch (error) {
        console.error("Error in /chat-image endpoint:", error);
        res.status(500).json({ response: "Failed to analyze the image. Please try again." });
    }
});

// Serve the main HTML file
app.get("/", (req, res) => {
    res.sendFile(path.join(__dirname, "public", "index.html"));
});

// Start the server
app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});
