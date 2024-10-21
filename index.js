const express = require('express');
const axios = require('axios');
const bodyParser = require('body-parser');

const app = express();
app.use(bodyParser.json());

// Endpoint to take crop prediction inputs
app.post('/predict-crop', async (req, res) => {
    const { rainfall, season, state, area } = req.body;

    try {
        // Make a request to the Python Flask API
        const response = await axios.post('http://localhost:5000/predict', {
            rainfall: rainfall,
            season: season,
            state: state,
            area: area
        });

        // Send the crop prediction result to the client
        res.json({ predictedCrop: response.data.predicted_crop });
    } catch (error) {
        res.status(500).send('Error predicting crop');
    }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});