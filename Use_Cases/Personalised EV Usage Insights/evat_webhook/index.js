// index.js
const express = require('express');
const { MongoClient } = require('mongodb');
const cors = require('cors');
const axios = require('axios');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

const uri = process.env.MONGO_URI;
const client = new MongoClient(uri);

// Connect to MongoDB
async function connectClient() {
  try {
    await client.connect();
    console.log('Connected to MongoDB');
  } catch (err) {
    console.error('MongoDB connection error:', err);
  }
}
connectClient();

// POST /api/save webhook
app.post('/api/save', async (req, res) => {
  console.log('Webhook triggered');
  console.log('Request body:', req.body);

  try {
    const db = client.db('EVAT');
    const collection = db.collection('user_responses');

    if (!req.body || Object.keys(req.body).length === 0) {
      return res.status(400).json({ message: 'No data provided' });
    }

    // Call prediction service
    let cluster = null;
    try {
      const predictUrl = process.env.PREDICT_URL;
      if (predictUrl) {
        console.log(`Calling prediction service at: ${predictUrl}`);
        const resp = await axios.post(predictUrl, req.body, { timeout: 20000 }); // 20s timeout
        if (resp.data && resp.data.cluster !== undefined) {
          cluster = parseInt(resp.data.cluster, 10);
          console.log(`Cluster assigned: ${cluster}`);
        }
      } else {
        console.warn('PREDICT_URL is not defined in .env');
      }
    } catch (e) {
      console.error('Prediction service error:', e.message);
      if (e.response) console.error('Response data:', e.response.data);
      if (e.request) console.error('Request made but no response:', e.request);
      console.warn('Proceeding without cluster assignment.');
    }

    // Save to MongoDB
    const docToInsert = { ...req.body };
    if (cluster !== null && !Number.isNaN(cluster)) {
      docToInsert.cluster = cluster;
    }

    const result = await collection.insertOne(docToInsert);
    console.log('Saved to MongoDB with ID:', result.insertedId);

    res.status(200).json({
      message: 'Data saved successfully',
      id: result.insertedId,
      cluster: cluster
    });

  } catch (err) {
    console.error('Error saving data:', err);
    res.status(500).json({ message: 'Internal Server Error' });
  }
});

// Health check
app.get('/', (req, res) => {
  res.send('EV Usage Insights API is live!');
});

// Warm-up ping for prediction service
const keepAlivePredictService = async () => {
  const url = process.env.PREDICT_URL;
  if (!url) return;

  try {
    console.log('Warming up prediction service...');
    await axios.post(url, {
      weekly_km: 0,
      fuel_efficiency: 0,
      monthly_fuel_spend: 0,
      trip_length: "",
      driving_frequency: "",
      driving_type: "",
      road_trips: "",
      car_ownership: "",
      home_charging: "",
      solar_panels: "",
      charging_preference: "",
      budget: "",
      priorities: "",
      postcode: ""
    }, { timeout: 10000 });
    console.log('Prediction service warmed up!');
  } catch (err) {
    console.warn('Warm-up ping failed (normal if service was cold):', err.message);
  }
};

// Call warm-up on server start
keepAlivePredictService();

// Optional: keep alive every 15 minutes
setInterval(keepAlivePredictService, 15 * 60 * 1000);

// Start server
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
