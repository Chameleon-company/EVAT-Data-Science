const fetch = require('node-fetch');

fetch('http://localhost:3000/api/save', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    weekly_km: 200,
    trip_length: 'medium',
    driving_frequency: 'weekly'
  })
})
  .then(res => res.json())
  .then(data => console.log(data))
  .catch(err => console.error(err));