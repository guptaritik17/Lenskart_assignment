# API Usage Examples

This document provides examples of how to use the Eyewear Visual Search API.

## Prerequisites

Make sure the API server is running:
```bash
python main.py serve
```

The API will be available at `http://localhost:8000`

## Using cURL

### 1. Health Check

```bash
curl http://localhost:8000/health
```

### 2. Ingest a Product

```bash
curl -X POST "http://localhost:8000/ingest?brand=RayBan&price=150.00&model_name=Aviator&material=Metal&frame_type=Aviator&color=Gold" \
  -F "image=@data/images/sample_aviator.jpg"
```

### 3. Search for Similar Products

```bash
curl -X POST "http://localhost:8000/search?top_k=5" \
  -F "image=@query_image.jpg"
```

### 4. Search with Filters

```bash
curl -X POST "http://localhost:8000/search?top_k=10&price_min=50&price_max=200&brand=RayBan&frame_type=Aviator" \
  -F "image=@query_image.jpg"
```

### 5. Classify Image Attributes

```bash
curl -X POST "http://localhost:8000/classify" \
  -F "image=@query_image.jpg"
```

### 6. Record Feedback

```bash
curl -X POST "http://localhost:8000/feedback" \
  -H "Content-Type: application/json" \
  -d '{
    "product_id": 1,
    "is_relevant": true,
    "similarity_score": 0.92,
    "clicked": true
  }'
```

### 7. Get Product Details

```bash
curl http://localhost:8000/product/1
```

### 8. Get System Statistics

```bash
curl http://localhost:8000/stats
```

## Using Python Requests

```python
import requests
from PIL import Image
import io

BASE_URL = "http://localhost:8000"

# 1. Health Check
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# 2. Ingest a Product
with open("data/images/sample.jpg", "rb") as f:
    files = {"image": f}
    params = {
        "brand": "RayBan",
        "price": 150.00,
        "model_name": "Aviator Classic",
        "material": "Metal",
        "frame_type": "Aviator",
        "color": "Gold"
    }
    response = requests.post(f"{BASE_URL}/ingest", files=files, params=params)
    print(response.json())

# 3. Search for Similar Products
with open("query_image.jpg", "rb") as f:
    files = {"image": f}
    params = {"top_k": 5}
    response = requests.post(f"{BASE_URL}/search", files=files, params=params)
    results = response.json()
    
    print(f"Found {results['total_results']} results in {results['search_time_ms']:.2f}ms")
    for i, product in enumerate(results['results'], 1):
        print(f"{i}. {product['brand']} - {product['model_name']}")
        print(f"   Similarity: {product['similarity_score']:.4f}")
        print(f"   Price: ${product['price']:.2f}")

# 4. Search with Filters
with open("query_image.jpg", "rb") as f:
    files = {"image": f}
    params = {
        "top_k": 10,
        "price_min": 50.0,
        "price_max": 200.0,
        "brand": "RayBan",
        "frame_type": "Aviator"
    }
    response = requests.post(f"{BASE_URL}/search", files=files, params=params)
    results = response.json()
    print(f"Filtered results: {results['total_results']}")

# 5. Classify Attributes
with open("query_image.jpg", "rb") as f:
    files = {"image": f}
    response = requests.post(f"{BASE_URL}/classify", files=files)
    predictions = response.json()
    
    for attr, preds in predictions['predictions'].items():
        print(f"\n{attr}:")
        for label, score in preds:
            print(f"  {label}: {score:.2%}")

# 6. Record Feedback
feedback = {
    "product_id": 1,
    "is_relevant": True,
    "similarity_score": 0.92,
    "clicked": True
}
response = requests.post(f"{BASE_URL}/feedback", json=feedback)
print(response.json())

# 7. Get Product Details
response = requests.get(f"{BASE_URL}/product/1")
product = response.json()
print(f"Product: {product['brand']} - {product['model_name']}")

# 8. Get Statistics
response = requests.get(f"{BASE_URL}/stats")
stats = response.json()
print(f"Total products: {stats['total_products']}")
print(f"Total feedback: {stats['total_feedback_entries']}")
```

## Using JavaScript/Fetch

```javascript
const BASE_URL = 'http://localhost:8000';

// 1. Search for Similar Products
async function searchSimilar(imageFile) {
  const formData = new FormData();
  formData.append('image', imageFile);
  
  const response = await fetch(`${BASE_URL}/search?top_k=5`, {
    method: 'POST',
    body: formData
  });
  
  const results = await response.json();
  console.log('Search results:', results);
  return results;
}

// 2. Classify Attributes
async function classifyImage(imageFile) {
  const formData = new FormData();
  formData.append('image', imageFile);
  
  const response = await fetch(`${BASE_URL}/classify`, {
    method: 'POST',
    body: formData
  });
  
  const predictions = await response.json();
  console.log('Predictions:', predictions);
  return predictions;
}

// 3. Record Feedback
async function recordFeedback(productId, isRelevant, score) {
  const response = await fetch(`${BASE_URL}/feedback`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      product_id: productId,
      is_relevant: isRelevant,
      similarity_score: score,
      clicked: true
    })
  });
  
  const result = await response.json();
  console.log('Feedback recorded:', result);
  return result;
}

// Usage example
const fileInput = document.getElementById('imageUpload');
fileInput.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (file) {
    const results = await searchSimilar(file);
    displayResults(results);
  }
});
```

## Response Examples

### Search Response
```json
{
  "results": [
    {
      "id": 1,
      "image_path": "data/images/aviator_001.jpg",
      "brand": "RayBan",
      "model_name": "Aviator Classic",
      "price": 150.0,
      "material": "Metal",
      "frame_type": "Aviator",
      "color": "Gold",
      "rim_type": "Full-rim",
      "similarity_score": 0.9234,
      "boosted_score": 0.9465
    },
    {
      "id": 5,
      "image_path": "data/images/aviator_005.jpg",
      "brand": "Oakley",
      "model_name": "Aviator Sport",
      "price": 180.0,
      "material": "Titanium",
      "frame_type": "Aviator",
      "color": "Silver",
      "rim_type": "Full-rim",
      "similarity_score": 0.8876,
      "boosted_score": 0.8876
    }
  ],
  "total_results": 2,
  "search_time_ms": 42.3,
  "query_attributes": null
}
```

### Classify Response
```json
{
  "success": true,
  "predictions": {
    "frame_type": [
      ["Aviator", 0.45],
      ["Wayfarer", 0.25],
      ["Round", 0.15]
    ],
    "material": [
      ["Metal", 0.50],
      ["Titanium", 0.30],
      ["Acetate", 0.10]
    ],
    "color": [
      ["Gold", 0.60],
      ["Silver", 0.25],
      ["Black", 0.10]
    ],
    "rim_type": [
      ["Full-rim", 0.70],
      ["Semi-rimless", 0.20],
      ["Rimless", 0.10]
    ]
  }
}
```

## Error Handling

All endpoints return appropriate HTTP status codes:
- `200 OK`: Successful request
- `400 Bad Request`: Invalid input
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

Error response format:
```json
{
  "detail": "Error message describing what went wrong"
}
```

## Rate Limiting

Currently, there are no rate limits. In production, implement rate limiting to prevent abuse.

## Interactive API Documentation

FastAPI provides interactive API documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These interfaces allow you to test all endpoints directly from your browser.
