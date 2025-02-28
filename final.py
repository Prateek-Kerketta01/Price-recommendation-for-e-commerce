from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import requests
from bs4 import BeautifulSoup
import pickle
import numpy as np

app = FastAPI()

with open('scaler.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

def scrape_flipkart():
    product_name = "cat food"
    url = f'https://www.flipkart.com/search?q={product_name}'

    headers = {
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "accept": "application/json",
        "accept-language": "en-US",
    }
    
    r = requests.get(url, headers=headers, verify=False)
    soup = BeautifulSoup(r.text, 'html.parser')
    item_card = soup.find('div', class_='slAVV4')
    
    if not item_card:
        return None, None

    price_card = item_card.find('div', class_='hl05eU')
    if not price_card:
        return None, None

    discount_price = price_card.find('div', class_='Nx9bqj')
    discount_price = discount_price.text if discount_price else "0"
    price_element = price_card.find('div', class_='yRaY8j')
    actual_price = price_element.text if price_element else "0"

    return float(actual_price.replace('₹', '').strip()), float(discount_price.replace('₹', '').strip())

def predict_price(actual_price, discount_price):
    price_diff = actual_price - discount_price
    features = np.array([[actual_price, price_diff]])
    scaled_features = features
    optimal_price = model.predict(scaled_features)[0]
    return max(0.9 * actual_price, min(optimal_price, 1.1 * actual_price))

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <body>
            <h2>Dynamic Pricing for Cat Food</h2>
            <form action="/get_price" method="get">
                <label for="product">Select Product:</label>
                <select id="product" name="product">
                    <option value="cat food">Cat Food</option>
                </select>
                <button type="submit">Get Price</button>
            </form>
        </body>
    </html>
    """

@app.get("/get_price", response_class=HTMLResponse)
def get_price():
    actual_price, discount_price = scrape_flipkart()
    
    if actual_price is None or discount_price is None:
        return "<h3>Product not found or unable to scrape prices.</h3>"
    
    optimal_price = predict_price(actual_price, discount_price)
    return (f"<h3>Flipkart Prices for 'Cat Food':</h3>"
            f"<p>Actual Price: ₹{actual_price:.2f}</p>"
            f"<p>Discounted Price: ₹{discount_price:.2f}</p>"
            f"<h3>Recommended Price: ₹{optimal_price:.2f}</h3>")
