import streamlit as st
import sqlite3
import pandas as pd
import random
import hashlib
import base64
import json
import os
from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager
from datetime import datetime

# --- CONFIG ---
DB_PATH = "agentiq.db"
LOGO_PATH = "logo.png"
SESSION_STATE_KEY = "recommendation_history"

# --- DATABASE MANAGEMENT ---
@contextmanager
def get_db_connection():
    """Context manager for database connections to ensure proper closure."""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        yield conn
    finally:
        if conn:
            conn.close()

def execute_query(query: str, params: tuple = (), fetch_all: bool = True) -> Any:
    """Execute a database query with proper error handling."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            if fetch_all:
                return cursor.fetchall()
            return cursor.fetchone()
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
        return [] if fetch_all else None

# --- DATA FUNCTIONS ---
@st.cache_data(ttl=300)
def load_customer_ids() -> List[str]:
    """Load all customer IDs from the database with caching."""
    result = execute_query("SELECT Customer_ID FROM customers")
    return [row['Customer_ID'] for row in result] if result else []

def get_customer_details(customer_id: str) -> Optional[Dict]:
    """Get customer details by ID."""
    result = execute_query(
        "SELECT * FROM customers WHERE Customer_ID = ?", 
        (customer_id,), 
        fetch_all=False
    )
    return dict(result) if result else None

@st.cache_data(ttl=300)
def load_products() -> pd.DataFrame:
    """Load all products with caching and proper type conversion."""
    with get_db_connection() as conn:
        df = pd.read_sql("SELECT * FROM products", conn)
    
    # Convert price to proper numeric type
    try:
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        # Fill NaN values with 0 if any conversion failed
        df['Price'] = df['Price'].fillna(0)
    except Exception as e:
        st.warning(f"Warning: Could not convert all prices to numeric values. Some features may be affected. Error: {e}")
    
    return df

@st.cache_data(ttl=600)
def get_unique_categories() -> List[str]:
    """Get unique product categories with caching."""
    with get_db_connection() as conn:
        df = pd.read_sql("SELECT DISTINCT Category FROM products", conn)
    return ['All'] + sorted(df['Category'].dropna().unique().tolist())

def parse_browsing_history(history_str: str) -> List[str]:
    """Safely parse browsing history string."""
    if not history_str:
        return []
    
    try:
        # Use json.loads instead of eval for safety
        if history_str.startswith('[') and history_str.endswith(']'):
            return json.loads(history_str)
        else:
            return [history_str]
    except json.JSONDecodeError:
        # If not valid JSON, treat as a single category
        return [history_str] if history_str else []

def match_browsing_categories(customer: Dict, products: pd.DataFrame) -> pd.DataFrame:
    """Match products to customer browsing history."""
    browsing = parse_browsing_history(customer.get('Browsing_History', ''))
    
    if not browsing:
        return products
    
    matched = products[(products['Category'].isin(browsing)) | 
                      (products['Subcategory'].isin(browsing))]
    
    return matched if not matched.empty else products

def get_diverse_recommendations(customer_id: str, items_df: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    """Get diverse product recommendations with history tracking to avoid repetition."""
    if items_df.empty:
        return pd.DataFrame()
    
    # Initialize session state for tracking recommendation history if it doesn't exist
    if SESSION_STATE_KEY not in st.session_state:
        st.session_state[SESSION_STATE_KEY] = {}
    
    if customer_id not in st.session_state[SESSION_STATE_KEY]:
        st.session_state[SESSION_STATE_KEY][customer_id] = []
    
    # Get previously recommended product IDs for this customer
    previous_recommendations = st.session_state[SESSION_STATE_KEY][customer_id]
    
    # Filter out previously recommended products when possible
    available_products = items_df.copy()
    if previous_recommendations and len(available_products) > k:
        # Only filter if we have history and enough products to recommend
        available_products = available_products[~available_products['Product_ID'].isin(previous_recommendations)]
        
        # If filtering makes dataset too small, use original dataset
        if len(available_products) < k:
            available_products = items_df.copy()
    
    # If still more products than needed, create a diverse selection
    if len(available_products) > k:
        # Ensure diversity through multiple strategies
        seed = int(hashlib.md5((customer_id + datetime.now().strftime('%Y%m%d')).encode()).hexdigest(), 16) % 10000
        random.seed(seed)
        
        # Strategy 1: Get products from different categories if possible
        unique_categories = available_products['Category'].unique()
        selected_products = pd.DataFrame()
        
        if len(unique_categories) >= k:
            # We can select one product from each of k different categories
            for i, category in enumerate(random.sample(list(unique_categories), k)):
                category_products = available_products[available_products['Category'] == category]
                if not category_products.empty:
                    selected_product = category_products.sample(n=1, random_state=seed+i)
                    selected_products = pd.concat([selected_products, selected_product])
        
        # If strategy 1 failed to get k products, use strategy 2
        if len(selected_products) < k:
            # Strategy 2: Sample with price range diversity
            available_products['PriceGroup'] = pd.qcut(available_products['Price'], 
                                                   min(4, len(available_products['Price'].unique())), 
                                                   labels=False, 
                                                   duplicates='drop')
            price_groups = available_products['PriceGroup'].unique()
            selected_products = pd.DataFrame()
            
            # Try to get products from different price groups
            products_needed = k
            for price_group in price_groups:
                group_products = available_products[available_products['PriceGroup'] == price_group]
                samples_from_group = min(max(1, products_needed // len(price_groups)), len(group_products))
                if samples_from_group > 0:
                    group_selection = group_products.sample(n=samples_from_group, random_state=seed)
                    selected_products = pd.concat([selected_products, group_selection])
                    products_needed -= samples_from_group
            
            # If we still need more products, get random ones
            if products_needed > 0 and not available_products.empty:
                remaining = available_products[~available_products['Product_ID'].isin(selected_products['Product_ID'])]
                if not remaining.empty:
                    additional = remaining.sample(n=min(products_needed, len(remaining)), random_state=seed)
                    selected_products = pd.concat([selected_products, additional])
    else:
        # Not enough products for advanced selection, use all available
        selected_products = available_products
    
    # If we still don't have k products, just take what we can from the original dataframe
    if len(selected_products) < k and len(items_df) >= k:
        selected_products = items_df.sample(n=k, random_state=int(hashlib.md5(customer_id.encode()).hexdigest(), 16) % 10000)
    elif len(selected_products) < k:
        selected_products = items_df
    
    # Update recommendation history
    new_recommendations = selected_products['Product_ID'].tolist()
    st.session_state[SESSION_STATE_KEY][customer_id] = (previous_recommendations + new_recommendations)[-20:]  # Keep last 20
    
    return selected_products.head(k)

def generate_personalized_reason(product: Dict, customer: Dict) -> str:
    """Generate personalized recommendation reason."""
    browsing = parse_browsing_history(customer.get('Browsing_History', ''))
    
    # Check if product matches browsing history
    if product['Category'] in browsing or product['Subcategory'] in browsing:
        return f"🧠 You're into {product['Category']} — and this is a perfect match!"
    
    # Calculate personalization score (higher means more relevant)
    personalization_score = 0
    
    # Age-based recommendations
    age = customer.get('Age', 0)
    if age < 25 and product['Category'] in ['Electronics', 'Fashion']:
        personalization_score += 2
    elif 25 <= age < 40 and product['Category'] in ['Home & Kitchen', 'Electronics']:
        personalization_score += 2
    elif age >= 40 and product['Category'] in ['Health', 'Home & Kitchen']:
        personalization_score += 2
    
    # Location-based recommendations
    location = customer.get('Location', '')
    if location in ['Urban'] and product['Category'] in ['Electronics', 'Fashion']:
        personalization_score += 1
    elif location in ['Suburban'] and product['Category'] in ['Home & Kitchen', 'Garden']:
        personalization_score += 1
    elif location in ['Rural'] and product['Category'] in ['Tools', 'Outdoors']:
        personalization_score += 1
    
    # Customer segment-based recommendations
    segment = customer.get('Customer_Segment', '')
    try:
        price = float(product.get('Price', 0))
        if segment in ['Premium'] and price > 2000:
            personalization_score += 2
        elif segment in ['Budget'] and price < 1000:
            personalization_score += 2
    except (ValueError, TypeError):
        # Handle case where price isn't a valid number
        pass
    
    # Choose appropriate reason based on relevance score
    if personalization_score >= 3:
        return f"🎯 Perfect match for your {customer['Customer_Segment']} preferences in {customer['Location']}!"
    elif personalization_score >= 2:
        return f"💖 Chosen specifically for {customer['Customer_Segment']} shoppers like you!"
    elif personalization_score >= 1:
        return f"🛍 Trending among {customer['Location']} shoppers in your age group ({customer['Age']})!"
    else:
        # Generic fallback reasons
        reasons = [
            f"💫 This would be a great addition to your collection!",
            f"🌟 Highly rated by customers with similar shopping patterns.",
            f"✨ Something new to explore based on your interests!",
            f"🔍 Discovered by our AI just for you!"
        ]
        return random.choice(reasons)

# Diverse set of fun facts to avoid repetition
fun_facts = [
    "🎉 Top-rated by users who also loved your past picks!",
    "🚀 Flying off the shelves — blink and it's gone!",
    "💡 Gen Z-approved and AI-recommended!",
    "🌟 A 5-star pick for a 5-star shopper like you!",
    "✨ Featured in our monthly top-selling products!",
    "💯 9 out of 10 similar customers loved this item!",
    "🏆 Award-winning quality at a price you'll love!",
    "⚡ This item has been trending for the past week!",
    "🌈 A perfect addition to complete your collection!",
    "🧠 Smart shoppers are adding this to their carts!",
    "🎁 Makes an excellent gift for someone special!",
    "💪 Built to last with premium materials!",
    "🔥 One of our hottest items this season!",
    "🌱 Eco-friendly choice for sustainable shopping!",
    "💎 Hidden gem discovered by our recommendation algorithm!"
]

# --- UI COMPONENTS ---
def get_base64_image(image_path: str) -> str:
    """Convert an image to base64 encoding."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except FileNotFoundError:
        st.warning(f"Logo file not found at {image_path}. Using text header instead.")
        return ""

def render_header():
    """Render the app header with logo."""
    # Apply gradient background and styling
    st.markdown("""
        <style>
            .stApp {
                background: linear-gradient(135deg, #d299c2, #fef9d7);
            }    
            .css-18e3th9 {
                background-color: rgba(255, 255, 255, 0.05);
                padding: 2rem;
                border-radius: 1rem;
            }
            h1 {
                text-align: center;
                color: #5f0a87;
                font-family: 'Segoe UI', sans-serif;
                font-weight: bold;
            }
            .recommendation-card {
                background-color: rgba(255, 255, 255, 0.7);
                border-radius: 10px;
                padding: 15px;
                margin-bottom: 15px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                transition: transform 0.2s;
            }
            .recommendation-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 6px 8px rgba(0,0,0,0.15);
            }
            .price-tag {
                font-size: 1.4rem;
                font-weight: bold;
                color: #5f0a87;
            }
            .badge {
                display: inline-block;
                padding: 3px 8px;
                border-radius: 12px;
                font-size: 0.8rem;
                color: white;
                background: linear-gradient(45deg, #a4508b, #5f0a87);
                margin-right: 5px;
            }
            .customer-info {
                background-color: rgba(255, 255, 255, 0.8);
                border-radius: 10px;
                padding: 10px 15px;
                margin-bottom: 20px;
            }
            .filters-container {
                background-color: rgba(255, 255, 255, 0.5);
                border-radius: 10px;
                padding: 15px;
                margin-bottom: 20px;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Try to load and display logo
    logo_base64 = get_base64_image(LOGO_PATH)
    if logo_base64:
        st.markdown(f"""
            <div style='text-align: center;'>
                <img src='data:image/png;base64,{logo_base64}' 
                    style='width: 180px; height: 180px; border-radius: 50%; object-fit: cover; box-shadow: 0 4px 10px rgba(0,0,0,0.2);'/>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<h1>AgentIQ — Your Cart's Got PhD Now</h1>", unsafe_allow_html=True)

def render_product_card(product, customer):
    """Render a beautiful product recommendation card."""
    # Handle price formatting with proper error checking
    try:
        price = float(product['Price'])
        price_display = f"₹{price:.2f}"
    except (ValueError, TypeError):
        # If conversion fails, display as is
        price_display = f"₹{product['Price']}"
    
    st.markdown(f"""
    <div class="recommendation-card">
        <h3>🛍 {product['Subcategory']}</h3>
        <span class="price-tag">{price_display}</span>
        <div style="margin: 10px 0;">
            <span class="badge">{product['Category']}</span>
            <span>⭐ {product['Product_Rating']}</span>
        </div>
        <p><strong>Brand:</strong> {product['Brand']}</p>
        <p><em>{generate_personalized_reason(product, customer)}</em></p>
        <p>{random.choice(fun_facts)}</p>
    </div>
    """, unsafe_allow_html=True)

# --- MAIN APP ---
def main():
    """Main application function."""
    st.set_page_config(
        page_title="AgentIQ - Smart Shopping AI",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    render_header()
    
    # Load customer data
    customer_ids = load_customer_ids()
    
    # Layout for customer selection and filters
    col1, col2 = st.columns([2, 1])
    
    with col1:
        customer_id = st.selectbox("👤 Choose your Customer ID", customer_ids)
    
    with col2:
        st.write(" ")  # Spacing
        st.write(" ")  # Spacing
        show_filters = st.checkbox("🔧 Show Filters", value=False)
    
    # Filters section
    if show_filters:
        st.markdown('<div class="filters-container">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            price_range = st.slider("💰 Price Range (₹)", 50, 10000, (500, 5000), step=50)
            only_browsing = st.checkbox("📚 Match my Browsing History", value=True)
        
        with col2:
            categories = get_unique_categories()
            category_filter = st.selectbox("🔍 Choose a Category", categories)
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Default values when filters are hidden
        price_range = (50, 10000)
        category_filter = 'All'
        only_browsing = True
    
    # Button for recommendations
    if st.button("🎯 Get Smart Recommendations", type="primary"):
        with st.spinner("AI is finding perfect matches for you..."):
            customer = get_customer_details(customer_id)
            
            if not customer:
                st.error("Customer not found. Please select a valid customer ID.")
            else:
                # Display customer info
                st.markdown(f"""
                <div class="customer-info">
                    <h3>👤 {customer['Customer_ID']}</h3>
                    <p>Age: {customer['Age']} | Segment: {customer['Customer_Segment']} | Location: {customer['Location']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Filter products
                products = load_products()
                
                if only_browsing:
                    products = match_browsing_categories(customer, products)
                
                if category_filter != 'All':
                    products = products[products['Category'] == category_filter]
                
                # Convert price column to numeric and handle filtering properly
                products['Price'] = pd.to_numeric(products['Price'], errors='coerce')
                products = products.dropna(subset=['Price'])  # Remove rows with non-numeric prices
                products = products[(products['Price'] >= price_range[0]) & 
                                   (products['Price'] <= price_range[1])]
                
                if products.empty:
                    st.warning("No products found matching your filters! Try adjusting your filter criteria.")
                else:
                    # Get diverse recommendations
                    recommended = get_diverse_recommendations(customer_id, products, 4)
                    
                    if recommended.empty:
                        st.warning("No recommendations available. Try different filters.")
                    else:
                        st.markdown("### 🌟 Your Smart Recommendations")
                        
                        # Display recommendations in a responsive grid
                        for _, product in recommended.iterrows():
                            render_product_card(product, customer)
                        
                        # Add a "Try Again" button for more recommendations
                        if st.button("🔄 Get More Recommendations"):
                            st.experimental_rerun()

if __name__ == "__main__":
    main()
